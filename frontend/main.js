const canvas = document.getElementById('comet-canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const modeBtn = document.getElementById('mode-btn');
const resetBtn = document.getElementById('reset-btn');
const linksBtn = document.getElementById('projection-btn');
const diagnosticsList = document.getElementById('diagnostics');
const modePill = document.getElementById('mode-pill');
const transcriptText = document.getElementById('transcript-text');

const state = {
  layers: [],
  audio: { times: [], rms: [] },
  speakerColors: {},
  diagnostics: {},
  transcript: '',
  mode: 'analysis',
  drawLinks: true,
};

const RIDGE_LAYERS = ['L10', 'L6', 'L2'];
const LAYER_STYLES = {
  L2: { baseColor: '#22d3ee', label: 'Layer 2 · voiceprint' },
  L6: { baseColor: '#f97316', label: 'Layer 6 · prosody' },
  L10: { baseColor: '#a855f7', label: 'Layer 10 · semantics' },
};
const AUDIO_STYLE = { baseColor: '#38bdf8', label: 'Audio waveform' };

let ws = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let listening = false;

const TARGET_SAMPLE_RATE = 16_000;

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

async function ensureSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/audio`;
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    console.log('WebSocket connected');
  };

  ws.onmessage = (event) => {
    handleMessage(event.data);
  };

  ws.onerror = (err) => {
    console.error('WebSocket error', err);
  };

  ws.onclose = () => {
    listening = false;
    updateButtons();
    setTimeout(() => ensureSocket().catch(() => undefined), 1500);
  };
}

async function startListening() {
  try {
    await ensureSocket();
    if (!mediaStream) {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: TARGET_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false,
      });
    }

    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(2048, 1, 1);

    const gainNode = audioContext.createGain();
    gainNode.gain.value = 0;

    processorNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const down = downsample(input, audioContext.sampleRate, TARGET_SAMPLE_RATE);
      if (down && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(down.buffer);
      }
    };

    sourceNode.connect(processorNode);
    processorNode.connect(gainNode);
    gainNode.connect(audioContext.destination);

    listening = true;
    updateButtons();
  } catch (err) {
    console.error('Failed to start listening', err);
    alert('Microphone permission or audio initialisation failed. Check console for details.');
  }
}

function stopListening() {
  listening = false;
  updateButtons();
  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

function downsample(buffer, inRate, outRate) {
  if (!buffer || inRate === outRate) {
    return Float32Array.from(buffer);
  }
  const ratio = inRate / outRate;
  const newLength = Math.round(buffer.length / ratio);
  if (!Number.isFinite(newLength) || newLength <= 0) {
    return null;
  }
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function handleMessage(message) {
  let data;
  try {
    data = typeof message === 'string' ? JSON.parse(message) : JSON.parse(new TextDecoder().decode(message));
  } catch (err) {
    console.warn('Failed to parse message', err);
    return;
  }
  if (data.type === 'hello') {
    return;
  }
  if (data.type === 'mode') {
    state.mode = data.mode || state.mode;
    updateModeUI();
    return;
  }
  if (data.type === 'layer_activity') {
    state.layers = data.layers || [];
    state.audio = data.audio || { times: [], rms: [] };
    state.diagnostics = data.meta?.diagnostics || {};
    state.mode = data.meta?.mode || state.mode;
    state.transcript = data.meta?.transcript || '';
    state.speakerColors = data.meta?.speaker_colors || {};
    updateTranscript();
    updateDiagnostics();
    updateModeUI();
    return;
  }
}

function updateTranscript() {
  const trimmed = state.transcript && state.transcript.trim();
  const text = trimmed && trimmed.length
    ? trimmed
    : 'Enable whisper-tiny to see live transcripts.';
  transcriptText.textContent = text;
}

function updateDiagnostics() {
  diagnosticsList.innerHTML = '';
  const entries = [
    { label: 'Semantic plane', value: state.diagnostics.projector_ready },
    { label: 'Speaker clusters', value: state.diagnostics.speaker_ready },
    { label: 'Whisper', value: state.diagnostics.keyword_ready },
  ];
  entries.forEach((entry) => {
    const li = document.createElement('li');
    const ok = Boolean(entry.value);
    li.innerHTML = `<span style="color:${ok ? 'rgba(125, 211, 252, 0.95)' : 'rgba(251, 191, 36, 0.85)'};font-weight:600">●</span> ${entry.label}: ${ok ? 'online' : 'warming'}`;
    diagnosticsList.appendChild(li);
  });
}

function updateModeUI() {
  const nextMode = state.mode === 'analysis' ? 'performance' : 'analysis';
  modeBtn.textContent = `${nextMode.charAt(0).toUpperCase() + nextMode.slice(1)} Mode`;
  modePill.textContent = `Mode: ${state.mode.charAt(0).toUpperCase() + state.mode.slice(1)}`;
  linksBtn.textContent = state.drawLinks ? 'Hide Grid' : 'Show Grid';
}

function updateButtons() {
  startBtn.textContent = listening ? 'Stop Listening' : 'Start Listening';
  startBtn.classList.toggle('primary', !listening);
  startBtn.classList.toggle('ghost', listening);
}

startBtn.addEventListener('click', () => {
  if (!listening) {
    startListening();
  } else {
    stopListening();
  }
});

modeBtn.addEventListener('click', () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  const nextMode = state.mode === 'analysis' ? 'performance' : 'analysis';
  ws.send(JSON.stringify({ type: 'mode', mode: nextMode }));
});

linksBtn.addEventListener('click', () => {
  state.drawLinks = !state.drawLinks;
  updateModeUI();
});

resetBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
  state.layers = [];
  state.transcript = '';
  updateTranscript();
});

let lastFrame = performance.now();
function render(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  drawScene(dt);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

function drawScene() {
  const width = canvas.width / (window.devicePixelRatio || 1);
  const height = canvas.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, width, height);

  const activeLayers = RIDGE_LAYERS.map((name) => state.layers.find((layer) => layer.name === name)).filter(Boolean);
  const hasAudio = Array.isArray(state.audio?.times) && state.audio.times.length > 0;

  if (!activeLayers.length && !hasAudio) {
    drawPlaceholder(width, height);
    return;
  }

  const domain = computeDomain(activeLayers, state.audio);
  const margin = 60;
  const totalRows = activeLayers.length + (hasAudio ? 1 : 0);
  const rowHeight = (height - margin * 2) / Math.max(totalRows, 1);

  if (state.drawLinks) {
    drawGuides(width, height, domain, margin);
  }

  let rowIndex = 0;
  activeLayers.forEach((layer, idx) => {
    const baseline = margin + rowIndex * rowHeight + rowHeight * 0.65;
    drawRidgeLayer(width, height, layer, domain, baseline, rowHeight, idx);
    rowIndex += 1;
  });

  if (hasAudio) {
    const baseline = margin + rowIndex * rowHeight + rowHeight * 0.65;
    drawAudioRidge(width, height, domain, baseline, rowHeight);
  }
}

function drawPlaceholder(width, height) {
  ctx.fillStyle = 'rgba(148, 197, 255, 0.4)';
  ctx.font = '14px "Inter", system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Speak to see neuron trajectories…', width / 2, height / 2);
  ctx.textAlign = 'left';
}

function computeDomain(layers, audio) {
  let minTime = Number.POSITIVE_INFINITY;
  let maxTime = 0;
  layers.forEach((layer) => {
    if (!layer.times.length) {
      return;
    }
    minTime = Math.min(minTime, layer.times[0]);
    maxTime = Math.max(maxTime, layer.times[layer.times.length - 1]);
  });
  if (audio?.times?.length) {
    minTime = Math.min(minTime, audio.times[0]);
    maxTime = Math.max(maxTime, audio.times[audio.times.length - 1]);
  }
  layers.forEach((layer) => {
    if (!layer.times.length) {
      return;
    }
    layer._min = Math.min(...layer.activities);
    layer._max = Math.max(...layer.activities);
    if (Math.abs(layer._max - layer._min) < 1e-6) {
      layer._max = layer._min + 1;
    }
  });
  if (!Number.isFinite(minTime)) {
    minTime = 0;
  }
  if (maxTime <= minTime) {
    maxTime = minTime + 1;
  }
  return { minTime, maxTime };
}

function drawGuides(width, height, domain, margin) {
  const usableWidth = width - margin * 2;
  ctx.strokeStyle = 'rgba(148, 197, 255, 0.18)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  const segments = 6;
  for (let i = 0; i <= segments; i += 1) {
    const ratio = i / segments;
    const x = margin + ratio * usableWidth;
    ctx.beginPath();
    ctx.moveTo(x, margin * 0.7);
    ctx.lineTo(x, height - margin * 0.6);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawRidgeLayer(width, height, layer, domain, baseline, rowHeight, index) {
  const margin = 60;
  const usableWidth = width - margin * 2;
  const amplitude = rowHeight * 0.4;
  const colorBase = LAYER_STYLES[layer.name]?.baseColor || '#cbd5f5';
  const label = LAYER_STYLES[layer.name]?.label || layer.name;
  const colorsBySpeaker = state.speakerColors || {};

  const minVal = layer._min ?? Math.min(...layer.activities);
  const maxVal = layer._max ?? Math.max(...layer.activities);

  ctx.lineWidth = 2;
  for (let i = 1; i < layer.times.length; i += 1) {
    const t0 = layer.times[i - 1];
    const t1 = layer.times[i];
    const x0 = margin + ((t0 - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
    const x1 = margin + ((t1 - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
    const y0 = baseline - ((layer.activities[i - 1] - minVal) / (maxVal - minVal) - 0.5) * amplitude * 2;
    const y1 = baseline - ((layer.activities[i] - minVal) / (maxVal - minVal) - 0.5) * amplitude * 2;
    const speaker = layer.speakers[i] ?? layer.speakers[i - 1];
    const color = colorsBySpeaker[String(speaker)] || colorBase;
    ctx.strokeStyle = applyAlpha(color, state.drawLinks ? 0.95 : 0.8);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }

  layer.times.forEach((time, idx) => {
    const x = margin + ((time - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
    const y = baseline - ((layer.activities[idx] - minVal) / (maxVal - minVal) - 0.5) * amplitude * 2;
    const speaker = layer.speakers[idx];
    const color = colorsBySpeaker[String(speaker)] || colorBase;
    ctx.fillStyle = applyAlpha(color, 0.9);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
  ctx.font = '12px "Inter", system-ui';
  ctx.textAlign = 'right';
  ctx.fillText(label, margin - 15, baseline - rowHeight * 0.25);
  ctx.textAlign = 'left';
}

function drawAudioRidge(width, height, domain, baseline, rowHeight) {
  const margin = 60;
  const usableWidth = width - margin * 2;
  const amplitude = rowHeight * 0.45;
  const times = state.audio.times || [];
  const values = state.audio.rms || [];
  if (!times.length || !values.length) {
    return;
  }
  const minVal = 0;
  const maxVal = 1;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = applyAlpha(AUDIO_STYLE.baseColor, 0.7);
  ctx.beginPath();
  times.forEach((time, idx) => {
    const x = margin + ((time - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
    const y = baseline - (values[idx] - minVal) / (maxVal - minVal) * amplitude;
    if (idx === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  ctx.fillStyle = 'rgba(148, 197, 255, 0.35)';
  ctx.textAlign = 'right';
  ctx.fillText(AUDIO_STYLE.label, margin - 15, baseline - rowHeight * 0.25);
  ctx.textAlign = 'left';
}
function applyAlpha(color, alpha) {
  if (!color || !color.startsWith('#')) {
    return `rgba(255,255,255,${alpha})`;
  }
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

updateButtons();
updateModeUI();
updateDiagnostics();
updateTranscript();
