const canvas = document.getElementById('activity-canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const modeBtn = document.getElementById('mode-btn');
const resetBtn = document.getElementById('reset-btn');
const gridBtn = document.getElementById('projection-btn');
const diagnosticsList = document.getElementById('diagnostics');
const modePill = document.getElementById('mode-pill');
const transcriptText = document.getElementById('transcript-text');

const TARGET_SAMPLE_RATE = 16_000;
const MAX_NEURON_SAMPLES = 48;
const HISTORY_SECONDS = 15;

const LAYER_ORDER = ['L10', 'L6', 'L2'];
const LAYER_CONFIG = {
  L10: { color: '#a855f7', offset: 90, label: 'Layer 10 · semantics' },
  L6: { color: '#f97316', offset: 55, label: 'Layer 6 · prosody' },
  L2: { color: '#22d3ee', offset: 20, label: 'Layer 2 · voiceprint' },
};
const AUDIO_STYLE = { color: '#38bdf8', offset: -10, label: 'Audio waveform' };

const state = {
  layers: [],
  audio: { times: [], rms: [] },
  speakerColors: {},
  diagnostics: {},
  transcript: '',
  mode: 'analysis',
  showGrid: true,
};

let ws = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let listening = false;

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
  }
}

window.addEventListener('resize', () => {
  resizeCanvas();
  renderScene();
});
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
  ws.onmessage = (event) => handleMessage(event.data);
  ws.onerror = (err) => console.error('WebSocket error', err);
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
    renderScene();
    return;
  }
}

function updateTranscript() {
  const trimmed = state.transcript && state.transcript.trim();
  transcriptText.textContent = trimmed && trimmed.length
    ? trimmed
    : 'Enable whisper-tiny to see live transcripts.';
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
  gridBtn.textContent = state.showGrid ? 'Hide Grid' : 'Show Grid';
}

function updateButtons() {
  startBtn.textContent = listening ? 'Stop Listening' : 'Start Listening';
  startBtn.classList.toggle('primary', !listening);
  startBtn.classList.toggle('ghost', listening);
}

function renderScene() {
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  const activeLayers = LAYER_ORDER
    .map((name) => state.layers.find((layer) => layer.name === name))
    .filter((layer) => layer && layer.times?.length && layer.vectors?.length);
  const hasAudio = state.audio?.times?.length;

  if (!activeLayers.length && !hasAudio) {
    drawPlaceholder(width, height);
    return;
  }

  const domain = computeDomain(activeLayers, state.audio);
  if (state.showGrid) {
    drawGrid(width, height, domain);
  }

  activeLayers.forEach((layer) => {
    drawLayerSurface(width, height, layer, domain, LAYER_CONFIG[layer.name]);
  });

  if (hasAudio) {
    drawAudioSurface(width, height, domain);
  }
}

function drawPlaceholder(width, height) {
  ctx.fillStyle = 'rgba(148, 197, 255, 0.35)';
  ctx.font = '15px "Inter", system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Speak to see neural trajectories…', width / 2, height / 2);
  ctx.textAlign = 'left';
}

function computeDomain(layers, audio) {
  let minTime = Number.POSITIVE_INFINITY;
  let maxTime = 0;
  const stats = {};
  let maxNeurons = 0;

  layers.forEach((layer) => {
    if (!layer.times.length || !layer.vectors.length) {
      return;
    }
    minTime = Math.min(minTime, layer.times[0]);
    maxTime = Math.max(maxTime, layer.times[layer.times.length - 1]);
    const sampleVector = layer.vectors[0] || [];
    if (!sampleVector.length) {
      return;
    }
    maxNeurons = Math.max(maxNeurons, sampleVector.length);
    const flat = layer.vectors.flat();
    let minVal = Math.min(...flat);
    let maxVal = Math.max(...flat);
    if (Math.abs(maxVal - minVal) < 1e-6) {
      maxVal = minVal + 1e-3;
    }
    stats[layer.name] = {
      min: minVal,
      max: maxVal,
      neuronCount: sampleVector.length,
    };
  });

  if (audio?.times?.length) {
    minTime = Math.min(minTime, audio.times[0]);
    maxTime = Math.max(maxTime, audio.times[audio.times.length - 1]);
  }
  if (!Number.isFinite(minTime)) {
    minTime = 0;
  }
  if (maxTime <= minTime) {
    maxTime = minTime + 1;
  }
  return { minTime, maxTime, stats, maxNeurons: Math.max(maxNeurons, 1) };
}

function drawGrid(width, height, domain) {
  const margin = 70;
  const usableWidth = width - margin * 2;
  ctx.strokeStyle = 'rgba(148, 197, 255, 0.16)';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 7]);
  const cols = 6;
  for (let i = 0; i <= cols; i += 1) {
    const ratio = i / cols;
    const x = margin + ratio * usableWidth;
    ctx.beginPath();
    ctx.moveTo(x, margin * 0.6);
    ctx.lineTo(x, height - margin * 0.5);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawLayerSurface(width, height, layer, domain, config = {}) {
  const margin = 70;
  const usableWidth = width - margin * 2;
  const depth = 90;
  const amplitude = 80;
  const angle = Math.PI / 6;
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const stats = domain.stats[layer.name];
  if (!stats) {
    return;
  }
  const baseline = height - margin - config.offset;
  const neuronCount = stats.neuronCount;
  const sampleStep = Math.max(1, Math.floor(neuronCount / MAX_NEURON_SAMPLES));

  ctx.globalAlpha = 0.3;
  layer.vectors.forEach((vector, timeIndex) => {
    const timeNorm = (layer.times[timeIndex] - domain.minTime) / (domain.maxTime - domain.minTime);
    ctx.beginPath();
    for (let neuron = 0; neuron < neuronCount; neuron += sampleStep) {
      const colNorm = neuronCount > 1 ? neuron / (neuronCount - 1) : 0;
      const activity = vector[neuron];
      const normActivity = (activity - stats.min) / (stats.max - stats.min);
      const x = margin + timeNorm * usableWidth + colNorm * depth * cosA;
      const y = baseline + colNorm * depth * sinA - (normActivity - 0.5) * amplitude;
      if (neuron === 0) {
        ctx.moveTo(x, baseline + colNorm * depth * sinA);
        ctx.lineTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.lineTo(margin + timeNorm * usableWidth + depth * cosA, baseline + depth * sinA);
    ctx.closePath();
    const gradient = ctx.createLinearGradient(0, baseline - amplitude, 0, baseline + depth * sinA);
    gradient.addColorStop(0, hexToRgba(config.color || '#cbd5f5', 0.55));
    gradient.addColorStop(1, hexToRgba(config.color || '#cbd5f5', 0.05));
    ctx.fillStyle = gradient;
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  ctx.lineWidth = 2;
  const colorsBySpeaker = state.speakerColors || {};
  for (let neuron = 0; neuron < neuronCount; neuron += sampleStep) {
    const colNorm = neuronCount > 1 ? neuron / (neuronCount - 1) : 0;
    ctx.beginPath();
    layer.vectors.forEach((vector, timeIndex) => {
      const timeNorm = (layer.times[timeIndex] - domain.minTime) / (domain.maxTime - domain.minTime);
      const activity = vector[neuron];
      const normActivity = (activity - stats.min) / (stats.max - stats.min);
      const speaker = layer.speakers[timeIndex];
      const color = colorsBySpeaker[String(speaker)] || config.color || '#cbd5f5';
      const x = margin + timeNorm * usableWidth + colNorm * depth * cosA;
      const y = baseline + colNorm * depth * sinA - (normActivity - 0.5) * amplitude;
      ctx.strokeStyle = hexToRgba(color, state.showGrid ? 0.95 : 0.8);
      if (timeIndex === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }

  ctx.fillStyle = hexToRgba('#e2e8f0', 0.65);
  ctx.font = '12px "Inter", system-ui';
  ctx.textAlign = 'right';
  ctx.fillText(config.label || layer.name, margin - 18, baseline - 8);
  ctx.textAlign = 'left';
}

function drawAudioSurface(width, height, domain) {
  const margin = 70;
  const usableWidth = width - margin * 2;
  const baseline = height - margin + AUDIO_STYLE.offset;
  const amplitude = 35;

  ctx.lineWidth = 2;
  ctx.strokeStyle = hexToRgba(AUDIO_STYLE.color, 0.8);
  ctx.beginPath();
  state.audio.times.forEach((time, idx) => {
    const x = margin + ((time - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
    const y = baseline - (state.audio.rms[idx] - 0.5) * amplitude * 2;
    if (idx === 0) {
      ctx.moveTo(x, baseline);
      ctx.lineTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.lineTo(margin + usableWidth, baseline);
  ctx.stroke();
  ctx.fillStyle = hexToRgba(AUDIO_STYLE.color, 0.15);
  ctx.fill();

  ctx.fillStyle = 'rgba(148, 197, 255, 0.65)';
  ctx.font = '12px "Inter", system-ui';
  ctx.textAlign = 'right';
  ctx.fillText(AUDIO_STYLE.label, margin - 18, baseline - 6);
  ctx.textAlign = 'left';
}

function hexToRgba(hex, alpha) {
  const clean = hex.replace('#', '');
  const bigint = parseInt(clean, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function btnGuarded(handler) {
  return () => {
    const result = handler();
    if (result && typeof result.catch === 'function') {
      result.catch((err) => console.error(err));
    }
  };
}

modeBtn.addEventListener('click', btnGuarded(async () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  const nextMode = state.mode === 'analysis' ? 'performance' : 'analysis';
  ws.send(JSON.stringify({ type: 'mode', mode: nextMode }));
}));

startBtn.addEventListener('click', btnGuarded(async () => {
  if (!listening) {
    await startListening();
  } else {
    stopListening();
  }
}));

gridBtn.addEventListener('click', () => {
  state.showGrid = !state.showGrid;
  updateModeUI();
  renderScene();
});

resetBtn.addEventListener('click', btnGuarded(async () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
  state.layers = [];
  state.audio = { times: [], rms: [] };
  renderScene();
  updateTranscript();
}));

updateButtons();
updateModeUI();
updateDiagnostics();
updateTranscript();
renderScene();

ensureSocket().catch(() => undefined);
