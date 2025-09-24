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
  speakerColors: {},
  diagnostics: {},
  transcript: '',
  mode: 'analysis',
  drawLinks: true,
};

const LAYER_STYLES = {
  L2: { baseColor: '#22d3ee' },
  L6: { baseColor: '#f97316' },
  L10: { baseColor: '#a855f7' },
};

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

function drawScene(dt) {
  const width = canvas.width / (window.devicePixelRatio || 1);
  const height = canvas.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, width, height);

  drawAxes(width, height);

  if (!state.layers.length) {
    drawPlaceholder(width, height);
    return;
  }

  const domain = computeDomain(state.layers);
  state.layers.forEach((layer) => {
    drawLayer(width, height, layer, domain);
  });
}

function drawAxes(width, height) {
  const margin = 60;
  if (state.drawLinks) {
    ctx.strokeStyle = 'rgba(148, 197, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(margin, height - margin);
    ctx.lineTo(width - margin, height - margin);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(margin, height - margin);
    ctx.lineTo(margin, margin);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(margin, height - margin);
    ctx.lineTo(margin + 80, margin + 40);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = 'rgba(226, 232, 240, 0.7)';
  ctx.font = '12px "Inter", system-ui';
  ctx.fillText('Time', width - margin - 40, height - margin + 20);
  ctx.save();
  ctx.translate(margin - 25, margin + (height - 2 * margin) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Neuron index', 0, 0);
  ctx.restore();
  ctx.fillText('Activity', margin + 70, margin + 10);
}

function drawPlaceholder(width, height) {
  ctx.fillStyle = 'rgba(148, 197, 255, 0.4)';
  ctx.font = '14px "Inter", system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Speak to see neuron trajectories…', width / 2, height / 2);
}

function computeDomain(layers) {
  let minTime = Number.POSITIVE_INFINITY;
  let maxTime = 0;
  let maxIndex = 0;
  let minActivity = Number.POSITIVE_INFINITY;
  let maxActivity = Number.NEGATIVE_INFINITY;
  layers.forEach((layer) => {
    const times = layer.times;
    if (!times.length) {
      return;
    }
    minTime = Math.min(minTime, times[0]);
    maxTime = Math.max(maxTime, times[times.length - 1]);
    layer.indices.forEach((idx) => {
      maxIndex = Math.max(maxIndex, idx);
    });
    layer.activities.forEach((val) => {
      minActivity = Math.min(minActivity, val);
      maxActivity = Math.max(maxActivity, val);
    });
  });
  if (!Number.isFinite(minTime)) {
    minTime = 0;
  }
  if (maxTime <= minTime) {
    maxTime = minTime + 1;
  }
  if (maxIndex <= 0) {
    maxIndex = 1;
  }
  if (!Number.isFinite(minActivity) || !Number.isFinite(maxActivity)) {
    minActivity = -0.5;
    maxActivity = 0.5;
  }
  if (Math.abs(maxActivity - minActivity) < 1e-6) {
    maxActivity = minActivity + 1;
  }
  return { minTime, maxTime, maxIndex, minActivity, maxActivity };
}

function drawLayer(width, height, layer, domain) {
  const margin = 60;
  const axisWidth = width - margin * 2;
  const axisHeight = height - margin * 2;
  const depth = axisWidth * 0.18;
  const angle = Math.PI / 6;
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  const colorBase = LAYER_STYLES[layer.name]?.baseColor || '#cbd5f5';
  const colorsBySpeaker = state.speakerColors || {};

  const points = layer.times.map((t, idx) => {
    const timeNorm = (t - domain.minTime) / (domain.maxTime - domain.minTime);
    const neuronNorm = Math.min(1, Math.max(0, layer.indices[idx] / (domain.maxIndex || 1)));
    const activity = layer.activities[idx];
    const activityNorm = (activity - domain.minActivity) / (domain.maxActivity - domain.minActivity);
    const x = margin + timeNorm * axisWidth + neuronNorm * depth * cosA;
    const y = height - margin - activityNorm * axisHeight + neuronNorm * depth * sinA;
    return {
      x,
      y,
      speaker: layer.speakers[idx],
    };
  });

  if (points.length < 2) {
    return;
  }

  ctx.lineWidth = 2;
  for (let i = 1; i < points.length; i += 1) {
    const from = points[i - 1];
    const to = points[i];
    const speaker = to.speaker ?? from.speaker;
    const color = colorsBySpeaker[String(speaker)] || colorsBySpeaker[speaker] || colorBase;
    const alpha = state.drawLinks ? 0.9 : 0.7;
    ctx.strokeStyle = applyAlpha(color, alpha);
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }

  points.forEach((pt) => {
    ctx.fillStyle = applyAlpha(colorsBySpeaker[String(pt.speaker)] || colorBase, 0.9);
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  });
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
