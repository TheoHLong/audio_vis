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
const HISTORY_SECONDS = 15;

const LAYER_ORDER = ['L10', 'L6', 'L2'];
const LAYER_CONFIG = {
  L10: { label: 'L10 · semantics' },
  L6: { label: 'L6 · prosody' },
  L2: { label: 'L2 · voiceprint' },
};
const AUDIO_STYLE = { color: '#38bdf8', offset: 10, label: 'waveform' };

const HEATMAP_GRADIENT = [
  { stop: 0.0, color: [0, 0, 131] },
  { stop: 0.125, color: [0, 60, 170] },
  { stop: 0.375, color: [5, 255, 255] },
  { stop: 0.625, color: [255, 255, 0] },
  { stop: 0.875, color: [250, 0, 0] },
  { stop: 1.0, color: [128, 0, 0] },
];
const HEATMAP_SIDE_MARGIN = 100;
const HEATMAP_TOP_MARGIN = 100;
const HEATMAP_BOTTOM_RESERVE = 150;
const HEATMAP_BAND_GAP = 18;
const HEATMAP_MIN_BAND_HEIGHT = 48;

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
  
  // Add very subtle background gradient for depth
  const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
  bgGradient.addColorStop(0, 'rgba(0, 0, 0, 0)');
  bgGradient.addColorStop(1, 'rgba(10, 15, 30, 0.08)');
  ctx.fillStyle = bgGradient;
  ctx.fillRect(0, 0, width, height);

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

  activeLayers.forEach((layer, index) => {
    drawLayerHeatmap(width, height, layer, domain, LAYER_CONFIG[layer.name], index, activeLayers.length);
  });

  if (hasAudio) {
    drawAudioSurface(width, height, domain);
  }
}

function drawPlaceholder(width, height) {
  ctx.fillStyle = 'rgba(148, 197, 255, 0.25)';
  ctx.font = '14px "Inter", system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Speak to see neural activation ridges...', width / 2, height / 2);
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
  const margin = 100;
  const usableWidth = width - margin * 2;
  
  // Draw minimal vertical time markers
  ctx.strokeStyle = 'rgba(148, 197, 255, 0.05)';
  ctx.lineWidth = 0.5;
  const cols = 5; // Just 5 time markers
  for (let i = 1; i < cols; i += 1) {
    const ratio = i / cols;
    const x = margin + ratio * usableWidth;
    ctx.beginPath();
    ctx.moveTo(x, margin);
    ctx.lineTo(x, height - margin * 0.5);
    ctx.stroke();
  }
  
  // Draw axis lines
  ctx.strokeStyle = 'rgba(148, 197, 255, 0.1)';
  ctx.beginPath();
  // Vertical axis
  ctx.moveTo(margin, margin);
  ctx.lineTo(margin, height - margin * 0.5);
  // Horizontal baseline
  ctx.moveTo(margin, height - margin * 0.5);
  ctx.lineTo(margin + usableWidth, height - margin * 0.5);
  ctx.stroke();
}

function drawLayerHeatmap(width, height, layer, domain, config = {}, index = 0, totalLayers = 1) {
  const usableWidth = width - HEATMAP_SIDE_MARGIN * 2;
  const stats = domain.stats[layer.name];
  if (!stats || usableWidth <= 0) {
    return;
  }

  const layerTimes = Array.isArray(layer.times) ? layer.times : [];
  const layerVectors = Array.isArray(layer.vectors) ? layer.vectors : [];
  const timeCount = Math.min(layerTimes.length, layerVectors.length);
  if (!timeCount) {
    return;
  }

  const neuronCount = stats.neuronCount;
  if (!neuronCount) {
    return;
  }

  const heatmapBottomLimit = height - HEATMAP_BOTTOM_RESERVE;
  const totalGap = HEATMAP_BAND_GAP * Math.max(0, totalLayers - 1);
  const rawBandHeight = (heatmapBottomLimit - HEATMAP_TOP_MARGIN - totalGap) / Math.max(totalLayers, 1);
  const bandHeight = Math.max(HEATMAP_MIN_BAND_HEIGHT, rawBandHeight);
  const totalHeight = bandHeight * totalLayers + totalGap;
  let bandTopStart = HEATMAP_TOP_MARGIN;
  if (bandTopStart + totalHeight > heatmapBottomLimit) {
    bandTopStart = Math.max(20, heatmapBottomLimit - totalHeight);
  }
  const top = bandTopStart + index * (bandHeight + HEATMAP_BAND_GAP);
  const left = HEATMAP_SIDE_MARGIN;

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, usableWidth, bandHeight);
  ctx.fillStyle = 'rgba(10, 18, 36, 0.52)';
  ctx.fill();
  ctx.clip();

  const timeRange = Math.max(1e-6, domain.maxTime - domain.minTime);
  const targetCols = Math.min(timeCount, 320);
  const targetRows = Math.min(neuronCount, 120);
  const timeStride = Math.max(1, Math.floor(timeCount / targetCols));
  const neuronStride = Math.max(1, Math.floor(neuronCount / targetRows));

  for (let t = 0; t < timeCount; t += timeStride) {
    const timeIndexEnd = Math.min(timeCount - 1, t + timeStride);
    const timeStart = layerTimes[t];
    const timeEnd = layerTimes[timeIndexEnd];
    let x0 = left + ((timeStart - domain.minTime) / timeRange) * usableWidth;
    let x1 = left + ((timeEnd - domain.minTime) / timeRange) * usableWidth;
    if (!Number.isFinite(x0) || !Number.isFinite(x1)) {
      continue;
    }
    if (x1 <= x0) {
      x1 = x0 + Math.max(1, usableWidth / timeCount);
    }
    const cellWidth = x1 - x0;

    for (let n = 0; n < neuronCount; n += neuronStride) {
      const neuronEnd = Math.min(neuronCount, n + neuronStride);
      let accum = 0;
      let count = 0;
      for (let ti = t; ti <= timeIndexEnd; ti += 1) {
        const vector = layerVectors[ti];
        if (!vector) {
          continue;
        }
        for (let ni = n; ni < neuronEnd; ni += 1) {
          const value = vector[ni];
          if (Number.isFinite(value)) {
            accum += value;
            count += 1;
          }
        }
      }
      if (!count) {
        continue;
      }
      const average = accum / count;
      const norm = Math.min(1, Math.max(0, (average - stats.min) / (stats.max - stats.min)));
      let y0 = top + (n / neuronCount) * bandHeight;
      let y1 = top + (neuronEnd / neuronCount) * bandHeight;
      if (y1 <= y0) {
        y1 = y0 + bandHeight / neuronCount;
      }
      const cellHeight = y1 - y0;
      ctx.fillStyle = heatmapColor(norm);
      ctx.fillRect(x0, y0, cellWidth, cellHeight);
    }
  }

  // Overlay a subtle gloss to keep the heatmap grounded in the scene.
  const gloss = ctx.createLinearGradient(0, top, 0, top + bandHeight);
  gloss.addColorStop(0, 'rgba(255, 255, 255, 0.04)');
  gloss.addColorStop(1, 'rgba(12, 16, 32, 0.55)');
  ctx.fillStyle = gloss;
  ctx.fillRect(left, top, usableWidth, bandHeight);

  ctx.restore();

  ctx.strokeStyle = 'rgba(148, 197, 255, 0.22)';
  ctx.lineWidth = 0.8;
  ctx.strokeRect(left, top, usableWidth, bandHeight);

  ctx.beginPath();
  ctx.moveTo(left - 6, top);
  ctx.lineTo(left - 6, top + bandHeight);
  ctx.strokeStyle = 'rgba(148, 197, 255, 0.18)';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.save();
  ctx.fillStyle = 'rgba(220, 230, 245, 0.9)';
  ctx.font = '12px "Inter", system-ui';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
  ctx.shadowBlur = 2;
  ctx.fillText(config.label || layer.name, left - 12, top + bandHeight / 2);
  ctx.restore();
}

function heatmapColor(value) {
  const t = Math.min(1, Math.max(0, value));
  for (let i = 0; i < HEATMAP_GRADIENT.length - 1; i += 1) {
    const current = HEATMAP_GRADIENT[i];
    const next = HEATMAP_GRADIENT[i + 1];
    if (t <= next.stop) {
      const span = next.stop - current.stop || 1;
      const localT = Math.min(1, Math.max(0, (t - current.stop) / span));
      const r = Math.round(current.color[0] + (next.color[0] - current.color[0]) * localT);
      const g = Math.round(current.color[1] + (next.color[1] - current.color[1]) * localT);
      const b = Math.round(current.color[2] + (next.color[2] - current.color[2]) * localT);
      return `rgba(${r}, ${g}, ${b}, 0.9)`;
    }
  }
  const last = HEATMAP_GRADIENT[HEATMAP_GRADIENT.length - 1];
  return `rgba(${last.color[0]}, ${last.color[1]}, ${last.color[2]}, 0.9)`;
}

function drawAudioSurface(width, height, domain) {
  const margin = 100;
  const usableWidth = width - margin * 2;
  const baseline = height - margin + AUDIO_STYLE.offset;
  const amplitude = 40; // Increased for better visibility

  // Draw audio as a simple filled waveform
  ctx.beginPath();
  ctx.moveTo(margin, baseline);
  
  state.audio.times.forEach((time, idx) => {
    const x = margin + ((time - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
    const y = baseline - state.audio.rms[idx] * amplitude;
    
    if (idx === 0) {
      ctx.lineTo(x, y);
    } else {
      // Smooth the line
      const prevX = margin + ((state.audio.times[idx-1] - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
      const cpX = (prevX + x) / 2;
      const prevY = baseline - state.audio.rms[idx-1] * amplitude;
      const cpY = (prevY + y) / 2;
      ctx.quadraticCurveTo(cpX, cpY, x, y);
    }
  });
  
  ctx.lineTo(margin + usableWidth, baseline);
  ctx.closePath();
  
  // Fill with stronger gradient for better visibility
  const gradient = ctx.createLinearGradient(0, baseline - amplitude, 0, baseline);
  gradient.addColorStop(0, hexToRgba(AUDIO_STYLE.color, 0.6));
  gradient.addColorStop(0.5, hexToRgba(AUDIO_STYLE.color, 0.3));
  gradient.addColorStop(1, hexToRgba(AUDIO_STYLE.color, 0.05));
  ctx.fillStyle = gradient;
  ctx.fill();
  
  // Add stronger outline
  ctx.strokeStyle = hexToRgba(AUDIO_STYLE.color, 0.5);
  ctx.lineWidth = 1;
  ctx.stroke();

  // Draw label - position to be fully visible
  ctx.save();
  ctx.fillStyle = 'rgba(220, 230, 245, 0.9)';
  ctx.font = '12px "Inter", system-ui';
  ctx.textAlign = 'right';
  const labelX = margin - 8;
  const labelY = baseline - 5; // Position just above the baseline
  // Add subtle text shadow for better readability
  ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
  ctx.shadowBlur = 2;
  ctx.fillText(AUDIO_STYLE.label, labelX, labelY);
  ctx.textAlign = 'left';
  ctx.restore();
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
