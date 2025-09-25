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
  L10: { color: '#a855f7', offset: 180, label: 'L10 · semantics' },
  L6: { color: '#f97316', offset: 120, label: 'L6 · prosody' },
  L2: { color: '#22d3ee', offset: 60, label: 'L2 · voiceprint' },
};
const AUDIO_STYLE = { color: '#38bdf8', offset: 10, label: 'waveform' };

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

  activeLayers.forEach((layer) => {
    drawLayerSurface(width, height, layer, domain, LAYER_CONFIG[layer.name]);
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

function drawLayerSurface(width, height, layer, domain, config = {}) {
  const margin = 100; // Increased margin for labels
  const usableWidth = width - margin * 2;
  const stats = domain.stats[layer.name];
  if (!stats) {
    return;
  }
  
  const baseline = height - margin - config.offset;
  const amplitude = 80; // Height of ridges - increased for better visibility
  const neuronCount = stats.neuronCount;
  
  // Create ridge plot with filled areas
  const ridgeCount = 6; // Draw 6 ridges per layer for better detail
  const ridgeStep = Math.max(1, Math.floor(neuronCount / ridgeCount));
  
  // Process each ridge from back to front for proper layering
  for (let ridgeIdx = ridgeCount - 1; ridgeIdx >= 0; ridgeIdx--) {
    const neuron = ridgeIdx * ridgeStep;
    if (neuron >= neuronCount) continue;
    
    const depthOffset = (ridgeIdx / (ridgeCount - 1)) * 35; // 3D depth effect
    const opacity = 0.3 + (ridgeIdx / (ridgeCount - 1)) * 0.4; // Higher base opacity for better visibility
    
    // Smooth the data with moving average
    const smoothedData = [];
    layer.vectors.forEach((vector, idx) => {
      let value = vector[neuron];
      if (idx > 0 && idx < layer.vectors.length - 1) {
        const prev = layer.vectors[idx - 1][neuron];
        const next = layer.vectors[idx + 1][neuron];
        value = (prev + value * 2 + next) / 4;
      }
      smoothedData.push(value);
    });
    
    // Draw filled ridge area
    ctx.beginPath();
    const ridgeBaseline = baseline + depthOffset;
    
    // Start from baseline
    ctx.moveTo(margin, ridgeBaseline);
    
    // Draw the ridge line
    layer.times.forEach((time, idx) => {
      const timeNorm = (time - domain.minTime) / (domain.maxTime - domain.minTime);
      const x = margin + timeNorm * usableWidth;
      const normActivity = (smoothedData[idx] - stats.min) / (stats.max - stats.min);
      const y = ridgeBaseline - normActivity * amplitude;
      
      if (idx === 0) {
        ctx.lineTo(x, y);
      } else {
        // Smooth curve
        const prevX = margin + ((layer.times[idx-1] - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
        const cpX = (prevX + x) / 2;
        const prevY = ridgeBaseline - ((smoothedData[idx-1] - stats.min) / (stats.max - stats.min)) * amplitude;
        const cpY = (prevY + y) / 2;
        ctx.quadraticCurveTo(cpX, cpY, x, y);
      }
    });
    
    // Complete the shape
    ctx.lineTo(margin + usableWidth, ridgeBaseline);
    ctx.closePath();
    
    // Fill with gradient - much brighter at peaks for clear elevation visibility
    const gradient = ctx.createLinearGradient(0, ridgeBaseline - amplitude, 0, ridgeBaseline + 10);
    gradient.addColorStop(0, hexToRgba(config.color, Math.min(1, opacity * 1.8))); // Much brighter peaks
    gradient.addColorStop(0.3, hexToRgba(config.color, opacity * 1.2));
    gradient.addColorStop(0.6, hexToRgba(config.color, opacity * 0.5));
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)'); // Fade to transparent at base
    ctx.fillStyle = gradient;
    ctx.fill();
    
    // Add ridge outline with stronger definition
    ctx.strokeStyle = hexToRgba(config.color, Math.min(1, opacity * 0.9));
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Add subtle shadow below ridge for depth
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(margin, ridgeBaseline + 1);
    ctx.lineTo(margin + usableWidth, ridgeBaseline + 1);
    ctx.stroke();
  }

  // Draw label - position to be fully visible
  ctx.save();
  ctx.fillStyle = 'rgba(220, 230, 245, 0.9)';
  ctx.font = '12px "Inter", system-ui';
  ctx.textAlign = 'right';
  const labelX = margin - 8;
  const labelY = baseline - 20; // Position above the baseline
  // Add subtle text shadow for better readability
  ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
  ctx.shadowBlur = 2;
  ctx.fillText(config.label || layer.name, labelX, labelY);
  ctx.textAlign = 'left';
  ctx.restore();
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
