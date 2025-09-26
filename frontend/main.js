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
  L10: { label: 'L10 · neural activations (jet)' },
  L6: { label: 'L6 · neural activations (jet)' },
  L2: { label: 'L2 · neural activations (jet)' },
};
const AUDIO_STYLE = { color: '#38bdf8', offset: 10, label: 'waveform' };

// Jet colormap for heatmaps
const JET_GRADIENT = [
  { stop: 0.0, color: [0, 0, 143] },      // Dark blue
  { stop: 0.125, color: [0, 0, 255] },    // Blue
  { stop: 0.25, color: [0, 127, 255] },   // Light blue
  { stop: 0.375, color: [0, 255, 255] },  // Cyan
  { stop: 0.5, color: [127, 255, 127] },  // Light green
  { stop: 0.625, color: [255, 255, 0] },  // Yellow
  { stop: 0.75, color: [255, 127, 0] },   // Orange
  { stop: 0.875, color: [255, 0, 0] },    // Red
  { stop: 1.0, color: [127, 0, 0] },      // Dark red
];

const RAINBOW_GRADIENT = [
  { stop: 0.0, color: [0, 0, 131] },
  { stop: 0.125, color: [0, 60, 170] },
  { stop: 0.375, color: [5, 255, 255] },
  { stop: 0.625, color: [255, 255, 0] },
  { stop: 0.875, color: [250, 0, 0] },
  { stop: 1.0, color: [128, 0, 0] },
];

const HEATMAP_LEFT_MARGIN = 100;
const HEATMAP_TOP_MARGIN = 110;
const HEATMAP_BOTTOM_MARGIN = 150;
const HEATMAP_MIN_BAND_HEIGHT = 120;

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
    const rawLayers = data.layers || [];
    state.layers = enrichLayers(rawLayers);
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

  const domain = computeHeatmapDomain(activeLayers, state.audio);
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

function enrichLayers(rawLayers) {
  if (!Array.isArray(rawLayers)) {
    return [];
  }
  return rawLayers.map((layer) => {
    const times = Array.isArray(layer?.times) ? layer.times : [];
    const vectors = Array.isArray(layer?.vectors) ? layer.vectors : [];
    return {
      ...layer,
      times,
      vectors,
    };
  });
}

function jetColormap(value, alpha = 1) {
  const t = Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
  for (let i = 0; i < JET_GRADIENT.length - 1; i += 1) {
    const current = JET_GRADIENT[i];
    const next = JET_GRADIENT[i + 1];
    if (t <= next.stop) {
      const span = next.stop - current.stop || 1;
      const localT = Math.min(1, Math.max(0, (t - current.stop) / span));
      const r = Math.round(current.color[0] + (next.color[0] - current.color[0]) * localT);
      const g = Math.round(current.color[1] + (next.color[1] - current.color[1]) * localT);
      const b = Math.round(current.color[2] + (next.color[2] - current.color[2]) * localT);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
  }
  const last = JET_GRADIENT[JET_GRADIENT.length - 1];
  return `rgba(${last.color[0]}, ${last.color[1]}, ${last.color[2]}, ${alpha})`;
}

function intensityToColor(value, alpha = 1) {
  const t = Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0.5));
  for (let i = 0; i < RAINBOW_GRADIENT.length - 1; i += 1) {
    const current = RAINBOW_GRADIENT[i];
    const next = RAINBOW_GRADIENT[i + 1];
    if (t <= next.stop) {
      const span = next.stop - current.stop || 1;
      const localT = Math.min(1, Math.max(0, (t - current.stop) / span));
      const r = Math.round(current.color[0] + (next.color[0] - current.color[0]) * localT);
      const g = Math.round(current.color[1] + (next.color[1] - current.color[1]) * localT);
      const b = Math.round(current.color[2] + (next.color[2] - current.color[2]) * localT);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
  }
  const last = RAINBOW_GRADIENT[RAINBOW_GRADIENT.length - 1];
  return `rgba(${last.color[0]}, ${last.color[1]}, ${last.color[2]}, ${alpha})`;
}

function drawPlaceholder(width, height) {
  ctx.fillStyle = 'rgba(148, 197, 255, 0.25)';
  ctx.font = '14px "Inter", system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Speak to see neural activation heatmaps...', width / 2, height / 2);
  ctx.textAlign = 'left';
}

function computeHeatmapDomain(layers, audio) {
  let minTime = Number.POSITIVE_INFINITY;
  let maxTime = 0;
  const stats = {};
  let maxNeurons = 0;

  layers.forEach((layer) => {
    if (!layer.times?.length || !layer.vectors?.length) {
      return;
    }

    minTime = Math.min(minTime, layer.times[0]);
    maxTime = Math.max(maxTime, layer.times[layer.times.length - 1]);

    const neuronCount = layer.vectors[0]?.length || 0;
    maxNeurons = Math.max(maxNeurons, neuronCount);

    // Compute activation range across all vectors
    let minActivation = Number.POSITIVE_INFINITY;
    let maxActivation = Number.NEGATIVE_INFINITY;

    layer.vectors.forEach(vector => {
      if (Array.isArray(vector)) {
        vector.forEach(value => {
          if (Number.isFinite(value)) {
            minActivation = Math.min(minActivation, value);
            maxActivation = Math.max(maxActivation, value);
          }
        });
      }
    });

    if (!Number.isFinite(minActivation) || !Number.isFinite(maxActivation)) {
      minActivation = -1;
      maxActivation = 1;
    } else if (Math.abs(maxActivation - minActivation) < 1e-6) {
      const mid = minActivation;
      minActivation = mid - 0.5;
      maxActivation = mid + 0.5;
    }

    stats[layer.name] = {
      neuronCount,
      minActivation,
      maxActivation,
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
  const layerTimes = Array.isArray(layer.times) ? layer.times : [];
  const layerVectors = Array.isArray(layer.vectors) ? layer.vectors : [];
  
  if (!layerTimes.length || !layerVectors.length) {
    return;
  }

  const stats = domain.stats[layer.name];
  if (!stats) {
    return;
  }

  const margin = HEATMAP_LEFT_MARGIN;
  const usableWidth = width - margin * 2;
  if (usableWidth <= 0) {
    return;
  }

  const topMargin = HEATMAP_TOP_MARGIN;
  const bottomMargin = HEATMAP_BOTTOM_MARGIN;
  const availableHeight = Math.max(HEATMAP_MIN_BAND_HEIGHT * totalLayers, height - topMargin - bottomMargin);
  const bandHeight = Math.max(HEATMAP_MIN_BAND_HEIGHT, availableHeight / Math.max(totalLayers, 1));
  
  const heatmapTop = topMargin + index * bandHeight;
  const heatmapHeight = bandHeight * 0.8; // Leave some space between layers
  
  const timeRange = Math.max(1e-6, domain.maxTime - domain.minTime);
  const activationRange = Math.max(1e-6, stats.maxActivation - stats.minActivation);
  
  // Create a temporary canvas for the heatmap data
  const heatmapCanvas = document.createElement('canvas');
  const heatmapCtx = heatmapCanvas.getContext('2d');
  
  const timeSteps = layerTimes.length;
  const neuronCount = stats.neuronCount;
  
  if (timeSteps === 0 || neuronCount === 0) {
    return;
  }
  
  heatmapCanvas.width = timeSteps;
  heatmapCanvas.height = neuronCount;
  
  const imageData = heatmapCtx.createImageData(timeSteps, neuronCount);
  const data = imageData.data;
  
  // Fill the heatmap data
  for (let t = 0; t < timeSteps; t++) {
    const vector = layerVectors[t];
    if (!Array.isArray(vector)) continue;
    
    for (let n = 0; n < neuronCount; n++) {
      const activation = vector[n] || 0;
      const normalizedActivation = (activation - stats.minActivation) / activationRange;
      
      // Convert to jet colormap
      const color = jetColormap(normalizedActivation, 1);
      const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
      
      if (rgbaMatch) {
        const pixelIndex = (n * timeSteps + t) * 4;
        data[pixelIndex] = parseInt(rgbaMatch[1]);     // R
        data[pixelIndex + 1] = parseInt(rgbaMatch[2]); // G
        data[pixelIndex + 2] = parseInt(rgbaMatch[3]); // B
        data[pixelIndex + 3] = 255;                    // A
      }
    }
  }
  
  heatmapCtx.putImageData(imageData, 0, 0);
  
  // Draw the heatmap to the main canvas
  ctx.save();
  ctx.imageSmoothingEnabled = false; // Keep pixel-perfect rendering
  ctx.drawImage(
    heatmapCanvas,
    0, 0, timeSteps, neuronCount,
    margin, heatmapTop, usableWidth, heatmapHeight
  );
  
  // Add border
  ctx.strokeStyle = 'rgba(148, 197, 255, 0.3)';
  ctx.lineWidth = 1;
  ctx.strokeRect(margin, heatmapTop, usableWidth, heatmapHeight);
  
  // Draw label
  const labelY = heatmapTop + heatmapHeight / 2;
  ctx.fillStyle = 'rgba(220, 232, 250, 0.92)';
  ctx.font = '12px "Inter", system-ui';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
  ctx.shadowBlur = 2;
  ctx.fillText(config.label || layer.name, margin - 18, labelY);
  
  // Draw neuron count label
  ctx.textAlign = 'left';
  ctx.font = '10px "Inter", system-ui';
  ctx.fillStyle = 'rgba(148, 197, 255, 0.7)';
  ctx.fillText(`${neuronCount} neurons`, margin + 5, heatmapTop + 15);
  
  ctx.restore();
}

function createIntensityGradient(points, margin, usableWidth, alpha = 1) {
  const gradient = ctx.createLinearGradient(margin, 0, margin + usableWidth, 0);
  const clampAlpha = Math.min(1, Math.max(0, alpha));
  if (!Array.isArray(points) || !points.length || !Number.isFinite(usableWidth) || usableWidth <= 0) {
    gradient.addColorStop(0, intensityToColor(0.5, clampAlpha));
    gradient.addColorStop(1, intensityToColor(0.5, clampAlpha));
    return gradient;
  }

  const range = Math.max(1e-6, usableWidth);
  const seen = new Set();

  points.forEach((point, index) => {
    if (!point || !Number.isFinite(point.x)) {
      return;
    }
    const ratio = Math.min(1, Math.max(0, (point.x - margin) / range));
    const key = Math.round(ratio * 1000);
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    gradient.addColorStop(ratio, intensityToColor(point.intensity, clampAlpha));
  });

  if (!seen.has(0)) {
    const start = points.find((point) => point && Number.isFinite(point.x));
    const color = start ? intensityToColor(start.intensity, clampAlpha) : intensityToColor(0.5, clampAlpha);
    gradient.addColorStop(0, color);
  }
  if (!seen.has(1000)) {
    const end = [...points].reverse().find((point) => point && Number.isFinite(point.x));
    const color = end ? intensityToColor(end.intensity, clampAlpha) : intensityToColor(0.5, clampAlpha);
    gradient.addColorStop(1, color);
  }

  return gradient;
}

function drawAudioSurface(width, height, domain) {
  const margin = 100;
  const usableWidth = width - margin * 2;
  const baseline = height - margin + AUDIO_STYLE.offset;
  const amplitude = 52;

  const audioPoints = [];

  // Draw audio as a simple filled waveform
  ctx.beginPath();
  ctx.moveTo(margin, baseline);
  
  state.audio.times.forEach((time, idx) => {
    const x = margin + ((time - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
    const intensity = Math.min(1, Math.max(0, state.audio.rms[idx] ?? 0));
    const y = baseline - intensity * amplitude;
    audioPoints.push({ x, intensity });
    
    if (idx === 0) {
      ctx.lineTo(x, y);
    } else {
      // Smooth the line
      const prevX = margin + ((state.audio.times[idx-1] - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
      const cpX = (prevX + x) / 2;
      const prevIntensity = Math.min(1, Math.max(0, state.audio.rms[idx - 1] ?? 0));
      const prevY = baseline - prevIntensity * amplitude;
      const cpY = (prevY + y) / 2;
      ctx.quadraticCurveTo(cpX, cpY, x, y);
    }
  });
  
  ctx.lineTo(margin + usableWidth, baseline);
  ctx.closePath();
  
  const fillGradient = createIntensityGradient(audioPoints, margin, usableWidth, 0.42);
  ctx.save();
  ctx.fillStyle = fillGradient;
  ctx.shadowColor = 'rgba(5, 12, 30, 0.32)';
  ctx.shadowBlur = 12;
  ctx.shadowOffsetY = 4;
  ctx.fill();

  ctx.globalCompositeOperation = 'source-atop';
  const audioShade = ctx.createLinearGradient(0, baseline - amplitude, 0, baseline + amplitude * 0.35);
  audioShade.addColorStop(0, 'rgba(255, 255, 255, 0.08)');
  audioShade.addColorStop(0.6, 'rgba(20, 40, 78, 0.12)');
  audioShade.addColorStop(1, 'rgba(6, 12, 24, 0.6)');
  ctx.fillStyle = audioShade;
  ctx.fill();
  ctx.restore();

  const strokeGradient = createIntensityGradient(audioPoints, margin, usableWidth, 0.85);
  ctx.beginPath();
  ctx.moveTo(margin, baseline);
  state.audio.times.forEach((time, idx) => {
    const x = margin + ((time - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
    const y = baseline - state.audio.rms[idx] * amplitude;
    if (idx === 0) {
      ctx.lineTo(x, y);
    } else {
      const prevX = margin + ((state.audio.times[idx - 1] - domain.minTime) / (domain.maxTime - domain.minTime)) * usableWidth;
      const cpX = (prevX + x) / 2;
      const prevY = baseline - state.audio.rms[idx - 1] * amplitude;
      const cpY = (prevY + y) / 2;
      ctx.quadraticCurveTo(prevX, prevY, cpX, cpY);
    }
  });
  ctx.lineTo(margin + usableWidth, baseline);
  ctx.strokeStyle = strokeGradient;
  ctx.lineWidth = 1.6;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(margin, baseline);
  ctx.lineTo(margin + usableWidth, baseline);
  ctx.strokeStyle = 'rgba(12, 18, 35, 0.5)';
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
