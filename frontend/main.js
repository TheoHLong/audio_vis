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
  L10: { label: 'L10 · semantics (PC1–2)' },
  L6: { label: 'L6 · prosody (PC1–2)' },
  L2: { label: 'L2 · voiceprint (PC1–2)' },
};
const AUDIO_STYLE = { color: '#38bdf8', offset: 10, label: 'waveform' };

const RAINBOW_GRADIENT = [
  { stop: 0.0, color: [0, 0, 131] },
  { stop: 0.125, color: [0, 60, 170] },
  { stop: 0.375, color: [5, 255, 255] },
  { stop: 0.625, color: [255, 255, 0] },
  { stop: 0.875, color: [250, 0, 0] },
  { stop: 1.0, color: [128, 0, 0] },
];
const RIDGE_LEFT_MARGIN = 100;
const RIDGE_TOP_MARGIN = 110;
const RIDGE_BOTTOM_MARGIN = 150;
const RIDGE_MIN_BAND_HEIGHT = 80;
const RIDGE_DEPTH_RATIO = 0.25;
const COMPONENTS_PER_LAYER = 2;

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
    .filter((layer) => layer && layer.times?.length && ((layer.components && layer.components.length) || layer.activities?.length));
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
    drawLayerRidge(width, height, layer, domain, LAYER_CONFIG[layer.name], index, activeLayers.length);
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
    const components = computeLayerComponents(vectors, COMPONENTS_PER_LAYER);
    return {
      ...layer,
      times,
      vectors,
      components,
    };
  });
}

function computeLayerComponents(vectors, componentCount) {
  if (!Array.isArray(vectors) || vectors.length === 0) {
    return [];
  }
  const sampleCount = vectors.length;
  const dim = Array.isArray(vectors[0]) ? vectors[0].length : 0;
  if (!dim) {
    return [];
  }

  const sanitized = vectors
    .filter((vec) => Array.isArray(vec) && vec.length === dim)
    .map((vec) => {
      const row = new Float64Array(dim);
      for (let i = 0; i < dim; i += 1) {
        const value = Number(vec[i]);
        row[i] = Number.isFinite(value) ? value : 0;
      }
      return row;
    });

  if (sanitized.length < 2) {
    return [];
  }

  const effectiveSamples = sanitized.length;
  const mean = new Float64Array(dim);
  sanitized.forEach((row) => {
    for (let i = 0; i < dim; i += 1) {
      mean[i] += row[i];
    }
  });
  for (let i = 0; i < dim; i += 1) {
    mean[i] /= effectiveSamples;
  }

  const centered = sanitized.map((row) => {
    const centeredRow = new Float64Array(dim);
    for (let i = 0; i < dim; i += 1) {
      centeredRow[i] = row[i] - mean[i];
    }
    return centeredRow;
  });

  const covariance = Array.from({ length: dim }, () => new Float64Array(dim));
  centered.forEach((row) => {
    for (let i = 0; i < dim; i += 1) {
      const vi = row[i];
      if (!Number.isFinite(vi)) {
        continue;
      }
      for (let j = i; j < dim; j += 1) {
        const vj = row[j];
        if (!Number.isFinite(vj)) {
          continue;
        }
        covariance[i][j] += vi * vj;
      }
    }
  });
  const denom = Math.max(1, effectiveSamples - 1);
  for (let i = 0; i < dim; i += 1) {
    for (let j = i; j < dim; j += 1) {
      const value = covariance[i][j] / denom;
      covariance[i][j] = value;
      if (j !== i) {
        covariance[j][i] = value;
      }
    }
  }

  const working = covariance.map((row) => Float64Array.from(row));
  const maxComponents = Math.min(componentCount, dim);
  const components = [];

  for (let c = 0; c < maxComponents; c += 1) {
    const eigen = powerIteration(working);
    if (!eigen || !Number.isFinite(eigen.eigenvalue) || eigen.eigenvalue <= 1e-6) {
      break;
    }
    const basis = eigen.vector;
    const projections = projectOntoComponent(centered, basis);
    const series = Array.from(projections, (value) => Number.isFinite(value) ? value : 0);

    let dominantIndex = 0;
    let dominantWeight = -1;
    for (let i = 0; i < basis.length; i += 1) {
      const weight = Math.abs(basis[i]);
      if (weight > dominantWeight) {
        dominantWeight = weight;
        dominantIndex = i;
      }
    }

    components.push({
      label: `PC${c + 1}`,
      values: series,
      eigenvalue: eigen.eigenvalue,
      basis: Array.from(basis),
      dominantIndex,
    });

    deflateMatrix(working, basis, eigen.eigenvalue);
  }

  return components;
}

function projectOntoComponent(vectors, component) {
  const length = vectors.length;
  const dim = component.length;
  const output = new Float64Array(length);
  for (let i = 0; i < length; i += 1) {
    const row = vectors[i];
    let sum = 0;
    for (let j = 0; j < dim; j += 1) {
      sum += row[j] * component[j];
    }
    output[i] = sum;
  }
  return output;
}

function deflateMatrix(matrix, vector, eigenvalue) {
  const dim = vector.length;
  for (let i = 0; i < dim; i += 1) {
    const vi = vector[i];
    for (let j = 0; j < dim; j += 1) {
      matrix[i][j] -= eigenvalue * vi * vector[j];
    }
  }
}

function powerIteration(matrix, maxIterations = 48, tolerance = 1e-6) {
  const dim = Array.isArray(matrix) ? matrix.length : 0;
  if (!dim) {
    return null;
  }

  let vector = new Float64Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i += 1) {
    const value = Math.random() - 0.5;
    vector[i] = value;
    norm += value * value;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i += 1) {
    vector[i] /= norm;
  }

  let eigenvalue = 0;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const mv = multiplyMatrixVector(matrix, vector);
    const mvNorm = vectorL2Norm(mv);
    if (!Number.isFinite(mvNorm) || mvNorm <= tolerance) {
      break;
    }
    for (let i = 0; i < dim; i += 1) {
      mv[i] /= mvNorm;
    }

    const diff = vectorDifferenceNorm(mv, vector);
    vector = mv;
    const mv2 = multiplyMatrixVector(matrix, vector);
    eigenvalue = dot(vector, mv2);
    if (diff <= tolerance) {
      break;
    }
  }

  return { vector, eigenvalue };
}

function multiplyMatrixVector(matrix, vector) {
  const dim = vector.length;
  const result = new Float64Array(dim);
  for (let i = 0; i < dim; i += 1) {
    const row = matrix[i];
    let sum = 0;
    for (let j = 0; j < dim; j += 1) {
      sum += row[j] * vector[j];
    }
    result[i] = sum;
  }
  return result;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function vectorL2Norm(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) {
    const value = vector[i];
    sum += value * value;
  }
  return Math.sqrt(sum);
}

function vectorDifferenceNorm(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
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
    if (!layer.times?.length) {
      return;
    }

    minTime = Math.min(minTime, layer.times[0]);
    maxTime = Math.max(maxTime, layer.times[layer.times.length - 1]);

    const sampleVector = (layer.vectors && layer.vectors[0]) || [];
    const components = Array.isArray(layer.components) ? layer.components : [];
    const indexSeries = Array.isArray(layer.indices) ? layer.indices : [];

    const neuronCount = Math.max(sampleVector.length, 1);
    maxNeurons = Math.max(maxNeurons, neuronCount);

    const statEntry = {
      neuronCount,
      activityMin: Number.POSITIVE_INFINITY,
      activityMax: Number.NEGATIVE_INFINITY,
      indexMin: Number.POSITIVE_INFINITY,
      indexMax: Number.NEGATIVE_INFINITY,
      components: [],
    };

    components.forEach((component) => {
      const values = Array.isArray(component?.values) ? component.values : [];
      let compMin = Number.POSITIVE_INFINITY;
      let compMax = Number.NEGATIVE_INFINITY;
      values.forEach((value) => {
        if (Number.isFinite(value)) {
          compMin = Math.min(compMin, value);
          compMax = Math.max(compMax, value);
        }
      });
      if (!Number.isFinite(compMin) || !Number.isFinite(compMax)) {
        compMin = -1;
        compMax = 1;
      } else if (Math.abs(compMax - compMin) < 1e-6) {
        const mid = compMin;
        compMin = mid - 0.5;
        compMax = mid + 0.5;
      }
      statEntry.components.push({ min: compMin, max: compMax });
      statEntry.activityMin = Math.min(statEntry.activityMin, compMin);
      statEntry.activityMax = Math.max(statEntry.activityMax, compMax);
    });

    if (!statEntry.components.length) {
      const fallback = Array.isArray(layer.activities) ? layer.activities : [];
      let compMin = Number.POSITIVE_INFINITY;
      let compMax = Number.NEGATIVE_INFINITY;
      fallback.forEach((value) => {
        if (Number.isFinite(value)) {
          compMin = Math.min(compMin, value);
          compMax = Math.max(compMax, value);
        }
      });
      if (!Number.isFinite(compMin) || !Number.isFinite(compMax)) {
        compMin = -1;
        compMax = 1;
      } else if (Math.abs(compMax - compMin) < 1e-6) {
        const mid = compMin;
        compMin = mid - 0.5;
        compMax = mid + 0.5;
      }
      statEntry.components.push({ min: compMin, max: compMax });
      statEntry.activityMin = Math.min(statEntry.activityMin, compMin);
      statEntry.activityMax = Math.max(statEntry.activityMax, compMax);
    }

    indexSeries.forEach((value) => {
      if (Number.isFinite(value)) {
        statEntry.indexMin = Math.min(statEntry.indexMin, value);
        statEntry.indexMax = Math.max(statEntry.indexMax, value);
      }
    });

    if (!Number.isFinite(statEntry.activityMin) || !Number.isFinite(statEntry.activityMax)) {
      statEntry.activityMin = -1;
      statEntry.activityMax = 1;
    }

    if (!Number.isFinite(statEntry.indexMin) || !Number.isFinite(statEntry.indexMax)) {
      statEntry.indexMin = 0;
      statEntry.indexMax = Math.max(neuronCount - 1, 1);
    } else if (Math.abs(statEntry.indexMax - statEntry.indexMin) < 1e-6) {
      const mid = statEntry.indexMin;
      statEntry.indexMin = mid - 0.5;
      statEntry.indexMax = mid + 0.5;
    }

    if (Number.isFinite(statEntry.indexMax)) {
      const impliedCount = Math.floor(statEntry.indexMax) + 1;
      maxNeurons = Math.max(maxNeurons, impliedCount);
      statEntry.neuronCount = Math.max(statEntry.neuronCount, impliedCount);
    }

    stats[layer.name] = statEntry;
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

function drawLayerRidge(width, height, layer, domain, config = {}, index = 0, totalLayers = 1) {
  const layerTimes = Array.isArray(layer.times) ? layer.times : [];
  if (!layerTimes.length) {
    return;
  }

  const seriesList = Array.isArray(layer.components) && layer.components.length
    ? layer.components.slice(0, COMPONENTS_PER_LAYER)
    : Array.isArray(layer.activities)
      ? [{ label: 'Activity', values: layer.activities.slice() }]
      : [];
  const componentCount = seriesList.length;
  if (!componentCount) {
    return;
  }

  const stats = domain.stats[layer.name];
  if (!stats) {
    return;
  }

  const margin = RIDGE_LEFT_MARGIN;
  const usableWidth = width - margin * 2;
  if (usableWidth <= 0) {
    return;
  }

  const topMargin = RIDGE_TOP_MARGIN;
  const bottomMargin = RIDGE_BOTTOM_MARGIN;
  const availableHeight = Math.max(RIDGE_MIN_BAND_HEIGHT * totalLayers, height - topMargin - bottomMargin);
  const bandHeight = Math.max(RIDGE_MIN_BAND_HEIGHT, availableHeight / Math.max(totalLayers, 1));
  const layerBaseline = topMargin + index * bandHeight + bandHeight * 0.78;
  const amplitude = bandHeight * 0.62;
  const depthScale = bandHeight * RIDGE_DEPTH_RATIO;
  const componentSpread = amplitude * 0.48;

  const timeRange = Math.max(1e-6, domain.maxTime - domain.minTime);
  const indexRange = Math.max(1e-6, stats.indexMax - stats.indexMin);
  const rawIndexSeries = Array.isArray(layer.indices) ? layer.indices : [];
  const smoothedIndex = smoothSeries(rawIndexSeries.slice(0, layerTimes.length));

  const bandTop = layerBaseline - amplitude - componentSpread * 0.75;
  const bandBottom = layerBaseline + amplitude + componentSpread * 0.85 + depthScale;
  const bandTopClamped = Math.max(20, bandTop);
  const bandBottomClamped = Math.min(height - Math.max(60, bottomMargin * 0.6), bandBottom);
  const bandHeightRect = Math.max(24, bandBottomClamped - bandTopClamped);
  ctx.save();
  ctx.fillStyle = 'rgba(12, 18, 35, 0.32)';
  ctx.fillRect(margin, bandTopClamped, usableWidth, bandHeightRect);
  ctx.restore();

  for (let componentIndex = 0; componentIndex < componentCount; componentIndex += 1) {
    const component = seriesList[componentIndex];
    const values = Array.isArray(component?.values) ? component.values : [];
    const sampleCount = Math.min(layerTimes.length, values.length);
    if (sampleCount < 2) {
      continue;
    }

    const compStats = (stats.components && stats.components[componentIndex])
      || { min: stats.activityMin, max: stats.activityMax };
    const activityRange = Math.max(1e-6, compStats.max - compStats.min);
    const smoothedValues = smoothSeries(values.slice(0, sampleCount));
    const componentOffset = (componentIndex - (componentCount - 1) / 2) * componentSpread;
    const fillAlpha = 0.32 + ((componentCount - componentIndex) / componentCount) * 0.18;
    const strokeAlpha = 0.7 + (componentIndex / Math.max(1, componentCount - 1)) * 0.22;

    const points = [];
    for (let i = 0; i < sampleCount; i += 1) {
      const time = layerTimes[i];
      const value = smoothedValues[i];
      if (!Number.isFinite(time) || !Number.isFinite(value)) {
        continue;
      }
      const x = margin + ((time - domain.minTime) / timeRange) * usableWidth;
      if (!Number.isFinite(x)) {
        continue;
      }

      const indexValue = Number.isFinite(smoothedIndex[i]) ? smoothedIndex[i] : stats.indexMin + indexRange / 2;
      const indexNorm = Math.min(1, Math.max(0, (indexValue - stats.indexMin) / indexRange));
      const depthOffset = (indexNorm - 0.5) * depthScale;

      const activityNorm = Math.min(1, Math.max(0, (value - compStats.min) / activityRange));
      const baselineForComponent = layerBaseline + componentOffset;
      const y = baselineForComponent - activityNorm * amplitude + depthOffset;
      points.push({ x, y, intensity: activityNorm });
    }

    if (points.length < 2) {
      continue;
    }

    const baselineY = layerBaseline + componentOffset + depthScale;
    const fillGradient = createIntensityGradient(points, margin, usableWidth, fillAlpha);
    const strokeGradient = createIntensityGradient(points, margin, usableWidth, strokeAlpha);

    const traceRidgeShape = () => {
      ctx.beginPath();
      ctx.moveTo(margin, baselineY);
      ctx.lineTo(points[0].x, baselineY);
      ctx.lineTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const curr = points[i];
        const cpX = (prev.x + curr.x) / 2;
        const cpY = (prev.y + curr.y) / 2;
        ctx.quadraticCurveTo(prev.x, prev.y, cpX, cpY);
      }
      const lastPoint = points[points.length - 1];
      ctx.lineTo(lastPoint.x, lastPoint.y);
      ctx.lineTo(lastPoint.x, baselineY);
      ctx.lineTo(margin + usableWidth, baselineY);
      ctx.closePath();
    };

    const traceCrest = () => {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i += 1) {
        const prev = points[i - 1];
        const curr = points[i];
        const cpX = (prev.x + curr.x) / 2;
        const cpY = (prev.y + curr.y) / 2;
        ctx.quadraticCurveTo(prev.x, prev.y, cpX, cpY);
      }
      const lastPoint = points[points.length - 1];
      ctx.lineTo(lastPoint.x, lastPoint.y);
    };

    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    traceRidgeShape();
    ctx.fillStyle = fillGradient;
    ctx.shadowColor = 'rgba(5, 12, 30, 0.4)';
    ctx.shadowBlur = 14;
    ctx.shadowOffsetY = 5;
    ctx.fill();

    ctx.save();
    ctx.globalCompositeOperation = 'source-atop';
    ctx.shadowColor = 'rgba(0, 0, 0, 0)';
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    traceRidgeShape();
    const verticalShade = ctx.createLinearGradient(0, layerBaseline - amplitude, 0, baselineY);
    verticalShade.addColorStop(0, 'rgba(255, 255, 255, 0.08)');
    verticalShade.addColorStop(0.55, 'rgba(46, 72, 120, 0.08)');
    verticalShade.addColorStop(1, 'rgba(6, 10, 22, 0.55)');
    ctx.fillStyle = verticalShade;
    ctx.fill();
    ctx.restore();

    ctx.shadowColor = 'rgba(0, 0, 0, 0)';
    traceCrest();
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = strokeGradient;
    ctx.stroke();

    traceCrest();
    ctx.lineWidth = 1.1;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(margin, baselineY);
    ctx.lineTo(margin + usableWidth, baselineY);
    ctx.strokeStyle = 'rgba(14, 21, 36, 0.38)';
    ctx.lineWidth = 0.9;
    ctx.stroke();

    ctx.restore();
  }

  const labelY = Math.max(36, layerBaseline - amplitude - componentSpread * 0.6 - depthScale - 12);
  ctx.save();
  ctx.fillStyle = 'rgba(220, 232, 250, 0.92)';
  ctx.font = '12px "Inter", system-ui';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
  ctx.shadowBlur = 2;
  ctx.fillText(config.label || layer.name, margin - 18, labelY);
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

function smoothSeries(series) {
  if (!Array.isArray(series)) {
    return [];
  }
  const result = series.slice();
  for (let i = 1; i < result.length - 1; i += 1) {
    const prev = series[i - 1];
    const curr = series[i];
    const next = series[i + 1];
    if (Number.isFinite(prev) && Number.isFinite(curr) && Number.isFinite(next)) {
      result[i] = (prev + curr * 2 + next) / 4;
    }
  }
  return result;
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
