import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.161/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.161/examples/jsm/controls/OrbitControls.js';

const container = document.getElementById('visual-container');
const startBtn = document.getElementById('start-btn');
const modeBtn = document.getElementById('mode-btn');
const resetBtn = document.getElementById('reset-btn');
const gridBtn = document.getElementById('projection-btn');
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
  showGrid: true,
};

const TARGET_SAMPLE_RATE = 16_000;
const HISTORY_SECONDS = 15;

const LAYER_CONFIGS = {
  L10: { color: '#a855f7', offset: 28, label: 'Layer 10 · semantics' },
  L6: { color: '#f97316', offset: 14, label: 'Layer 6 · prosody' },
  L2: { color: '#22d3ee', offset: 0, label: 'Layer 2 · voiceprint' },
};
const AUDIO_STYLE = { color: '#38bdf8', offset: -18, label: 'Audio waveform' };

let ws = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let listening = false;

/* --------------------- THREE.JS SETUP --------------------- */
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070d);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
camera.position.set(45, 35, 70);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 25;
controls.maxDistance = 160;
controls.maxPolarAngle = Math.PI * 0.85;
controls.target.set(0, 0, 0);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
keyLight.position.set(60, 80, 40);
const fillLight = new THREE.DirectionalLight(0x88aaff, 0.4);
fillLight.position.set(-40, 30, -60);
scene.add(ambientLight, keyLight, fillLight);

const gridGroup = new THREE.Group();
scene.add(gridGroup);

const layerMeshes = new Map();
const audioLineGroup = new THREE.Group();
scene.add(audioLineGroup);

function ensureSurface(name, color) {
  if (layerMeshes.has(name)) {
    return layerMeshes.get(name);
  }
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.MeshPhongMaterial({
    color: new THREE.Color(color),
    transparent: true,
    opacity: 0.65,
    side: THREE.DoubleSide,
    shininess: 60,
    specular: new THREE.Color(0x222222),
    vertexColors: true,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  scene.add(mesh);
  layerMeshes.set(name, mesh);
  return mesh;
}

function updateRendererSize() {
  const { width, height } = container.getBoundingClientRect();
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

window.addEventListener('resize', updateRendererSize);
updateRendererSize();

/* --------------------- SOCKET / AUDIO --------------------- */
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

/* --------------------- SOCKET HANDLING --------------------- */
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
    updateVisualization();
    return;
  }
}

function updateVisualization() {
  updateTranscript();
  updateDiagnostics();
  updateModeUI();
  refreshSurfaces();
}

/* --------------------- VISUAL UPDATE --------------------- */
function refreshSurfaces() {
  const activeLayers = Object.keys(LAYER_CONFIGS)
    .map((name) => state.layers.find((layer) => layer.name === name))
    .filter(Boolean);
  const hasAudio = Array.isArray(state.audio?.times) && state.audio.times.length > 0;

  updateGridHelpers(activeLayers, hasAudio);

  if (!activeLayers.length && !hasAudio) {
    disposeSurfaces();
    return;
  }

  const domain = computeDomain(activeLayers, state.audio);

  activeLayers.forEach((layer, idx) => {
    const config = LAYER_CONFIGS[layer.name];
    const mesh = ensureSurface(layer.name, config.color);
    const geometry = buildSurfaceGeometry(layer, domain, config, idx);
    mesh.geometry.dispose();
    mesh.geometry = geometry;
  });

  // remove any meshes for missing layers
  [...layerMeshes.keys()].forEach((name) => {
    if (!activeLayers.find((layer) => layer.name === name)) {
      const mesh = layerMeshes.get(name);
      scene.remove(mesh);
      mesh.geometry.dispose();
      layerMeshes.delete(name);
    }
  });

  updateAudioGeometry(domain, hasAudio);
}

function disposeSurfaces() {
  [...layerMeshes.values()].forEach((mesh) => {
    scene.remove(mesh);
    mesh.geometry.dispose();
  });
  layerMeshes.clear();
  audioLineGroup.clear();
}

function computeDomain(layers, audio) {
  let minTime = Number.POSITIVE_INFINITY;
  let maxTime = 0;
  const stats = {};

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
    const flat = layer.vectors.flat();
    let minVal = Math.min(...flat);
    let maxVal = Math.max(...flat);
    if (Math.abs(maxVal - minVal) < 1e-6) {
      maxVal = minVal + 1e-3;
    }
    stats[layer.name] = { min: minVal, max: maxVal, neurons: sampleVector.length };
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
  return { minTime, maxTime, stats };
}

function buildSurfaceGeometry(layer, domain, config, stackIndex) {
  const times = layer.times;
  const vectors = layer.vectors;
  const speakers = layer.speakers;
  const stats = domain.stats[layer.name];
  const rows = Math.min(times.length, vectors.length);
  const cols = stats ? stats.neurons : (vectors[0] ? vectors[0].length : 0);
  if (!rows || !cols || rows < 2 || cols < 2) {
    return new THREE.BufferGeometry();
  }
  const colorsBySpeaker = state.speakerColors || {};

  const width = 60;
  const depthScale = 30;
  const heightScale = 18;
  const yOffset = config.offset || 0;

  const positions = new Float32Array(rows * cols * 3);
  const colors = new Float32Array(rows * cols * 3);
  const indices = new Uint32Array((rows - 1) * (cols - 1) * 6);

  const speakerColorCache = new Map();
  function colorForSpeaker(id) {
    if (id === null || id === undefined || id < 0) {
      return new THREE.Color(config.color);
    }
    const key = String(id);
    if (!speakerColorCache.has(key)) {
      const hex = colorsBySpeaker[key] || config.color;
      speakerColorCache.set(key, new THREE.Color(hex));
    }
    return speakerColorCache.get(key);
  }

  for (let r = 0; r < rows; r += 1) {
    const t = times[r];
    const timeNorm = (t - domain.minTime) / (domain.maxTime - domain.minTime);
    const vector = vectors[r];
    const speaker = speakers[r] ?? null;
    for (let c = 0; c < cols; c += 1) {
      const idx = r * cols + c;
      const posIndex = idx * 3;
      const colNorm = cols > 1 ? c / (cols - 1) : 0;
      const activity = vector[c];
      const normActivity = (activity - stats.min) / (stats.max - stats.min);
      const height = (normActivity - 0.5) * heightScale;

      positions[posIndex] = (timeNorm - 0.5) * width;
      positions[posIndex + 1] = yOffset + (colNorm - 0.5) * depthScale;
      positions[posIndex + 2] = height;

      const color = colorForSpeaker(speaker);
      colors[posIndex] = color.r;
      colors[posIndex + 1] = color.g;
      colors[posIndex + 2] = color.b;
    }
  }

  let idxPtr = 0;
  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const a = r * cols + c;
      const b = r * cols + (c + 1);
      const d = (r + 1) * cols + c;
      const e = (r + 1) * cols + (c + 1);
      indices[idxPtr++] = a;
      indices[idxPtr++] = d;
      indices[idxPtr++] = b;
      indices[idxPtr++] = b;
      indices[idxPtr++] = d;
      indices[idxPtr++] = e;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

function updateAudioGeometry(domain, hasAudio) {
  audioLineGroup.clear();
  if (!hasAudio) {
    return;
  }
  const times = state.audio.times;
  const values = state.audio.rms;
  if (!times.length || !values.length) {
    return;
  }
  const width = 60;
  const baselineY = AUDIO_STYLE.offset;
  const amplitude = 10;
  const positions = new Float32Array(times.length * 3);
  for (let i = 0; i < times.length; i += 1) {
    const timeNorm = (times[i] - domain.minTime) / (domain.maxTime - domain.minTime);
    const x = (timeNorm - 0.5) * width;
    const y = baselineY;
    const z = (values[i] - 0.5) * amplitude;
    const idx = i * 3;
    positions[idx] = x;
    positions[idx + 1] = y;
    positions[idx + 2] = z;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: new THREE.Color(AUDIO_STYLE.color), transparent: true, opacity: 0.7 });
  const line = new THREE.Line(geometry, material);
  audioLineGroup.add(line);
}

function updateGridHelpers(layers, hasAudio) {
  gridGroup.clear();
  if (!state.showGrid || (!layers.length && !hasAudio)) {
    gridGroup.visible = false;
    return;
  }
  gridGroup.visible = true;
  const width = 60;
  const depth = 60;
  const grid = new THREE.GridHelper(width, 10, 0x284a63, 0x284a63);
  grid.rotation.x = Math.PI / 2;
  grid.position.y = -22;
  gridGroup.add(grid);
}

/* --------------------- UI HELPERS --------------------- */
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

/* --------------------- UI EVENTS --------------------- */
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

gridBtn.addEventListener('click', () => {
  state.showGrid = !state.showGrid;
  updateModeUI();
  refreshSurfaces();
});

resetBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
  state.layers = [];
  state.audio = { times: [], rms: [] };
  refreshSurfaces();
  updateTranscript();
});

updateButtons();
updateModeUI();
updateDiagnostics();
updateTranscript();

/* --------------------- RENDER LOOP --------------------- */
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

/* --------------------- EXPORTS --------------------- */
// Ready for interactions
