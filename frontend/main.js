const canvas = document.getElementById('comet-canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const modeBtn = document.getElementById('mode-btn');
const resetBtn = document.getElementById('reset-btn');
const valenceBar = document.getElementById('valence-bar');
const arousalBar = document.getElementById('arousal-bar');
const keywordList = document.getElementById('keyword-list');
const diagnosticsList = document.getElementById('diagnostics');
const modePill = document.getElementById('mode-pill');

const state = {
  frames: [],
  head: null,
  emotion: { valence: 0.5, arousal: 0.5 },
  keywords: [],
  diagnostics: {},
  mode: 'analysis',
  palette: [],
  defaultColor: '#4b5563',
  particles: [],
  connected: false,
};

let ws = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let listening = false;

const TARGET_SAMPLE_RATE = 16_000;
const MODE_SETTINGS = {
  analysis: { widthScale: 0.9, glowScale: 1.0, pulse: 1.0 },
  performance: { widthScale: 1.5, glowScale: 1.8, pulse: 1.3 },
};

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.scale(dpr, dpr);
}

window.addEventListener('resize', () => {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  resizeCanvas();
});
resizeCanvas();

function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return Promise.resolve();
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/audio`;
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  return new Promise((resolve, reject) => {
    ws.onopen = () => {
      state.connected = true;
      resolve();
    };

    ws.onmessage = (event) => {
      handleMessage(event.data);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error', err);
      reject(err);
    };

    ws.onclose = () => {
      state.connected = false;
      listening = false;
      updateButtons();
      setTimeout(() => connectWebSocket(), 1500).catch(() => undefined);
    };
  });
}

async function startListening() {
  try {
    await connectWebSocket();
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
      const down = downsampleBuffer(input, audioContext.sampleRate, TARGET_SAMPLE_RATE);
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

function downsampleBuffer(buffer, inRate, outRate) {
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
    state.palette = data.palette || [];
    state.defaultColor = data.defaultColor || '#4b5563';
    return;
  }
  if (data.type === 'mode') {
    state.mode = data.mode || 'analysis';
    updateModeUI();
    return;
  }
  if (data.type === 'reset') {
    state.frames = [];
    state.head = null;
    state.particles = [];
    return;
  }
  if (data.type === 'frame_batch') {
    state.frames = data.frames || [];
    state.head = data.head || null;
    state.emotion = data.emotion || state.emotion;
    state.keywords = data.keywords || [];
    state.diagnostics = data.diagnostics || {};
    state.mode = data.mode || state.mode;
    const particles = data.particles || [];
    particles.forEach((particle) => {
      state.particles.push({
        x: particle.x,
        y: particle.y,
        color: particle.color,
        life: 1.0,
      });
    });
    updateBars();
    updateKeywords();
    updateDiagnostics();
    updateModeUI();
  }
}

function updateBars() {
  const clamp = (value) => Math.max(0, Math.min(1, value));
  valenceBar.style.transform = `scaleX(${clamp(state.emotion.valence)})`;
  arousalBar.style.transform = `scaleX(${clamp(state.emotion.arousal)})`;
}

function updateKeywords() {
  keywordList.innerHTML = '';
  const end = state.keywords.slice(-4);
  end.forEach((keyword) => {
    const li = document.createElement('li');
    li.textContent = keyword.text.toUpperCase();
    li.style.opacity = keyword.confidence ?? 0.6;
    keywordList.appendChild(li);
  });
  if (end.length === 0) {
    const empty = document.createElement('li');
    empty.textContent = '—';
    empty.style.opacity = 0.4;
    keywordList.appendChild(empty);
  }
}

function updateDiagnostics() {
  diagnosticsList.innerHTML = '';
  const entries = [
    { key: 'projector_ready', label: 'Projection', value: state.diagnostics.projector_ready },
    { key: 'speaker_ready', label: 'Speaker Clusters', value: state.diagnostics.speaker_ready },
    { key: 'keyword_ready', label: 'Keywords', value: state.diagnostics.keyword_ready },
  ];
  entries.forEach((entry) => {
    const li = document.createElement('li');
    const status = entry.value ? 'online' : 'warming';
    const color = entry.value ? 'rgba(125, 211, 252, 0.95)' : 'rgba(251, 191, 36, 0.85)';
    li.innerHTML = `<span style="color:${color};font-weight:600">●</span> ${entry.label}: ${status}`;
    diagnosticsList.appendChild(li);
  });
}

function updateModeUI() {
  const nextMode = state.mode === 'analysis' ? 'performance' : 'analysis';
  modeBtn.textContent = `${nextMode.charAt(0).toUpperCase() + nextMode.slice(1)} Mode`;
  modePill.textContent = `Mode: ${state.mode.charAt(0).toUpperCase() + state.mode.slice(1)}`;
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

resetBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
  state.frames = [];
  state.head = null;
  state.particles = [];
});

let lastFrame = performance.now();
function render(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  drawScene(dt);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

function withAlpha(hex, alpha) {
  const safeHex = hex && hex.startsWith('#') ? hex : state.defaultColor;
  const r = parseInt(safeHex.slice(1, 3), 16);
  const g = parseInt(safeHex.slice(3, 5), 16);
  const b = parseInt(safeHex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function drawScene(dt) {
  const width = canvas.width / (window.devicePixelRatio || 1);
  const height = canvas.height / (window.devicePixelRatio || 1);
  const { valence, arousal } = state.emotion;

  const baseHue = 200 + (valence - 0.5) * 80;
  const intensity = 0.08 + arousal * 0.18;
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, `rgba(${hslToRgb(baseHue, 60, 18)}, ${Math.min(0.45, intensity)})`);
  gradient.addColorStop(1, 'rgba(5, 9, 18, 0.95)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  const centerX = width / 2;
  const centerY = height / 2;
  const scale = Math.min(width, height) * 0.42;
  const settings = MODE_SETTINGS[state.mode] || MODE_SETTINGS.analysis;

  const frames = state.frames;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (let i = 1; i < frames.length; i += 1) {
    const prev = frames[i - 1];
    const curr = frames[i];
    const x0 = centerX + scale * prev.x;
    const y0 = centerY - scale * prev.y;
    const x1 = centerX + scale * curr.x;
    const y1 = centerY - scale * curr.y;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.lineWidth = Math.max(1.2, curr.width * settings.widthScale);
    ctx.strokeStyle = withAlpha(curr.color, Math.min(curr.alpha * 1.1, 1));
    ctx.shadowBlur = 40 * curr.glow * settings.glowScale;
    ctx.shadowColor = withAlpha(curr.color, Math.min(curr.glow * 1.4, 1));
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Particles / emphasis pulses
  const nextParticles = [];
  state.particles.forEach((particle) => {
    const decay = Math.exp(-dt * 2.8);
    particle.life *= decay;
    if (particle.life < 0.05) return;
    const px = centerX + scale * particle.x;
    const py = centerY - scale * particle.y;
    const radius = 4 + (1 - particle.life) * 12 * settings.pulse;
    ctx.beginPath();
    ctx.fillStyle = withAlpha(particle.color, particle.life);
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fill();
    nextParticles.push(particle);
  });
  state.particles = nextParticles;

  // Draw head glow
  if (state.head) {
    const hx = centerX + scale * state.head.x;
    const hy = centerY - scale * state.head.y;
    const radius = 8 + state.head.width * 2.5 * settings.widthScale;
    const headGlow = ctx.createRadialGradient(hx, hy, 0, hx, hy, radius * 2.6);
    headGlow.addColorStop(0, withAlpha(state.head.color, Math.min(0.75 + state.head.glow * 0.3, 1)));
    headGlow.addColorStop(1, 'rgba(5,6,16,0)');
    ctx.fillStyle = headGlow;
    ctx.beginPath();
    ctx.arc(hx, hy, radius * 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = 'rgba(241, 245, 249, 0.9)';
    ctx.font = '600 14px "Inter", system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    state.keywords.slice(-3).forEach((keyword, idx) => {
      const offsetY = (idx - 1) * 20;
      const alpha = Math.min(1, keyword.confidence ?? 0.7);
      ctx.fillStyle = `rgba(226, 232, 255, ${alpha})`;
      ctx.fillText(keyword.text.toUpperCase(), hx + 18, hy + offsetY);
    });
  }
}

function hslToRgb(h, s, l) {
  const saturation = s / 100;
  const lightness = l / 100;
  const k = (n) => (n + h / 30) % 12;
  const a = saturation * Math.min(lightness, 1 - lightness);
  const f = (n) => lightness - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return `${Math.round(f(0) * 255)}, ${Math.round(f(8) * 255)}, ${Math.round(f(4) * 255)}`;
}

updateButtons();
updateModeUI();
updateKeywords();
updateDiagnostics();
updateBars();
