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
  stars: [],
  connections: [],
  nebula: { hue: 210, intensity: 0.3, level: 0.5 },
  themes: [],
  transcript: '',
  diagnostics: {},
  mode: 'analysis',
  rhythm: [],
  drawLinks: true,
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
  if (data.type === 'constellation') {
    state.stars = data.stars || [];
    state.connections = data.connections || [];
    state.nebula = data.nebula || state.nebula;
    state.rhythm = data.rhythm || [];
    state.diagnostics = data.meta?.diagnostics || {};
    state.mode = data.meta?.mode || state.mode;
    state.transcript = data.meta?.transcript || '';
    state.themes = data.meta?.themes || [];
    updateTranscript();
    updateThemes();
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

function updateThemes() {
  const themePanel = document.getElementById('themes-list');
  if (!themePanel) {
    return;
  }
  themePanel.innerHTML = '';
  if (!state.themes.length) {
    const li = document.createElement('li');
    li.textContent = 'Themes forming…';
    li.style.opacity = 0.6;
    themePanel.appendChild(li);
    return;
  }
  state.themes.forEach((theme) => {
    const li = document.createElement('li');
    li.textContent = `${theme.text.toUpperCase()} ×${theme.count}`;
    themePanel.appendChild(li);
  });
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
  linksBtn.textContent = state.drawLinks ? 'Hide Links' : 'Show Links';
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
  state.stars = [];
  state.connections = [];
  state.keywords = [];
  state.themes = [];
  state.transcript = '';
  updateTranscript();
  updateKeywords();
  updateThemes();
});

let lastFrame = performance.now();
function render(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  drawScene(now / 1000, dt);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

function drawScene(timeSeconds, dt) {
  const width = canvas.width / (window.devicePixelRatio || 1);
  const height = canvas.height / (window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, width, height);

  drawNebula(width, height);
  drawRhythm(width, height, timeSeconds);

  if (!state.stars.length) {
    return;
  }

  const bounds = computeBounds(state.stars);
  const scaleX = bounds.scaleX * width * 0.45;
  const scaleY = bounds.scaleY * height * 0.45;
  const centerX = width / 2;
  const centerY = height / 2;

  const positions = new Map();
  state.stars.forEach((star) => {
    const px = centerX + (star.x - bounds.midX) * scaleX;
    const py = centerY - (star.y - bounds.midY) * scaleY;
    positions.set(star.id, { x: px, y: py });
  });

  if (state.drawLinks) {
    ctx.lineWidth = 1;
    ctx.lineCap = 'round';
    state.connections.forEach((edge) => {
      const from = positions.get(edge.source);
      const to = positions.get(edge.target);
      if (!from || !to) {
        return;
      }
      const alpha = Math.max(0.05, Math.min(0.8, edge.strength));
      ctx.strokeStyle = `rgba(148, 197, 255, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    });
  }

  state.stars.forEach((star) => {
    const pos = positions.get(star.id);
    if (!pos) {
      return;
    }
    const baseRadius = 4 + star.semantic * 10;
    const twinkle = 0.6 + Math.sin(timeSeconds * 4 + star.id) * 0.4 * star.twinkle;
    const radius = baseRadius * twinkle;

    const brightness = Math.max(0.3, star.brightness);
    const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius * 1.8);
    gradient.addColorStop(0, applyAlpha(star.color, 0.9));
    gradient.addColorStop(1, applyAlpha(star.color, 0.0));
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius * 1.8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = applyAlpha('#ffffff', brightness);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius * 0.6, 0, Math.PI * 2);
    ctx.fill();

    if (star.theme) {
      ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
      ctx.font = '600 11px "Inter", system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(star.theme.toUpperCase(), pos.x, pos.y - radius - 6);
    }
  });
}

function drawNebula(width, height) {
  const hue = state.nebula.hue ?? 210;
  const intensity = state.nebula.intensity ?? 0.3;
  const gradient = ctx.createRadialGradient(width / 2, height / 2, 0, width / 2, height / 2, Math.max(width, height));
  gradient.addColorStop(0, `hsla(${hue}, 70%, 12%, ${intensity})`);
  gradient.addColorStop(1, 'rgba(4, 7, 12, 0.95)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
}

function drawRhythm(width, height, timeSeconds) {
  if (!state.rhythm.length) {
    return;
  }
  const baseY = height - 40;
  const barHeight = 20;
  ctx.lineCap = 'round';
  state.rhythm.forEach((pulse) => {
    const age = Math.max(0, timeSeconds - pulse.t);
    if (age > 3) {
      return;
    }
    const alpha = Math.max(0, 1 - age / 3);
    const strength = pulse.strength ?? 0.6;
    const x = width / 2 + Math.sin(pulse.t * 2) * (width * 0.4);
    const thickness = 4 + strength * 14;
    ctx.strokeStyle = `rgba(148, 197, 255, ${alpha * 0.6})`;
    ctx.lineWidth = thickness;
    ctx.beginPath();
    ctx.moveTo(x, baseY);
    ctx.lineTo(x, baseY + barHeight);
    ctx.stroke();
  });
}

function computeBounds(stars) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  stars.forEach((star) => {
    minX = Math.min(minX, star.x);
    maxX = Math.max(maxX, star.x);
    minY = Math.min(minY, star.y);
    maxY = Math.max(maxY, star.y);
  });
  if (!Number.isFinite(minX) || minX === maxX) {
    minX = -1;
    maxX = 1;
  }
  if (!Number.isFinite(minY) || minY === maxY) {
    minY = -1;
    maxY = 1;
  }
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  const rangeX = Math.max(maxX - minX, 0.5);
  const rangeY = Math.max(maxY - minY, 0.5);
  return {
    midX,
    midY,
    scaleX: 1 / rangeX,
    scaleY: 1 / rangeY,
  };
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
updateThemes();
updateDiagnostics();
updateTranscript();
