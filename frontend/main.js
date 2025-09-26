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
  L10: { label: 'L10 · Semantic Processing (high-level meaning)' },
  L6: { label: 'L6 · Prosodic Features (rhythm, emotion)' },
  L2: { label: 'L2 · Acoustic Features (low-level sounds)' },
};
const AUDIO_STYLE = { color: '#38bdf8', offset: 10, label: 'spectrogram' };

// FFT and Spectrogram configuration
const FFT_SIZE = 512;
const FREQ_BINS = FFT_SIZE / 2; // 256 frequency bins
const MAX_FREQ = TARGET_SAMPLE_RATE / 2; // Nyquist frequency (8kHz)

// Viridis colormap for better perceptual uniformity
const VIRIDIS_GRADIENT = [
  { stop: 0.0, color: [68, 1, 84] },       // Dark purple
  { stop: 0.125, color: [71, 44, 122] },   // Purple  
  { stop: 0.25, color: [59, 81, 139] },    // Blue-purple
  { stop: 0.375, color: [44, 113, 142] },  // Blue-green
  { stop: 0.5, color: [33, 144, 141] },    // Teal
  { stop: 0.625, color: [39, 173, 129] },  // Green
  { stop: 0.75, color: [92, 200, 99] },    // Light green
  { stop: 0.875, color: [170, 220, 50] },  // Yellow-green
  { stop: 1.0, color: [253, 231, 37] },    // Bright yellow
];

// Magma colormap as alternative
const MAGMA_GRADIENT = [
  { stop: 0.0, color: [0, 0, 4] },         // Black
  { stop: 0.125, color: [28, 16, 68] },    // Dark purple
  { stop: 0.25, color: [79, 18, 123] },    // Purple
  { stop: 0.375, color: [129, 37, 129] },  // Magenta
  { stop: 0.5, color: [181, 54, 122] },    // Pink
  { stop: 0.625, color: [229, 80, 100] },  // Coral
  { stop: 0.75, color: [251, 135, 97] },   // Orange
  { stop: 0.875, color: [254, 194, 135] }, // Light orange
  { stop: 1.0, color: [252, 253, 191] },   // Pale yellow
];

const HEATMAP_LEFT_MARGIN = 120;
const HEATMAP_TOP_MARGIN = 150;
const HEATMAP_BOTTOM_MARGIN = 50;
const HEATMAP_MIN_BAND_HEIGHT = 100;
const PC_TRACK_HEIGHT = 40;
const STATS_TRACK_HEIGHT = 30;
const TIMELINE_HEIGHT = 40;

// FFT implementation for spectrogram computation
function fft(real, imag) {
  const n = real.length;
  if (n <= 1) return;
  
  // Bit-reverse reordering
  for (let i = 0; i < n; i++) {
    let j = 0;
    for (let k = 0; k < Math.log2(n); k++) {
      j = (j << 1) | ((i >> k) & 1);
    }
    if (j > i) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  
  // Cooley-Tukey FFT
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const w_real = Math.cos(2 * Math.PI / size);
    const w_imag = -Math.sin(2 * Math.PI / size);
    
    for (let i = 0; i < n; i += size) {
      let wr = 1, wi = 0;
      for (let j = 0; j < halfSize; j++) {
        const u_real = real[i + j];
        const u_imag = imag[i + j];
        const v_real = real[i + j + halfSize] * wr - imag[i + j + halfSize] * wi;
        const v_imag = real[i + j + halfSize] * wi + imag[i + j + halfSize] * wr;
        
        real[i + j] = u_real + v_real;
        imag[i + j] = u_imag + v_imag;
        real[i + j + halfSize] = u_real - v_real;
        imag[i + j + halfSize] = u_imag - v_imag;
        
        const wr_new = wr * w_real - wi * w_imag;
        wi = wr * w_imag + wi * w_real;
        wr = wr_new;
      }
    }
  }
}

// Color interpolation for gradient maps
function interpolateColor(gradient, value) {
  value = Math.max(0, Math.min(1, value));
  
  let lowerBound = gradient[0];
  let upperBound = gradient[gradient.length - 1];
  
  for (let i = 0; i < gradient.length - 1; i++) {
    if (value >= gradient[i].stop && value <= gradient[i + 1].stop) {
      lowerBound = gradient[i];
      upperBound = gradient[i + 1];
      break;
    }
  }
  
  const range = upperBound.stop - lowerBound.stop;
  const t = range > 0 ? (value - lowerBound.stop) / range : 0;
  
  const color = [
    Math.round(lowerBound.color[0] + (upperBound.color[0] - lowerBound.color[0]) * t),
    Math.round(lowerBound.color[1] + (upperBound.color[1] - lowerBound.color[1]) * t),
    Math.round(lowerBound.color[2] + (upperBound.color[2] - lowerBound.color[2]) * t),
  ];
  
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

// WebSocket and data management
let layerHistory = {};
let audioHistory = { times: [], rms: [], raw_audio: [] };
let keywords = [];
let currentTranscript = '';
let speakerColors = {};
let neuronAnalysisData = {};  // Store neuron sorting and PC data
let ws = null;
let isRecording = false;
let audioContext = null;
let processor = null;
let source = null;
let showProjection = true;

// Z-score normalization per neuron
function zScoreNormalize(vectors) {
  if (!vectors || vectors.length === 0) return vectors;
  
  const n_time = vectors.length;
  const n_neurons = vectors[0].length;
  const normalized = [];
  
  for (let t = 0; t < n_time; t++) {
    normalized.push(new Array(n_neurons));
  }
  
  // Calculate mean and std per neuron
  for (let n = 0; n < n_neurons; n++) {
    let sum = 0;
    let sumSq = 0;
    
    for (let t = 0; t < n_time; t++) {
      const val = vectors[t][n];
      sum += val;
      sumSq += val * val;
    }
    
    const mean = sum / n_time;
    const variance = (sumSq / n_time) - (mean * mean);
    const std = Math.sqrt(Math.max(0, variance)) || 1;
    
    // Normalize
    for (let t = 0; t < n_time; t++) {
      normalized[t][n] = (vectors[t][n] - mean) / std;
    }
  }
  
  return normalized;
}

// Apply neuron sorting to vectors
function applySorting(vectors, sortedIndices) {
  if (!sortedIndices || sortedIndices.length === 0) return vectors;
  
  return vectors.map(vec => {
    const sorted = new Array(vec.length);
    for (let i = 0; i < sortedIndices.length && i < vec.length; i++) {
      sorted[i] = vec[sortedIndices[i]];
    }
    return sorted;
  });
}

// Draw timeline with landmarks
function drawTimeline(ctx, x, y, width, height, times, keywords) {
  ctx.save();
  
  // Background
  ctx.fillStyle = 'rgba(30, 30, 30, 0.5)';
  ctx.fillRect(x, y, width, height);
  
  // Draw time axis
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, y + height);
  ctx.lineTo(x + width, y + height);
  ctx.stroke();
  
  // Draw time ticks and labels
  if (times && times.length > 0) {
    const maxTime = times[times.length - 1];
    const tickInterval = 1; // 1 second intervals
    
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    
    for (let t = 0; t <= maxTime; t += tickInterval) {
      const xPos = x + (t / maxTime) * width;
      
      // Draw tick
      ctx.beginPath();
      ctx.moveTo(xPos, y + height - 5);
      ctx.lineTo(xPos, y + height);
      ctx.stroke();
      
      // Draw label
      ctx.fillText(`${t}s`, xPos - 10, y + height - 8);
    }
  }
  
  // Draw keywords as markers
  if (keywords && keywords.length > 0 && times && times.length > 0) {
    const maxTime = times[times.length - 1];
    
    keywords.forEach(kw => {
      const xPos = x + ((kw.t - times[0]) / maxTime) * width;
      
      // Draw marker
      ctx.fillStyle = `rgba(255, 200, 50, ${kw.confidence})`;
      ctx.fillRect(xPos - 1, y, 2, height - 10);
      
      // Draw text
      ctx.save();
      ctx.font = '11px sans-serif';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.translate(xPos, y + 15);
      ctx.rotate(-Math.PI / 6);
      ctx.fillText(kw.text, 0, 0);
      ctx.restore();
    });
  }
  
  ctx.restore();
}

// Draw PC summary tracks
function drawPCTracks(ctx, x, y, width, height, pcData, times, label) {
  if (!pcData || !pcData.pc_timeseries || pcData.pc_timeseries.length === 0) return;
  
  ctx.save();
  
  // Background
  ctx.fillStyle = 'rgba(40, 40, 40, 0.3)';
  ctx.fillRect(x, y, width, height);
  
  // Draw label
  ctx.font = '11px sans-serif';
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillText(`${label} PCs`, x + 5, y + 12);
  
  const pcColors = ['#ff6b6b', '#4ecdc4', '#45b7d1'];
  const n_pcs = Math.min(3, pcData.pc_timeseries[0].length);
  const pcHeight = (height - 15) / n_pcs;
  
  // Draw each PC
  for (let pc = 0; pc < n_pcs; pc++) {
    const yOffset = y + 15 + pc * pcHeight + pcHeight / 2;
    
    ctx.strokeStyle = pcColors[pc];
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.8;
    
    ctx.beginPath();
    for (let t = 0; t < pcData.pc_timeseries.length; t++) {
      const xPos = x + (t / pcData.pc_timeseries.length) * width;
      const value = pcData.pc_timeseries[t][pc];
      const yPos = yOffset - (value * pcHeight * 0.4); // Scale to fit
      
      if (t === 0) {
        ctx.moveTo(xPos, yPos);
      } else {
        ctx.lineTo(xPos, yPos);
      }
    }
    ctx.stroke();
    
    // Draw variance label
    if (pcData.explained_variance && pcData.explained_variance[pc]) {
      const variance = (pcData.explained_variance[pc] * 100).toFixed(1);
      ctx.font = '9px monospace';
      ctx.fillStyle = pcColors[pc];
      ctx.fillText(`PC${pc+1}: ${variance}%`, x + width - 60, yOffset);
    }
  }
  
  ctx.restore();
}

// Draw layer statistics track
function drawStatsTrack(ctx, x, y, width, height, stats, label) {
  if (!stats) return;
  
  ctx.save();
  
  // Background
  ctx.fillStyle = 'rgba(30, 30, 30, 0.2)';
  ctx.fillRect(x, y, width, height);
  
  // Draw sparsity
  if (stats.sparsity) {
    ctx.strokeStyle = 'rgba(255, 150, 50, 0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    for (let t = 0; t < stats.sparsity.length; t++) {
      const xPos = x + (t / stats.sparsity.length) * width;
      const yPos = y + height - (stats.sparsity[t] * height);
      
      if (t === 0) {
        ctx.moveTo(xPos, yPos);
      } else {
        ctx.lineTo(xPos, yPos);
      }
    }
    ctx.stroke();
  }
  
  // Draw change rate
  if (stats.change_rate) {
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.7)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    // Normalize change rate
    const maxChange = Math.max(...stats.change_rate);
    
    for (let t = 0; t < stats.change_rate.length; t++) {
      const xPos = x + (t / stats.change_rate.length) * width;
      const yPos = y + height - ((stats.change_rate[t] / maxChange) * height);
      
      if (t === 0) {
        ctx.moveTo(xPos, yPos);
      } else {
        ctx.lineTo(xPos, yPos);
      }
    }
    ctx.stroke();
  }
  
  ctx.restore();
}

// Enhanced heatmap drawing with sorted neurons
function drawHeatmap(ctx, x, y, width, height, layer) {
  const { vectors, times } = layer;
  if (!vectors || vectors.length === 0) return;
  
  // Get neuron analysis data if available
  const analysis = layer.neuron_analysis;
  let processedVectors = vectors;
  
  // Apply neuron sorting if available
  if (analysis && analysis.sorted_indices) {
    processedVectors = applySorting(vectors, analysis.sorted_indices);
  }
  
  // Apply z-score normalization
  processedVectors = zScoreNormalize(processedVectors);
  
  const n_neurons = processedVectors[0].length;
  const n_time = processedVectors.length;
  const neuronHeight = height / n_neurons;
  const timeWidth = width / n_time;
  
  // Draw heatmap
  for (let t = 0; t < n_time; t++) {
    for (let n = 0; n < n_neurons; n++) {
      const value = processedVectors[t][n];
      // Map z-score to [0, 1] using tanh for better contrast
      const normalizedValue = (Math.tanh(value / 2) + 1) / 2;
      
      ctx.fillStyle = interpolateColor(VIRIDIS_GRADIENT, normalizedValue);
      ctx.fillRect(
        x + t * timeWidth,
        y + n * neuronHeight,
        Math.ceil(timeWidth) + 1,
        Math.ceil(neuronHeight) + 1
      );
    }
  }
  
  // Draw neuron grouping indicators if we have clustering info
  if (analysis && analysis.sorted_indices) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    
    // Draw subtle dividers between neuron groups (every 10 neurons for now)
    for (let n = 10; n < n_neurons; n += 10) {
      ctx.beginPath();
      ctx.moveTo(x, y + n * neuronHeight);
      ctx.lineTo(x + width, y + n * neuronHeight);
      ctx.stroke();
    }
  }
}

// Main drawing function
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Background
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  const heatmapWidth = canvas.width - HEATMAP_LEFT_MARGIN - 50;
  
  // Draw title and status
  ctx.font = 'bold 18px sans-serif';
  ctx.fillStyle = '#ffffff';
  ctx.fillText('Neural Audio Visualization', 20, 30);
  
  ctx.font = '12px sans-serif';
  ctx.fillStyle = '#888888';
  ctx.fillText(`Mode: ${showProjection ? 'Analysis' : 'Performance'}`, 20, 50);
  
  // Draw transcript
  if (currentTranscript) {
    ctx.font = '14px sans-serif';
    ctx.fillStyle = '#cccccc';
    const lines = wrapText(ctx, currentTranscript, canvas.width - 40);
    lines.forEach((line, i) => {
      ctx.fillText(line, 20, 80 + i * 18);
    });
  }
  
  // Calculate layout positions
  let currentY = HEATMAP_TOP_MARGIN;
  
  // Draw timeline at top
  if (audioHistory.times && audioHistory.times.length > 0) {
    drawTimeline(ctx, HEATMAP_LEFT_MARGIN, currentY, heatmapWidth, TIMELINE_HEIGHT, 
                 audioHistory.times, keywords);
    currentY += TIMELINE_HEIGHT + 10;
  }
  
  // Draw spectrogram
  const spectrogramHeight = 80;
  drawSpectrogram(ctx, HEATMAP_LEFT_MARGIN, currentY, heatmapWidth, spectrogramHeight);
  
  // Draw spectrogram label
  ctx.font = '12px sans-serif';
  ctx.fillStyle = AUDIO_STYLE.color;
  ctx.fillText(AUDIO_STYLE.label, 10, currentY + spectrogramHeight / 2);
  
  currentY += spectrogramHeight + 20;
  
  // Calculate space for each layer
  const remainingHeight = canvas.height - currentY - HEATMAP_BOTTOM_MARGIN;
  const layerTotalHeight = remainingHeight / LAYER_ORDER.length;
  
  // Draw each layer with its components
  LAYER_ORDER.forEach((layerName, idx) => {
    const layer = layerHistory[layerName];
    if (!layer || !layer.vectors || layer.vectors.length === 0) return;
    
    const layerY = currentY + idx * layerTotalHeight;
    const heatmapHeight = Math.min(HEATMAP_MIN_BAND_HEIGHT, layerTotalHeight - PC_TRACK_HEIGHT - STATS_TRACK_HEIGHT - 20);
    
    // Draw layer label
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(LAYER_CONFIG[layerName].label, 10, layerY + heatmapHeight / 2);
    
    // Draw heatmap
    drawHeatmap(ctx, HEATMAP_LEFT_MARGIN, layerY, heatmapWidth, heatmapHeight, layer);
    
    // Draw PC tracks if available
    if (layer.neuron_analysis) {
      drawPCTracks(ctx, HEATMAP_LEFT_MARGIN, layerY + heatmapHeight + 5, 
                   heatmapWidth, PC_TRACK_HEIGHT, layer.neuron_analysis, 
                   layer.times, layerName);
      
      // Draw stats track
      if (layer.neuron_analysis.stats) {
        drawStatsTrack(ctx, HEATMAP_LEFT_MARGIN, 
                      layerY + heatmapHeight + PC_TRACK_HEIGHT + 10,
                      heatmapWidth, STATS_TRACK_HEIGHT, 
                      layer.neuron_analysis.stats, layerName);
      }
    }
  });
  
  requestAnimationFrame(draw);
}

// Spectrogram drawing function
function drawSpectrogram(ctx, x, y, width, height) {
  if (!audioHistory.raw_audio || audioHistory.raw_audio.length === 0) return;
  
  // Background
  ctx.fillStyle = 'rgba(20, 20, 30, 0.5)';
  ctx.fillRect(x, y, width, height);
  
  const spectrogramData = computeSpectrogram(audioHistory.raw_audio);
  if (!spectrogramData || spectrogramData.length === 0) return;
  
  const timeSteps = spectrogramData.length;
  const freqBins = spectrogramData[0].length;
  const timeWidth = width / timeSteps;
  const freqHeight = height / freqBins;
  
  // Draw spectrogram
  for (let t = 0; t < timeSteps; t++) {
    for (let f = 0; f < freqBins; f++) {
      const magnitude = spectrogramData[t][f];
      const normalizedMag = Math.log(1 + magnitude) / 10; // Log scale
      
      ctx.fillStyle = interpolateColor(MAGMA_GRADIENT, normalizedMag);
      ctx.fillRect(
        x + t * timeWidth,
        y + height - (f + 1) * freqHeight,
        Math.ceil(timeWidth) + 1,
        Math.ceil(freqHeight) + 1
      );
    }
  }
}

// Compute spectrogram from raw audio
function computeSpectrogram(audioFrames) {
  if (!audioFrames || audioFrames.length === 0) return [];
  
  const spectrogram = [];
  const hopSize = FFT_SIZE / 2;
  
  for (let i = 0; i < audioFrames.length; i++) {
    const frame = audioFrames[i];
    if (!frame || frame.length < FFT_SIZE) continue;
    
    // Apply Hanning window
    const windowed = new Float32Array(FFT_SIZE);
    for (let j = 0; j < FFT_SIZE; j++) {
      const window = 0.5 - 0.5 * Math.cos(2 * Math.PI * j / (FFT_SIZE - 1));
      windowed[j] = frame[j] * window;
    }
    
    // Compute FFT
    const real = Array.from(windowed);
    const imag = new Array(FFT_SIZE).fill(0);
    fft(real, imag);
    
    // Calculate magnitude spectrum
    const magnitudes = [];
    for (let j = 0; j < FREQ_BINS; j++) {
      const mag = Math.sqrt(real[j] * real[j] + imag[j] * imag[j]);
      magnitudes.push(mag);
    }
    
    spectrogram.push(magnitudes);
  }
  
  return spectrogram;
}

// Text wrapping helper
function wrapText(ctx, text, maxWidth) {
  const words = text.split(' ');
  const lines = [];
  let currentLine = '';
  
  for (const word of words) {
    const testLine = currentLine ? `${currentLine} ${word}` : word;
    const metrics = ctx.measureText(testLine);
    
    if (metrics.width > maxWidth && currentLine) {
      lines.push(currentLine);
      currentLine = word;
    } else {
      currentLine = testLine;
    }
  }
  
  if (currentLine) {
    lines.push(currentLine);
  }
  
  return lines;
}

// WebSocket message handling
function handleMessage(data) {
  if (data.type === 'layer_activity') {
    // Update layer history
    data.layers.forEach(layer => {
      layerHistory[layer.name] = layer;
    });
    
    // Update audio history
    if (data.audio) {
      audioHistory = data.audio;
    }
    
    // Update metadata
    if (data.meta) {
      if (data.meta.transcript) {
        currentTranscript = data.meta.transcript;
        transcriptText.textContent = currentTranscript;
      }
      
      if (data.meta.keywords) {
        keywords = data.meta.keywords;
      }
      
      if (data.meta.speaker_colors) {
        speakerColors = data.meta.speaker_colors;
      }
      
      // Update diagnostics
      if (data.meta.diagnostics) {
        const diag = data.meta.diagnostics;
        diagnosticsList.innerHTML = `
          <li>Projector: ${diag.projector_ready ? '✓' : '○'}</li>
          <li>Speaker: ${diag.speaker_ready ? '✓' : '○'}</li>
          <li>Keywords: ${diag.keyword_ready ? '✓' : '○'}</li>
        `;
      }
    }
  }
}

// WebSocket connection
function connectWebSocket() {
  ws = new WebSocket('ws://localhost:8000/ws/audio');
  
  ws.onopen = () => {
    console.log('WebSocket connected');
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleMessage(data);
    } catch (e) {
      console.error('Failed to parse message:', e);
    }
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  ws.onclose = () => {
    console.log('WebSocket disconnected');
    if (isRecording) {
      setTimeout(connectWebSocket, 1000);
    }
  };
}

// Audio processing setup
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    source = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    
    processor.onaudioprocess = (e) => {
      if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
      
      const inputData = e.inputBuffer.getChannelData(0);
      const samples = new Float32Array(inputData);
      
      // Send audio data to server
      ws.send(samples.buffer);
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
    
    isRecording = true;
    startBtn.textContent = 'Stop';
    startBtn.classList.add('recording');
    
    connectWebSocket();
  } catch (error) {
    console.error('Failed to start recording:', error);
    alert('Failed to access microphone');
  }
}

function stopRecording() {
  isRecording = false;
  
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  
  if (source) {
    source.disconnect();
    source = null;
  }
  
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  
  if (ws) {
    ws.close();
    ws = null;
  }
  
  startBtn.textContent = 'Start';
  startBtn.classList.remove('recording');
}

// Event listeners
startBtn.addEventListener('click', () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

modeBtn.addEventListener('click', () => {
  const currentMode = modePill.textContent.includes('Analysis') ? 'analysis' : 'performance';
  const newMode = currentMode === 'analysis' ? 'performance' : 'analysis';
  modePill.textContent = `Mode: ${newMode.charAt(0).toUpperCase() + newMode.slice(1)}`;
  
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'mode', mode: newMode }));
  }
});

resetBtn.addEventListener('click', () => {
  layerHistory = {};
  audioHistory = { times: [], rms: [], raw_audio: [] };
  keywords = [];
  currentTranscript = '';
  speakerColors = {};
  neuronAnalysisData = {};
  
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
});

gridBtn.addEventListener('click', () => {
  showProjection = !showProjection;
  gridBtn.textContent = showProjection ? 'Hide Grid' : 'Show Grid';
});

// Canvas setup
function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight - 100;
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Start animation loop
draw();
