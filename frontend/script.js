// Connects to backend WebSocket, sends frames, receives analysis, and updates UI.

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const fpsInput = document.getElementById('fpsInput');
const ctx = overlay.getContext('2d');

let localStream = null;
let ws = null;
let captureInterval = null;
let audioContext = null;
let audioSource = null;
let audioAnalyser = null;
let audioInterval = null;

function getWSUrl() {
  // Use location.host (includes port if present) so WS connects to same origin
  // This avoids constructing incorrect hostnames in Codespaces / forwarded URLs.
  const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
  const host = location.host || location.hostname || 'localhost';
  return `${proto}://${host}/ws`;
}

async function startCamera() {
  const msgEl = document.getElementById('messages');
  try {
    // Request microphone access as well to compute audio intensity locally.
    // We only compute RMS/intensity; no speech content is analyzed.
    localStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: true });
    video.srcObject = localStream;
    console.log('startCamera: stream assigned to video.srcObject', localStream);
    // Ensure video is visible and scales nicely
    video.style.objectFit = 'cover';
    video.style.visibility = 'visible';

    // Some browsers require an explicit play() call after assigning srcObject
    try {
      await video.play();
      console.log('startCamera: video.play() succeeded, paused=', video.paused);
    } catch (errPlay) {
      console.warn('startCamera: video.play() failed (may require user gesture)', errPlay);
    }

    // Wait for metadata/canplay to know natural video size, prefer loadedmetadata/canplay
    await new Promise((resolve) => {
      const onMeta = () => {
        // Use actual video dimensions for canvas overlay sizing
        overlay.width = video.videoWidth || 640;
        overlay.height = video.videoHeight || 480;
        overlay.style.width = overlay.width + 'px';
        overlay.style.height = overlay.height + 'px';
        video.removeEventListener('loadedmetadata', onMeta);
        video.removeEventListener('canplay', onMeta);
        resolve();
      };
      video.addEventListener('loadedmetadata', onMeta);
      video.addEventListener('canplay', onMeta);
      // fallback timeout in case events don't fire
      setTimeout(() => {
        overlay.width = video.videoWidth || 640;
        overlay.height = video.videoHeight || 480;
        overlay.style.width = overlay.width + 'px';
        overlay.style.height = overlay.height + 'px';
        resolve();
      }, 1200);
    });
    console.log('startCamera: video size', overlay.width, overlay.height, 'videoWidth', video.videoWidth);
    msgEl.innerText = '';
  } catch (err) {
    console.error('getUserMedia error', err);
    msgEl.innerText = 'Kamera açılamadı: ' + (err.message || err);
    throw err;
  }
}

function stopCamera() {
  if (localStream) {
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
  }
  // stop audio processing
  if (audioInterval) {
    clearInterval(audioInterval);
    audioInterval = null;
  }
  if (audioAnalyser) {
    audioAnalyser.disconnect();
    audioAnalyser = null;
  }
  if (audioSource) {
    audioSource.disconnect();
    audioSource = null;
  }
  if (audioContext) {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }
}

function connectWS() {
  const url = getWSUrl();
  console.log('connectWS: connecting to', url);
  document.getElementById('wsStatus').innerText = 'connecting';
  ws = new WebSocket(url);
  // Expose ws for debug in console
  window.__face_ws = ws;
  ws.addEventListener('error', (e) => console.warn('WebSocket error', e));

  ws.addEventListener('open', () => {
    console.log('WebSocket connected');
    document.getElementById('wsStatus').innerText = 'connected';
    // Start sending audio intensity once WS is open
    tryStartAudioProcessing();
  });

    // Fetch current config to sync UI toggles
    fetch('/api/state').then(r => r.json()).then(s => {
      // no-op for now; UI toggles control backend config explicitly
    }).catch(() => {});

  ws.addEventListener('error', (e) => {
    console.warn('WebSocket error', e);
    document.getElementById('wsStatus').innerText = 'error';
  });

  ws.addEventListener('message', (ev) => {
    // Log raw message for debugging
    console.log('ws message raw', ev.data);
    try {
      const p = JSON.parse(ev.data);
      console.log('ws message parsed', p);
      if (p.type === 'analysis') {
        updateStatus(p);
      }
    } catch (err) {
      console.warn('Invalid server message', err);
    }
  });

  ws.addEventListener('close', () => console.log('WebSocket closed'));
}

function updateStatus(p) {
  document.getElementById('faceDetected').innerText = p.face ? 'Yes' : 'No';
  if (p.features) {
    // Guard against missing numeric values from server
    const mf = p.features;
    document.getElementById('mouthOpen').innerText = (mf.mouth_open_ratio || 0).toFixed(3);
    document.getElementById('eyeOpen').innerText = (mf.eye_openness || 0).toFixed(3);
    document.getElementById('browIntensity').innerText = (mf.eyebrow_intensity || 0).toFixed(3);
  }

  // Audio
  if (typeof p.audio_intensity !== 'undefined') {
    document.getElementById('audioIntensity').innerText = (p.audio_intensity || 0).toFixed(3);
    document.getElementById('audioSpeaking').innerText = (p.audio_intensity && p.audio_intensity > 0.25) ? 'Yes' : 'No';
  }

  // Action Units
  if (p.action_units && document.getElementById('toggleAUs').checked) {
    const aus = p.action_units;
    const setAU = (idFill, idVal, key) => {
      const v = (aus[key] && aus[key].score) ? Number(aus[key].score) : 0.0;
      document.getElementById(idFill).style.width = Math.round(v * 100) + '%';
      document.getElementById(idVal).innerText = v.toFixed(2);
    }
    setAU('au1Fill','au1Val','AU1');
    setAU('au2Fill','au2Val','AU2');
    setAU('au4Fill','au4Val','AU4');
    setAU('au6Fill','au6Val','AU6');
    setAU('au12Fill','au12Val','AU12');
    setAU('au15Fill','au15Val','AU15');
  }

  // Append contextual events to micro timeline log for download
  if (p.contextual_events && p.contextual_events.length) {
    p.contextual_events.forEach(ev => appendTimelineEvent(ev));
  }

  // AU micro-expressions timeline (if enabled)
  if (document.getElementById('toggleMicros').checked) {
    if (p.au_micro_expressions && p.au_micro_expressions.length) {
      // display as list of recent events
      const lines = p.au_micro_expressions.map(e => `${e.au} @ ${new Date(e.start_time*1000).toISOString()} dur=${(e.duration*1000).toFixed(0)}ms peak=${e.peak.toFixed(2)}`);
      document.getElementById('microTimeline').innerText = lines.join('\n');
    } else {
      document.getElementById('microTimeline').innerText = '-';
    }
    if (p.micro_expressions) document.getElementById('facsMicro').innerText = JSON.stringify(p.micro_expressions, null, 2);
  } else {
    document.getElementById('microTimeline').innerText = '';
    document.getElementById('facsMicro').innerText = '';
  }

  // FACS feature-based micro-expressions
  if (p.micro_expressions) {
    document.getElementById('facsMicro').innerText = JSON.stringify(p.micro_expressions, null, 2);
  }

  // Contextual events
  if (p.contextual_events) {
    document.getElementById('contextList').innerText = JSON.stringify(p.contextual_events, null, 2);
  }
  // Raw landmarks (sample first 10)
  if (p.landmarks) {
    try {
      const keys = Object.keys(p.landmarks).filter(k => k !== 'face_width');
      const sample = {};
      keys.slice(0, 10).forEach(k => { sample[k] = p.landmarks[k]; });
      document.getElementById('landmarksRaw').innerText = JSON.stringify(sample, null, 2);
    } catch (e) { document.getElementById('landmarksRaw').innerText = '-'; }
  }

  // Normalized landmarks (sample)
  if (p.normalized_landmarks) {
    if (document.getElementById('toggleNormalized').checked) {
      try {
        const sample = p.normalized_landmarks.slice(0, 10);
        document.getElementById('landmarksNorm').innerText = JSON.stringify(sample, null, 2);
      } catch (e) { document.getElementById('landmarksNorm').innerText = '-'; }
    } else {
      document.getElementById('landmarksNorm').innerText = '';
    }
  }

  // Smoothed normalized landmarks (sample)
  if (p.smoothed_normalized_landmarks) {
    try {
      const sample = p.smoothed_normalized_landmarks.slice(0, 10);
      document.getElementById('landmarksSmooth').innerText = JSON.stringify(sample, null, 2);
    } catch (e) { document.getElementById('landmarksSmooth').innerText = '-'; }
  }

  // Audio visualization (bar)
  if (typeof p.audio_intensity !== 'undefined' && document.getElementById('toggleAudio').checked) {
    const intensity = p.audio_intensity || 0;
    let audioFill = document.querySelector('#audioViz .audioFill');
    if (!audioFill) {
      const audioViz = document.getElementById('audio');
      const wrap = document.createElement('div'); wrap.id='audioViz';
      wrap.innerHTML = '<div class="audioBar"><div class="audioFill"></div></div>';
      audioViz.appendChild(wrap);
      audioFill = document.querySelector('#audioViz .audioFill');
    }
    audioFill.style.width = Math.round(intensity*100) + '%';
  }
  // Draw landmarks if present
  if (p.landmarks && p.face) {
    drawLandmarks(p.landmarks);
  } else {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }
}

function drawLandmarks(pts) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.fillStyle = '#00ff7f';
  Object.keys(pts).forEach(k => {
    if (k === 'face_width') return;
    const pt = pts[k];
    if (!pt) return;
    // Support either [x,y] arrays or objects {x:.., y:..}
    const x = Array.isArray(pt) ? pt[0] : (pt.x ?? pt[0]);
    const y = Array.isArray(pt) ? pt[1] : (pt.y ?? pt[1]);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!video || video.paused || video.ended) return;
  // keep a lightweight counter to log occasionally
  if (typeof sendFrame._count === 'undefined') sendFrame._count = 0;
  sendFrame._count++;

  const c = document.createElement('canvas');
  c.width = overlay.width;
  c.height = overlay.height;
  const cctx = c.getContext('2d');
  cctx.drawImage(video, 0, 0, c.width, c.height);
  const data = c.toDataURL('image/jpeg', 0.6);
  ws.send(JSON.stringify({ type: 'frame', data }));
  if (sendFrame._count % 10 === 0) console.log('sendFrame: sent frame #', sendFrame._count);
}

function startCapture() {
  connectWS();
  const fps = Number(fpsInput.value) || 10;
  const intervalMs = Math.floor(1000 / Math.max(1, Math.min(fps, 20)));
  console.log('startCapture: fps', fps, 'intervalMs', intervalMs);
  captureInterval = setInterval(sendFrame, intervalMs);
}

function stopCapture() {
  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  if (!localStream) await startCamera();
  startCapture();
  // ensure UI toggles reflect initial state
  document.getElementById('toggleOverlay').addEventListener('change', (e) => { overlay.style.display = e.target.checked ? 'block' : 'none'; });
  document.getElementById('toggleNormalized').addEventListener('change', (e) => {});
  document.getElementById('toggleAUs').addEventListener('change', (e) => { document.getElementById('auBars').style.display = e.target.checked ? 'block' : 'none'; });
  document.getElementById('toggleMicros').addEventListener('change', (e) => { document.getElementById('microTimeline').style.display = e.target.checked ? 'block' : 'none'; document.getElementById('facsMicro').style.display = e.target.checked ? 'block' : 'none'; });
  document.getElementById('toggleAudio').addEventListener('change', (e) => { document.getElementById('audio').style.display = e.target.checked ? 'block' : 'none'; });
  // initialize visibility
  overlay.style.display = document.getElementById('toggleOverlay').checked ? 'block' : 'none';
  document.getElementById('auBars').style.display = document.getElementById('toggleAUs').checked ? 'block' : 'none';
  document.getElementById('microTimeline').style.display = document.getElementById('toggleMicros').checked ? 'block' : 'none';
  document.getElementById('facsMicro').style.display = document.getElementById('toggleMicros').checked ? 'block' : 'none';
  document.getElementById('audio').style.display = document.getElementById('toggleAudio').checked ? 'block' : 'none';
  // when toggle changes, send config to server
  ['toggleAUs','toggleMicros','toggleNormalized','toggleOverlay','toggleAudio'].forEach(id => {
    document.getElementById(id).addEventListener('change', sendConfigToServer);
  });
  document.getElementById('downloadTimeline').addEventListener('click', downloadTimeline);
  // Send initial config to backend based on toggles
  sendConfigToServer();
});

stopBtn.addEventListener('click', () => {
  startBtn.disabled = false;
  stopBtn.disabled = true;
  stopCapture();
  stopCamera();
});

// Auto connect when opening page
window.addEventListener('load', async () => {
  const msgEl = document.getElementById('messages');
  // If camera permission already granted, auto-start for convenience
  try {
    if (navigator.permissions) {
      const status = await navigator.permissions.query({ name: 'camera' });
      if (status.state === 'granted') {
        try {
          await startCamera();
          startCapture();
          msgEl.innerText = '';
        } catch (err) {
          console.warn('Auto-start failed', err);
        }
      }
    }
  } catch (err) {
    // permission API may not be available or allowed
  }
});

function tryStartAudioProcessing() {
  if (!localStream || !ws) return;
  try {
    if (audioContext) return; // already started
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    audioSource = audioContext.createMediaStreamSource(localStream);
    audioAnalyser = audioContext.createAnalyser();
    audioAnalyser.fftSize = 2048;
    audioSource.connect(audioAnalyser);

    const buf = new Float32Array(audioAnalyser.fftSize);
    audioInterval = setInterval(() => {
      audioAnalyser.getFloatTimeDomainData(buf);
      // compute RMS
      let sum = 0.0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length + 1e-12);
      const db = 20 * Math.log10(rms + 1e-12);
      // map db floor -60..0 to 0..1
      const minDb = -60.0;
      let intensity = (db - minDb) / (0 - minDb);
      if (!isFinite(intensity)) intensity = 0.0;
      intensity = Math.max(0, Math.min(1, intensity));
      const speaking = intensity > 0.25;
      // send audio context to server (type: audio)
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'audio', intensity }));
      }
      // update local UI immediately
      document.getElementById('audioIntensity').innerText = intensity.toFixed(3);
      document.getElementById('audioSpeaking').innerText = speaking ? 'Yes' : 'No';
    }, 100);
  } catch (err) {
    console.warn('Audio processing start failed', err);
  }
}

// Timeline storage and download
const timelineEvents = [];
function appendTimelineEvent(ev) {
  timelineEvents.push(ev);
  // render last 20 events
  const out = timelineEvents.slice(-20).map(e => `${e.type||e.au||'event'} @ ${new Date((e.start_time||e.timestamp||Date.now())*1000).toISOString()} score=${(e.score||e.peak||'').toString()}`);
  const el = document.getElementById('microTimeline');
  el.innerText = out.join('\n');
}

// Download timeline as JSON
function downloadTimeline() {
  const blob = new Blob([JSON.stringify(timelineEvents, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'timeline.json';
  a.click();
  URL.revokeObjectURL(url);
}

function sendConfigToServer() {
  // Map UI toggles to backend processing flags
  const cfg = {
    enable_aus: document.getElementById('toggleAUs').checked,
    enable_au_micro: document.getElementById('toggleMicros').checked,
    enable_contextual: true, // keep contextual on; user can control via server API later
    enable_smoothing: document.getElementById('toggleNormalized').checked,
    enable_facs: true,
  };
  fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg) })
    .then(r => r.json()).then(j => console.log('config updated', j)).catch(e => console.warn('config update failed', e));
}
