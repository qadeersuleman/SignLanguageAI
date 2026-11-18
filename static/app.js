/* SignBridge AI — Real-Time Frontend Controller */

const state = {
  signs: [],
  signMap: {},
  voiceEnabled: true,
  voiceEngine: "browser", // "browser" | "python"
  fullPhrase: true,
  autoSentence: true,
  speechRate: 1.0,
  speechPitch: 1.0,
  cooldownSec: 2.8,
  selectedVoiceURI: "",
  showSkeleton: true,
  mirror: true,
  paused: false,
  lastSpokenSign: null,
  lastSpokenAt: 0,
  currentSign: null,
  sentenceWords: [],
  history: [],
  ws: null,
  pythonTtsBackend: "unknown",
  manualPreviewUntil: 0,
};

const el = (id) => document.getElementById(id);

function showToast(msg) {
  const toast = el("toast");
  toast.textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => toast.classList.add("hidden"), 2600);
}

/* ==================== VOICE OUTPUT ENGINE ==================== */
function setSpeakingVisual(active) {
  const viz = el("voiceVisualizer");
  const step5 = el("step5");
  if (active) {
    viz.classList.add("speaking");
    step5.classList.add("active");
    el("metricStep5").textContent = "Speaking output aloud...";
  } else {
    viz.classList.remove("speaking");
    el("metricStep5").textContent = state.voiceEnabled ? "Voice & Screen Ready" : "Screen Only (Muted)";
  }
}

function loadBrowserVoices() {
  if (!("speechSynthesis" in window)) return;
  const voices = window.speechSynthesis.getVoices();
  const select = el("voiceSelect");
  if (!select || !voices.length) return;

  const prev = state.selectedVoiceURI || select.value;
  select.innerHTML = "";
  const sorted = [...voices].sort((a, b) => {
    const aEn = a.lang.toLowerCase().startsWith("en") ? 0 : 1;
    const bEn = b.lang.toLowerCase().startsWith("en") ? 0 : 1;
    return aEn - bEn || a.name.localeCompare(b.name);
  });

  sorted.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v.voiceURI;
    opt.textContent = `${v.name} (${v.lang})`;
    select.appendChild(opt);
  });

  if (prev && sorted.some((v) => v.voiceURI === prev)) {
    select.value = prev;
  } else if (sorted.length) {
    state.selectedVoiceURI = sorted[0].voiceURI;
    select.value = sorted[0].voiceURI;
  }
}

async function speakText(text, force = false) {
  if (!state.voiceEnabled && !force) return;
  const clean = (text || "").trim();
  if (!clean) return;

  if (state.voiceEngine === "python") {
    setSpeakingVisual(true);
    try {
      await fetch("/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: clean, rate: state.speechRate, force }),
      });
    } catch (e) {
      console.error("Python TTS error:", e);
    } finally {
      setTimeout(() => setSpeakingVisual(false), Math.max(900, clean.length * 65));
    }
    return;
  }

  // Browser Web Speech API
  if (!("speechSynthesis" in window)) {
    // Fallback automatically to Python TTS if browser lacks speechSynthesis
    state.voiceEngine = "python";
    return speakText(clean, force);
  }

  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(clean);
    utter.rate = state.speechRate;
    utter.pitch = state.speechPitch;
    const voices = window.speechSynthesis.getVoices();
    const match = voices.find((v) => v.voiceURI === state.selectedVoiceURI);
    if (match) utter.voice = match;

    utter.onstart = () => setSpeakingVisual(true);
    utter.onend = () => setSpeakingVisual(false);
    utter.onerror = () => setSpeakingVisual(false);
    window.speechSynthesis.speak(utter);
  } catch (err) {
    console.error("Browser speech error:", err);
  }
}

/* ==================== UI RENDERING ==================== */
function renderSignsGrid() {
  const grid = el("signsGrid");
  const select = el("trainerSignSelect");
  grid.innerHTML = "";
  select.innerHTML = "";

  state.signs.forEach((sign) => {
    const card = document.createElement("button");
    card.className = "sign-card";
    card.dataset.sign = sign.id;
    card.style.setProperty("--card-color", sign.color);
    card.style.setProperty("--card-tint", sign.bg_tint);
    card.title = `${sign.label}: ${sign.how_to}`;
    card.innerHTML = `
      <div class="sign-emoji">${sign.emoji}</div>
      <div class="sign-ribbon">${sign.label}</div>
      <div class="sign-badge">${sign.badge}</div>
    `;
    card.addEventListener("click", () => previewSignCard(sign));
    grid.appendChild(card);

    const opt = document.createElement("option");
    opt.value = sign.id;
    opt.textContent = `${sign.emoji} ${sign.label}`;
    select.appendChild(opt);
  });
}

function previewSignCard(sign) {
  state.manualPreviewUntil = Date.now() + 3000;
  state.currentSign = sign.id;
  highlightSignCard(sign.id);
  updatePredictionHud(sign, 1.0);
  updatePipelineSteps({
    cameraActive: true,
    handsCount: 1,
    inferenceMs: 2.1,
    modelAccuracy: 99.0,
    sign: sign.id,
    confidence: 1.0,
    speaking: true,
    fps: 30,
  });

  const phrase = state.fullPhrase ? sign.spoken_text : sign.short_speech;
  speakText(phrase, true);
  pushHistory(sign);
  if (state.autoSentence) appendSentenceWord(sign.id);

  el("activeSignTip").innerHTML = `
    <div class="tip-icon">${sign.emoji}</div>
    <div class="tip-text">
      <strong>${sign.label} (${sign.meaning}):</strong> ${sign.how_to}
    </div>
  `;
}

function highlightSignCard(signId) {
  document.querySelectorAll(".sign-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.sign === signId);
  });
}

function updatePredictionHud(signObj, confidence) {
  const hud = el("predictionHud");
  const camCard = el("cameraCard");

  if (!signObj) {
    hud.classList.remove("detected");
    hud.style.borderColor = "rgba(148, 163, 184, 0.28)";
    camCard.style.borderColor = "";
    el("predEmoji").textContent = "🤖";
    el("predWord").textContent = "Show a Hand Sign";
    el("predConf").textContent = "--%";
    el("predPhrase").textContent = "Hold any of the 9 signs steady in front of your webcam";
    el("confBarFill").style.width = "0%";
    return;
  }

  const pct = Math.round(confidence * 100);
  hud.classList.add("detected");
  hud.style.borderColor = signObj.color;
  camCard.style.borderColor = signObj.color;

  el("predEmoji").textContent = signObj.emoji;
  el("predWord").textContent = signObj.label;
  el("predConf").textContent = `${pct}%`;
  el("predConf").style.color = signObj.color;
  el("predPhrase").textContent = `"${signObj.spoken_text}" — ${signObj.meaning}`;
  el("confBarFill").style.width = `${pct}%`;

  // Subtle pop animation on change
  hud.classList.remove("pop");
  void hud.offsetWidth;
  hud.classList.add("pop");

  const mascot = el("mascotWrap");
  mascot.classList.add("excited");
  clearTimeout(updatePredictionHud._mascotTimer);
  updatePredictionHud._mascotTimer = setTimeout(() => mascot.classList.remove("excited"), 450);
}

function renderProbabilities(probs) {
  const container = el("probBars");
  const entries = Object.entries(probs || {});
  if (!entries.length) {
    container.innerHTML = `<div class="empty-hint">Waiting for hand detection...</div>`;
    return;
  }

  container.innerHTML = entries
    .slice(0, 3)
    .map(([label, score]) => {
      const sign = state.signMap[label];
      const color = sign ? sign.color : "#38bdf8";
      const pct = Math.round(score * 100);
      return `
        <div class="prob-row">
          <span class="prob-name">${sign ? sign.emoji : ""} ${label}</span>
          <div class="prob-track">
            <div class="prob-fill" style="width:${pct}%; background:${color};"></div>
          </div>
          <span class="prob-pct">${pct}%</span>
        </div>
      `;
    })
    .join("");
}

function updatePipelineSteps({ cameraActive, handsCount, inferenceMs, modelAccuracy, sign, confidence, fps }) {
  const s1 = cameraActive;
  const s2 = handsCount > 0;
  const s3 = s2 && inferenceMs >= 0;
  const s4 = Boolean(sign);
  const s5 = s4;

  el("step1").classList.toggle("active", s1);
  el("conn1").classList.toggle("active", s1 && s2);
  el("metricStep1").textContent = s1 ? `Live Stream · ${fps || 0} FPS` : "Camera paused / offline";

  el("step2").classList.toggle("active", s2);
  el("conn2").classList.toggle("active", s2 && s3);
  el("metricStep2").textContent = s2
    ? `${handsCount} Hand${handsCount > 1 ? "s" : ""} · ${handsCount * 21} Landmarks`
    : "Scanning for hands...";

  el("step3").classList.toggle("active", s3);
  el("conn3").classList.toggle("active", s3 && s4);
  el("metricStep3").textContent = `RandomForest · ${modelAccuracy || 99}% Acc · ${inferenceMs || 0} ms`;

  el("step4").classList.toggle("active", s4);
  el("conn4").classList.toggle("active", s4 && s5);
  el("metricStep4").textContent = s4
    ? `"${sign}" (${Math.round((confidence || 0) * 100)}% conf)`
    : "Awaiting stable gesture";

  if (!el("voiceVisualizer").classList.contains("speaking")) {
    el("step5").classList.toggle("active", s5);
    el("metricStep5").textContent = s5 ? `Displayed: ${sign}` : "Voice & Screen Ready";
  }
}

/* ==================== SENTENCE BUILDER & HISTORY ==================== */
function renderSentence() {
  const box = el("sentenceBox");
  if (!state.sentenceWords.length) {
    box.innerHTML = `<span class="sentence-placeholder">Detected signs will form a sentence here — e.g. "Hello · Please · Help · Thank You"</span>`;
    return;
  }

  box.innerHTML = state.sentenceWords
    .map((word) => {
      const sign = state.signMap[word];
      const color = sign ? sign.color : "#38bdf8";
      return `<span class="word-chip" style="border-color:${color}; background:${sign ? sign.bg_tint : ""}">${sign ? sign.emoji : ""} ${word}</span>`;
    })
    .join("");
}

function appendSentenceWord(word) {
  if (state.sentenceWords[state.sentenceWords.length - 1] === word) return;
  state.sentenceWords.push(word);
  if (state.sentenceWords.length > 14) state.sentenceWords.shift();
  renderSentence();
}

function pushHistory(signObj) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  state.history.unshift({ sign: signObj, time: timeStr });
  if (state.history.length > 12) state.history.pop();
  renderHistory();
}

function renderHistory() {
  const list = el("historyList");
  if (!state.history.length) {
    list.innerHTML = `<div class="empty-hint">Perform any sign or click a card on the right to begin.</div>`;
    return;
  }
  list.innerHTML = state.history
    .map(
      (item) => `
      <div class="history-item">
        <span class="history-left"><span>${item.sign.emoji}</span><span>${item.sign.label}</span></span>
        <span class="history-time">${item.time}</span>
      </div>
    `
    )
    .join("");
}

/* ==================== LIVE WEBSOCKET STREAM ==================== */
function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => {
    el("connectionPill").classList.add("connected");
    el("connectionText").textContent = "AI Pipeline Live";
  };

  ws.onclose = () => {
    el("connectionPill").classList.remove("connected");
    el("connectionText").textContent = "Reconnecting...";
    setTimeout(connectWebSocket, 1800);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleTelemetry(data);
  };
}

function handleTelemetry(data) {
  el("fpsChip").textContent = `${data.fps || 0} FPS`;
  el("handsChip").textContent = `${data.hands_count || 0} Hand${data.hands_count === 1 ? "" : "s"}`;

  // Camera availability state
  const fallback = el("videoFallback");
  if (!data.camera_ok || data.paused) {
    fallback.classList.remove("hidden");
  } else {
    fallback.classList.add("hidden");
  }

  // If user clicked a card for manual preview, keep preview for 3s unless real hand detected
  if (Date.now() < state.manualPreviewUntil && !data.hands_count) {
    return;
  }

  const signObj = data.sign ? state.signMap[data.sign] : null;
  updatePipelineSteps({
    cameraActive: data.camera_ok && !data.paused,
    handsCount: data.hands_count || 0,
    inferenceMs: data.inference_ms || 0,
    modelAccuracy: data.model_accuracy || 99,
    sign: data.sign,
    confidence: data.confidence || 0,
    fps: data.fps || 0,
  });

  renderProbabilities(data.probabilities || {});

  if (signObj) {
    state.currentSign = signObj.id;
    highlightSignCard(signObj.id);
    updatePredictionHud(signObj, data.confidence || 0);

    // Trigger Voice + Sentence + History when prediction is stable
    if (data.stable && (data.confidence || 0) >= 0.52) {
      const now = Date.now();
      const cooldownMs = state.cooldownSec * 1000;
      const isNewSign = signObj.id !== state.lastSpokenSign;
      const cooledDown = now - state.lastSpokenAt >= cooldownMs;

      if (isNewSign || cooledDown) {
        state.lastSpokenSign = signObj.id;
        state.lastSpokenAt = now;
        const phrase = state.fullPhrase ? signObj.spoken_text : signObj.short_speech;
        speakText(phrase);
        pushHistory(signObj);
        if (state.autoSentence) appendSentenceWord(signObj.id);

        el("activeSignTip").innerHTML = `
          <div class="tip-icon">${signObj.emoji}</div>
          <div class="tip-text">
            <strong>Detected: ${signObj.label}</strong> — ${signObj.meaning}.
          </div>
        `;
      }
    }
  } else if (data.hands_count === 0) {
    highlightSignCard(null);
    updatePredictionHud(null, 0);
  }
}

/* ==================== TRAINER MODAL ==================== */
async function refreshTrainerCounts() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    state.pythonTtsBackend = data.voice_backend || "unknown";
    el("pythonTtsInfo").textContent = `Python Local Voice Backend: ${state.pythonTtsBackend} · Model Accuracy: ${data.model_accuracy}%`;

    const counts = data.custom_counts || {};
    el("trainerCounts").innerHTML = state.signs
      .map((s) => {
        const c = counts[s.id] || 0;
        return `<span class="count-pill ${c > 0 ? "has-data" : ""}">${s.emoji} ${s.label}: ${c}</span>`;
      })
      .join("");
  } catch (e) {
    console.error(e);
  }
}

async function captureSample(count = 1) {
  const label = el("trainerSignSelect").value;
  const status = el("trainerStatus");
  let captured = 0;

  for (let i = 0; i < count; i++) {
    status.textContent = `Capturing sample ${i + 1} / ${count} for "${label}"... Hold pose steady!`;
    const res = await fetch("/api/train/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    const data = await res.json();
    if (res.ok && data.ok) {
      captured++;
    } else {
      status.textContent = data.error || "No hand detected! Show your hand clearly to the camera.";
      break;
    }
    if (count > 1) await new Promise((r) => setTimeout(r, 170));
  }

  if (captured > 0) {
    status.textContent = `Captured ${captured} sample(s) for "${label}". Click "Retrain AI Model Now" to apply!`;
    showToast(`Recorded ${captured} sample(s) for ${label}`);
    refreshTrainerCounts();
  }
}

/* ==================== EVENT BINDINGS ==================== */
async function postSetting(payload) {
  await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function bindEvents() {
  // Theme toggle
  el("btnTheme").addEventListener("click", () => {
    const root = document.documentElement;
    const isDark = root.getAttribute("data-theme") === "dark";
    root.setAttribute("data-theme", isDark ? "light" : "dark");
    el("themeIcon").textContent = isDark ? "🌙" : "☀️";
  });

  // Skeleton toggle
  const btnSkel = el("btnToggleSkeleton");
  btnSkel.classList.add("active");
  btnSkel.addEventListener("click", async () => {
    state.showSkeleton = !state.showSkeleton;
    btnSkel.classList.toggle("active", state.showSkeleton);
    await postSetting({ show_skeleton: state.showSkeleton });
  });

  // Mirror toggle
  const btnMirror = el("btnToggleMirror");
  btnMirror.classList.add("active");
  btnMirror.addEventListener("click", async () => {
    state.mirror = !state.mirror;
    btnMirror.classList.toggle("active", state.mirror);
    await postSetting({ mirror: state.mirror });
  });

  // Camera pause/resume
  el("btnCameraToggle").addEventListener("click", async () => {
    state.paused = !state.paused;
    el("btnCameraToggle").textContent = state.paused ? "▶️ Resume" : "⏸️ Pause";
    await postSetting({ paused: state.paused });
  });

  el("btnRetryCamera").addEventListener("click", async () => {
    state.paused = false;
    el("btnCameraToggle").textContent = "⏸️ Pause";
    await fetch("/api/camera/reconnect", { method: "POST" });
    el("videoStream").src = `/video_feed?t=${Date.now()}`;
    showToast("Reconnecting webcam...");
  });

  // Quick voice mute/unmute
  el("btnVoiceQuick").addEventListener("click", () => {
    state.voiceEnabled = !state.voiceEnabled;
    const btn = el("btnVoiceQuick");
    btn.classList.toggle("active", state.voiceEnabled);
    btn.classList.toggle("muted", !state.voiceEnabled);
    btn.textContent = state.voiceEnabled ? "🔊 Voice ON" : "🔇 Voice OFF";
    showToast(state.voiceEnabled ? "Voice output enabled" : "Voice output muted");
  });

  // Replay current sign speech
  el("btnReplaySpeech").addEventListener("click", () => {
    const sign = state.signMap[state.currentSign];
    if (sign) {
      speakText(state.fullPhrase ? sign.spoken_text : sign.short_speech, true);
    } else {
      speakText("Hello! Show any hand sign to translate it into text and voice.", true);
    }
  });

  // Voice Engine segmented control
  el("voiceEngineSeg").querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      el("voiceEngineSeg").querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.voiceEngine = btn.dataset.engine;
      showToast(
        state.voiceEngine === "browser"
          ? "Switched to Browser Natural Voice"
          : `Switched to Python Local TTS (${state.pythonTtsBackend})`
      );
    });
  });

  el("chkFullPhrase").addEventListener("change", (e) => {
    state.fullPhrase = e.target.checked;
  });

  el("chkAutoSentence").addEventListener("change", (e) => {
    state.autoSentence = e.target.checked;
  });

  // Sentence Builder Controls
  el("btnSpeakSentence").addEventListener("click", () => {
    if (!state.sentenceWords.length) {
      showToast("Sentence is empty — perform or click a few signs first!");
      return;
    }
    const sentence = state.sentenceWords.join(", ");
    speakText(sentence, true);
  });

  el("btnCopySentence").addEventListener("click", async () => {
    if (!state.sentenceWords.length) return;
    await navigator.clipboard.writeText(state.sentenceWords.join(" "));
    showToast("Sentence copied to clipboard!");
  });

  el("btnUndoWord").addEventListener("click", () => {
    state.sentenceWords.pop();
    renderSentence();
  });

  el("btnClearSentence").addEventListener("click", () => {
    state.sentenceWords = [];
    renderSentence();
  });

  el("btnClearHistory").addEventListener("click", () => {
    state.history = [];
    renderHistory();
  });

  // Trainer Modal
  el("btnOpenTrainer").addEventListener("click", () => {
    refreshTrainerCounts();
    el("trainerModal").classList.remove("hidden");
  });
  el("btnCloseTrainer").addEventListener("click", () => el("trainerModal").classList.add("hidden"));
  el("btnCaptureOne").addEventListener("click", () => captureSample(1));
  el("btnCaptureBurst").addEventListener("click", () => captureSample(8));

  el("btnRetrainModel").addEventListener("click", async () => {
    el("trainerStatus").textContent = "Training RandomForest AI model...";
    const res = await fetch("/api/train/retrain", { method: "POST" });
    const data = await res.json();
    el("trainerStatus").textContent = `Model retrained! Accuracy: ${data.accuracy}% (${data.samples_trained} samples)`;
    showToast(`AI Model retrained (${data.accuracy}% accuracy)!`);
  });

  el("btnResetSamples").addEventListener("click", async () => {
    await fetch("/api/train/reset", { method: "POST" });
    refreshTrainerCounts();
    el("trainerStatus").textContent = "Custom samples cleared and base AI model restored.";
    showToast("Custom samples reset");
  });

  // Voice Studio Modal
  el("btnVoiceSettings").addEventListener("click", () => {
    loadBrowserVoices();
    refreshTrainerCounts();
    el("voiceModal").classList.remove("hidden");
  });
  el("btnCloseVoice").addEventListener("click", () => el("voiceModal").classList.add("hidden"));
  el("btnSaveVoice").addEventListener("click", () => el("voiceModal").classList.add("hidden"));

  el("voiceSelect").addEventListener("change", (e) => {
    state.selectedVoiceURI = e.target.value;
  });

  el("rateSlider").addEventListener("input", (e) => {
    state.speechRate = parseFloat(e.target.value);
    el("rateVal").textContent = `${state.speechRate.toFixed(2)}x`;
  });

  el("pitchSlider").addEventListener("input", (e) => {
    state.speechPitch = parseFloat(e.target.value);
    el("pitchVal").textContent = state.speechPitch.toFixed(2);
  });

  el("cooldownSlider").addEventListener("input", (e) => {
    state.cooldownSec = parseFloat(e.target.value);
    el("cooldownVal").textContent = `${state.cooldownSec.toFixed(1)}s`;
  });

  el("btnTestVoice").addEventListener("click", () => {
    speakText("Hello! SignBridge AI is ready to translate your hand signs into speech.", true);
  });

  // Close modals on backdrop click
  ["trainerModal", "voiceModal"].forEach((id) => {
    el(id).addEventListener("click", (e) => {
      if (e.target === el(id)) el(id).classList.add("hidden");
    });
  });
}

/* ==================== BOOTSTRAP ==================== */
async function init() {
  try {
    const res = await fetch("/api/signs");
    const data = await res.json();
    state.signs = data.signs || [];
    state.signs.forEach((s) => (state.signMap[s.id] = s));
    renderSignsGrid();
  } catch (e) {
    console.error("Failed to fetch signs:", e);
  }

  bindEvents();
  loadBrowserVoices();
  if ("speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = loadBrowserVoices;
  }
  refreshTrainerCounts();
  connectWebSocket();
}

document.addEventListener("DOMContentLoaded", init);
