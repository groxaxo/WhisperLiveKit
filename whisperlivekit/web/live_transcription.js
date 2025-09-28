/* Notepad-centric UI that streams dictation and optional translation results. */

let isRecording = false;
let userClosing = false;
let audioContext = null;
let analyser = null;
let microphone = null;
let recorder = null;
let wakeLock = null;
let animationFrame = null;
let startTime = null;
let timerInterval = null;
const chunkDuration = 100;

const waveCanvas = document.getElementById("waveCanvas");
const waveCtx = waveCanvas.getContext("2d");
const statusText = document.getElementById("status");
const recordButton = document.getElementById("recordButton");
const timerElement = document.querySelector(".timer");
const themeRadios = document.querySelectorAll('input[name="theme"]');
const microphoneSelect = document.getElementById("microphoneSelect");
const websocketInput = document.getElementById("websocketInput");
const translationWebsocketInput = document.getElementById("translationWebsocketInput");
const translationToggle = document.getElementById("translationToggle");
const translationPane = document.getElementById("translationPane");
const dictationNote = document.getElementById("dictationNote");
const translationNote = document.getElementById("translationNote");
const dictationPreviewText = document.querySelector("#dictationPreview .preview-text");
const translationPreviewText = document.querySelector("#translationPreview .preview-text");
const saveNoteButton = document.getElementById("saveNoteButton");
const newNoteButton = document.getElementById("newNoteButton");
const continueNoteButton = document.getElementById("continueNoteButton");
const saveTranslationButton = document.getElementById("saveTranslationButton");

const LOCAL_STORAGE_NOTE_KEY = "whisper-notepad-current";
const LOCAL_STORAGE_ARCHIVE_KEY = "whisper-notepad-archive";
const LOCAL_STORAGE_TRANSLATION_ENABLED = "whisper-notepad-translation";
const LOCAL_STORAGE_MIC = "selectedMicrophone";

let translationEnabled = false;
let selectedMicrophoneId = null;
let availableMicrophones = [];

const streamControllers = {
  dictation: {
    mode: "dictation",
    socket: null,
    waitingForStop: false,
    lastData: null,
    lastSignature: null,
    baseText: "",
    noteElement: dictationNote,
    previewTextElement: dictationPreviewText,
    urlInput: websocketInput,
    url: "",
  },
  translation: {
    mode: "translation",
    socket: null,
    waitingForStop: false,
    lastData: null,
    lastSignature: null,
    baseText: "",
    noteElement: translationNote,
    previewTextElement: translationPreviewText,
    urlInput: translationWebsocketInput,
    url: "",
  },
};

function getWaveStroke() {
  const styles = getComputedStyle(document.documentElement);
  const v = styles.getPropertyValue("--wave-stroke").trim();
  return v || "#2563eb";
}

let waveStroke = getWaveStroke();
function updateWaveStroke() {
  waveStroke = getWaveStroke();
}

function applyTheme(pref) {
  if (pref === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else if (pref === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  updateWaveStroke();
}

const savedThemePref = localStorage.getItem("themePreference") || "system";
applyTheme(savedThemePref);
if (themeRadios.length) {
  themeRadios.forEach((r) => {
    r.checked = r.value === savedThemePref;
    r.addEventListener("change", () => {
      if (r.checked) {
        localStorage.setItem("themePreference", r.value);
        applyTheme(r.value);
      }
    });
  });
}

const darkMq = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
const handleOsThemeChange = () => {
  const pref = localStorage.getItem("themePreference") || "system";
  if (pref === "system") updateWaveStroke();
};
if (darkMq && darkMq.addEventListener) {
  darkMq.addEventListener("change", handleOsThemeChange);
} else if (darkMq && darkMq.addListener) {
  darkMq.addListener(handleOsThemeChange);
}

function restoreNoteFromStorage() {
  const saved = localStorage.getItem(LOCAL_STORAGE_NOTE_KEY);
  if (saved) {
    dictationNote.value = saved;
    streamControllers.dictation.baseText = saved.trimEnd();
  }
}

function persistNote() {
  localStorage.setItem(LOCAL_STORAGE_NOTE_KEY, dictationNote.value);
}

restoreNoteFromStorage();

function restoreTranslationPreference() {
  const saved = localStorage.getItem(LOCAL_STORAGE_TRANSLATION_ENABLED);
  if (saved === "1") {
    translationEnabled = true;
    translationToggle.checked = true;
    translationPane.classList.remove("is-hidden");
  } else {
    translationEnabled = false;
    translationToggle.checked = false;
    translationPane.classList.add("is-hidden");
  }
}

restoreTranslationPreference();

const host = window.location.hostname || "localhost";
const port = window.location.port;
const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const defaultWebSocketUrl = `${protocol}://${host}${port ? ":" + port : ""}/asr`;
const defaultTranslationUrl = `${defaultWebSocketUrl}?task=translate`;

websocketInput.value = defaultWebSocketUrl;
translationWebsocketInput.value = defaultTranslationUrl;
streamControllers.dictation.url = defaultWebSocketUrl;
streamControllers.translation.url = defaultTranslationUrl;

function getActiveModes() {
  const modes = ["dictation"];
  if (translationEnabled) {
    modes.push("translation");
  }
  return modes;
}

function isAnyWaiting() {
  return getActiveModes().some((mode) => streamControllers[mode].waitingForStop);
}

function combineBaseWithNew(base, finalText) {
  if (!base) return finalText;
  if (!finalText) return base;
  const separator = base.endsWith("\n") ? "\n" : "\n\n";
  return base + separator + finalText;
}

function buildTextFromLines(data, finalizing = false) {
  const lines = data?.lines || [];
  const bufferTranscription = (data?.buffer_transcription || "").trim();
  const bufferDiarization = (data?.buffer_diarization || "").trim();
  const paragraphs = [];

  lines.forEach((item, idx) => {
    if (!item) return;
    if (item.speaker === -2) return;
    let text = (item.text || "").trim();
    if (item.speaker && item.speaker > 0) {
      text = `Speaker ${item.speaker}: ${text}`;
    }

    if (idx === lines.length - 1 && finalizing) {
      if (bufferDiarization) {
        text += (text ? " " : "") + bufferDiarization;
      }
      if (bufferTranscription) {
        text += (text ? " " : "") + bufferTranscription;
      }
    }

    paragraphs.push(text);
  });

  const finalText = paragraphs.filter((p) => p.length > 0).join("\n\n");

  let previewText = "";
  if (lines.length) {
    const last = lines[lines.length - 1];
    previewText = (last?.text || "").trim();
    if (!finalizing) {
      if (bufferDiarization) previewText += (previewText ? " " : "") + bufferDiarization;
      if (bufferTranscription) previewText += (previewText ? " " : "") + bufferTranscription;
    }
  } else if (!finalizing) {
    previewText = bufferTranscription;
  }

  if (finalizing) {
    previewText = "";
  }

  return { finalText, previewText };
}

function updateStream(mode, data, finalizing = false) {
  const ctrl = streamControllers[mode];
  if (!ctrl) return;

  if (data?.status === "no_audio_detected") {
    ctrl.previewTextElement.textContent = "No audio detected...";
    return;
  }

  const signature = JSON.stringify({
    lines: (data?.lines || []).map((it) => ({ speaker: it.speaker, text: it.text })),
    buffer_transcription: data?.buffer_transcription || "",
    buffer_diarization: data?.buffer_diarization || "",
    finalizing,
  });

  if (!finalizing && ctrl.lastSignature === signature) {
    return;
  }

  ctrl.lastSignature = signature;

  const { finalText, previewText } = buildTextFromLines(data, finalizing);
  const combined = combineBaseWithNew(ctrl.baseText, finalText);
  ctrl.noteElement.value = combined;
  ctrl.previewTextElement.textContent = previewText;
  ctrl.noteElement.scrollTop = ctrl.noteElement.scrollHeight;

  if (mode === "dictation" && finalizing) {
    persistNote();
  }

  if (finalizing) {
    ctrl.baseText = ctrl.noteElement.value.trimEnd();
    ctrl.previewTextElement.textContent = "";
    ctrl.lastData = null;
  }
}

function finalizeStopIfNeeded() {
  if (isRecording) return;
  if (isAnyWaiting()) return;
  if (getActiveModes().some((mode) => streamControllers[mode].socket)) return;

  dictationNote.readOnly = false;
  dictationNote.classList.remove("locked");
  streamControllers.dictation.baseText = dictationNote.value.trimEnd();
  persistNote();

  translationNote.readOnly = true;
  translationNote.classList.remove("locked");
  streamControllers.translation.baseText = translationNote.value.trimEnd();

  userClosing = false;
  statusText.textContent = "Finished processing audio! Ready to record again.";
  updateUI();
}

function closeStream(mode) {
  const ctrl = streamControllers[mode];
  if (ctrl.socket) {
    ctrl.socket.onopen = null;
    ctrl.socket.onmessage = null;
    ctrl.socket.onclose = null;
    ctrl.socket.onerror = null;
    try {
      ctrl.socket.close();
    } catch (err) {
      console.warn(`Error closing ${mode} socket`, err);
    }
  }
  ctrl.socket = null;
  ctrl.waitingForStop = false;
  ctrl.lastData = null;
  ctrl.lastSignature = null;
  finalizeStopIfNeeded();
}

function updateUI() {
  const waiting = isAnyWaiting();
  recordButton.classList.toggle("recording", isRecording);
  recordButton.disabled = waiting;

  if (waiting) {
    statusText.textContent = "Please wait for processing to complete...";
  } else if (isRecording) {
    statusText.textContent = translationEnabled
      ? "Streaming dictation and translation..."
      : "Recording...";
  } else if (!waiting && !isRecording && !userClosing) {
    if (!streamControllers.dictation.noteElement.value) {
      statusText.textContent = "Click to start transcription";
    }
  }
}

function drawWaveform() {
  if (!analyser) return;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(dataArray);

  waveCtx.clearRect(0, 0, waveCanvas.width / (window.devicePixelRatio || 1), waveCanvas.height / (window.devicePixelRatio || 1));
  waveCtx.lineWidth = 1;
  waveCtx.strokeStyle = waveStroke;
  waveCtx.beginPath();

  const sliceWidth = (waveCanvas.width / (window.devicePixelRatio || 1)) / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0;
    const y = (v * (waveCanvas.height / (window.devicePixelRatio || 1))) / 2;

    if (i === 0) {
      waveCtx.moveTo(x, y);
    } else {
      waveCtx.lineTo(x, y);
    }

    x += sliceWidth;
  }

  waveCtx.lineTo(waveCanvas.width / (window.devicePixelRatio || 1), (waveCanvas.height / (window.devicePixelRatio || 1)) / 2);
  waveCtx.stroke();

  animationFrame = requestAnimationFrame(drawWaveform);
}

function updateTimer() {
  if (!startTime) return;

  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const minutes = Math.floor(elapsed / 60).toString().padStart(2, "0");
  const seconds = (elapsed % 60).toString().padStart(2, "0");
  timerElement.textContent = `${minutes}:${seconds}`;
}

async function startRecording() {
  try {
    try {
      wakeLock = await navigator.wakeLock.request("screen");
    } catch (err) {
      console.log("Wake lock not available", err);
    }

    const audioConstraints = selectedMicrophoneId
      ? { audio: { deviceId: { exact: selectedMicrophoneId } } }
      : { audio: true };

    const stream = await navigator.mediaDevices.getUserMedia(audioConstraints);

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);

    recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    recorder.ondataavailable = (e) => {
      getActiveModes().forEach((mode) => {
        const ctrl = streamControllers[mode];
        if (ctrl.socket && ctrl.socket.readyState === WebSocket.OPEN) {
          ctrl.socket.send(e.data);
        }
      });
    };
    recorder.start(chunkDuration);

    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);
    drawWaveform();

    dictationNote.readOnly = true;
    dictationNote.classList.add("locked");
    if (translationEnabled) {
      translationNote.readOnly = true;
      translationNote.classList.add("locked");
    }

    isRecording = true;
    updateUI();
  } catch (err) {
    if (window.location.hostname === "0.0.0.0") {
      statusText.textContent = "Error accessing microphone. Browsers may block microphone access on 0.0.0.0. Try using localhost:8000 instead.";
    } else {
      statusText.textContent = "Error accessing microphone. Please allow microphone access.";
    }
    console.error(err);
    isRecording = false;
    updateUI();
    throw err;
  }
}

async function stopRecording() {
  if (wakeLock) {
    try {
      await wakeLock.release();
    } catch (err) {
      console.warn("Error releasing wake lock", err);
    }
    wakeLock = null;
  }

  userClosing = true;

  const emptyBlob = new Blob([], { type: "audio/webm" });
  getActiveModes().forEach((mode) => {
    const ctrl = streamControllers[mode];
    if (ctrl.socket && ctrl.socket.readyState === WebSocket.OPEN) {
      ctrl.waitingForStop = true;
      ctrl.socket.send(emptyBlob);
    }
  });

  if (recorder) {
    recorder.stop();
    recorder = null;
  }

  if (microphone) {
    microphone.disconnect();
    microphone = null;
  }

  if (analyser) {
    analyser = null;
  }

  if (audioContext && audioContext.state !== "closed") {
    try {
      await audioContext.close();
    } catch (e) {
      console.warn("Could not close audio context:", e);
    }
    audioContext = null;
  }

  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }

  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerElement.textContent = "00:00";
  startTime = null;

  isRecording = false;
  updateUI();
}

async function setupStream(mode) {
  const ctrl = streamControllers[mode];
  if (!ctrl) return;
  if (ctrl.socket && ctrl.socket.readyState === WebSocket.OPEN) {
    return;
  }

  return new Promise((resolve, reject) => {
    let ws;
    try {
      ws = new WebSocket(ctrl.url);
    } catch (error) {
      reject(error);
      return;
    }
    ctrl.socket = ws;

    ws.onopen = () => {
      statusText.textContent = `Connected to ${mode} server.`;
      resolve();
    };

    ws.onclose = () => {
      ctrl.socket = null;
      ctrl.waitingForStop = false;
      ctrl.lastData = null;
      ctrl.lastSignature = null;
      if (!userClosing) {
        statusText.textContent = `${mode === "translation" ? "Translation" : "Dictation"} connection closed.`;
      }
      finalizeStopIfNeeded();
    };

    ws.onerror = (evt) => {
      console.error(`WebSocket error on ${mode}`, evt);
      reject(new Error(`Error connecting to ${mode} WebSocket.`));
    };

    ws.onmessage = (event) => {
      if (typeof event.data !== "string") return;
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        console.warn(`Could not parse ${mode} message`, err);
        return;
      }
      if (data.type === "ready_to_stop") {
        ctrl.waitingForStop = false;
        if (ctrl.lastData) {
          updateStream(mode, ctrl.lastData, true);
        }
        ctrl.lastSignature = null;
        ctrl.lastData = null;
        if (ctrl.socket && ctrl.socket.readyState === WebSocket.OPEN) {
          ctrl.socket.close();
        }
        finalizeStopIfNeeded();
        return;
      }

      ctrl.lastData = data;
      updateStream(mode, data, false);
    };
  });
}

async function toggleRecording() {
  if (isRecording) {
    await stopRecording();
    return;
  }

  if (isAnyWaiting()) {
    return;
  }

  const activeModes = getActiveModes();

  for (const mode of activeModes) {
    const ctrl = streamControllers[mode];
    const urlValue = (ctrl.urlInput?.value || "").trim();
    if (!urlValue) {
      statusText.textContent = `Please provide a ${mode} WebSocket URL.`;
      return;
    }
    if (!urlValue.startsWith("ws://") && !urlValue.startsWith("wss://")) {
      statusText.textContent = "Invalid WebSocket URL (must start with ws:// or wss://).";
      return;
    }
    ctrl.url = urlValue;
    ctrl.baseText = ctrl.noteElement.value.trimEnd();
    ctrl.lastSignature = null;
    ctrl.previewTextElement.textContent = "";
  }

  try {
    for (const mode of activeModes) {
      await setupStream(mode);
    }
    await startRecording();
  } catch (error) {
    console.error(error);
    statusText.textContent = "Could not connect to WebSocket or access mic. Aborted.";
    activeModes.forEach((mode) => closeStream(mode));
    userClosing = false;
  }
}

function handleMicrophoneChange() {
  selectedMicrophoneId = microphoneSelect.value || null;
  localStorage.setItem(LOCAL_STORAGE_MIC, selectedMicrophoneId || "");

  const selectedDevice = availableMicrophones.find((mic) => mic.deviceId === selectedMicrophoneId);
  const deviceName = selectedDevice ? selectedDevice.label : "Default Microphone";

  statusText.textContent = `Microphone set to: ${deviceName}`;

  if (isRecording) {
    statusText.textContent = "Switching microphone... Please wait.";
    stopRecording().then(() => {
      setTimeout(() => {
        toggleRecording();
      }, 800);
    });
  }
}

async function enumerateMicrophones() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((track) => track.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    availableMicrophones = devices.filter((device) => device.kind === "audioinput");
    populateMicrophoneSelect();
  } catch (error) {
    console.error("Error enumerating microphones:", error);
    statusText.textContent = "Error accessing microphones. Please grant permission.";
  }
}

function populateMicrophoneSelect() {
  if (!microphoneSelect) return;

  microphoneSelect.innerHTML = '<option value="">Default Microphone</option>';

  availableMicrophones.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Microphone ${index + 1}`;
    microphoneSelect.appendChild(option);
  });

  const savedMicId = localStorage.getItem(LOCAL_STORAGE_MIC);
  if (savedMicId && availableMicrophones.some((mic) => mic.deviceId === savedMicId)) {
    microphoneSelect.value = savedMicId;
    selectedMicrophoneId = savedMicId;
  }
}

function downloadText(filename, text) {
  const blob = new Blob([text || ""], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function formatTimestamp() {
  const now = new Date();
  const pad = (n) => `${n}`.padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
}

function handleSaveNote() {
  const text = dictationNote.value || "";
  downloadText(`whisper-note-${formatTimestamp()}.txt`, text);
  localStorage.setItem(LOCAL_STORAGE_ARCHIVE_KEY, text);
  persistNote();
  statusText.textContent = "Note saved locally.";
}

function handleSaveTranslation() {
  const text = translationNote.value || "";
  downloadText(`whisper-translation-${formatTimestamp()}.txt`, text);
  statusText.textContent = "Translation saved.";
}

function handleNewNote() {
  dictationNote.value = "";
  translationNote.value = "";
  dictationPreviewText.textContent = "";
  translationPreviewText.textContent = "";
  streamControllers.dictation.baseText = "";
  streamControllers.translation.baseText = "";
  persistNote();
  localStorage.removeItem(LOCAL_STORAGE_ARCHIVE_KEY);
  statusText.textContent = "Started a new note.";
}

function handleContinueNote() {
  const saved = localStorage.getItem(LOCAL_STORAGE_ARCHIVE_KEY) || localStorage.getItem(LOCAL_STORAGE_NOTE_KEY);
  if (saved) {
    dictationNote.value = saved;
    streamControllers.dictation.baseText = saved.trimEnd();
    persistNote();
    statusText.textContent = "Restored saved note.";
  } else {
    statusText.textContent = "No saved note available.";
  }
}

recordButton.addEventListener("click", toggleRecording);

if (microphoneSelect) {
  microphoneSelect.addEventListener("change", handleMicrophoneChange);
}

dictationNote.addEventListener("input", () => {
  if (!isRecording && !isAnyWaiting()) {
    streamControllers.dictation.baseText = dictationNote.value.trimEnd();
    persistNote();
  }
});

saveNoteButton.addEventListener("click", handleSaveNote);
newNoteButton.addEventListener("click", handleNewNote);
continueNoteButton.addEventListener("click", handleContinueNote);
saveTranslationButton.addEventListener("click", handleSaveTranslation);

translationToggle.addEventListener("change", () => {
  translationEnabled = translationToggle.checked;
  localStorage.setItem(LOCAL_STORAGE_TRANSLATION_ENABLED, translationEnabled ? "1" : "0");
  translationPane.classList.toggle("is-hidden", !translationEnabled);
  if (!translationEnabled) {
    translationPreviewText.textContent = "";
    closeStream("translation");
  }
  updateUI();
});

websocketInput.addEventListener("change", () => {
  const value = websocketInput.value.trim();
  if (!value.startsWith("ws://") && !value.startsWith("wss://")) {
    statusText.textContent = "Invalid WebSocket URL (must start with ws:// or wss://).";
    return;
  }
  streamControllers.dictation.url = value;
  statusText.textContent = "Dictation WebSocket updated.";
});

translationWebsocketInput.addEventListener("change", () => {
  const value = translationWebsocketInput.value.trim();
  if (!value.startsWith("ws://") && !value.startsWith("wss://")) {
    statusText.textContent = "Invalid translation WebSocket URL (must start with ws:// or wss://).";
    return;
  }
  streamControllers.translation.url = value;
  statusText.textContent = "Translation WebSocket updated.";
});

window.addEventListener("beforeunload", () => {
  getActiveModes().forEach((mode) => closeStream(mode));
  if (recorder) {
    recorder.stop();
  }
});

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await enumerateMicrophones();
  } catch (error) {
    console.warn("Could not enumerate microphones on load", error);
  }
});

if (navigator.mediaDevices) {
  const handler = async () => {
    try {
      await enumerateMicrophones();
    } catch (error) {
      console.log("Error re-enumerating microphones:", error);
    }
  };
  if (navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange", handler);
  } else {
    navigator.mediaDevices.ondevicechange = handler;
  }
}

updateUI();
