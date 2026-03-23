# WhisperLiveKit Chrome Extension

> **Version 0.1.1** — Capture, transcribe, diarize & translate audio from any browser tab

<p align="center">
  <img src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/chrome-extension/demo-extension.png" alt="WhisperLiveKit Chrome Extension Demo" width="730">
</p>

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Tab Audio Capture** | Capture audio from any Chrome tab in real-time |
| 🗣️ **Speaker Diarization** | Identify multiple speakers in the conversation |
| 🌍 **Live Translation** | Translate speech to 200+ languages simultaneously |
| ⚡ **Low Latency** | Streaming transcription with sub-100ms latency |
| 🔒 **Privacy-First** | All processing happens locally on your server |
| 🖥️ **CPU Mode** | Works with Parakeet — no GPU required |

## 🚀 Quick Start

### 1. Start WhisperLiveKit Server

```bash
# GPU mode (recommended)
wlk --model base --language en

# CPU mode (no GPU required)
pip install -e ".[cpu]"
wlk --backend openai-api
```

### 2. Sync Extension Files

```bash
python scripts/sync_extension.py
```

### 3. Load Extension in Chrome

1. Open `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `chrome-extension` directory

### 4. Use the Extension

1. Navigate to any page with audio (YouTube, podcast, meeting, etc.)
2. Click the WhisperLiveKit icon in your toolbar
3. Grant tab audio permission when prompted
4. Watch live transcription appear!

## 🔧 Configuration

Connect to a custom server:

```javascript
// In sidepanel.js - modify the WebSocket URL
const ws = new WebSocket("ws://your-server:8000/asr");
```

## 🐛 Known Limitations

- **Panel audio capture**: Tab audio cannot be captured from side panels ([Chromium Issue #40926394](https://issues.chromium.org/issues/40926394))
- **Microphone input**: Requires additional configuration — see [these tricks](https://github.com/justinmann/sidepanel-audio-issue)

## 🛠️ For Developers

```bash
# Rebuild frontend assets
python scripts/sync_extension.py

# Test with different backends
wlk --backend faster-whisper --model medium
wlk --backend openai-api  # Parakeet CPU mode
```

---

**Made with ❤️ by the WhisperLiveKit team**
