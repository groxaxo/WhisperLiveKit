## WhisperLiveKit Chrome Extension v1.1.0
Capture the audio of your current tab, transcribe, diarize and translate it using WhisperLiveKit, in Chrome and other Chromium-based browsers.

> Currently, only the tab audio is captured; your microphone audio is not recorded.

<img src="https://raw.githubusercontent.com/QuentinFuxa/WhisperLiveKit/refs/heads/main/chrome-extension/demo-extension.png" alt="WhisperLiveKit Demo" width="730">

## Prerequisites

Before using the extension, make sure you have WhisperLiveKit running:

```bash
# Install WhisperLiveKit
pip install whisperlivekit

# Start the server
wlk --model base --language en --host 0.0.0.0
```

## Installing the Extension

1. **Sync the extension files** (if building from source):
   ```bash
   python scripts/sync_extension.py
   ```

2. **Load the extension in Chrome**:
   - Open Chrome and navigate to `chrome://extensions/`
   - Enable "Developer mode" (toggle in the top right)
   - Click "Load unpacked"
   - Select the `chrome-extension` directory from your WhisperLiveKit installation

## Using the Extension

1. Navigate to any webpage with audio (e.g., YouTube, podcasts, video conferences)
2. Click the WhisperLiveKit extension icon in your browser toolbar
3. Configure the server URL (default: `ws://localhost:8000/asr`)
4. Click "Start Capturing" to begin transcription
5. The audio from the tab will be transcribed in real-time!

## Configuration

You can configure the extension to connect to:
- **Local server**: `ws://localhost:8000/asr` (default)
- **Remote server**: `ws://your-server-ip:8000/asr`
- **HTTPS server**: `wss://your-domain.com/asr`

## Troubleshooting

### Extension Not Capturing Audio
- Ensure WhisperLiveKit server is running and accessible
- Check that you've granted the necessary permissions
- Verify the WebSocket URL is correct
- Some sites may block audio capture for security reasons

### Connection Issues
- If using a remote server, ensure it's configured with `--host 0.0.0.0`
- For HTTPS sites, your WhisperLiveKit server may need SSL certificates
- Check browser console for detailed error messages

## Technical Notes

### Developers

- **Tab audio capture limitation**: Extensions cannot capture audio when used as a side panel due to Chromium limitations:
  - https://issues.chromium.org/issues/40926394
  - https://groups.google.com/a/chromium.org/g/chromium-extensions/c/DET2SXCFnDg
  - https://issues.chromium.org/issues/40916430

- **Microphone capture**: To capture microphone audio in an extension, see these workarounds:
  - https://github.com/justinmann/sidepanel-audio-issue
  - https://medium.com/@lynchee.owo/how-to-enable-microphone-access-in-chrome-extensions-by-code-924295170080 (check comments)

## Features

- ✅ Real-time transcription of tab audio
- ✅ Speaker diarization (when enabled on server)
- ✅ Translation support (when enabled on server)
- ✅ Works with any audio source in the browser
- ✅ Configurable server connection

## Privacy

All audio processing happens on your WhisperLiveKit server. No data is sent to third parties.
