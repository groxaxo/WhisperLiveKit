# Open WebUI Integration Guide

This guide explains how to integrate WhisperLiveKit as a plug-and-play speech-to-text (STT) solution for [Open WebUI](https://github.com/open-webui/open-webui).

## Overview

WhisperLiveKit now provides an OpenAI Whisper API-compatible endpoint that can be used directly with Open WebUI's STT configuration. This allows you to:

- Use WhisperLiveKit as your local, self-hosted STT provider
- Benefit from ultra-low-latency transcription with speaker identification
- Avoid sending audio data to external services
- Support 200+ languages with optional translation
- Get real-time transcription during conversations with LLMs

## Features

- ✅ **OpenAI API Compatible**: Drop-in replacement for OpenAI's Whisper API
- ✅ **Multiple Response Formats**: json, text, srt, vtt, verbose_json
- ✅ **Language Support**: Auto-detection or specify any supported language
- ✅ **Self-Hosted**: Keep your audio data private
- ✅ **Low Latency**: Optimized for real-time transcription
- ✅ **Easy Setup**: Works out of the box with Open WebUI

## Quick Start

### 1. Start WhisperLiveKit Server

Start the WhisperLiveKit server with your desired configuration:

```bash
# Basic setup with English
wlk --model base --language en --host 0.0.0.0 --port 8000

# With larger model for better accuracy
wlk --model medium --language auto --host 0.0.0.0 --port 8000

# With diarization for speaker identification
wlk --model base --language en --diarization --host 0.0.0.0 --port 8000
```

The server will start and provide:
- WebSocket endpoint for real-time transcription: `ws://localhost:8000/asr`
- OpenAI-compatible API endpoint: `http://localhost:8000/v1/audio/transcriptions`
- Web interface: `http://localhost:8000`

### 2. Configure Open WebUI

#### Option A: Using the Web Interface

1. Log in to Open WebUI as an administrator
2. Go to **Admin Settings** → **Audio**
3. Configure Speech-to-Text settings:
   - **STT Engine**: Select "OpenAI"
   - **API Base URL**: `http://localhost:8000/v1` (or your WhisperLiveKit server URL)
   - **API Key**: Enter any string (e.g., "wlk-key") - it's not validated but required by Open WebUI
   - **STT Model**: Enter "whisper-1" (this is ignored by WhisperLiveKit, which uses the configured model)

4. Save the settings

#### Option B: Using Environment Variables

Add these environment variables to your Open WebUI configuration:

```bash
# For Docker deployment
docker run -d \
  -e AUDIO_STT_ENGINE=openai \
  -e AUDIO_STT_OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e AUDIO_STT_OPENAI_API_KEY=wlk-key \
  -e AUDIO_STT_MODEL=whisper-1 \
  -p 3000:8080 \
  ghcr.io/open-webui/open-webui:main
```

For non-Docker deployments, export these variables:

```bash
export AUDIO_STT_ENGINE="openai"
export AUDIO_STT_OPENAI_API_BASE_URL="http://localhost:8000/v1"
export AUDIO_STT_OPENAI_API_KEY="wlk-key"
export AUDIO_STT_MODEL="whisper-1"
```

### 3. Test the Integration

1. Open Open WebUI in your browser
2. Start a new chat
3. Click the microphone icon in the input field
4. Speak your message
5. Your speech will be transcribed by WhisperLiveKit and appear in the input field

## Advanced Configuration

### Network Configuration

#### Same Machine (Localhost)
If both WhisperLiveKit and Open WebUI run on the same machine:
```bash
API_BASE_URL=http://localhost:8000/v1
```

#### Docker to Host
If Open WebUI runs in Docker and WhisperLiveKit on the host:
```bash
API_BASE_URL=http://host.docker.internal:8000/v1
```

#### Different Machines
If they run on different machines, use the WhisperLiveKit server's IP:
```bash
API_BASE_URL=http://192.168.1.100:8000/v1
```

#### HTTPS/Secure Deployment

For production deployments with HTTPS:

1. Start WhisperLiveKit with SSL:
```bash
wlk --model base --language en \
    --host 0.0.0.0 --port 8000 \
    --ssl-certfile /path/to/cert.pem \
    --ssl-keyfile /path/to/key.pem
```

2. Configure Open WebUI with HTTPS URL:
```bash
AUDIO_STT_OPENAI_API_BASE_URL=https://your-domain.com/v1
```

### Language Configuration

WhisperLiveKit will use the language specified when starting the server:

```bash
# English only
wlk --model base --language en

# French
wlk --model base --language fr

# Auto-detect language
wlk --model base --language auto

# Multiple languages with translation to English
wlk --model base --language auto --target-language en
```

### Model Selection

Choose the appropriate Whisper model for your use case:

```bash
# Faster, less accurate (good for testing)
wlk --model tiny --language en

# Balanced (recommended for most uses)
wlk --model base --language en

# Better accuracy, more compute
wlk --model small --language en
wlk --model medium --language en

# Best accuracy (requires GPU)
wlk --model large-v3 --language en
```

### Speaker Diarization

Enable speaker identification for multi-speaker scenarios:

```bash
wlk --model base --language en --diarization
```

## API Endpoint Details

### POST /v1/audio/transcriptions

OpenAI-compatible transcription endpoint.

#### Request Format

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer wlk-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Audio file (mp3, mp4, m4a, opus, webm, etc.) |
| `model` | String | Yes | Model name (e.g., "whisper-1") - value is ignored |
| `response_format` | String | No | Output format: json, text, srt, vtt, verbose_json (default: json) |
| `language` | String | No | Language code (e.g., "en", "fr") - overrides server config |
| `prompt` | String | No | Optional prompt (currently not used) |
| `temperature` | Float | No | Sampling temperature (currently not used) |

#### Response Formats

**JSON** (`response_format=json`):
```json
{
  "text": "This is the transcribed text."
}
```

**Text** (`response_format=text`):
```
This is the transcribed text.
```

**Verbose JSON** (`response_format=verbose_json`):
```json
{
  "text": "This is the transcribed text.",
  "language": "en",
  "duration": 5.2
}
```

**SRT** (`response_format=srt`):
```srt
1
00:00:00,000 --> 00:00:05,200
This is the transcribed text.
```

**WebVTT** (`response_format=vtt`):
```vtt
WEBVTT

00:00:00.000 --> 00:00:05.200
This is the transcribed text.
```

## Docker Deployment

### WhisperLiveKit in Docker

```bash
# Build and run WhisperLiveKit with GPU
docker build -t wlk .
docker run --gpus all -p 8000:8000 \
  --name wlk \
  wlk --model base --language en --host 0.0.0.0 --port 8000

# CPU only
docker build -f Dockerfile.cpu -t wlk .
docker run -p 8000:8000 \
  --name wlk \
  wlk --model base --language en --host 0.0.0.0 --port 8000
```

### Docker Compose Example

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  whisperlivekit:
    build: .
    ports:
      - "8000:8000"
    command: >
      --model base
      --language en
      --host 0.0.0.0
      --port 8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - AUDIO_STT_ENGINE=openai
      - AUDIO_STT_OPENAI_API_BASE_URL=http://whisperlivekit:8000/v1
      - AUDIO_STT_OPENAI_API_KEY=wlk-key
      - AUDIO_STT_MODEL=whisper-1
    depends_on:
      - whisperlivekit
    restart: unless-stopped
```

Start both services:
```bash
docker-compose up -d
```

## Troubleshooting

### Connection Issues

**Problem**: Open WebUI can't connect to WhisperLiveKit

**Solutions**:
- Verify WhisperLiveKit is running: `curl http://localhost:8000/`
- Check the API endpoint: `curl http://localhost:8000/v1/audio/transcriptions`
- Ensure correct base URL (must end with `/v1`)
- Check firewall rules
- For Docker: use `host.docker.internal` instead of `localhost`

### Audio Format Issues

**Problem**: Audio upload fails or returns errors

**Solutions**:
- Ensure audio file format is supported (mp3, mp4, m4a, opus, webm)
- Check file size (large files may timeout)
- Verify audio file is not corrupted
- Check WhisperLiveKit logs for specific errors

### Empty Transcriptions

**Problem**: Transcription returns empty text

**Solutions**:
- Verify audio file contains speech
- Try a different Whisper model (larger models are more accurate)
- Check the language setting matches your audio
- Ensure audio quality is sufficient
- Check WhisperLiveKit logs for processing errors

### Performance Issues

**Problem**: Transcription is slow

**Solutions**:
- Use a smaller model (tiny, base, small)
- Ensure GPU is being used (check with `nvidia-smi`)
- Reduce audio file length
- Disable diarization if not needed
- Check system resources (CPU, memory, GPU)

### Docker Networking

**Problem**: Docker containers can't communicate

**Solutions**:
- Use service names in docker-compose (e.g., `http://whisperlivekit:8000`)
- Use `host.docker.internal` when Open WebUI container needs to reach host
- Ensure containers are on the same network
- Check docker network configuration: `docker network ls`

## Performance Tips

1. **Model Selection**: Start with `base` model, upgrade to `small` or `medium` for better accuracy
2. **GPU Acceleration**: Use GPU for faster processing with medium/large models
3. **Language Specification**: Specify language explicitly for faster processing
4. **Disable Features**: Turn off diarization if you don't need speaker identification
5. **Batch Processing**: For many files, consider using the batch API

## Supported Audio Formats

- MP3 (`.mp3`)
- MP4 audio (`.mp4`, `.m4a`)
- Opus (`.opus`)
- WebM (`.webm`)
- WAV (`.wav`)
- FLAC (`.flac`)
- OGG (`.ogg`)

## Comparison with OpenAI Whisper

| Feature | WhisperLiveKit | OpenAI Whisper API |
|---------|---------------|-------------------|
| Cost | Free (self-hosted) | $0.006/minute |
| Privacy | 100% local | Cloud-based |
| Latency | Ultra-low (<500ms) | Variable (network dependent) |
| Speaker Diarization | ✅ Yes | ❌ No |
| Real-time Streaming | ✅ Yes | ❌ No (batch only) |
| Translation | ✅ 200 languages | ✅ Limited |
| Customization | ✅ Full control | ❌ Limited |
| GPU Acceleration | ✅ Supported | N/A |

## Additional Resources

- [WhisperLiveKit Documentation](https://github.com/QuentinFuxa/WhisperLiveKit)
- [Open WebUI Documentation](https://docs.openwebui.com/)
- [Whisper Model Selection Guide](docs/default_and_custom_models.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

## Support

For issues specific to:
- **WhisperLiveKit**: [GitHub Issues](https://github.com/QuentinFuxa/WhisperLiveKit/issues)
- **Open WebUI**: [Open WebUI GitHub](https://github.com/open-webui/open-webui/issues)
- **Integration**: File an issue in the WhisperLiveKit repository with tag `open-webui`

## Contributing

Contributions to improve the Open WebUI integration are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
