# OpenVINO Backend Integration

This guide explains how to use the OpenVINO backend for fast CPU inference with WhisperLiveKit.

## Overview

The OpenVINO backend provides optimized CPU inference for Whisper models using Intel's OpenVINO toolkit. It's particularly well-suited for:

- **Intel CPUs**: Leverages Intel CPU extensions (AVX, AVX-512)
- **Fast CPU Inference**: 6-10x realtime performance on modern Intel CPUs
- **Low Memory**: Optimized INT8/INT4 quantized models (~500MB-1GB)
- **Intel GPU Support**: Can utilize Intel integrated or discrete GPUs

## Installation

### 1. Install OpenVINO GenAI

```bash
pip install openvino-genai openvino
```

### 2. Prepare OpenVINO Model

The OpenVINO backend requires Whisper models in OpenVINO IR (Intermediate Representation) format. Standard Whisper PyTorch models need to be converted first.

**Option A: Use Pre-converted Models**

For a standalone OpenVINO Whisper server with pre-configured models and setup scripts, see the related project:
- **[Whisper-Fast-Cpu-OpenVino](https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino)**

This project provides:
- Automated setup scripts
- Pre-converted INT8/INT4 quantized models
- OpenAI-compatible API
- Global dictation client
- Performance-optimized configurations

**Option B: Convert Models Manually**

To convert a Whisper model to OpenVINO format:

```bash
# Install model conversion tools
pip install optimum-intel openvino-dev

# Convert a Hugging Face Whisper model
optimum-cli export openvino \
  --model openai/whisper-tiny \
  --task automatic-speech-recognition \
  --weight-format int8 \
  openvino_model_int8_tiny/
```

## Usage

### Basic Example

```bash
# Start server with OpenVINO backend
wlk --backend openvino \
    --backend-policy localagreement \
    --openvino-model-dir /path/to/openvino_model \
    --language en
```

### Advanced Configuration

```bash
# OpenVINO with custom device and threads
wlk --backend openvino \
    --backend-policy localagreement \
    --openvino-model-dir ./openvino_model_int8_turbo \
    --openvino-device CPU \
    --openvino-threads 8 \
    --language en \
    --host 0.0.0.0
```

### Docker Deployment

```bash
# Build OpenVINO Docker image
docker build -f Dockerfile.openvino -t wlk-openvino .

# Run with custom model directory
docker run -p 8000:8000 \
  -v /path/to/models:/models \
  wlk-openvino \
  --backend openvino \
  --openvino-model-dir /models/openvino_whisper_int8
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backend openvino` | Enable OpenVINO backend | Required |
| `--openvino-model-dir` | Path to OpenVINO IR model directory | Uses `--model-dir` if not specified |
| `--openvino-device` | Device: `CPU`, `GPU`, or `AUTO` | `CPU` |
| `--openvino-threads` | Number of CPU threads (0 = auto) | `0` |
| `--backend-policy` | Must be `localagreement` for OpenVINO | Required |

## Performance Optimization

### CPU Optimization

For best CPU performance on Intel processors:

```bash
# Set thread count to number of P-cores (for hybrid architectures)
# or total physical cores (for non-hybrid)
wlk --backend openvino \
    --openvino-threads 8 \
    --openvino-device CPU
```

### Intel GPU Acceleration

To use Intel integrated or discrete GPU:

```bash
wlk --backend openvino \
    --openvino-device GPU
```

Requirements:
- Intel GPU drivers installed
- OpenVINO GPU plugin (`intel-opencl-icd` or similar)

### Auto Device Selection

Let OpenVINO choose the best device:

```bash
wlk --backend openvino \
    --openvino-device AUTO
```

## Performance Benchmarks

Tested on Intel Core i5-1240P (12th Gen):

| Model | Size | Speed | Latency | Memory |
|-------|------|-------|---------|--------|
| INT8 Turbo | ~1GB | 6-10x RT | <1s | ~500-800MB |
| INT4 | ~600MB | Fastest | <500ms | ~400-600MB |

*RT = Realtime (1x = same duration as audio)*

## Supported Backends

The OpenVINO backend currently works with:
- ✅ LocalAgreement policy (`--backend-policy localagreement`)
- ❌ SimulStreaming policy (not yet supported)

## Limitations

1. **Model Format**: Requires OpenVINO IR format models
2. **Word Timestamps**: May not provide word-level timestamps (depends on model conversion)
3. **Backend Policy**: Only works with LocalAgreement policy
4. **Experimental Status**: This backend is experimental and may have issues

## Troubleshooting

### "openvino-genai not installed" Error

```bash
pip install openvino-genai openvino
```

### "Model directory not found" Error

Ensure you're pointing to a valid OpenVINO IR model directory containing:
- `openvino_model.xml`
- `openvino_model.bin`
- Configuration files

### Slow Performance

1. Check thread count matches your CPU cores
2. Verify you're using quantized (INT8/INT4) models
3. Consider using GPU device if available
4. Ensure no other heavy processes are running

### Intel GPU Not Working

```bash
# Install Intel GPU drivers and OpenCL runtime
sudo apt install intel-opencl-icd intel-level-zero-gpu

# Verify GPU is detected
clinfo
```

## Related Projects

### Whisper-Fast-Cpu-OpenVino

For a complete, standalone OpenVINO Whisper solution:
- **GitHub**: https://github.com/groxaxo/Whisper-Fast-Cpu-OpenVino
- **Features**: 
  - Automated setup and model download
  - OpenAI-compatible API
  - Global dictation client (Ctrl+Alt+Space)
  - Open-WebUI integration
  - Optimized for Intel CPUs

## Python API Example

```python
from whisperlivekit import TranscriptionEngine

# Initialize with OpenVINO backend
engine = TranscriptionEngine(
    backend="openvino",
    backend_policy="localagreement",
    openvino_model_dir="./openvino_model_int8_turbo",
    openvino_device="CPU",
    openvino_threads=8,
    lan="en"
)

# Use with your application
# (See basic_server.py for complete example)
```

## Contributing

The OpenVINO backend is experimental. Contributions are welcome:
- Model conversion improvements
- Performance optimizations
- SimulStreaming policy support
- Better error handling

## See Also

- [WhisperCpp Optimization Guide](WHISPERCPP_OPTIMIZATION.md) - Alternative fast backend
- [API Documentation](API.md) - Complete API reference
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
