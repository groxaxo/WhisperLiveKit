# WhisperCpp Realtime Optimization Guide

## Overview

This guide explains how to configure WhisperLiveKit's WhisperCpp backend for optimal realtime transcription performance with low CPU usage, specifically tuned for Intel 1240P processors.

## Quick Start: Optimal Realtime Configuration

```bash
python -m whisperlivekit \
  --backend whispercpp \
  --backend-policy localagreement \
  --model-dir /path/to/ggml-large-v3-turbo-q5_0.bin \
  --lan en \
  --whispercpp-threads 8 \
  --whispercpp-beam-size 1 \
  --whispercpp-best-of 1 \
  --whispercpp-no-fallback \
  --whispercpp-max-context 448 \
  --whispercpp-step-ms 500 \
  --whispercpp-window-ms 5000 \
  --vad
```

## Performance Parameters Explained

### Core Performance Knobs

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--whispercpp-threads` | 8 | 8 | Number of CPU threads. 8 is optimal for Intel 1240P (8 P-cores) |
| `--whispercpp-beam-size` | 1 | 1 | Beam search width. 1 = greedy decoding (fastest) |
| `--whispercpp-best-of` | 1 | 1 | Number of candidates to keep. 1 = fastest |
| `--whispercpp-no-fallback` | false | true | Disable temperature fallback for speed |
| `--whispercpp-max-context` | -1 | 448 | Cap context tokens to prevent unbounded growth |

### Timestamp Control

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--whispercpp-no-timestamps` | false | false | Disable all timestamps (fastest, but no timing info) |
| `--whispercpp-max-len` | 0 | 0 | Set to 1 only if you need word-level timestamps (slower) |

**Note**: For realtime captions, keep segment timestamps enabled (`--whispercpp-no-timestamps` false) but avoid word-level timing (`--whispercpp-max-len 0`) unless absolutely necessary.

### Chunking & Buffering

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--whispercpp-step-ms` | 500 | 400-700 | Processing step size in milliseconds |
| `--whispercpp-window-ms` | 5000 | 4000-6000 | Rolling audio buffer window in milliseconds |

These values are based on whisper.cpp's proven realtime example (`--step 500 --length 5000`).

### Language Setting

**Always set a fixed language when possible:**
```bash
--lan en  # or es, fr, de, etc.
```

Language auto-detection adds latency. Specifying the language upfront improves speed significantly.

### VAD (Voice Activity Detection)

**Enable VAD to skip processing silence:**
```bash
--vad
```

This ensures decoding only runs during speech, dramatically reducing CPU usage.

## Recommended Model: Turbo v3 Q5

Download the quantized Turbo v3 model for best speed/quality balance:

```bash
# Download ggml-large-v3-turbo-q5_0.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin

# Use it
--model-dir /path/to/ggml-large-v3-turbo-q5_0.bin
```

**Why Turbo v3 Q5?**
- **Turbo**: 8x faster than large-v3 with minimal quality loss
- **Q5**: 5-bit quantization provides excellent speed with good accuracy
- **Size**: ~1.5GB (vs 3GB for full precision)

## Building whisper.cpp with BLAS (Intel CPU Optimization)

For maximum performance on Intel CPUs, build whisper.cpp with OpenBLAS support:

### 1. Install OpenBLAS

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libopenblas-dev
```

**Arch Linux:**
```bash
sudo pacman -S openblas
```

### 2. Build whispercpp with BLAS

```bash
# Clone whispercpp if you haven't
pip uninstall whispercpp  # Remove pip version
git clone https://github.com/stlukey/whispercpp.git
cd whispercpp

# Build with BLAS support
export GGML_BLAS=1
pip install -e .
```

**Expected speedup**: 2-3x faster than CPU-only build on Intel 1240P.

### 3. Verify BLAS is Active

When you run WhisperLiveKit, you should see log messages indicating BLAS usage:
```
ggml_init_cublas: found X BLAS devices
```

## OpenVINO Encoder Offload (Intel iGPU)

For Intel systems with integrated graphics (e.g., Iris Xe on 12th gen+), you can offload the encoder to the iGPU using OpenVINO, reducing CPU load significantly.

### Prerequisites

1. **Intel OpenVINO Runtime**: Install from [Intel's OpenVINO toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
2. **libwhisper.so with OpenVINO**: WhisperLiveKit includes a built-in `ctypes` wrapper. To use it, you must build `libwhisper.so` from the included `whispercpp_official` submodule with OpenVINO enabled.
3. **OpenVINO encoder XML file**: Generated from the Whisper model

### Generating OpenVINO Encoder Files

Use the whisper.cpp conversion script:

```bash
# Clone whisper.cpp
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp

# Generate OpenVINO encoder for large-v3-turbo
# Note: Use base model name without quant suffix
python models/convert-whisper-to-openvino.py \
  --model large-v3-turbo \
  --output models/large-v3-turbo-encoder-openvino
```

This creates:
- `large-v3-turbo-encoder-openvino.xml`
- `large-v3-turbo-encoder-openvino.bin`

### Building libwhisper.so with OpenVINO

To use the built-in OpenVINO support, build the shared library:

```bash
cd whispercpp_official
mkdir build && cd build
cmake .. -DWHISPER_OPENVINO=1 -DBUILD_SHARED_LIBS=ON
make -j$(nproc) whisper
```

The resulting `libwhisper.so` should be located at `whispercpp_official/build/src/libwhisper.so`. WhisperLiveKit will automatically detect and use it.

### CLI Usage

```bash
python -m whisperlivekit \
  --backend whispercpp \
  --backend-policy localagreement \
  --model-dir /path/to/ggml-large-v3-turbo-q5_0.bin \
  --lan en \
  --whispercpp-openvino \
  --whispercpp-ov-encoder /path/to/large-v3-turbo-encoder-openvino.xml \
  --whispercpp-ov-device GPU
```

### OpenVINO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--whispercpp-openvino` | off | Enable OpenVINO encoder offload |
| `--whispercpp-ov-encoder` | (required) | Path to OpenVINO encoder XML file |
| `--whispercpp-ov-device` | CPU | Target device: CPU, GPU, or NPU |

### Expected Performance

With OpenVINO GPU offload on Intel Iris Xe:
- **CPU usage**: Reduced by 30-50%
- **First inference**: Slow (OpenVINO compilation, cached for subsequent runs)
- **Subsequent inferences**: Faster due to caching

> **Note**: First run triggers OpenVINO graph compilation. WhisperLiveKit's warmup handles this automatically.


## Performance Tuning Tips

### 1. Single Instance + Queue

WhisperCpp performs best with:
- **One model instance** per process
- **Serialized decode calls** through a queue
- **No parallel decoding** (it thrashes CPU caches)

WhisperLiveKit already implements this pattern via `TranscriptionEngine` singleton.

### 2. Audio Format

Ensure audio is:
- **16kHz sample rate**
- **Mono channel**
- **Float32 format**

Resampling in Python per chunk is expensive. Configure your audio capture to deliver 16kHz mono directly.

### 3. Uvicorn Workers

For the FastAPI server, use:
```bash
uvicorn whisperlivekit.server:app --workers 1
```

Scale horizontally by running multiple containers/processes, not multiple workers per process.

## Benchmark: Expected Performance

On Intel 1240P with optimal settings:

| Configuration | RTF* | CPU Usage | Latency |
|---------------|------|-----------|---------|
| Default (no optimization) | 0.8-1.2 | 60-80% | 800-1200ms |
| With BLAS + tuned params | 0.2-0.4 | 25-40% | 300-500ms |
| + Turbo v3 Q5 model | 0.1-0.2 | 15-25% | 200-350ms |

*RTF = Real-Time Factor (lower is better; 0.1 = 10x faster than realtime)

## Example Configurations

### Maximum Speed (no word timestamps)

```bash
python -m whisperlivekit \
  --backend whispercpp \
  --backend-policy localagreement \
  --model-dir ggml-large-v3-turbo-q5_0.bin \
  --lan en \
  --whispercpp-threads 8 \
  --whispercpp-beam-size 1 \
  --whispercpp-best-of 1 \
  --whispercpp-no-fallback \
  --whispercpp-max-context 448 \
  --whispercpp-no-timestamps \
  --vad
```

### Balanced (segment timestamps, no word-level)

```bash
python -m whisperlivekit \
  --backend whispercpp \
  --backend-policy localagreement \
  --model-dir ggml-large-v3-turbo-q5_0.bin \
  --lan en \
  --whispercpp-threads 8 \
  --whispercpp-beam-size 1 \
  --whispercpp-best-of 1 \
  --whispercpp-no-fallback \
  --whispercpp-max-context 448 \
  --whispercpp-step-ms 500 \
  --whispercpp-window-ms 5000 \
  --vad
```

### High Quality (with word timestamps)

```bash
python -m whisperlivekit \
  --backend whispercpp \
  --backend-policy localagreement \
  --model-dir ggml-large-v3-turbo-q5_0.bin \
  --lan en \
  --whispercpp-threads 8 \
  --whispercpp-beam-size 2 \
  --whispercpp-best-of 2 \
  --whispercpp-max-context 448 \
  --whispercpp-max-len 1 \
  --whispercpp-step-ms 600 \
  --whispercpp-window-ms 6000 \
  --vad
```

## Troubleshooting

### High CPU Usage

1. Verify BLAS is enabled (check logs)
2. Ensure `--whispercpp-beam-size 1` and `--whispercpp-best-of 1`
3. Enable `--whispercpp-no-fallback`
4. Reduce `--whispercpp-threads` if hyperthreading causes issues

### High Latency

1. Reduce `--whispercpp-step-ms` to 400-500ms
2. Ensure `--lan` is set (not auto)
3. Enable `--vad` to skip silence
4. Use Turbo v3 model instead of large-v3

### Poor Accuracy

1. Increase `--whispercpp-beam-size` to 2-3
2. Remove `--whispercpp-no-fallback`
3. Increase `--whispercpp-window-ms` to 6000-8000
4. Use higher quality model (Q8 instead of Q5)

## References

- [whisper.cpp GitHub](https://github.com/ggml-org/whisper.cpp)
- [whisper.cpp CLI README](https://huggingface.co/spaces/natasa365/whisper.cpp/blob/main/examples/cli/README.md)
- [whisper.cpp Models](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
- [OpenBLAS](https://www.openblas.net/)
