import io
import logging
import math
import sys
from typing import List

import numpy as np
import soundfile as sf

from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper.transcribe import transcribe as whisper_transcribe
from whisperlivekit.backend_support import whispercpp_backend_available
from whisperlivekit.libwhisper import WhisperCpp

logger = logging.getLogger(__name__)

# Check for pywhispercpp (OpenVINO friendly backend)
try:
    import _pywhispercpp as pw
    HAS_PYWHISPERCPP = True
except ImportError:
    HAS_PYWHISPERCPP = False

class LibWhisperParamsWrapper:
    def __init__(self, params):
        self.params = params

    def with_n_threads(self, n):
        self.params.n_threads = n
        return self

    def with_beam_size(self, n):
        self.params.beam_size = n
        return self

    def with_best_of(self, n):
        self.params.best_of = n
        return self

    def with_temperature_inc(self, val):
        self.params.temperature_inc = val
        return self

    def with_n_max_text_ctx(self, n):
        self.params.n_max_text_ctx = n
        return self

    def with_token_timestamps(self, val):
        self.params.token_timestamps = val
        return self

    def with_timestamps(self, val):
        self.params.no_timestamps = not val
        return self

    def with_max_len(self, n):
        self.params.max_len = n
        return self

    def with_no_context(self, val):
        self.params.no_context = val
        return self

    def with_language(self, lang):
        if lang == "auto":
            self.params.language = None
            self.params.detect_language = True
        else:
            self.params.language = lang.encode('utf-8')
        return self
    
    def set_tokens(self, tokens):
        # TODO: implement in libwhisper.py if needed
        pass

class LibWhisperContextWrapper:
    def __init__(self, model_path):
        self.w = WhisperCpp(model_path)
        
    def full(self, params, audio):
        real_params = params.params if isinstance(params, LibWhisperParamsWrapper) else params
        return self.w.full(audio, real_params)
        
    def full_lang_id(self):
        return self.w.full_lang_id()
        
    def lang_id_to_str(self, lang_id):
        return self.w.lang_id_to_str(lang_id)
        
    def full_n_segments(self):
        return self.w.n_segments()
        
    def full_get_segment_text(self, s):
        return self.w.get_segment_text(s)
        
    def full_get_segment_start(self, s):
        return self.w.get_segment_t0(s)
        
    def full_get_segment_end(self, s):
        return self.w.get_segment_t1(s)
        
    def full_n_tokens(self, s):
        return self.w.n_tokens(s)
        
    def full_get_token_text(self, s, i):
        return self.w.get_token_text(s, i)
        
    def full_get_token_data(self, s, i):
        return self.w.get_token_data(s, i)
        
    def tokenize(self, text, n_max):
        return self.w.tokenize(text, n_max)

    def openvino_init(self, model_path, device, cache_dir):
        return self.w.init_openvino(model_path, device, cache_dir)

    def get_full_params(self):
        return self.w.get_full_params()

class PyWhisperCppParamsWrapper:
    def __init__(self, params):
        self.params = params

    def with_n_threads(self, n):
        self.params.n_threads = n
        return self

    def with_beam_size(self, n):
        if hasattr(self.params, 'beam_search'):
             self.params.beam_search.beam_size = n
        return self

    def with_best_of(self, n):
        if hasattr(self.params, 'greedy'):
             self.params.greedy.best_of = n
        return self

    def with_temperature_inc(self, val):
        self.params.temperature_inc = val
        return self

    def with_n_max_text_ctx(self, n):
        self.params.n_max_text_ctx = n
        return self

    def with_token_timestamps(self, val):
        self.params.token_timestamps = val
        return self

    def with_timestamps(self, val):
        # pywhispercpp doesn't expose 'print_timestamps' in the same way for logic
        # but token_timestamps handles the timing info we need
        return self

    def with_max_len(self, n):
        self.params.max_len = n
        return self

    def with_no_context(self, val):
        self.params.no_context = val
        return self

    def with_language(self, lang):
        self.params.language = lang
        return self
    
    def set_tokens(self, tokens):
        pass

class PyWhisperCppContextWrapper:
    def __init__(self, ctx):
        self.ctx = ctx
        
    def full(self, params, audio):
        real_params = params.params if isinstance(params, PyWhisperCppParamsWrapper) else params
        return pw.whisper_full(self.ctx, real_params, audio, audio.size)
        
    def full_lang_id(self):
        return pw.whisper_full_lang_id(self.ctx)
        
    def lang_id_to_str(self, lang_id):
        return pw.whisper_lang_str(lang_id)
        
    def full_n_segments(self):
        return pw.whisper_full_n_segments(self.ctx)
        
    def full_get_segment_text(self, s):
        res = pw.whisper_full_get_segment_text(self.ctx, s)
        if isinstance(res, bytes):
            return res.decode('utf-8', errors='ignore')
        return res
        
    def full_get_segment_start(self, s):
        return pw.whisper_full_get_segment_t0(self.ctx, s)
        
    def full_get_segment_end(self, s):
        return pw.whisper_full_get_segment_t1(self.ctx, s)
        
    def full_n_tokens(self, s):
        return pw.whisper_full_n_tokens(self.ctx, s)
        
    def full_get_token_text(self, s, i):
        res = pw.whisper_full_get_token_text(self.ctx, s, i)
        if isinstance(res, bytes):
            return res.decode('utf-8', errors='ignore')
        return res
        
    def full_get_token_data(self, s, i):
        return pw.whisper_full_get_token_data(self.ctx, s, i)
        
    def tokenize(self, text, n_max):
        return pw.whisper_tokenize(self.ctx, text, n_max)

    def openvino_init(self, model_path, device, cache_dir):
        return pw.whisper_ctx_init_openvino_encoder(self.ctx, model_path, device, cache_dir)

class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, lora_path=None, logfile=sys.stderr, **kwargs):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.lora_path = lora_path
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    def load_model(self, model_size, cache_dir, model_dir):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperASR(ASRBase):
    """Uses WhisperLiveKit's built-in Whisper implementation."""
    sep = " "

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from whisperlivekit.whisper import load_model as load_whisper_model

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)            
            if resolved_path.is_dir():
                model_info = detect_model_format(resolved_path)
                if not model_info.has_pytorch:
                    raise FileNotFoundError(
                        f"No supported PyTorch checkpoint found under {resolved_path}"
                    )            
            logger.debug(f"Loading Whisper model from custom path {resolved_path}")
            return load_whisper_model(str(resolved_path), lora_path=self.lora_path)

        if model_size is None:
            raise ValueError("Either model_size or model_dir must be set for WhisperASR")

        return load_whisper_model(model_size, download_root=cache_dir, lora_path=self.lora_path)

    def transcribe(self, audio, init_prompt=""):
        options = dict(self.transcribe_kargs)
        options.pop("vad", None)
        options.pop("vad_filter", None)
        language = self.original_language if self.original_language else None

        result = whisper_transcribe(
            self.model,
            audio,
            language=language,
            initial_prompt=init_prompt,
            condition_on_previous_text=True,
            word_timestamps=True,
            **options,
        )
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the Whisper result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                token = ASRToken(
                    word["start"],
                    word["end"],
                    word["word"],
                    probability=word.get("probability"),
                )
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        logger.warning("VAD is not currently supported for WhisperASR backend and will be ignored.")

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading faster-whisper model from {resolved_path}. "
                         f"model_size and cache_dir parameters are not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("Either model_size or model_dir must be set")
        device = "auto" # Allow CTranslate2 to decide available device
        compute_type = "auto" # Allow CTranslate2 to decide faster compute type
                              

        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder, transcribe

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading MLX Whisper model from {resolved_path}. model_size parameter is not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = self.translate_model_name(model_size)
            logger.debug(f"Loading whisper model {model_size}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either model_size or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                probability=word["probability"]
                token = ASRToken(word["start"], word["end"], word["word"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.direct_english_translation = False

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        tokens = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if not self.direct_english_translation and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True


class WhisperCppASR(ASRBase):
    """
    whisper.cpp backend via whispercpp Python bindings.
    Works with LocalAgreement (not SimulStreaming).
    Optimized for realtime transcription with low CPU usage.
    
    Supports OpenVINO encoder offload for Intel iGPU acceleration.
    """
    sep = ""

    _TIME_UNIT_TO_SECONDS = 0.01  # whisper.cpp timestamps are typically centiseconds

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, 
                 threads=8, beam_size=1, best_of=1, no_fallback=False,
                 max_context=-1, no_timestamps=False, max_len=0,
                 step_ms=500, window_ms=5000,
                 openvino=False, ov_encoder=None, ov_device="CPU",
                 **kwargs):
        # Store performance parameters
        self.threads = threads
        self.beam_size = beam_size
        self.best_of = best_of
        self.no_fallback = no_fallback
        self.max_context = max_context
        self.no_timestamps = no_timestamps
        self.max_len = max_len
        self.step_ms = step_ms
        self.window_ms = window_ms
        
        # OpenVINO parameters
        self.openvino = openvino
        self.ov_encoder = ov_encoder
        self.ov_device = ov_device
        
        # Validate OpenVINO configuration
        if self.openvino and not self.ov_encoder:
            raise ValueError(
                "--whispercpp-openvino requires --whispercpp-ov-encoder (path to encoder XML)"
            )
        
        # Call parent init
        super().__init__(lan, model_size, cache_dir, model_dir, **kwargs)
        
        logger.info(
            f"WhisperCpp configured: threads={threads}, beam_size={beam_size}, "
            f"best_of={best_of}, no_fallback={no_fallback}, max_context={max_context}, "
            f"no_timestamps={no_timestamps}, max_len={max_len}, "
            f"step_ms={step_ms}, window_ms={window_ms}"
        )
        
        if self.openvino:
            logger.info(
                f"OpenVINO encoder offload requested: device={self.ov_device}, "
                f"encoder={self.ov_encoder}"
            )

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        # We prefer our own LibWhisper wrapper as it is guaranteed to have the 
        # features we need (like OpenVINO) if we built it ourselves.
        try:
            from whisperlivekit.libwhisper import lib
            if lib is not None:
                return self._load_libwhisper(model_size, cache_dir, model_dir)
        except Exception as e:
            logger.debug(f"LibWhisper not available, falling back to other bindings: {e}")

        if not whispercpp_backend_available(warn_on_missing=True):
            raise ImportError("whispercpp is not installed and libwhisper.so not found. Install with: pip install whispercpp")

        # Determine which backend to use
        use_pywhispercpp = False
        has_whispercpp = False
        try:
            from whispercpp import api, Whisper
            self._api = api
            has_whispercpp = True
        except ImportError:
            has_whispercpp = False

        if HAS_PYWHISPERCPP:
            if not has_whispercpp:
                use_pywhispercpp = True
            elif self.openvino:
                # If OpenVINO requested, check if standard backend supports it
                if has_whispercpp and not hasattr(api.Context, 'openvino_init'):
                    logger.info("Standard whispercpp lacks OpenVINO support; switching to pywhispercpp.")
                    use_pywhispercpp = True
        
        if use_pywhispercpp:
            return self._load_pywhispercpp(model_size, cache_dir, model_dir)
        
        # Standard whispercpp path
        from whispercpp import api, Whisper
        self._api = api

        # If user provided a path, it should be a ggml/gguf model file OR a folder containing one.
        if model_dir is not None:
            resolved = resolve_model_path(model_dir)
            ggml_file = None

            if resolved.is_file():
                ggml_file = resolved
            elif resolved.is_dir():
                # pick first ggml*.bin file (common whisper.cpp distribution)
                for f in resolved.iterdir():
                    if f.is_file() and f.name.lower().startswith("ggml") and f.suffix.lower() == ".bin":
                        ggml_file = f
                        break

            if ggml_file is None:
                raise FileNotFoundError(
                    f"whispercpp backend requires a ggml model file (e.g. ggml-*.bin). "
                    f"Got: {resolved}"
                )

            self._ctx = api.Context.from_file(str(ggml_file))
            self._params = api.Params.from_enum(api.SAMPLING_GREEDY).build()
            
            # Attempt OpenVINO initialization if requested
            self._init_openvino_if_available()
            
            return self._ctx  # stored in self._ctx

        # Otherwise, allow whispercpp to download a converted model by name (tiny/base/small/...)
        if model_size is None:
            raise ValueError("Either model_size or model_dir must be set for WhisperCppASR")

        w = Whisper.from_pretrained(model_size, basedir=cache_dir)
        self._ctx = w.context
        self._params = w.params
        
        # Attempt OpenVINO initialization if requested
        self._init_openvino_if_available()
        
        return self._ctx

    def _load_libwhisper(self, model_size, cache_dir, model_dir):
        ggml_file = None
        if model_dir:
            resolved = resolve_model_path(model_dir)
            if resolved.is_file():
                ggml_file = str(resolved)
            elif resolved.is_dir():
                for f in resolved.iterdir():
                    if f.is_file() and f.name.lower().startswith("ggml") and f.suffix.lower() == ".bin":
                        ggml_file = str(f)
                        break
        
        if not ggml_file:
             raise ValueError("Could not find model file for libwhisper. Please provide --model-dir with a GGML model.")
             
        self._ctx = LibWhisperContextWrapper(ggml_file)
        self._params = LibWhisperParamsWrapper(self._ctx.get_full_params())
        
        self._init_openvino_if_available()
        return self._ctx

    def _load_pywhispercpp(self, model_size, cache_dir, model_dir):
        ggml_file = None
        if model_dir:
            resolved = resolve_model_path(model_dir)
            if resolved.is_file():
                ggml_file = str(resolved)
            elif resolved.is_dir():
                for f in resolved.iterdir():
                    if f.is_file() and f.name.lower().startswith("ggml") and f.suffix.lower() == ".bin":
                        ggml_file = str(f)
                        break
        
        if not ggml_file and model_size:
             try:
                 from pywhispercpp.utils import download_model
                 ggml_file = download_model(model_size, cache_dir)
             except ImportError:
                 pass
        
        if not ggml_file:
             raise ValueError("Could not find model file for pywhispercpp (and auto-download failed). Please provide --model-dir.")
             
        ctx_ptr = pw.whisper_init_from_file(ggml_file)
        self._ctx = PyWhisperCppContextWrapper(ctx_ptr)
        
        strategy = pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
        raw_params = pw.whisper_full_default_params(strategy)
        self._params = PyWhisperCppParamsWrapper(raw_params)
        
        self._init_openvino_if_available()
        return self._ctx
    
    def _init_openvino_if_available(self):
        """
        Attempt to initialize OpenVINO encoder offload.
        
        Note: The current whispercpp Python bindings (by aarnphm) do not expose
        whisper_openvino_init(). This method is a placeholder that logs the status
        and will work when bindings add support.
        """
        if not self.openvino:
            return
        
        # Check if the bindings expose openvino_init
        if hasattr(self._ctx, 'openvino_init'):
            try:
                success = self._ctx.openvino_init(
                    self.ov_encoder,
                    self.ov_device,
                    ""  # cache_dir - empty string for default
                )
                if success:
                    logger.info(f"OpenVINO encoder initialized successfully on {self.ov_device}")
                else:
                    logger.warning("OpenVINO encoder initialization returned False")
            except Exception as e:
                logger.error(f"OpenVINO initialization failed: {e}")
        else:
            # Current bindings don't expose the function
            logger.warning(
                "OpenVINO encoder offload requested but current whispercpp bindings "
                "do not expose openvino_init(). OpenVINO will NOT be used. "
                "To enable OpenVINO, use bindings that expose whisper_openvino_init() "
                "or build custom bindings with OpenVINO support."
            )

    def _configure_params(self, init_prompt: str):
        p = self._params

        # Apply performance parameters
        # Threads
        if hasattr(p, "with_n_threads"):
            p.with_n_threads(self.threads)
        
        # Beam size (greedy decoding when beam_size=1)
        if hasattr(p, "with_beam_size"):
            p.with_beam_size(self.beam_size)
        
        # Best of
        if hasattr(p, "with_best_of"):
            p.with_best_of(self.best_of)
        
        # No fallback
        if self.no_fallback and hasattr(p, "with_temperature_inc"):
            p.with_temperature_inc(0.0)  # Disable temperature fallback
        
        # Max context
        if self.max_context > 0 and hasattr(p, "with_n_max_text_ctx"):
            p.with_n_max_text_ctx(self.max_context)
        
        # Timestamp control
        if self.no_timestamps:
            # Disable all timestamps for maximum speed
            if hasattr(p, "with_token_timestamps"):
                p.with_token_timestamps(False)
            if hasattr(p, "with_timestamps"):
                p.with_timestamps(False)
        else:
            # Enable token timestamps only if max_len > 0 (word-level timing needed)
            if hasattr(p, "with_token_timestamps"):
                p.with_token_timestamps(self.max_len > 0)
            if hasattr(p, "with_timestamps"):
                p.with_timestamps(True)
        
        # Max length
        if self.max_len > 0 and hasattr(p, "with_max_len"):
            p.with_max_len(self.max_len)

        # Keep context across calls (closer to condition_on_previous_text=True)
        if hasattr(p, "with_no_context"):
            p.with_no_context(False)

        # Language
        if hasattr(p, "with_language"):
            if self.original_language:
                p.with_language(self.original_language)
            else:
                p.with_language("auto")

        # Prompt: best-effort (bindings differ). Ignore if unsupported.
        if init_prompt:
            try:
                if hasattr(self._ctx, "tokenize") and hasattr(p, "set_tokens"):
                    ids = self._ctx.tokenize(init_prompt, 512)
                    p.set_tokens(ids)
            except Exception:
                logger.warning("whispercpp init_prompt could not be applied; continuing without it.")

        return p

    def transcribe(self, audio, init_prompt=""):
        audio = np.asarray(audio, dtype=np.float32)
        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio)

        params = self._configure_params(init_prompt)
        self._ctx.full(params, audio)

        # best-effort detected language
        detected_language = None
        try:
            lang_id = self._ctx.full_lang_id()
            if hasattr(self._ctx, "lang_id_to_str"):
                detected_language = self._ctx.lang_id_to_str(lang_id)
        except Exception:
            pass

        segments = []
        n_seg = self._ctx.full_n_segments()
        for s in range(n_seg):
            text = self._ctx.full_get_segment_text(s)
            t0 = self._ctx.full_get_segment_start(s) * self._TIME_UNIT_TO_SECONDS
            t1 = self._ctx.full_get_segment_end(s) * self._TIME_UNIT_TO_SECONDS

            # token-level timing -> reconstruct "word-ish" items
            token_items = []
            n_tok = self._ctx.full_n_tokens(s)
            for i in range(n_tok):
                ttxt = self._ctx.full_get_token_text(s, i)
                tdat = self._ctx.full_get_token_data(s, i)  # has t0/t1/p in most builds
                token_items.append(
                    {
                        "text": ttxt,
                        "t0": float(getattr(tdat, "t0", 0)) * self._TIME_UNIT_TO_SECONDS,
                        "t1": float(getattr(tdat, "t1", 0)) * self._TIME_UNIT_TO_SECONDS,
                        "p": float(getattr(tdat, "p", 0.0)),
                    }
                )

            words = self._tokens_to_words(token_items)

            segments.append({"start": t0, "end": t1, "text": text, "words": words})

        return {"segments": segments, "language": detected_language}

    def _tokens_to_words(self, token_items):
        words = []
        cur_text = ""
        cur_t0 = None
        cur_t1 = None
        probs = []

        def flush():
            nonlocal cur_text, cur_t0, cur_t1, probs
            if not cur_text or cur_t0 is None or cur_t1 is None:
                cur_text, cur_t0, cur_t1, probs = "", None, None, []
                return
            # geometric mean of probs (best-effort)
            safe = [p for p in probs if p and p > 0.0]
            if safe:
                prob = math.exp(sum(math.log(p) for p in safe) / len(safe))
            else:
                prob = None
            words.append({"start": cur_t0, "end": cur_t1, "word": cur_text, "probability": prob})
            cur_text, cur_t0, cur_t1, probs = "", None, None, []

        for it in token_items:
            txt = it["text"]

            # skip special tokens best-effort
            if txt.startswith("<|") and txt.endswith("|>"):
                continue

            # boundary: token begins with a space => new word
            if txt.startswith(" ") and cur_text:
                flush()

            if cur_t0 is None:
                cur_t0 = it["t0"]
            cur_t1 = it["t1"]
            cur_text += txt
            probs.append(it.get("p", 0.0))

        flush()
        return words

    def ts_words(self, r) -> List[ASRToken]:
        tokens = []
        for seg in r.get("segments", []):
            for w in seg.get("words", []):
                tokens.append(
                    ASRToken(
                        w["start"],
                        w["end"],
                        w["word"],
                        probability=w.get("probability"),
                    )
                )
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [seg["end"] for seg in res.get("segments", [])]

    def use_vad(self):
        logger.warning("VAD is not supported for WhisperCppASR backend and will be ignored.")


class OpenVINOASR(ASRBase):
    """Uses OpenVINO GenAI as the backend for optimized CPU inference."""
    sep = ""

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, 
                 device="CPU", threads=0, **kwargs):
        self.device = device
        self.threads = threads
        super().__init__(lan, model_size, cache_dir, model_dir, **kwargs)

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        try:
            import openvino_genai as ov_genai
        except ImportError:
            raise ImportError(
                "OpenVINO GenAI is not installed. "
                "Install it with: pip install openvino-genai openvino"
            )

        if model_dir is None:
            raise ValueError(
                "OpenVINO backend requires --openvino-model-dir or --model-dir "
                "pointing to an OpenVINO IR model directory. "
                "Standard Whisper models need to be converted to OpenVINO format first."
            )

        resolved_path = resolve_model_path(model_dir)
        logger.info(f"Loading OpenVINO Whisper model from {resolved_path} on device {self.device}")
        
        # Initialize OpenVINO Whisper pipeline
        config = {}
        if self.threads > 0:
            config["NUM_STREAMS"] = "1"
            config["INFERENCE_NUM_THREADS"] = str(self.threads)
        
        try:
            pipeline = ov_genai.WhisperPipeline(str(resolved_path), self.device, **config)
            logger.info("OpenVINO Whisper model loaded successfully")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            raise RuntimeError(
                f"Could not load OpenVINO model from {resolved_path}. "
                f"Ensure the directory contains a valid OpenVINO IR model. Error: {e}"
            )

    def transcribe(self, audio, init_prompt=""):
        """Transcribe audio using OpenVINO GenAI pipeline."""
        import openvino_genai as ov_genai
        
        # Prepare generation config
        config = ov_genai.WhisperGenerationConfig()
        config.max_new_tokens = 448  # Standard for Whisper
        config.return_timestamps = True
        
        if self.original_language:
            config.language = f"<|{self.original_language}|>"
        
        if init_prompt:
            config.initial_prompt = init_prompt
        
        # Convert audio to the format expected by OpenVINO
        # OpenVINO expects float32 audio at 16kHz
        if isinstance(audio, np.ndarray):
            audio_data = audio.astype(np.float32)
        else:
            audio_data = np.array(audio, dtype=np.float32)
        
        # Run inference
        try:
            result = self.model.generate(audio_data, config)
            
            # Convert OpenVINO result to expected format
            # Note: This is a simplified conversion - full implementation
            # would need to properly parse OpenVINO's output format
            segments = []
            if hasattr(result, 'chunks'):
                for chunk in result.chunks:
                    segments.append({
                        "start": chunk.start_ts,
                        "end": chunk.end_ts,
                        "text": chunk.text,
                        "words": []  # OpenVINO may not provide word-level timestamps
                    })
            else:
                # Fallback if chunks not available
                segments.append({
                    "start": 0.0,
                    "end": len(audio_data) / 16000.0,
                    "text": str(result),
                    "words": []
                })
            
            return {"segments": segments, "language": self.original_language or "en"}
        except Exception as e:
            logger.error(f"OpenVINO transcription failed: {e}")
            raise

    def ts_words(self, r) -> List[ASRToken]:
        """Convert OpenVINO result to ASRToken list."""
        tokens = []
        for segment in r.get("segments", []):
            # If word-level timestamps are available
            if segment.get("words"):
                for word in segment["words"]:
                    token = ASRToken(
                        word["start"],
                        word["end"],
                        word["word"],
                        probability=word.get("probability"),
                    )
                    tokens.append(token)
            else:
                # Fallback: create a single token for the whole segment
                text = segment.get("text", "").strip()
                if text:
                    token = ASRToken(
                        segment["start"],
                        segment["end"],
                        text,
                    )
                    tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [seg["end"] for seg in res.get("segments", [])]

    def use_vad(self):
        logger.warning("VAD is not currently supported for OpenVINO backend and will be ignored.")
