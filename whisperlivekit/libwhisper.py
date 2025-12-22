import ctypes
import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "..", "whispercpp_official", "build", "src", "libwhisper.so")
if not os.path.exists(lib_path):
    # Try another common location if not found (e.g. if installed)
    lib_path = "libwhisper.so"

try:
    lib = ctypes.CDLL(lib_path)
except Exception as e:
    logger.debug(f"Could not load libwhisper.so from {lib_path}: {e}")
    lib = None

# Define types
whisper_context_p = ctypes.c_void_p
whisper_state_p = ctypes.c_void_p
whisper_full_params_p = ctypes.c_void_p
whisper_token = ctypes.c_int32

class whisper_ahead(ctypes.Structure):
    _fields_ = [
        ("n_text_layer", ctypes.c_int),
        ("n_head", ctypes.c_int),
    ]

class whisper_aheads(ctypes.Structure):
    _fields_ = [
        ("n_heads", ctypes.c_size_t),
        ("heads", ctypes.POINTER(whisper_ahead)),
    ]

class whisper_context_params(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("gpu_device", ctypes.c_int),
        ("dtw_token_timestamps", ctypes.c_bool),
        ("dtw_aheads_preset", ctypes.c_int),
        ("dtw_n_top", ctypes.c_int),
        ("dtw_aheads", whisper_aheads),
        ("dtw_mem_size", ctypes.c_size_t),
    ]

class whisper_token_data(ctypes.Structure):
    _fields_ = [
        ("id", whisper_token),
        ("tid", whisper_token),
        ("p", ctypes.c_float),
        ("plog", ctypes.c_float),
        ("pt", ctypes.c_float),
        ("ptsum", ctypes.c_float),
        ("t0", ctypes.c_int64),
        ("t1", ctypes.c_int64),
        ("t_dtw", ctypes.c_int64),
        ("vlen", ctypes.c_float),
    ]

# Callback types
whisper_new_segment_callback = ctypes.CFUNCTYPE(None, whisper_context_p, whisper_state_p, ctypes.c_int, ctypes.c_void_p)
whisper_progress_callback = ctypes.CFUNCTYPE(None, whisper_context_p, whisper_state_p, ctypes.c_int, ctypes.c_void_p)
whisper_encoder_begin_callback = ctypes.CFUNCTYPE(ctypes.c_bool, whisper_context_p, whisper_state_p, ctypes.c_void_p)
whisper_logits_filter_callback = ctypes.CFUNCTYPE(None, whisper_context_p, whisper_state_p, ctypes.POINTER(whisper_token_data), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p)
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

class whisper_vad_params(ctypes.Structure):
    _fields_ = [
        ("threshold", ctypes.c_float),
        ("min_speech_duration_ms", ctypes.c_int),
        ("min_silence_duration_ms", ctypes.c_int),
        ("max_speech_duration_s", ctypes.c_float),
        ("speech_pad_ms", ctypes.c_int),
        ("samples_overlap", ctypes.c_float),
    ]

class whisper_full_params(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("no_timestamps", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        ("debug_mode", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        ("tdrz_enable", ctypes.c_bool),
        ("suppress_regex", ctypes.c_char_p),
        ("initial_prompt", ctypes.c_char_p),
        ("carry_initial_prompt", ctypes.c_bool),
        ("prompt_tokens", ctypes.POINTER(whisper_token)),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("detect_language", ctypes.c_bool),
        ("suppress_blank", ctypes.c_bool),
        ("suppress_nst", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        # greedy
        ("best_of", ctypes.c_int),
        # beam_search
        ("beam_size", ctypes.c_int),
        ("patience", ctypes.c_float),
        # callbacks
        ("new_segment_callback", whisper_new_segment_callback),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", whisper_progress_callback),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", whisper_encoder_begin_callback),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", whisper_logits_filter_callback),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
        
        ("grammar_rules", ctypes.c_void_p), # const whisper_grammar_element **
        ("n_grammar_rules", ctypes.c_size_t),
        ("i_start_rule", ctypes.c_size_t),
        ("grammar_penalty", ctypes.c_float),
        
        ("vad", ctypes.c_bool),
        ("vad_model_path", ctypes.c_char_p),
        ("vad_params", whisper_vad_params),
    ]

if lib:
    lib.whisper_init_from_file_with_params.restype = whisper_context_p
    lib.whisper_init_from_file_with_params.argtypes = [ctypes.c_char_p, whisper_context_params]
    
    lib.whisper_context_default_params.restype = whisper_context_params
    lib.whisper_context_default_params.argtypes = []
    
    lib.whisper_full_default_params.restype = whisper_full_params
    lib.whisper_full_default_params.argtypes = [ctypes.c_int]
    
    lib.whisper_full.restype = ctypes.c_int
    lib.whisper_full.argtypes = [whisper_context_p, whisper_full_params, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    
    lib.whisper_full_n_segments.restype = ctypes.c_int
    lib.whisper_full_n_segments.argtypes = [whisper_context_p]
    
    lib.whisper_full_get_segment_text.restype = ctypes.c_char_p
    lib.whisper_full_get_segment_text.argtypes = [whisper_context_p, ctypes.c_int]
    
    lib.whisper_full_get_segment_t0.restype = ctypes.c_int64
    lib.whisper_full_get_segment_t0.argtypes = [whisper_context_p, ctypes.c_int]
    
    lib.whisper_full_get_segment_t1.restype = ctypes.c_int64
    lib.whisper_full_get_segment_t1.argtypes = [whisper_context_p, ctypes.c_int]
    
    lib.whisper_full_n_tokens.restype = ctypes.c_int
    lib.whisper_full_n_tokens.argtypes = [whisper_context_p, ctypes.c_int]
    
    lib.whisper_full_get_token_text.restype = ctypes.c_char_p
    lib.whisper_full_get_token_text.argtypes = [whisper_context_p, ctypes.c_int, ctypes.c_int]
    
    lib.whisper_full_get_token_data.restype = whisper_token_data
    lib.whisper_full_get_token_data.argtypes = [whisper_context_p, ctypes.c_int, ctypes.c_int]
    
    lib.whisper_free.restype = None
    lib.whisper_free.argtypes = [whisper_context_p]
    
    lib.whisper_ctx_init_openvino_encoder.restype = ctypes.c_int
    lib.whisper_ctx_init_openvino_encoder.argtypes = [whisper_context_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

    lib.whisper_full_lang_id.restype = ctypes.c_int
    lib.whisper_full_lang_id.argtypes = [whisper_context_p]

    lib.whisper_lang_str.restype = ctypes.c_char_p
    lib.whisper_lang_str.argtypes = [ctypes.c_int]

    lib.whisper_tokenize.restype = ctypes.c_int
    lib.whisper_tokenize.argtypes = [whisper_context_p, ctypes.c_char_p, ctypes.POINTER(whisper_token), ctypes.c_int]

class WhisperCpp:
    def __init__(self, model_path):
        if not lib:
            raise RuntimeError("libwhisper.so not loaded")
        
        cparams = lib.whisper_context_default_params()
        self.ctx = lib.whisper_init_from_file_with_params(model_path.encode('utf-8'), cparams)
        if not self.ctx:
            raise RuntimeError(f"Failed to initialize whisper context from {model_path}")
            
    def init_openvino(self, model_path=None, device="CPU", cache_dir=None):
        m_path = model_path.encode('utf-8') if model_path else None
        d_device = device.encode('utf-8') if device else b"CPU"
        c_dir = cache_dir.encode('utf-8') if cache_dir else None
        
        res = lib.whisper_ctx_init_openvino_encoder(self.ctx, m_path, d_device, c_dir)
        if res != 0:
            logger.warning(f"whisper_ctx_init_openvino_encoder failed with return code {res}")
            return False
        return True
        
    def full(self, audio, params):
        audio_ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        res = lib.whisper_full(self.ctx, params, audio_ptr, len(audio))
        return res
        
    def get_full_params(self, strategy=0): # 0 = WHISPER_SAMPLING_GREEDY
        return lib.whisper_full_default_params(strategy)
        
    def n_segments(self):
        return lib.whisper_full_n_segments(self.ctx)
        
    def get_segment_text(self, i):
        return lib.whisper_full_get_segment_text(self.ctx, i).decode('utf-8')
        
    def get_segment_t0(self, i):
        return lib.whisper_full_get_segment_t0(self.ctx, i)
        
    def get_segment_t1(self, i):
        return lib.whisper_full_get_segment_t1(self.ctx, i)
        
    def n_tokens(self, i):
        return lib.whisper_full_n_tokens(self.ctx, i)
        
    def get_token_text(self, s, i):
        return lib.whisper_full_get_token_text(self.ctx, s, i).decode('utf-8')
        
    def get_token_data(self, s, i):
        return lib.whisper_full_get_token_data(self.ctx, s, i)

    def full_lang_id(self):
        return lib.whisper_full_lang_id(self.ctx)

    def lang_id_to_str(self, lang_id):
        return lib.whisper_lang_str(lang_id).decode('utf-8')

    def tokenize(self, text, n_max):
        tokens = (whisper_token * n_max)()
        n = lib.whisper_tokenize(self.ctx, text.encode('utf-8'), tokens, n_max)
        if n < 0:
            return []
        return [tokens[i] for i in range(n)]

    def __del__(self):
        if hasattr(self, 'ctx') and self.ctx:
            lib.whisper_free(self.ctx)
            self.ctx = None
