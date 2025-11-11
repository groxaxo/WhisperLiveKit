import sys
import numpy as np
import logging
from typing import List, Tuple, Optional
import logging
from whisperlivekit.timed_objects import ASRToken, Transcript
from whisperlivekit.warmup import load_file
from whisperlivekit.simul_whisper.license_simulstreaming import SIMULSTREAMING_LICENSE
from .whisper import load_model, tokenizer
from .whisper.audio import TOKENS_PER_SECOND

import os
import gc
logger = logging.getLogger(__name__)

import torch
from whisperlivekit.simul_whisper.config import AlignAttConfig
from whisperlivekit.simul_whisper.simul_whisper import PaddedAlignAttWhisper
from whisperlivekit.simul_whisper.whisper import tokenizer

try:
    from .mlx_encoder import mlx_model_mapping, load_mlx_encoder
    HAS_MLX_WHISPER = True
except ImportError:
    HAS_MLX_WHISPER = False
if HAS_MLX_WHISPER:
    HAS_FASTER_WHISPER = False
else:
    try:
        from faster_whisper import WhisperModel
        HAS_FASTER_WHISPER = True
    except ImportError:
        HAS_FASTER_WHISPER = False


# TOO_MANY_REPETITIONS = 3

class SimulStreamingOnlineProcessor:
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        logfile=sys.stderr,
        warmup_file=None
    ):        
        self.asr = asr
        self.logfile = logfile
        self.end = 0.0
        self.global_time_offset = 0.0
        
        self.committed: List[ASRToken] = []
        self.last_result_tokens: List[ASRToken] = []
        self.load_new_backend()
        
        #can be moved
        if asr.tokenizer:
            self.model.tokenizer = asr.tokenizer

    def load_new_backend(self):
        model = self.asr.get_new_model_instance()
        self.model = PaddedAlignAttWhisper(
            cfg=self.asr.cfg,
            loaded_model=model,
            mlx_encoder=self.asr.mlx_encoder,
            fw_encoder=self.asr.fw_encoder,
            )

    def insert_silence(self, silence_duration, offset):
        """
        If silences are > 5s, we do a complete context clear. Otherwise, we just insert a small silence and shift the last_attend_frame
        """
        if silence_duration < 5:
            gap_silence = torch.zeros(int(16000*silence_duration))
            self.model.insert_audio(gap_silence)
            # self.global_time_offset += silence_duration
        else:
            self.process_iter(is_last=True) #we want to totally process what remains in the buffer.
            self.model.refresh_segment(complete=True)
            self.global_time_offset += silence_duration + offset


        
    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time):
        """Append an audio chunk to be processed by SimulStreaming."""
            
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        self.end = audio_stream_end_time #Only to be aligned with what happens in whisperstreaming backend.
        self.model.insert_audio(audio_tensor)

    def get_buffer(self):
        return Transcript(
            start=None, 
            end=None, 
            text='', 
            probability=None
        )

    def timestamped_text(self, tokens, generation):
        """
        generate timestamped text from tokens and generation data.
        
        args:
            tokens: List of tokens to process
            generation: Dictionary containing generation progress and optionally results
            
        returns:
            List of tuples containing (start_time, end_time, word) for each word
        """
        FRAME_DURATION = 0.02    
        if "result" in generation:
            split_words = generation["result"]["split_words"]
            split_tokens = generation["result"]["split_tokens"]
        else:
            split_words, split_tokens = self.model.tokenizer.split_to_word_tokens(tokens)
        progress = generation["progress"]
        frames = [p["most_attended_frames"][0] for p in progress]
        absolute_timestamps = [p["absolute_timestamps"][0] for p in progress]
        tokens_queue = tokens.copy()
        timestamped_words = []
        
        for word, word_tokens in zip(split_words, split_tokens):
            # start_frame = None
            # end_frame = None
            for expected_token in word_tokens:
                if not tokens_queue or not frames:
                    raise ValueError(f"Insufficient tokens or frames for word '{word}'")
                    
                actual_token = tokens_queue.pop(0)
                current_frame = frames.pop(0)
                current_timestamp = absolute_timestamps.pop(0)
                if actual_token != expected_token:
                    raise ValueError(
                        f"Token mismatch: expected '{expected_token}', "
                        f"got '{actual_token}' at frame {current_frame}"
                    )
                # if start_frame is None:
                #     start_frame = current_frame
                # end_frame = current_frame
            # start_time = start_frame * FRAME_DURATION
            # end_time = end_frame * FRAME_DURATION
            start_time = current_timestamp
            end_time = current_timestamp + 0.1
            timestamp_entry = (start_time, end_time, word)
            timestamped_words.append(timestamp_entry)
            logger.debug(f"TS-WORD:\t{start_time:.2f}\t{end_time:.2f}\t{word}")
        return timestamped_words

    def process_iter(self, is_last=False) -> Tuple[List[ASRToken], float]:
        """
        Process accumulated audio chunks using SimulStreaming.
        
        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        try:
            tokens, generation_progress = self.model.infer(is_last=is_last)
            ts_words = self.timestamped_text(tokens, generation_progress)
            
            new_tokens = []
            for ts_word in ts_words:
                
                start, end, word = ts_word
                token = ASRToken(
                    start=start,
                    end=end,
                    text=word,
                    probability=0.95  # fake prob. Maybe we can extract it from the model?
                ).with_offset(
                    self.global_time_offset
                )
                new_tokens.append(token)
                
            # identical_tokens = 0
            # n_new_tokens = len(new_tokens)
            # if n_new_tokens:
            
            self.committed.extend(new_tokens)
            
            # if token in self.committed:
            #     pos = len(self.committed) - 1 - self.committed[::-1].index(token)
            # if pos:
            #     for i in range(len(self.committed) - n_new_tokens, -1, -n_new_tokens):
            #         commited_segment = self.committed[i:i+n_new_tokens]
            #         if commited_segment == new_tokens:
            #             identical_segments +=1
            #             if identical_tokens >= TOO_MANY_REPETITIONS:
            #                 logger.warning('Too many repetition, model is stuck. Load a new one')
            #                 self.committed = self.committed[:i]
            #                 self.load_new_backend()
            #                 return [], self.end

            # pos = self.committed.rindex(token)

            
            
            return new_tokens, self.end

            
        except Exception as e:
            logger.exception(f"SimulStreaming processing error: {e}")
            return [], self.end

    def warmup(self, audio, init_prompt=""):
        """Warmup the SimulStreaming model."""
        try:
            self.model.insert_audio(audio)
            self.model.infer(True)
            self.model.refresh_segment(complete=True)
            logger.info("SimulStreaming model warmed up successfully")
        except Exception as e:
            logger.exception(f"SimulStreaming warmup failed: {e}")

    def __del__(self):
        # free the model and add a new model to stack.
        # del self.model
        gc.collect()
        torch.cuda.empty_cache()
        # self.asr.new_model_to_stack()
        self.model.remove_hooks()

class SimulStreamingASR():
    """SimulStreaming backend with AlignAtt policy."""
    sep = ""

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr, **kwargs):
        logger.warning(SIMULSTREAMING_LICENSE)
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = lan
        
        self.model_path = kwargs.get('model_path', './large-v3.pt')
        self.frame_threshold = kwargs.get('frame_threshold', 25)
        self.audio_max_len = kwargs.get('audio_max_len', 20.0)
        self.audio_min_len = kwargs.get('audio_min_len', 0.0)
        self.segment_length = kwargs.get('segment_length', 0.5)
        self.beams = kwargs.get('beams', 1)
        self.decoder_type = kwargs.get('decoder_type', 'greedy' if self.beams == 1 else 'beam')
        self.task = kwargs.get('task', 'transcribe')
        self.cif_ckpt_path = kwargs.get('cif_ckpt_path', None)
        self.never_fire = kwargs.get('never_fire', False)
        self.init_prompt = kwargs.get('init_prompt', None)
        self.static_init_prompt = kwargs.get('static_init_prompt', None)
        self.max_context_tokens = kwargs.get('max_context_tokens', None)
        self.warmup_file = kwargs.get('warmup_file', None)
        self.preload_model_count = kwargs.get('preload_model_count', 1)
        self.compute_type = kwargs.get('compute_type', None)
        
        if model_dir is not None:
            self.model_path = model_dir
        elif modelsize is not None:
            model_mapping = {
                'tiny': './tiny.pt',
                'base': './base.pt',
                'small': './small.pt',
                'medium': './medium.pt',
                'medium.en': './medium.en.pt',
                'large-v1': './large-v1.pt',
                'base.en': './base.en.pt',
                'small.en': './small.en.pt',
                'tiny.en': './tiny.en.pt',
                'large-v2': './large-v2.pt',
                'large-v3': './large-v3.pt',
                'large': './large-v3.pt'
            }
            self.model_path = model_mapping.get(modelsize, f'./{modelsize}.pt')
        
        self.cfg = AlignAttConfig(
                model_path=self.model_path,
                segment_length=self.segment_length,
                frame_threshold=self.frame_threshold,
                language=self.original_language,
                audio_max_len=self.audio_max_len,
                audio_min_len=self.audio_min_len,
                cif_ckpt_path=self.cif_ckpt_path,
                decoder_type="beam",
                beam_size=self.beams,
                task=self.task,
                never_fire=self.never_fire,
                init_prompt=self.init_prompt,
                max_context_tokens=self.max_context_tokens,
                static_init_prompt=self.static_init_prompt,
        )  
        
        # Set up tokenizer for translation if needed
        if self.task == "translate":
            self.tokenizer = self.set_translate_task()
        else:
            self.tokenizer = None
        
        self.model_name = os.path.basename(self.cfg.model_path).replace(".pt", "")
        self.model_path = os.path.dirname(os.path.abspath(self.cfg.model_path))
        self.models = [self.load_model() for i in range(self.preload_model_count)]
    
        self.mlx_encoder, self.fw_encoder = None, None
        if HAS_MLX_WHISPER:
            print('Simulstreaming will use MLX whisper for a faster encoder.')
            mlx_model_name = mlx_model_mapping[self.model_name]
            self.mlx_encoder = load_mlx_encoder(path_or_hf_repo=mlx_model_name)
        elif HAS_FASTER_WHISPER:
            from whisperlivekit.gpu_utils import get_optimal_compute_type
            # Use override if provided, otherwise auto-detect
            if self.compute_type is not None:
                optimal_compute_type = self.compute_type
                print(f'Simulstreaming will use Faster Whisper for the encoder with manually specified compute_type={optimal_compute_type}.')
            else:
                optimal_compute_type = get_optimal_compute_type('auto')
                print(f'Simulstreaming will use Faster Whisper for the encoder with auto-detected compute_type={optimal_compute_type}.')
            self.fw_encoder = WhisperModel(
                self.model_name,
                device='auto',
                compute_type=optimal_compute_type,
            )

    def load_model(self):
        whisper_model = load_model(name=self.model_name, download_root=self.model_path)
        warmup_audio = load_file(self.warmup_file)
        whisper_model.transcribe(warmup_audio, language=self.original_language if self.original_language != 'auto' else None)
        return whisper_model
    
    def get_new_model_instance(self):
        """
        SimulStreaming cannot share the same backend because it uses global forward hooks on the attention layers.
        Therefore, each user requires a separate model instance, which can be memory-intensive. To maintain speed, we preload the models into memory.
        """
        if len(self.models) == 0:
            self.models.append(self.load_model())
        new_model = self.models.pop()
        return new_model
        # self.models[0]

    def new_model_to_stack(self):
        self.models.append(self.load_model())
        

    def set_translate_task(self):
        """Set up translation task."""
        if self.cfg.language == 'auto':
            raise Exception('Translation cannot be done with language = auto')
        return tokenizer.get_tokenizer(
            multilingual=True,
            language=self.cfg.language,
            num_languages=99,
            task="translate"
        )

    def transcribe(self, audio):
        """
        Warmup is done directly in load_model
        """
        pass