"""
OpenAI-compatible API endpoint for Open WebUI integration.

This module provides an OpenAI Whisper API-compatible endpoint that allows
WhisperLiveKit to be used as a drop-in replacement for OpenAI's transcription
service in Open WebUI and other compatible applications.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from whisperlivekit.core import TranscriptionEngine

logger = logging.getLogger(__name__)


def create_openai_routes(app: FastAPI, transcription_engine: TranscriptionEngine):
    """
    Add OpenAI-compatible API routes to the FastAPI app.
    
    Args:
        app: FastAPI application instance
        transcription_engine: Initialized TranscriptionEngine instance
    """
    
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(...),
        response_format: str = Form("json"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        temperature: Optional[float] = Form(0.0),
    ):
        """
        OpenAI Whisper API-compatible transcription endpoint.
        
        This endpoint accepts audio files and returns transcriptions in various formats,
        compatible with Open WebUI's STT integration.
        
        Args:
            file: Audio file (supports mp3, mp4, m4a, opus, webm, etc.)
            model: Model name (e.g., "whisper-1", ignored as we use configured model)
            response_format: Output format: "json", "text", "srt", "vtt", "verbose_json"
            language: Optional language code (e.g., "en", "fr")
            prompt: Optional prompt to guide transcription
            temperature: Sampling temperature (0.0 - 1.0)
        
        Returns:
            Transcription in requested format
        """
        return await _process_transcription(
            transcription_engine, file, model, response_format, 
            language, prompt, temperature
        )


def create_openai_routes_deferred(app: FastAPI, engine_getter: callable):
    """
    Add OpenAI-compatible API routes with deferred engine access.
    
    This version allows registering routes before the engine is initialized.
    
    Args:
        app: FastAPI application instance
        engine_getter: Callable that returns the TranscriptionEngine instance
    """
    
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(...),
        response_format: str = Form("json"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        temperature: Optional[float] = Form(0.0),
    ):
        """
        OpenAI Whisper API-compatible transcription endpoint.
        
        This endpoint accepts audio files and returns transcriptions in various formats,
        compatible with Open WebUI's STT integration.
        
        Args:
            file: Audio file (supports mp3, mp4, m4a, opus, webm, etc.)
            model: Model name (e.g., "whisper-1", ignored as we use configured model)
            response_format: Output format: "json", "text", "srt", "vtt", "verbose_json"
            language: Optional language code (e.g., "en", "fr")
            prompt: Optional prompt to guide transcription
            temperature: Sampling temperature (0.0 - 1.0)
        
        Returns:
            Transcription in requested format
        """
        engine = engine_getter()
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Transcription engine not yet initialized"
            )
        return await _process_transcription(
            engine, file, model, response_format, 
            language, prompt, temperature
        )


async def _process_transcription(
    transcription_engine: TranscriptionEngine,
    file: UploadFile,
    model: str,
    response_format: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: Optional[float]
):
    """
    Process the transcription request.
    
    Internal helper function used by both route registration methods.
    """
    logger.info(f"Received transcription request: model={model}, format={response_format}, language={language}")
    
    # Validate response format
    valid_formats = ["json", "text", "srt", "vtt", "verbose_json"]
    if response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {', '.join(valid_formats)}"
        )
    
    # Save uploaded file to temporary location
    temp_file = None
    try:
        # Create temporary file for the upload
        suffix = Path(file.filename).suffix if file.filename else ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp_file = temp.name
            content = await file.read()
            temp.write(content)
        
        logger.info(f"Saved uploaded file to {temp_file}, size: {len(content)} bytes")
        
        # Load and process the audio file
        audio_data, sample_rate = sf.read(temp_file)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except ImportError:
                logger.warning("librosa not available, using original sample rate")
        
        logger.info(f"Loaded audio: duration={len(audio_data)/sample_rate:.2f}s, sample_rate={sample_rate}")
        
        # Perform transcription using simpler batch approach
        transcription_result = await transcribe_audio_simple(
            transcription_engine,
            audio_data,
            sample_rate,
            language=language
        )
        
        # Format response based on requested format
        if response_format == "text":
            return PlainTextResponse(content=transcription_result["text"])
        
        elif response_format == "json":
            return JSONResponse(content={
                "text": transcription_result["text"]
            })
        
        elif response_format == "verbose_json":
            return JSONResponse(content={
                "text": transcription_result["text"],
                "language": transcription_result.get("language"),
                "duration": transcription_result.get("duration"),
            })
        
        elif response_format == "srt":
            # For SRT, we'll provide a simple single-segment format
            srt_content = format_as_srt(transcription_result["text"], transcription_result.get("duration", 0))
            return PlainTextResponse(content=srt_content, media_type="application/x-subrip")
        
        elif response_format == "vtt":
            # For VTT, we'll provide a simple single-segment format
            vtt_content = format_as_vtt(transcription_result["text"], transcription_result.get("duration", 0))
            return PlainTextResponse(content=vtt_content, media_type="text/vtt")
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file and Path(temp_file).exists():
            try:
                Path(temp_file).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")


async def transcribe_audio_simple(
    engine: TranscriptionEngine,
    audio_data: np.ndarray,
    sample_rate: int,
    language: Optional[str] = None
) -> dict:
    """
    Transcribe audio data using the transcription engine's underlying Whisper backend.
    
    This is a simplified batch transcription approach that bypasses the streaming
    components and directly uses the Whisper model for file transcription.
    
    Args:
        engine: TranscriptionEngine instance
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        language: Optional language code
    
    Returns:
        Dictionary with transcription results
    """
    try:
        duration = len(audio_data) / sample_rate
        
        # Use the underlying Whisper model for batch transcription
        if hasattr(engine.asr, 'model'):
            # Access the underlying Whisper model
            model = engine.asr.model
            
            # Use the model's transcribe method directly
            if hasattr(model, 'transcribe'):
                result = model.transcribe(
                    audio_data,
                    language=language if language else engine.args.lan if engine.args.lan != "auto" else None
                )
                
                return {
                    "text": result.get("text", "").strip(),
                    "language": result.get("language", language),
                    "duration": duration
                }
        
        # Fallback: use faster-whisper if available
        try:
            from faster_whisper import WhisperModel
            
            # Try to create a temporary model instance
            model_size = getattr(engine.args, 'model_size', 'base')
            temp_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            
            segments, info = temp_model.transcribe(
                audio_data,
                language=language if language else None
            )
            
            # Collect all segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            return {
                "text": " ".join(text_parts).strip(),
                "language": info.language if hasattr(info, 'language') else language,
                "duration": duration
            }
        except ImportError:
            logger.warning("faster-whisper not available")
        
        # Final fallback: return empty transcription
        logger.warning("No suitable transcription method found, returning empty result")
        return {
            "text": "",
            "language": language,
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"Error in batch transcription: {str(e)}", exc_info=True)
        return {
            "text": "",
            "language": language,
            "duration": len(audio_data) / sample_rate if sample_rate > 0 else 0
        }


def format_as_srt(text: str, duration: float) -> str:
    """
    Format text as SRT subtitles (simple single segment).
    
    Args:
        text: Transcription text
        duration: Audio duration in seconds
    
    Returns:
        SRT formatted string
    """
    if not text:
        return ""
    
    start_time = "00:00:00,000"
    end_time = format_timestamp_srt(duration)
    
    return f"1\n{start_time} --> {end_time}\n{text}\n"


def format_as_vtt(text: str, duration: float) -> str:
    """
    Format text as WebVTT subtitles (simple single segment).
    
    Args:
        text: Transcription text
        duration: Audio duration in seconds
    
    Returns:
        WebVTT formatted string
    """
    if not text:
        return "WEBVTT\n\n"
    
    start_time = "00:00:00.000"
    end_time = format_timestamp_vtt(duration)
    
    return f"WEBVTT\n\n{start_time} --> {end_time}\n{text}\n"


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as WebVTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
