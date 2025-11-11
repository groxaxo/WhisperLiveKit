"""GPU utility functions for optimal compute type selection."""

import logging

logger = logging.getLogger(__name__)


def get_optimal_compute_type(device="auto"):
    """
    Determine the optimal compute type for CTranslate2/faster-whisper based on GPU capabilities.
    
    For NVIDIA Ampere GPUs (compute capability >= 8.0), use int8_bfloat16 for best performance.
    This includes: A100, A30, RTX 3000 series, RTX 4000 series.
    
    Args:
        device: Device specification ("auto", "cuda", "cpu")
    
    Returns:
        str: Optimal compute type ("int8_bfloat16", "int8_float16", or "auto")
    """
    if device == "cpu":
        return "int8"
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.debug("CUDA not available, using CPU compute type")
            return "int8"
        
        # Get GPU compute capability
        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability
        
        logger.info(f"Detected GPU compute capability: {major}.{minor}")
        
        # Ampere (8.x) and newer support bfloat16
        # Ada Lovelace (8.9), Hopper (9.x), Ampere (8.0, 8.6, 8.7)
        if major >= 8:
            logger.info("Ampere or newer GPU detected - using int8_bfloat16 for optimal performance")
            return "int8_bfloat16"
        # Turing and Volta support float16 but not bfloat16
        elif major >= 7:
            logger.info("Turing/Volta GPU detected - using int8_float16")
            return "int8_float16"
        else:
            logger.info(f"Older GPU architecture detected - using auto compute type")
            return "auto"
            
    except ImportError:
        logger.warning("PyTorch not available, cannot detect GPU capabilities")
        return "auto"
    except Exception as e:
        logger.warning(f"Error detecting GPU capabilities: {e}")
        return "auto"


def log_gpu_info():
    """Log information about available GPU for debugging."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.info("No CUDA GPU available")
            return
        
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            compute_cap = torch.cuda.get_device_capability(i)
            logger.info(
                f"GPU {i}: {props.name}, "
                f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}, "
                f"Total Memory: {props.total_memory / 1024**3:.2f} GB"
            )
    except Exception as e:
        logger.debug(f"Could not log GPU info: {e}")
