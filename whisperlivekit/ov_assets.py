"""Utilities for locating bundled OpenVINO Whisper assets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

# Layout: project_root/whisper-large-v3-int8-ov/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "whisper-large-v3-int8-ov"

_ENCODER_NAME = "openvino_encoder_model.xml"
_DECODER_NAME = "openvino_decoder_model.xml"
_TOKENIZER_NAME = "openvino_tokenizer.xml"
_DETOKENIZER_NAME = "openvino_detokenizer.xml"


def get_model_dir() -> Optional[Path]:
    """Return the default directory containing the bundled OpenVINO model files."""
    if _DEFAULT_MODEL_DIR.exists():
        return _DEFAULT_MODEL_DIR
    return None


def _existing(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def get_encoder_xml() -> Optional[Path]:
    model_dir = get_model_dir()
    if not model_dir:
        return None
    return _existing(model_dir / _ENCODER_NAME)


def get_decoder_xml() -> Optional[Path]:
    model_dir = get_model_dir()
    if not model_dir:
        return None
    return _existing(model_dir / _DECODER_NAME)


def get_tokenizer_assets() -> Dict[str, Path]:
    """Return available tokenizer/detokenizer XMLs bundled with the project."""
    model_dir = get_model_dir()
    if not model_dir:
        return {}

    assets = {}
    tokenizer_path = model_dir / _TOKENIZER_NAME
    detokenizer_path = model_dir / _DETOKENIZER_NAME
    if tokenizer_path.exists():
        assets["tokenizer_xml"] = tokenizer_path
    if detokenizer_path.exists():
        assets["detokenizer_xml"] = detokenizer_path
    return assets
