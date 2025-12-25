from .whisper import WhisperTorch, WhisperONNX
from .owl import Owlv2, OwlViT
from .paddle import PaddleOCR
from .jina_clip import JinaClip

__all__ = ["WhisperTorch", "WhisperONNX", "Owlv2", "OwlViT", "PaddleOCR", "JinaClip"]
