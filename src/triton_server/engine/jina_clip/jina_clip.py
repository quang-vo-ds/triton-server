import torch
from transformers import AutoModel
from PIL import Image
import numpy as np

from .base import BaseJinaClip
from ...configs import settings


class JinaClip(BaseJinaClip):
    def __init__(
        self,
        model_name: str = settings.jina_clip_settings.MODEL_NAME,
        device: str = settings.jina_clip_settings.DEVICE,
    ):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()

    def encode_text(self, texts: np.ndarray) -> np.ndarray:
        """Encode text inputs into embeddings.

        Args:
            texts: np.ndarray of shape (batch_size,) containing text strings.

        Returns:
            np.ndarray of shape (batch_size, embedding_dim) containing text embeddings.
        """
        # Convert numpy array to list of strings
        texts_list = texts.tolist()

        with torch.no_grad():
            embeddings = self.model.encode_text(texts_list)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def encode_image(self, images: np.ndarray) -> np.ndarray:
        """Encode image inputs into embeddings.

        Args:
            images: np.ndarray of shape (batch_size, height, width, channels).

        Returns:
            np.ndarray of shape (batch_size, embedding_dim) containing image embeddings.
        """
        # Convert numpy arrays to PIL Images
        pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]

        with torch.no_grad():
            embeddings = self.model.encode_image(pil_images)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        return embeddings

