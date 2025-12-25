from abc import ABC, abstractmethod

import numpy as np


class BaseJinaClip(ABC):
    @abstractmethod
    def encode_text(self, texts: np.ndarray) -> np.ndarray:
        """Encode text inputs into embeddings."""
        pass

    @abstractmethod
    def encode_image(self, images: np.ndarray) -> np.ndarray:
        """Encode image inputs into embeddings."""
        pass

