from abc import ABC, abstractmethod

import numpy as np

from .transcript import Transcript


class BaseWhisper(ABC):
    @abstractmethod
    def transcribe(self, audio_arr: np.ndarray) -> list[list[Transcript]]:
        pass
