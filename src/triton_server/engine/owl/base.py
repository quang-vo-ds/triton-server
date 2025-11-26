from abc import ABC, abstractmethod

import numpy as np


class BaseOwl(ABC):
    @abstractmethod
    def detect(self, image_arr: np.ndarray, prompts: list[str]) -> np.ndarray:
        pass
