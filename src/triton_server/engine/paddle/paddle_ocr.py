from pathlib import Path

import numpy as np
from paddleocr import PaddleOCR as _PaddleOCR

from ...configs import settings

# Resolve absolute paths for models
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class PaddleOCR:
    def __init__(
        self,
        det_model_dir: str = settings.paddle_settings.DET_MODEL_NAME,
        rec_model_dir: str = settings.paddle_settings.REC_MODEL_NAME,
        device: str = settings.paddle_settings.DEVICE,
    ):
        self.model = _PaddleOCR(
            det_model_dir=str(_PROJECT_ROOT / det_model_dir),
            rec_model_dir=str(_PROJECT_ROOT / rec_model_dir),
            use_angle_cls=False,
            use_gpu=(device == "cuda"),
            lang="en",
        )

    def recognize(self, image_arr: np.ndarray) -> np.ndarray:
        """Run OCR on batch of images.

        Args:
            image_arr: Image array with shape (batch, H, W, C) or (H, W, C)

        Returns:
            np.ndarray of shape (batch, 1) with concatenated text per image
        """
        if len(image_arr.shape) == 3:
            image_arr = np.expand_dims(image_arr, axis=0)

        batch_results = []
        for img in image_arr:
            result = self.model.ocr(img, cls=False)
            dets = []
            if result and result[0]:
                for line in result[0]:
                    box, (text, _) = line
                    y1, x1 = box[0][1], box[0][0]
                    dets.append((y1, x1, text))
            # Sort by y1 (top to bottom), then x1 (left to right)
            dets.sort(key=lambda d: (d[0], d[1]))
            text = " ".join(d[2] for d in dets) if dets else ""
            batch_results.append([text])

        return np.array(batch_results)

