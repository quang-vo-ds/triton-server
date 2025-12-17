import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import numpy as np

from .base import BaseOwl
from ...configs import settings


class Owlv2(BaseOwl):
    def __init__(self, model_name: str = settings.owl_settings.MODEL_NAME, device: str = settings.owl_settings.DEVICE):
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)

    def detect(
        self, image_arr: np.ndarray, prompts: list[str], score_thresh: float = 0.3
    ) -> np.ndarray:
        # Handle single image case by adding batch dimension
        if len(image_arr.shape) == 3:
            image_arr = np.expand_dims(image_arr, axis=0)

        text_labels = [prompts] * image_arr.shape[0]

        inputs = self.processor(
            text=text_labels, images=image_arr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [(image.shape[0], image.shape[1]) for image in image_arr]

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_thresh,
            text_labels=text_labels,
        )

        batch_dets = []
        for res in results:
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()

            dets = []
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                if s >= score_thresh:
                    dets.append((float(x1), float(y1), float(x2), float(y2), float(s)))

            batch_dets.append(dets)

        return batch_dets
