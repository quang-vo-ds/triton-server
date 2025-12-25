import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np

from .base import BaseOwl
from ...configs import settings


class OwlViT(BaseOwl):
    def __init__(
        self,
        model_name: str = settings.owl_settings.MODEL_NAME,
        device: str = settings.owl_settings.DEVICE,
    ):
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)

    def detect(
        self,
        image_arr: np.ndarray,
        prompts: np.ndarray,
        score_thresh: float = 0.001,
        max_detections: int | None = None,
    ) -> np.ndarray:
        # Handle single image case by adding batch dimension
        if len(image_arr.shape) == 3:
            image_arr = np.expand_dims(image_arr, axis=0)
        if len(prompts.shape) == 1:
            prompts = np.expand_dims(prompts, axis=0)

        assert (
            prompts.shape[0] == image_arr.shape[0]
        ), f"Prompts batch size ({prompts.shape[0]}) must match image batch size ({image_arr.shape[0]})"

        prompts_list = prompts.tolist()

        inputs = self.processor(
            text=prompts_list, images=image_arr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [(image.shape[0], image.shape[1]) for image in image_arr]
        )
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_thresh,
        )

        batch_dets = []
        for idx, res in enumerate(results):
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()
            image_prompts = prompts_list[idx]

            dets = []
            for (x1, y1, x2, y2), label_idx, score in zip(boxes, labels, scores):
                if score >= score_thresh:
                    label = image_prompts[label_idx]
                    dets.append(
                        [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                            label,
                            float(score),
                        ]
                    )

            batch_dets.append(dets)

        # Determine max_detections: use min of provided value and actual max across batch
        actual_max = max(len(dets) for dets in batch_dets) if batch_dets else 0
        max_detections = (
            min(max_detections, actual_max)
            if max_detections is not None
            else actual_max
        )
        max_detections = max(
            max_detections, 1
        )  # Ensure at least 1 slot for proper array shape

        # Pad/truncate detections to max_detections
        empty_det = [0.0, 0.0, 0.0, 0.0, "", 0.0]
        batch_dets = [
            (dets + [empty_det] * max_detections)[:max_detections]
            for dets in batch_dets
        ]

        return np.array(batch_dets, dtype=object)
