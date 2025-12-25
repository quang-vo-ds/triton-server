import os
import cv2
import numpy as np


def _is_empty_detection(x1, y1, x2, y2, label, score):
    """Check if detection is an empty placeholder."""
    return x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0 and label == "" and score == 0


def draw_detections(image_path: str, detections: list, dest_path: str):
    """Draw detection bounding boxes on image and save to destination.

    Args:
        image_path: Path to input image
        detections: List of [x1, y1, x2, y2, label, score] lists
        dest_path: Path to save result image
    """
    img = cv2.imread(image_path)
    for x1, y1, x2, y2, label, score in detections:
        if _is_empty_detection(x1, y1, x2, y2, label, score):
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label}: {score:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(dest_path, img)


def crop_detections(image_path: str, detections: list, dest_dir: str):
    """Crop each detection from image and save to destination directory.

    Args:
        image_path: Path to input image
        detections: List of [x1, y1, x2, y2, label, score] lists
        dest_dir: Directory to save cropped images
    """
    os.makedirs(dest_dir, exist_ok=True)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    for i, (x1, y1, x2, y2, label, score) in enumerate(detections):
        if _is_empty_detection(x1, y1, x2, y2, label, score):
            continue
        # Clamp coordinates to image bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(dest_dir, f"{i}_{label}_{score:.2f}.jpg"), crop)
