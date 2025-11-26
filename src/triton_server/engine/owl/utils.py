import cv2
import numpy as np


def draw_detections(image_path: str, detections: list, dest_path: str):
    """Draw detection bounding boxes on image and save to destination.

    Args:
        image_path: Path to input image
        detections: List of (x1, y1, x2, y2, score) tuples
        dest_path: Path to save result image
    """
    img = cv2.imread(image_path)
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{score:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(dest_path, img)
