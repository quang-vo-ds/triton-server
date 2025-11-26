import time
import json
import os
from pathlib import Path

import cv2
import numpy as np

from triton_server.engine.sam import SAMEngine

IMAGE_FILE = "data/image.jpg"
MASKS_FILE = "data/masks.json"
RESULT_FILE = "data/result.jpg"


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()  # Convert scalar numpy types to Python types
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj


def test_sam_model():
    model = SAMEngine(model_name="sam2.1_hiera_tiny", device="cpu")

    image_arr = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)

    image_arrs = np.stack([image_arr, image_arr], axis=0)

    start_time = time.time()
    result = model.predict(image_arrs)
    end_time = time.time()

    # Save result
    model.save_image(result[1], RESULT_FILE)

    print(f"Time taken: {end_time - start_time} seconds")
    assert result is not None


if __name__ == "__main__":
    test_sam_model()
