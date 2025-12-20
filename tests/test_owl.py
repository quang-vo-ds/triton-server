import time
import cv2
import numpy as np

from triton_server.engine.owl import Owlv2, draw_detections, crop_detections

IMAGE_FILE = "data/image.jpg"
RESULT_FILE = "data/result.jpg"
CROP_DIR = "data/crops"


def test_owlv2_model():
    model = Owlv2()

    image_arr = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
    prompt = ["scoreboard", "athletes"]

    image_arrs = np.stack([image_arr, image_arr], axis=0)
    prompts = np.array([prompt, prompt])

    start_time = time.time()
    result = model.detect(image_arrs, prompts=prompts, max_detections=10)
    end_time = time.time()

    print(result)
    print(f"Time taken: {end_time - start_time} seconds")

    draw_detections(IMAGE_FILE, result[0], RESULT_FILE)
    crop_detections(IMAGE_FILE, result[0], CROP_DIR)

    assert result is not None


if __name__ == "__main__":
    test_owlv2_model()
