import time
import cv2
import numpy as np

from triton_server.engine.paddle import PaddleOCR

IMAGE_FILE = "data/scoreboard.jpg"


def test_paddle_model():
    model = PaddleOCR()

    image_arr = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
    image_arrs = np.stack([image_arr, image_arr], axis=0)

    start_time = time.time()
    result = model.recognize(image_arrs)
    end_time = time.time()

    for i, text in enumerate(result):
        print(f"Image {i}: {text}")

    print(f"Time taken: {end_time - start_time} seconds")
    assert result is not None


if __name__ == "__main__":
    test_paddle_model()

