import numpy as np
import cv2
import time
from pytriton.client import ModelClient

IMAGE_FILE = "data/scoreboard.jpg"
SERVER_URL = "http://localhost:8000"
MODEL_NAME = "paddle-ocr"

INIT_TIMEOUT_S = 300.0
INFERENCE_TIMEOUT_S = 300.0


def test_client():
    image_arr = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
    image_batch = np.stack([image_arr, image_arr], axis=0)

    with ModelClient(
        SERVER_URL, MODEL_NAME,
        init_timeout_s=INIT_TIMEOUT_S, inference_timeout_s=INFERENCE_TIMEOUT_S,
    ) as client:
        print("Sending request...")
        start_time = time.time()
        result_dict = client.infer_batch(image_batch)
        end_time = time.time()

        for i, text in enumerate(result_dict["text"]):
            print(f"Image {i}: {text[0].decode()}")

        print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    test_client()

