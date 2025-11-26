import numpy as np
import cv2
import time
from pytriton.client import ModelClient

IMAGE_FILE = "data/image.jpg"
SERVER_URL = "http://localhost:8000"
MODEL_NAME = "sam2.1_hiera_tiny"
MODEL_VERSION = "1"

INIT_TIMEOUT_S = 300.0
INFERENCE_TIMEOUT_S = 300.0


def test_client():
    image_arr = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)

    batch_1 = np.stack([image_arr, image_arr], axis=0)
    batch_2 = np.stack([image_arr, image_arr], axis=0)

    with ModelClient(
        SERVER_URL,
        MODEL_NAME,
        init_timeout_s=INIT_TIMEOUT_S,
        inference_timeout_s=INFERENCE_TIMEOUT_S,
    ) as client:
        # Send first request
        print("Sending request (1).")
        result_dict_1 = client.infer_batch(batch_1)

        # Wait 2 seconds immediately after sending first request
        print("Waiting 2 seconds...")
        time.sleep(2)

        # Send second request
        print("Sending request (2).")
        result_dict_2 = client.infer_batch(batch_2)

        # Process first result
        for output_name, output_data in result_dict_1.items():
            print(f"{output_name}: {output_data} for request (1).")

        # Process second result
        for output_name, output_data in result_dict_2.items():
            print(f"{output_name}: {output_data} for request (2).")


if __name__ == "__main__":
    test_client()
