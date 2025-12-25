import numpy as np
import cv2
import time
from pytriton.client import ModelClient

IMAGE_FILE = "data/image.jpg"
SERVER_URL = "http://localhost:8000"
MODEL_NAME = "jina-clip-v1"

INIT_TIMEOUT_S = 300.0
INFERENCE_TIMEOUT_S = 300.0


def test_client_text():
    """Test text encoding via Triton client."""
    texts = np.array([[b"a photo of a cat"], [b"a photo of a dog"]])

    with ModelClient(
        SERVER_URL, f"{MODEL_NAME}-text",
        init_timeout_s=INIT_TIMEOUT_S, inference_timeout_s=INFERENCE_TIMEOUT_S,
    ) as client:
        print("Sending text request...")
        start_time = time.time()
        result_dict = client.infer_batch(texts)
        end_time = time.time()

        for output_name, output_data in result_dict.items():
            print(f"{output_name}: shape={output_data.shape}")

        print(f"Time taken: {end_time - start_time:.2f} seconds")


def test_client_image():
    """Test image encoding via Triton client."""
    image_arr = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
    image_batch = np.stack([image_arr, image_arr], axis=0)

    with ModelClient(
        SERVER_URL, f"{MODEL_NAME}-image",
        init_timeout_s=INIT_TIMEOUT_S, inference_timeout_s=INFERENCE_TIMEOUT_S,
    ) as client:
        print("Sending image request...")
        start_time = time.time()
        result_dict = client.infer_batch(image_batch)
        end_time = time.time()

        for output_name, output_data in result_dict.items():
            print(f"{output_name}: shape={output_data.shape}")

        print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    test_client_text()
    test_client_image()

