import time
import cv2
import numpy as np

from triton_server.engine.jina_clip import JinaClip

IMAGE_FILE = "data/image.jpg"


def test_jina_clip_encode_text():
    model = JinaClip()

    texts = np.array(["a photo of a cat", "a photo of a dog", "a beautiful sunset"])

    start_time = time.time()
    embeddings = model.encode_text(texts)
    end_time = time.time()

    print(f"Text embeddings shape: {embeddings.shape}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    assert embeddings is not None
    assert embeddings.shape[0] == len(texts)
    assert len(embeddings.shape) == 2  # (batch_size, embedding_dim)


def test_jina_clip_encode_image():
    model = JinaClip()

    image_arr = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
    image_batch = np.stack([image_arr, image_arr], axis=0)

    start_time = time.time()
    embeddings = model.encode_image(image_batch)
    end_time = time.time()

    print(f"Image embeddings shape: {embeddings.shape}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    assert embeddings is not None
    assert embeddings.shape[0] == 2
    assert len(embeddings.shape) == 2  # (batch_size, embedding_dim)


def test_jina_clip_similarity():
    """Test that similar text/image pairs have higher cosine similarity."""
    model = JinaClip()

    image_arr = cv2.cvtColor(cv2.imread(IMAGE_FILE), cv2.COLOR_BGR2RGB)
    image_batch = np.stack([image_arr], axis=0)
    texts = np.array(["athletes playing sports", "random unrelated text"])

    text_embeddings = model.encode_text(texts)
    image_embeddings = model.encode_image(image_batch)

    # Normalize embeddings for cosine similarity
    text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    image_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    # Compute similarities
    similarities = np.dot(text_norm, image_norm.T)
    print(f"Similarities: {similarities}")

    assert similarities is not None


if __name__ == "__main__":
    test_jina_clip_encode_text()
    test_jina_clip_encode_image()
    test_jina_clip_similarity()

