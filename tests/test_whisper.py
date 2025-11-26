import time
import json

import librosa
import numpy as np

from triton_server.engine.whisper import WhisperTorch, WhisperONNX

AUDIO_FILE = "data/audio.wav"


def test_whisper_model():
    # Test with timestamps enabled (default)
    model = WhisperONNX(return_timestamps=True)

    audio_arr, sampling_rate = librosa.load(AUDIO_FILE, sr=16000)

    print(f"Audio dtype: {audio_arr.dtype}")
    print(f"Audio shape: {audio_arr.shape}")
    print(f"Sampling rate: {sampling_rate}")

    # Test with batch of audio arrays
    audio_arrs = np.stack([audio_arr, audio_arr], axis=0)
    start_time = time.time()
    result = model.transcribe(audio_arrs)
    end_time = time.time()
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")

    all_chunks = []
    for batch_idx, r in enumerate(result):
        print(f"\n=== Batch {batch_idx} ===")
        print(f"Number of segments: {len(r)}")
        for seg_idx, transcript in enumerate(r):
            print(
                f"  [{seg_idx}] {transcript.start_time}s - {transcript.end_time}s: {transcript.text}"
            )

        r_json = json.dumps([chunk.model_dump() for chunk in r])
        all_chunks.append(r_json)

    print("\nEncoded output:")
    print(np.char.encode(all_chunks, "utf-8"))

    assert result is not None


if __name__ == "__main__":
    test_whisper_model()
