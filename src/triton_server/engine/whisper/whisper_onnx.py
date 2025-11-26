import os
import numpy as np
import onnxruntime as ort
from transformers import WhisperProcessor, WhisperFeatureExtractor, GenerationConfig

from .base import BaseWhisper
from .transcript import Transcript
from ...configs import settings


class WhisperONNX(BaseWhisper):
    def __init__(
        self,
        model_name: str = settings.whisper_settings.MODEL_NAME,
        device: str = settings.whisper_settings.DEVICE,
        language: str = "english",
        task: str = "transcribe",
        return_timestamps: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.return_timestamps = return_timestamps

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        # Get prompt tokens that control the decoder's behavior
        self.gen_config = GenerationConfig.from_pretrained(model_name)
        self.prompt_ids = self.tokenizer.get_decoder_prompt_ids(
            language=language, task=task, no_timestamps=not return_timestamps
        )

        # Load ONNX Runtime sessions for encoder and decoder models
        self._load_onnx_sessions()

    def transcribe(self, audio_arr: np.ndarray) -> list[list[Transcript]]:
        # Extract mel-spectrogram features from audio: Audio (160000 samples @ 16kHz = 10s) -> Mel-spec (80, 3000)
        feats = self._extract_features(audio_arr)

        # Encode audio features into hidden states: single forward pass
        encoder_out = self.encoder.run(
            None, {self.encoder.get_inputs()[0].name: feats}
        )[0]
        batch_size = encoder_out.shape[0]

        # Build initial prompt tokens to condition the decoder: [<|startoftranscript|>, <|en|>, <|transcribe|>]
        prompt_tokens = [self.gen_config.decoder_start_token_id] + [
            tok for _, tok in self.prompt_ids
        ]
        sequences = [prompt_tokens.copy() for _ in range(batch_size)]

        # Process prompt tokens through decoder to initialize KV-cache
        decoder_cache = encoder_cache = None
        for i, tok in enumerate(prompt_tokens):
            # Create batch of current token: [tok, tok, ..., tok] (batch_size times)
            current_token = np.full(batch_size, tok, dtype=np.int64)

            if i == 0:
                # First token uses decoder_init (no past KV cache)
                next_token, decoder_cache, encoder_cache = self._init_decoder(
                    encoder_out, current_token
                )
            else:
                # Subsequent tokens use decoder_with_past (reuses KV cache)
                next_token, decoder_cache = self._step_decoder(
                    current_token, decoder_cache, encoder_cache
                )

        # Generate transcription tokens autoregressively
        finished = np.zeros(batch_size, dtype=bool)
        for _ in range(200 - len(prompt_tokens)):
            if finished.all():
                break

            # Predict next token using cached attention states
            next_token, decoder_cache = self._step_decoder(
                next_token, decoder_cache, encoder_cache
            )

            # Append predicted token to each batch's sequence
            for i in range(batch_size):
                if not finished[i]:
                    sequences[i].append(int(next_token[i]))
                    # Stop when end-of-sequence token is generated
                    if next_token[i] == self.tokenizer.eos_token_id:
                        finished[i] = True

        # Decode token sequences into text transcripts
        if self.return_timestamps:
            # Parse timestamp tokens and segment text by time
            return [self._parse_timestamps(seq) for seq in sequences]
        else:
            return [
                [
                    Transcript(
                        text=self.tokenizer.decode(
                            seq, skip_special_tokens=True
                        ).strip()
                    )
                ]
                for seq in sequences
            ]

    def _parse_timestamps(self, sequence: list[int]) -> list[Transcript]:
        # Whisper Timestamp Encoding:
        # - Timestamp tokens start at ID 50364 and increment by 1 for each 0.02s
        # - Token 50364 = 0.00s, 50365 = 0.02s, 50414 = 1.00s, 51864 = 30.00s
        # - Formula: time_seconds = (token_id - 50364) * 0.02
        timestamp_begin = 50364

        # Tokens to ignore when parsing (special control tokens)
        skip_tokens = {
            self.gen_config.decoder_start_token_id,  # 50258 = <|startoftranscript|>
            self.gen_config.no_timestamps_token_id,  # 50363 = <|notimestamps|>
            self.tokenizer.eos_token_id,  # 50257 = <|endoftext|>
        } | {
            tok for _, tok in self.prompt_ids
        }  # Language/task tokens (50259, 50359, etc.)

        transcripts = []  # Final list of time-segmented transcripts
        current_tokens = []  # Accumulator for text tokens between timestamps
        current_start_time = None  # Start time of current segment

        for token_id in sequence:
            # Skip special control tokens
            if token_id in skip_tokens:
                continue

            # Check if this is a timestamp token (50364+)
            if token_id >= timestamp_begin:
                # Convert token ID to seconds. Example: 50414 -> (50414 - 50364) * 0.02 = 1.00s
                time_seconds = (token_id - timestamp_begin) * 0.02

                if current_start_time is None:
                    # First timestamp = start of segment
                    current_start_time = time_seconds
                else:
                    # Second timestamp = end of segment, create Transcript. Example: 50414 (1.00s) marks end, decode tokens between timestamps
                    if current_tokens:
                        text = self.tokenizer.decode(
                            current_tokens, skip_special_tokens=True
                        ).strip()
                        if text:
                            transcripts.append(
                                Transcript(
                                    text=text,
                                    start_time=int(current_start_time),
                                    end_time=int(time_seconds),
                                )
                            )
                    # Start new segment
                    current_tokens, current_start_time = [], time_seconds
            else:
                # Regular text token (not a timestamp)
                current_tokens.append(token_id)

        # Handle remaining tokens at end of sequence (no closing timestamp)
        # Assume 30s duration as default
        if current_tokens and current_start_time is not None:
            text = self.tokenizer.decode(
                current_tokens, skip_special_tokens=True
            ).strip()
            if text:
                transcripts.append(
                    Transcript(
                        text=text,
                        start_time=int(current_start_time),
                        end_time=int(current_start_time + 30),  # Default 30s chunk
                    )
                )

        # Fallback: If no timestamps found in sequence, return all text with time 0
        if not transcripts:
            text = self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
            if text:
                transcripts.append(Transcript(text=text, start_time=0, end_time=0))

        return transcripts

    def _step_decoder(
        self,
        tokens: np.ndarray,
        decoder_cache: list[np.ndarray],
        encoder_cache: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if not (self._use_past and decoder_cache is not None):
            raise RuntimeError("decoder_cache should always exist after init")

        # Prepare inputs: current token IDs with sequence length 1
        # Shape: [batch_size, 1] (we only process ONE new token at a time)
        inputs = {"input_ids": tokens[:, None].astype(np.int64)}
        num_layers = len(decoder_cache) // 2  # Each layer has 2 tensors (K, V)

        # Interleave decoder and encoder cache in ONNX's expected format
        # For each layer: [decoder_K, decoder_V, encoder_K, encoder_V]
        # Example for 2 layers: [dec_K₀, dec_V₀, enc_K₀, enc_V₀, dec_K₁, dec_V₁, enc_K₁, enc_V₁]
        full_past = [
            cache[i]
            for layer in range(num_layers)
            for cache in [
                decoder_cache[layer * 2 : layer * 2 + 2],  # Get K, V for this layer
                encoder_cache[
                    layer * 2 : layer * 2 + 2
                ],  # Get encoder K, V for this layer
            ]
            for i in range(len(cache))  # Iterate over K, V
        ]

        # Add all past KV tensors to ONNX input dictionary
        # ONNX input names are like: "past_key_values.0.decoder.key", "past_key_values.0.decoder.value", ...
        inputs.update(
            {
                self.decoder_with_past.get_inputs()[1 + i].name: p
                for i, p in enumerate(full_past)
            }
        )

        # Run ONNX decoder inference
        # Output structure (for 4-layer model):
        # [logits, dec_K₀, dec_V₀, enc_K₀, enc_V₀, dec_K₁, dec_V₁, enc_K₁, enc_V₁, ...]
        outs = self.decoder_with_past.run(None, inputs)

        # Get predicted token: argmax over vocabulary dimension
        # logits shape: [batch_size, 1, vocab_size] -> take last position, find max
        next_token = np.argmax(outs[0][:, -1, :], axis=-1)

        # Return predicted token and updated KV cache (everything after logits)
        return next_token, outs[1:]

    def _init_decoder(
        self, encoder_out: np.ndarray, init_token: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        # Run decoder_init model (no past KV cache inputs)
        # Inputs: token IDs + encoder hidden states
        outs = self.decoder_init.run(
            None,
            {
                "input_ids": init_token[:, None].astype(np.int64),  # [batch, 1]
                "encoder_hidden_states": encoder_out,  # [batch, seq_len, hidden_dim]
            },
        )

        if self._use_past:
            # Extract KV cache from outputs (skip first output which is logits)
            # Format: [dec_K₀, dec_V₀, enc_K₀, enc_V₀, dec_K₁, dec_V₁, enc_K₁, enc_V₁, ...]
            all_present = outs[1:]
            num_layers = (
                len(all_present) // 4
            )  # Each layer produces 4 tensors (2 dec + 2 enc)

            # Extract decoder self-attention cache: indices 0, 1 for each layer
            # [dec_K₀, dec_V₀, dec_K₁, dec_V₁, ...]
            decoder_cache = [
                all_present[layer * 4 + i]
                for layer in range(num_layers)
                for i in range(2)  # K, V for decoder
            ]

            # Extract encoder cross-attention cache: indices 2, 3 for each layer
            # [enc_K₀, enc_V₀, enc_K₁, enc_V₁, ...]
            encoder_cache = [
                all_present[layer * 4 + i]
                for layer in range(num_layers)
                for i in range(2, 4)  # K, V for encoder
            ]
        else:
            # No KV caching (slower, not recommended for production)
            decoder_cache = encoder_cache = None

        # Get predicted next token from logits
        # logits shape: [batch, 1, vocab_size] -> argmax over vocab dimension
        next_token = np.argmax(outs[0][:, -1, :], axis=-1)

        return next_token, decoder_cache, encoder_cache

    def _extract_features(self, audio_arr: np.ndarray) -> np.ndarray:
        # Handle both single audio and batched audio
        batch = (
            [audio_arr[i] for i in range(audio_arr.shape[0])]
            if audio_arr.ndim == 2
            else [audio_arr]
        )

        # Extract mel-spectrogram for each audio in batch
        # WhisperFeatureExtractor handles:
        # - Padding/truncating to 30s (480,000 samples)
        # - STFT with 400 sample window, 160 sample hop
        # - Mel filterbank with 80 bins
        # - Log scaling and normalization
        feats = [
            self.feature_extractor(
                arr,
                sampling_rate=settings.whisper_settings.SAMPLING_RATE,  # 16000 Hz
                return_tensors="np",
            ).input_features[
                0
            ]  # Extract numpy array from dict
            for arr in batch
        ]

        # Stack batch and ensure float32 for ONNX Runtime
        return np.stack(feats, axis=0).astype(np.float32)  # [batch_size, 80, 3000]

    def _load_onnx_sessions(self):
        # Select execution provider based on device
        providers = (
            ["CPUExecutionProvider"]
            if self.device == "cpu"
            else ["CUDAExecutionProvider"]
        )

        # Load encoder model: processes entire audio mel-spectrogram at once
        self.encoder = ort.InferenceSession(
            os.path.join(self.model_name, "encoder_model.onnx"), providers=providers
        )

        # Load initial decoder model (no past KV cache inputs)
        self.decoder_init = ort.InferenceSession(
            os.path.join(self.model_name, "decoder_model.onnx"), providers=providers
        )

        # Try to load decoder with KV cache support
        try:
            self.decoder_with_past = ort.InferenceSession(
                os.path.join(self.model_name, "decoder_with_past_model.onnx"),
                providers=providers,
            )
            self._use_past = True
        except Exception:
            # Fallback: use decoder_init for all tokens
            self.decoder_with_past, self._use_past = None, False
