"""Configuration module"""

import os
import logging
from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(str, Enum):
    DEVELOPMENT = "dev"
    PRODUCTION = "prod"


class WhisperModel(BaseModel):
    """Agent model settings."""

    MODEL_NAME: str = Field(default="")
    SAMPLING_RATE: int = Field(default=16000)
    DEVICE: str = Field(default="")
    BATCH_SIZE: int = Field(default=8)


class OwlModel(BaseModel):
    """Owl model settings."""

    MODEL_NAME: str = Field(default="")
    DEVICE: str = Field(default="")


class SAMModel(BaseModel):
    """SAM model settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )

    MODEL_NAME: str = Field(default="")
    CHECKPOINT_DIR_NAME: str = Field(default="checkpoints")
    CONFIG_DIR_NAME: str = Field(default="configs")
    COLOR_MAP: str = Field(default="Set3")
    DEVICE: str = Field(default="")


class PaddleModel(BaseModel):
    """PaddleOCR model settings."""

    DET_MODEL_NAME: str = Field(default="")
    REC_MODEL_NAME: str = Field(default="")
    DEVICE: str = Field(default="")


class TritonModel(BaseModel):
    """Triton model settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )

    TRITON_MAX_BATCH_SIZE: int = Field(default=8)
    TRITON_MAX_QUEUE_DELAY_MICROSECONDS: int = Field(default=1_000_000)


class AppSettings(BaseSettings):
    """Configuration settings for the agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )

    # Environment
    APP_ENV: Environment = Field(default=Environment.DEVELOPMENT)
    HF_TOKEN: str = Field(default="")
    LOG_DIR_NAME: str = Field(default="logs")

    # Models
    whisper_settings: WhisperModel = Field(default=WhisperModel())
    owl_settings: OwlModel = Field(default=OwlModel())
    sam_settings: SAMModel = Field(default=SAMModel())
    paddle_settings: PaddleModel = Field(default=PaddleModel())
    triton_settings: TritonModel = Field(default=TritonModel())
