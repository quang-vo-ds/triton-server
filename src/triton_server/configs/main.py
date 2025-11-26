import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

import yaml

from .models import AppSettings


CONFIG_DIR_PATH = Path(__file__).parent / "conf"


def load_settings() -> AppSettings:
    load_dotenv()

    # Decide environment (default to dev)
    env = os.getenv("APP_ENV", "dev")

    # Load YAML file
    yaml_file = CONFIG_DIR_PATH / f"{env}.yaml"
    with open(yaml_file, "r") as f:
        yaml_data: Dict[str, Any] = yaml.safe_load(f)

    # Feed YAML into Pydantic model (env + .env will override YAML)
    return AppSettings(**yaml_data)


settings = load_settings()
