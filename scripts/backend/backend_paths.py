"""Shared persistent root for the staged backend modules."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ["SAIGONBOT_DATA_DIR"])
