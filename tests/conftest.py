import os
from pathlib import Path

from match_forecast.common import runtime

runtime(Path(__file__).resolve().parents[1])
