"""Keep downloads local and CPU tests fast on small matrix workloads."""
import os
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / 'model_cache'))

import torch

torch.set_num_threads(1)
