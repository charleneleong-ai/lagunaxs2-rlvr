"""Reproducibility: seed python / numpy / torch (CPU + CUDA) in one call."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0) -> int:
    """Seed all RNGs that affect a run (projector init, data shuffling, sampling). Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
