"""Global determinism: one seed for python `random`, numpy (if present), and torch (cpu+cuda)."""
from __future__ import annotations

import os
import random

import torch

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG the project touches so training runs and tests are reproducible.

    Returns the seed so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # numpy is an optional transitive dep (transformers/datasets use it internally); seed it only
    # if importable — function-local import is the documented exception for conditional deps.
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    return seed
