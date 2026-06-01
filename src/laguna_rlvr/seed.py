"""Global determinism: one seed for python `random`, plus numpy and torch when importable."""
from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG the project touches so training runs and tests are reproducible.

    torch and numpy are heavy/optional deps (only the visual/training track pulls torch), so they
    are seeded only when importable — function-local guarded imports per the conditional-dep rule.
    This keeps the core RL pipeline and its tests torch-free. Returns the seed so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    return seed
