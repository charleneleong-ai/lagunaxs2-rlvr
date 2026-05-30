import pytest

from laguna_rlvr.seed import seed_everything


@pytest.fixture(autouse=True)
def _global_seed():
    """Seed every RNG before each test so the suite is deterministic (no flaky inits)."""
    seed_everything(42)
