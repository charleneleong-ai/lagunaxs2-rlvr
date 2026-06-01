import pytest

from laguna_rlvr.visual.grounding import parse_box
from laguna_rlvr.visual.screenspot import _to_norm_box


class TestToNormBox:
    """Pixel [x, y, w, h] -> normalized [x1, y1, x2, y2] text, parseable back by grounding.parse_box."""

    def test_known_box(self):
        # ScreenSpot row: bbox [910, 78, 44, 34] on a 960x540 image.
        assert _to_norm_box([910, 78, 44, 34], 960, 540) == "[0.948, 0.144, 0.994, 0.207]"

    def test_full_image_maps_to_unit_square(self):
        assert _to_norm_box([0, 0, 100, 50], 100, 50) == "[0.000, 0.000, 1.000, 1.000]"

    def test_roundtrips_through_parse_box(self):
        box = parse_box(_to_norm_box([910, 78, 44, 34], 960, 540))
        assert box == pytest.approx((910 / 960, 78 / 540, 954 / 960, 112 / 540), abs=1e-3)
