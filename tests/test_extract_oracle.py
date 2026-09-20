import pytest

from shapeprim.data.primitives import Primitive
from shapeprim.extract.oracle import OracleExtractor


def test_oracle_returns_ground_truth_unchanged():
    prims = [
        Primitive(type="circle", cx=1, cy=2, width=3, height=3, color=(1, 1, 1)),
        Primitive(type="triangle", cx=4, cy=5, width=6, height=7, color=(1, 1, 1), is_distractor=True),
    ]
    out = OracleExtractor().extract(image=None, ground_truth=prims)
    assert out == prims
    assert out is not prims  # returns a copy, not the same list object


def test_oracle_requires_ground_truth():
    with pytest.raises(ValueError):
        OracleExtractor().extract(image=None)
