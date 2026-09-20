"""Oracle extractor: returns the true primitives, unchanged.

Upper bound for the stage-2 relational classifier -- comparing oracle vs.
classical/learned results separates "is the primitive-graph representation
any good" from "did stage 1 fail to recover it" (PLAN.md section 4).
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from ..data.primitives import Primitive
from .base import PrimitiveExtractor


class OracleExtractor(PrimitiveExtractor):
    name = "oracle"

    def extract(self, image: Image.Image, ground_truth: Optional[List[Primitive]] = None) -> List[Primitive]:
        if ground_truth is None:
            raise ValueError("OracleExtractor.extract requires ground_truth primitives")
        return list(ground_truth)
