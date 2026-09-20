"""Common interface every primitive extractor implements: extract(image) -> list[Primitive].

``ground_truth`` is an extra, optional argument only ``OracleExtractor``
consumes (it needs the true primitives to "extract" them); every other
extractor ignores it. Keeping it on the shared signature lets evaluation
code loop over extractors uniformly without special-casing the oracle.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image

from ..data.primitives import Primitive


class PrimitiveExtractor:
    name = "base"

    def extract(self, image: Image.Image, ground_truth: Optional[List[Primitive]] = None) -> List[Primitive]:
        raise NotImplementedError
