"""torch.utils.data.Dataset wrappers around SynthShapeDataset.

``ImageClassificationDataset`` feeds the CNN baseline raw pixels, with
optional augmentation (see ``augment.py``).
``GraphClassificationDataset`` runs a primitive extractor on the rendered
image and turns the result into a ``ShapeGraph`` for the GNN. Both share
the same underlying rendering (same seed and split -> same image), so a
CNN run and a GNN run configured alike see exactly the same images.

Extraction is cached (``PrimitiveCache``). The classical extractor costs
roughly a millisecond per image, which is invisible for one pass and
dominant across a 50-epoch run: without a cache the same deterministic
image is re-thresholded and re-contoured every epoch. The cache makes the
cost O(dataset) instead of O(dataset x epochs), and persisting it to disk
makes a sweep over training-set sizes or model hyperparameters pay it once
in total rather than once per run.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..extract.base import PrimitiveExtractor
from ..graph.build import NUM_EDGE_FEATURES, NUM_NODE_FEATURES, ShapeGraph, build_graph
from .augment import AugmentConfig, build_transform, resolve_augment
from .objects import CLASS_NAMES, DATASET_VERSION
from .primitives import Primitive
from .synth_dataset import GenerationConfig, SynthShapeDataset

# Label indices for the full 10-class vocabulary. Kept as a module-level
# convenience, but datasets index against *their own* class list: a
# dataset restricted to a subset must emit 0..len(subset)-1, not the
# global positions, or the labels run past the classifier's output layer.
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def class_to_idx(classes: Sequence[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(classes)}


def _identity_key(
    classes: Sequence[str],
    n_per_class: int,
    cfg: GenerationConfig,
    seed: int,
    split: str,
    extractor_name: str,
) -> str:
    """A short stable digest of everything that determines the extraction.

    Anything that changes the rendered pixels or the extractor must change
    this key, or a stale cache would silently feed one experiment another
    experiment's primitives.
    """
    payload = {
        "classes": list(classes),
        "n_per_class": n_per_class,
        "cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
        "seed": seed,
        "split": split,
        "extractor": extractor_name,
        "dataset_version": DATASET_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class PrimitiveCache:
    """Index -> extracted primitives, in memory and optionally on disk.

    Not shared across DataLoader worker processes: each worker would fill
    its own copy and none would survive the epoch. Call ``prewarm`` in the
    main process before handing the dataset to a multi-worker DataLoader
    (the driver does this), which is faster anyway for a deterministic
    dataset.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._data: Dict[int, List[Primitive]] = {}
        self.hits = 0
        self.misses = 0
        if self.path is not None and self.path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self._data)

    def get(self, index: int) -> Optional[List[Primitive]]:
        found = self._data.get(index)
        if found is None:
            self.misses += 1
            return None
        self.hits += 1
        return found

    def put(self, index: int, primitives: List[Primitive]) -> None:
        self._data[index] = primitives

    def load(self) -> None:
        assert self.path is not None
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            # A truncated cache (interrupted write, full disk) must not take
            # an experiment down with it -- extraction is reproducible, so
            # the right recovery is to drop the cache and recompute.
            self._data = {}
            return
        self._data = {
            int(k): [Primitive.from_dict(p) for p in prims] for k, prims in raw.get("items", {}).items()
        }

    def save(self) -> None:
        if self.path is None or not self._data:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": {str(k): [p.to_dict() for p in v] for k, v in self._data.items()}}
        # Atomic replace: a crash mid-write leaves the old cache intact
        # rather than a half-written file that every later run must discard.
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


class ImageClassificationDataset(Dataset):
    """Rendered images + integer labels, with optional augmentation.

    Augmentation applies to this dataset only, and callers should enable it
    for training splits only -- an augmented validation set measures a
    different distribution than the one used for model selection, and an
    augmented test set is not the test set.
    """

    def __init__(
        self,
        classes: Sequence[str] = CLASS_NAMES,
        n_per_class: int = 100,
        cfg: Optional[GenerationConfig] = None,
        seed: int = 0,
        split: str = "",
        augment=None,
    ):
        self.inner = SynthShapeDataset(classes=classes, n_per_class=n_per_class, cfg=cfg, seed=seed, split=split)
        self.class_to_idx = class_to_idx(self.inner.classes)
        self.augment_cfg: AugmentConfig = resolve_augment(augment)
        self.transform = build_transform(self.augment_cfg, background=self.inner.cfg.background)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.inner[idx]
        image = sample.image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C, H, W)
        return tensor, self.class_to_idx[sample.label]


class GraphClassificationDataset(Dataset):
    """Extracted primitives -> ShapeGraph + integer label, with caching."""

    def __init__(
        self,
        extractor: PrimitiveExtractor,
        classes: Sequence[str] = CLASS_NAMES,
        n_per_class: int = 100,
        cfg: Optional[GenerationConfig] = None,
        seed: int = 0,
        split: str = "",
        cache_dir: Optional[str | Path] = None,
    ):
        self.inner = SynthShapeDataset(classes=classes, n_per_class=n_per_class, cfg=cfg, seed=seed, split=split)
        self.class_to_idx = class_to_idx(self.inner.classes)
        self.extractor = extractor
        self.cache_key = _identity_key(
            self.inner.classes, n_per_class, self.inner.cfg, seed, split, getattr(extractor, "name", type(extractor).__name__)
        )
        path = Path(cache_dir) / f"{self.cache_key}.json" if cache_dir else None
        self.cache = PrimitiveCache(path)

    def __len__(self) -> int:
        return len(self.inner)

    def primitives_at(self, idx: int) -> Tuple[List[Primitive], List[Primitive], str]:
        """(extracted, ground truth, label) for index ``idx``.

        Ground truth is always rendered fresh; only the extractor output is
        cached, since that is the expensive and deterministic part.
        """
        sample = self.inner[idx]
        cached = self.cache.get(idx)
        if cached is None:
            cached = self.extractor.extract(sample.image, ground_truth=sample.primitives)
            self.cache.put(idx, cached)
        return cached, sample.primitives, sample.label

    def prewarm(self, verbose: bool = False, save: bool = True) -> None:
        """Extract every sample once, in this process, then persist.

        Call before wrapping in a multi-worker DataLoader: workers do not
        share a cache, so without this each worker re-extracts its shard
        every epoch.
        """
        missing = [i for i in range(len(self)) if self.cache.get(i) is None]
        # ``get`` counted those as misses; they are about to be filled.
        self.cache.misses -= len(missing)
        if not missing:
            return
        if verbose:
            print(f"    extracting {len(missing)}/{len(self)} samples ({self.extractor.name})...", flush=True)
        for i in missing:
            self.primitives_at(i)
        if save:
            self.cache.save()

    def __getitem__(self, idx: int) -> Tuple[ShapeGraph, int]:
        primitives, _gt, label = self.primitives_at(idx)
        graph = build_graph(primitives, label=label)
        return graph, self.class_to_idx[label]


def collate_graphs(batch: List[Tuple[ShapeGraph, int]]):
    """Merge a list of (ShapeGraph, label) into one disjoint-union batch.

    Returns (node_features, edge_index, edge_features, node_batch, num_graphs, labels).
    ``node_batch[i]`` is the index of the graph node ``i`` belongs to -- the
    standard "batch vector" trick for pooling variable-sized graphs without
    padding.
    """
    graphs, labels = zip(*batch)
    num_graphs = len(graphs)

    node_feat_parts, batch_parts = [], []
    src_parts, dst_parts, edge_feat_parts = [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        n = g.num_nodes
        if n > 0:
            node_feat_parts.append(torch.from_numpy(g.node_features))
            batch_parts.append(torch.full((n,), gi, dtype=torch.long))
        if g.num_edges > 0:
            src_parts.append(torch.from_numpy(g.edge_index[0]) + offset)
            dst_parts.append(torch.from_numpy(g.edge_index[1]) + offset)
            edge_feat_parts.append(torch.from_numpy(g.edge_features))
        offset += n

    node_features = torch.cat(node_feat_parts, dim=0).float() if node_feat_parts else torch.zeros(0, NUM_NODE_FEATURES)
    node_batch = torch.cat(batch_parts, dim=0) if batch_parts else torch.zeros(0, dtype=torch.long)
    if src_parts:
        edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)
        edge_features = torch.cat(edge_feat_parts, dim=0).float()
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_features = torch.zeros(0, NUM_EDGE_FEATURES)

    return node_features, edge_index, edge_features, node_batch, num_graphs, torch.tensor(labels, dtype=torch.long)
