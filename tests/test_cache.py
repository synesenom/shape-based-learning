"""Extraction caching: correctness first, speed second.

A cache that returns another condition's primitives would corrupt every
downstream number silently, so the key has to cover everything that
determines the extraction.
"""

import json

from shapeprim.data.primitives import Primitive
from shapeprim.data.synth_dataset import GenerationConfig
from shapeprim.data.torch_datasets import GraphClassificationDataset, PrimitiveCache
from shapeprim.extract.base import PrimitiveExtractor
from shapeprim.extract.oracle import OracleExtractor

CFG = GenerationConfig(image_size=48, distractor_prob=0.0)


class CountingExtractor(PrimitiveExtractor):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def extract(self, image, ground_truth=None):
        self.calls += 1
        return list(ground_truth or [])


def _dataset(extractor, cache_dir=None, seed=0, split="train", n=3, cfg=CFG):
    return GraphClassificationDataset(
        extractor, classes=["house", "tree"], n_per_class=n, cfg=cfg, seed=seed, split=split, cache_dir=cache_dir
    )


def test_extraction_happens_once_per_sample_in_memory():
    ex = CountingExtractor()
    ds = _dataset(ex)
    for _ in range(3):  # three "epochs"
        for i in range(len(ds)):
            ds[i]
    assert ex.calls == len(ds)


def test_prewarm_fills_the_cache_exactly_once():
    ex = CountingExtractor()
    ds = _dataset(ex)
    ds.prewarm()
    assert ex.calls == len(ds)
    ds.prewarm()
    assert ex.calls == len(ds)


def test_cache_persists_across_dataset_instances(tmp_path):
    ex1 = CountingExtractor()
    ds1 = _dataset(ex1, cache_dir=tmp_path)
    ds1.prewarm()
    assert ex1.calls == len(ds1)

    ex2 = CountingExtractor()
    ds2 = _dataset(ex2, cache_dir=tmp_path)
    ds2.prewarm()
    assert ex2.calls == 0, "second dataset re-extracted instead of loading the cache"


def test_cached_primitives_match_freshly_extracted(tmp_path):
    fresh = _dataset(OracleExtractor())
    cached_ds = _dataset(OracleExtractor(), cache_dir=tmp_path)
    cached_ds.prewarm()
    reloaded = _dataset(OracleExtractor(), cache_dir=tmp_path)
    for i in range(len(fresh)):
        a = fresh.primitives_at(i)[0]
        b = reloaded.primitives_at(i)[0]
        assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


def test_cache_key_separates_splits_seeds_configs_and_extractors(tmp_path):
    base = _dataset(OracleExtractor(), cache_dir=tmp_path).cache_key
    assert _dataset(OracleExtractor(), cache_dir=tmp_path, split="val").cache_key != base
    assert _dataset(OracleExtractor(), cache_dir=tmp_path, seed=1).cache_key != base
    assert _dataset(OracleExtractor(), cache_dir=tmp_path, n=4).cache_key != base
    assert _dataset(CountingExtractor(), cache_dir=tmp_path).cache_key != base
    other_cfg = GenerationConfig(image_size=64, distractor_prob=0.0)
    assert _dataset(OracleExtractor(), cache_dir=tmp_path, cfg=other_cfg).cache_key != base


def test_cache_key_is_stable_across_instances(tmp_path):
    assert _dataset(OracleExtractor(), cache_dir=tmp_path).cache_key == _dataset(
        OracleExtractor(), cache_dir=tmp_path
    ).cache_key


def test_corrupt_cache_file_is_discarded_not_fatal(tmp_path):
    """A truncated cache (interrupted write, full disk) must not be fatal.

    Extraction is reproducible, so the right recovery is to drop the cache
    and recompute rather than take the experiment down.
    """
    ex = CountingExtractor()
    key = _dataset(ex, cache_dir=tmp_path).cache_key
    (tmp_path / f"{key}.json").write_text("{ truncated")

    recovered = _dataset(ex, cache_dir=tmp_path)
    recovered.prewarm()
    assert ex.calls == len(recovered)
    # And the recovered cache is written back intact.
    assert json.loads((tmp_path / f"{key}.json").read_text())["items"]


def test_primitive_cache_round_trip(tmp_path):
    path = tmp_path / "c.json"
    cache = PrimitiveCache(path)
    cache.put(0, [Primitive(type="circle", cx=1.0, cy=2.0, width=3.0, height=4.0)])
    cache.save()

    reloaded = PrimitiveCache(path)
    got = reloaded.get(0)
    assert got is not None and got[0].type == "circle" and got[0].cx == 1.0
    assert reloaded.get(1) is None


def test_empty_cache_writes_nothing(tmp_path):
    path = tmp_path / "empty.json"
    PrimitiveCache(path).save()
    assert not path.exists()


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    cache = PrimitiveCache(tmp_path / "c.json")
    cache.put(0, [Primitive(type="line", cx=1.0, cy=1.0, width=8.0, height=2.0)])
    cache.save()
    cache.save()
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads((tmp_path / "c.json").read_text())["items"]
