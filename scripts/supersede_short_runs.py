#!/usr/bin/env python3
"""Move runs that early-stopped before ``min_steps`` optimizer steps aside.

Protocol rule 3 (docs/experiment_protocol.md) forbids early stopping
before 500 optimizer steps. Runs recorded before the rule existed that
stopped earlier are moved to ``results/<phase>/superseded_patience8/``
(kept, not deleted, so the change is auditable) and are re-run by
``run_experiment.py --resume``. Runs that trained past the floor are
unaffected by the rule and stay where they are.

Usage:
    python scripts/supersede_short_runs.py [--min-steps 500] [--dry-run]
"""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-steps", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-precise-bn", action="store_true",
        help="instead: supersede CNN runs with trainable BatchNorm recorded before precise BN (rule 3c)",
    )
    args = ap.parse_args()
    moved = 0
    for metrics in sorted(ROOT.glob("results/*/*/*/metrics.json")):
        run_dir = metrics.parent
        if "superseded" in run_dir.parts[-3] or "superseded" in run_dir.parts[-2]:
            continue
        m = json.loads(metrics.read_text())
        cfg = json.loads((run_dir / "config.json").read_text())["config"]
        phase, experiment = run_dir.parts[-3], run_dir.parts[-2]
        if args.no_precise_bn:
            model_cfg = cfg.get("model_spec", {}).get("model_config", m.get("kind"))
            frozen = model_cfg == "cnn_probe"
            if m.get("kind") != "cnn" or frozen or "precise_bn" in (m.get("train_params") or {}):
                continue
            dest = ROOT / "results" / phase / "superseded_no_precise_bn" / experiment / run_dir.name
            print(f"{phase}/{experiment}/{run_dir.name}: trained without precise BN")
        else:
            if "steps" in m:  # recorded under the rule
                continue
            n_classes = len(cfg.get("classes") or [])
            batch = cfg["experiment"]["batch_size"]
            steps = m["epochs_run"] * math.ceil(m["n_train_per_class"] * n_classes / batch)
            if not m.get("stopped_early") or steps >= args.min_steps:
                continue
            dest = ROOT / "results" / phase / "superseded_patience8" / experiment / run_dir.name
            print(f"{phase}/{experiment}/{run_dir.name}: stopped after {steps} steps")
        moved += 1
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(metrics)], cwd=ROOT,
                                     capture_output=True).returncode == 0
            if tracked:
                subprocess.run(["git", "mv", str(run_dir), str(dest)], cwd=ROOT, check=True)
            else:
                shutil.move(str(run_dir), str(dest))
    print(f"{moved} run(s) superseded")


if __name__ == "__main__":
    main()
