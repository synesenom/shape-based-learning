#!/usr/bin/env bash
# Run every PLAN.md experiment, one job at a time, resumably, pushing results.
#
# The machine running this can be reclaimed at any moment, so:
#   - every step skips work that is already done (run_experiment --resume
#     skips runs whose metrics.json exists; detectors are not retrained if
#     their weights exist; SAM-fit and extraction are cached);
#   - results are committed and pushed every CHECKPOINT_MINUTES and after
#     every step, so a fresh clone resumes where the last push left off;
#   - exactly one training job runs at a time, with all cores. Running
#     several PyTorch jobs side by side on 4 cores oversubscribes the
#     thread pools and made each job ~10x slower, not ~3x.
#
# Usage:
#   scripts/run_all.sh [--wait-pid PID] [--no-push]
set -u
cd "$(dirname "$0")/.."

WAIT_PID=""
PUSH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --wait-pid) WAIT_PID="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done

CHECKPOINT_MINUTES=${CHECKPOINT_MINUTES:-15}
BRANCH=$(git rev-parse --abbrev-ref HEAD)
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

checkpoint() {
  [ "$PUSH" = 1 ] || return 0
  (
    flock -w 300 9 || exit 0
    git add -A results >/dev/null 2>&1
    if ! git diff --cached --quiet; then
      git commit -q -m "Checkpoint experiment results ($1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SojNaaWHiJoRRuxsTd6pEi" || exit 0
    fi
    for delay in 2 4 8 16; do
      git push -q origin "$BRANCH" && exit 0
      sleep "$delay"
    done
    echo "checkpoint push failed ($1)" >&2
  ) 9>.git/checkpoint.lock
}

# Run a command with a periodic checkpoint while it runs.
step() {
  local name="$1"; shift
  echo "=== $(date -u +%FT%TZ) $name" | tee -a "$LOG_DIR/run_all.log"
  "$@" >>"$LOG_DIR/$name.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    for _ in $(seq $((CHECKPOINT_MINUTES * 6))); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 10
    done
    kill -0 "$pid" 2>/dev/null && checkpoint "$name, in progress"
  done
  wait "$pid"
  local rc=$?
  echo "=== $(date -u +%FT%TZ) $name exit $rc" | tee -a "$LOG_DIR/run_all.log"
  checkpoint "$name"
  return $rc
}

experiment() {  # experiment <phase> <name> [plot args...]
  local phase="$1" name="$2"; shift 2
  step "$name" python scripts/run_experiment.py --config "configs/$phase/$name.yaml" --resume
  python scripts/plot_results.py --summary "results/$phase/$name/summary.json" >>"$LOG_DIR/$name.log" 2>&1
  if [ $# -gt 0 ]; then
    python scripts/plot_conditions.py --summary "results/$phase/$name/summary.json" "$@" >>"$LOG_DIR/$name.log" 2>&1
  fi
  checkpoint "$name plots"
}

if [ -n "$WAIT_PID" ]; then
  echo "waiting for pid $WAIT_PID to finish" | tee -a "$LOG_DIR/run_all.log"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

# ---- Phase 1 ----------------------------------------------------------------
experiment phase1 novel_composition
experiment phase1 shift_position_scale_fewshot
experiment phase1 shift_position_scale
experiment phase1 learning_curve

# ---- Phase 2 ----------------------------------------------------------------
[ -f results/phase2/learned_extractor/learned_phase2_0-30.pt ] || \
  step learned_extractor python scripts/train_learned_extractor.py --config configs/phase2/learned_extractor.yaml
experiment phase2 angle_extrapolation --prefix angle --train-range 0 30 --xlabel "viewing angle (degrees)"
experiment phase2 relation_rotation --prefix rot --train-range 0 30 --xlabel "in-plane rotation (degrees)"
experiment phase2 full_range --prefix angle --train-range 0 70 --xlabel "viewing angle (degrees)"

# ---- Phase 3a ---------------------------------------------------------------
[ -f results/phase3/learned_extractor_appearance/learned_appearance.pt ] || \
  step learned_extractor_appearance python scripts/train_learned_extractor.py --config configs/phase3/learned_extractor_appearance.yaml
experiment phase3 appearance_shift
experiment phase3 appearance_reverse

# ---- Phase 3b ---------------------------------------------------------------
step fetch_quickdraw python scripts/fetch_quickdraw.py
[ -f results/phase3/strokefit_proxy/summary.json ] || step strokefit_proxy python scripts/eval_strokefit.py
experiment phase3 quickdraw_fewshot
experiment phase3 quickdraw_cross
experiment phase3 quickdraw_cross_reverse

# ---- Phase 3c ---------------------------------------------------------------
step fetch_real python scripts/fetch_real_data.py
step precompute_sam python scripts/precompute_sam.py --threads 4
experiment phase3 real_images

echo "=== $(date -u +%FT%TZ) all done" | tee -a "$LOG_DIR/run_all.log"
