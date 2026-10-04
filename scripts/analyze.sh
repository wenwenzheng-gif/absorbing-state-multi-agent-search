#!/usr/bin/env bash
# Summarize and estimate d_c for every finished run of one config tag, then
# build the 3x3 (policy x cell) d_c table.
#
# Usage: scripts/analyze.sh --tag qwen3.5-4b_seed1 [--run-root DIR] [--convention pre_own]
#
# For each of the 9 cell x policy run dirs under runs/agents/<TAG>/ (or
# --run-root) that has an episodes.jsonl, this runs:
#   python -m src.agent_system.analysis.summarize      --run-dir <dir>
#   python -m src.agent_system.analysis.critical_degree --run-dir <dir> --convention pre_own
#   python -m src.agent_system.analysis.critical_degree --run-dir <dir> --convention pre_own --first-round 2
# The first call is the default all-rounds sum (--convention defaults to
# pre_own here, which is also dc_table.py's main table convention, so this
# call's unsuffixed output files -- analysis/critical_degree.json,
# d_c_estimates.csv, log_R_by_degree.csv -- carry the pre_own curve as
# their primary one). The all-rounds pre_own sum is the empirical counterpart
# to references.delayed_dc, including the round-1 branching contribution.
# The second call sums from round 2 and is kept as a secondary diagnostic (dc_table.py's
# emp_pre_own_t2 column -- the previous version's main empirical value); it
# also always uses --convention pre_own: pre_own is the frozen src/utils.py
# convention (the A-to-A count read after the branch step, before the
# agent's own experiments/pruning) -- post_own additionally folds in
# self-pruning (1-p_own), which the paper's own d_c derivation does not include. It writes to analysis/critical_degree_t2.json
# etc., alongside the untouched default (first_round=1) outputs from the
# first call. Note critical_degree already reports every convention side by
# side in its JSON/CSV output regardless of which one is passed as
# --convention, so this only changes which convention's curve gets the
# unsuffixed (primary) filename; dc_table.py reads specific convention rows
# out of the JSON either way. Then calls scripts/dc_table.py to assemble
# runs/agents/<TAG>/dc_table.{csv,md} from whatever analysis output exists.
# Missing run dirs are left as empty cells, not errors.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=""
RUN_ROOT=""
CONVENTION="pre_own"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --convention) CONVENTION="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$TAG" ]]; then
  echo "error: --tag is required, e.g. --tag qwen3.5-4b_seed1" >&2
  exit 1
fi
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="runs/agents/$TAG"
fi

CELLS=(synthetic tcas physics)
POLICIES=(C0 C1 I1)

echo "tag=$TAG  run_root=$RUN_ROOT  convention=$CONVENTION"

for cell in "${CELLS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    run_dir="$RUN_ROOT/${TAG}_${cell}_${policy}"
    if [[ ! -f "$run_dir/episodes.jsonl" ]]; then
      echo "skip (no episodes.jsonl): $cell/$policy -> $run_dir"
      continue
    fi
    echo "== $cell/$policy -> $run_dir =="
    python -m src.agent_system.analysis.summarize --run-dir "$run_dir" > /dev/null
    python -m src.agent_system.analysis.critical_degree --run-dir "$run_dir" \
      --convention "$CONVENTION" > /dev/null
    python -m src.agent_system.analysis.critical_degree --run-dir "$run_dir" \
      --convention pre_own --first-round 2 > /dev/null
  done
done

python3 scripts/dc_table.py --tag "$TAG" --run-root "$RUN_ROOT"
