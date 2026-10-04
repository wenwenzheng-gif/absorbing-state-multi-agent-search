#!/usr/bin/env bash
# Run the C0/C1/I1 x {synthetic, tcas, physics} grid for one
# already-generated config tag (see scripts/make_configs.py).
#
# LLM-only run: every config here has policy.kind == "llm_full" -- the LLM
# branches and chooses experiments (and, for C1/I1, writes the free-text
# messages); evidence pruning is applied by code, not by the model. No
# rule/uniform/adaptive config is ever run by this script.
#
# Usage:
#   scripts/run_experiments.sh --tag qwen3.5-4b_seed1 [options]
#
# Options:
#   --tag TAG           required. Names configs/agents/<TAG>/*.json and the
#                        output root runs/agents/<TAG> (runs/agents/<TAG>_smoke
#                        with --smoke).
#   --env-file PATH      default .env
#   --env E[,E...]       restrict to environment(s): synthetic|tcas|physics.
#                        Repeatable and/or comma-separated. Default: all.
#   --policy P[,P...]    restrict to polic(y/ies): C0|C1|I1. Repeatable
#                        and/or comma-separated. Default: all.
#   -j N                 run up to N cells concurrently (default 3). The
#                        vLLM server is 4 data-parallel replicas x
#                        max-num-seqs 64, and each run has llm concurrency
#                        12, so -j 3..9 is a reasonable range.
#   --workers N          --workers passed to src.agent_system.shard, i.e.
#                        how many shards of ONE cell run in parallel
#                        (default 1: with a single pilot seed and --split
#                        seed this cannot help; --split degree or a later
#                        multi-seed run may raise it).
#   --split seed|degree  --split passed to src.agent_system.shard (default
#                        seed, unchanged behaviour: one process per task
#                        seed, each covering every degree). degree: one
#                        process per (degree, task_seed) pair, so
#                        --split degree --workers <#degrees> runs every
#                        degree of a cell at once against the local vLLM
#                        server.
#   --concurrency N      --concurrency passed to src.agent_system.shard,
#                        overriding policy.llm.concurrency in every shard
#                        config (default: unset, i.e. use the config's own
#                        value).
#   --smoke              run only one degree (see --smoke-degree) for the
#                        selected cells, into runs/agents/<TAG>_smoke instead
#                        of runs/agents/<TAG>. Generates a temporary config
#                        copy with degrees overridden; the real
#                        configs/agents/<TAG>/*.json files are untouched.
#   --smoke-degree D     degree to use in --smoke mode (default 8).
#   --force              re-run cells whose merged run directory already
#                        looks complete.
#   --dry-run            print the commands that would run, without running
#                        anything (including scripts/check_vllm.sh).
#
# Before running anything for real, this script (a) runs
# scripts/check_vllm.sh, (b) checks OPENAI_MODEL in --env-file is non-empty
# and was confirmed served by check_vllm.sh, and (c) checks every selected
# config has policy.kind == "llm_full", policy.llm.prompt_version == "v4"
# (if that key is present), and policy.llm.model equal to OPENAI_MODEL (if
# the config's model field is non-empty). Any violation aborts before any
# validate/shard call is made.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=""
ENV_FILE=".env"
JOBS=3
WORKERS=1
SPLIT="seed"
CONCURRENCY=""
SMOKE=0
SMOKE_DEGREE=8
DRY_RUN=0
FORCE=0
ENV_FILTERS=()
POLICY_FILTERS=()

usage() {
  sed -n '2,59p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --env)
      IFS=',' read -ra _parts <<< "$2"
      ENV_FILTERS+=("${_parts[@]}")
      shift 2
      ;;
    --policy)
      IFS=',' read -ra _parts <<< "$2"
      POLICY_FILTERS+=("${_parts[@]}")
      shift 2
      ;;
    -j) JOBS="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --split)
      SPLIT="$2"
      if [[ "$SPLIT" != "seed" && "$SPLIT" != "degree" ]]; then
        echo "error: --split must be 'seed' or 'degree', got '$SPLIT'" >&2
        exit 1
      fi
      shift 2
      ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --smoke-degree) SMOKE_DEGREE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$TAG" ]]; then
  echo "error: --tag is required, e.g. --tag qwen3.5-4b_seed1" >&2
  exit 1
fi

CONFIG_DIR="configs/agents/$TAG"
if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "error: config dir not found: $CONFIG_DIR" >&2
  echo "  run: python3 scripts/make_configs.py --model <id> --seed-index <I>" >&2
  exit 1
fi

CELLS=(synthetic tcas physics)
declare -A CELL_ENV=( [synthetic]=synthetic [tcas]=tcas [physics]=physics )
POLICIES=(C0 C1 I1)

env_selected() {
  local env="$1" x
  (( ${#ENV_FILTERS[@]} == 0 )) && return 0
  for x in "${ENV_FILTERS[@]}"; do [[ "$x" == "$env" ]] && return 0; done
  return 1
}
policy_selected() {
  local policy="$1" x
  (( ${#POLICY_FILTERS[@]} == 0 )) && return 0
  for x in "${POLICY_FILTERS[@]}"; do [[ "$x" == "$policy" ]] && return 0; done
  return 1
}

SELECTED=()
for cell in "${CELLS[@]}"; do
  env_selected "${CELL_ENV[$cell]}" || continue
  for policy in "${POLICIES[@]}"; do
    policy_selected "$policy" || continue
    SELECTED+=("${cell}:${policy}")
  done
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "error: no cells selected -- check --env/--policy filters" >&2
  exit 1
fi

echo "tag=$TAG  selected=${SELECTED[*]}"

if [[ $SMOKE -eq 1 ]]; then
  RUN_ROOT="runs/agents/${TAG}_smoke"
else
  RUN_ROOT="runs/agents/${TAG}"
fi
LOG_DIR="$RUN_ROOT/logs"

# -- 1. vLLM endpoint check ---------------------------------------------
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] would run: scripts/check_vllm.sh --env-file $ENV_FILE"
else
  echo "== scripts/check_vllm.sh =="
  if ! scripts/check_vllm.sh --env-file "$ENV_FILE"; then
    echo "error: vLLM endpoint check failed; refusing to launch any run" >&2
    exit 1
  fi
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: env file not found: $ENV_FILE" >&2
  exit 1
fi
OPENAI_MODEL_ENV=$(grep -E '^OPENAI_MODEL=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)
if [[ -z "$OPENAI_MODEL_ENV" ]]; then
  echo "error: OPENAI_MODEL is empty in $ENV_FILE" >&2
  exit 1
fi

# -- 2. LLM-only + model-consistency gate, every selected config --------
GATE_FAILED=0
for pair in "${SELECTED[@]}"; do
  cell="${pair%%:*}"; policy="${pair##*:}"
  cfg="$CONFIG_DIR/${cell}_${policy}.json"
  if [[ ! -f "$cfg" ]]; then
    echo "FAIL: missing config $cfg" >&2
    GATE_FAILED=1
    continue
  fi
  if ! gate_msg=$(python3 - "$cfg" "$OPENAI_MODEL_ENV" <<'PY'
import json, sys

cfg_path, expected_model = sys.argv[1], sys.argv[2]
data = json.load(open(cfg_path, encoding="utf-8"))
kind = (data.get("policy") or {}).get("kind")
if kind != "llm_full":
    print(f"policy.kind={kind!r} is not 'llm_full' (this pilot is LLM-only)")
    sys.exit(1)
llm = (data.get("policy") or {}).get("llm") or {}
prompt_version = llm.get("prompt_version")
if prompt_version is not None and prompt_version != "v4":
    print(f"policy.llm.prompt_version={prompt_version!r} != 'v4' "
          "(free communication requires llm_full + prompt_version v4)")
    sys.exit(1)
cfg_model = llm.get("model")
if cfg_model and cfg_model != expected_model:
    print(f"policy.llm.model={cfg_model!r} != OPENAI_MODEL={expected_model!r}")
    sys.exit(1)
PY
  ); then
    echo "FAIL: $cfg -- $gate_msg" >&2
    GATE_FAILED=1
  fi
done
if [[ $GATE_FAILED -ne 0 ]]; then
  echo "refusing to continue: one or more configs failed the LLM-only / model gate" >&2
  exit 1
fi
echo "gate ok: all selected configs are llm_full, prompt_version v4 (if set), model=$OPENAI_MODEL_ENV"

# -- 3. build the actual (possibly --smoke) config + run-id per cell ----
SMOKE_CONFIG_DIR="$RUN_ROOT/_configs"

resolve_run_cfg() {
  # Echoes the config path to actually validate/run for one cell:policy.
  local cell="$1" policy="$2" cfg="$3"
  if [[ $SMOKE -eq 0 ]]; then
    echo "$cfg"
    return
  fi
  local smoke_cfg="$SMOKE_CONFIG_DIR/${cell}_${policy}.json"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "$smoke_cfg"
    return
  fi
  mkdir -p "$SMOKE_CONFIG_DIR"
  python3 - "$cfg" "$smoke_cfg" "$SMOKE_DEGREE" <<'PY'
import json, sys
src, dst, degree = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(src, encoding="utf-8"))
data["degrees"] = [degree]
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
PY
  echo "$smoke_cfg"
}

run_id_of() {
  # The run id shard.py will use: config["name"], read without mutating it.
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$1"
}

run_is_complete() {
  # Exit 0 if the merged run dir under $1 for config $2 already has every
  # episode recorded with no failed shard.
  local merged_dir="$1" cfg="$2"
  [[ -f "$merged_dir/run_summary.json" && -f "$merged_dir/episodes.jsonl" ]] || return 1
  python3 - "$merged_dir" "$cfg" <<'PY'
import json, sys
from pathlib import Path

merged_dir, cfg_path = Path(sys.argv[1]), sys.argv[2]
config = json.load(open(cfg_path, encoding="utf-8"))
expected = len(config["degrees"]) * len(config["task_seeds"]) * config.get("repeats", 1)

summary = json.load(open(merged_dir / "run_summary.json", encoding="utf-8"))
if summary.get("failed_shards"):
    sys.exit(1)

episodes = (merged_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
episodes = [line for line in episodes if line.strip()]
sys.exit(0 if len(episodes) >= expected else 1)
PY
}

# -- 4. validate every selected config -----------------------------------
echo "== validate =="
VALIDATE_FAILED=0
declare -A RUN_CFG
declare -A RUN_ID
for pair in "${SELECTED[@]}"; do
  cell="${pair%%:*}"; policy="${pair##*:}"
  cfg="$CONFIG_DIR/${cell}_${policy}.json"
  run_cfg=$(resolve_run_cfg "$cell" "$policy" "$cfg")
  RUN_CFG["$pair"]="$run_cfg"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would run: python -m src.agent_system.cli validate --config $run_cfg --env-file $ENV_FILE"
    RUN_ID["$pair"]="${TAG}_${cell}_${policy}"
    continue
  fi
  RUN_ID["$pair"]=$(run_id_of "$run_cfg")
  if ! python -m src.agent_system.cli validate --config "$run_cfg" --env-file "$ENV_FILE"; then
    echo "FAIL: validate failed for $run_cfg (see output above)" >&2
    VALIDATE_FAILED=1
  fi
done
if [[ $DRY_RUN -eq 0 && $VALIDATE_FAILED -ne 0 ]]; then
  echo "refusing to run: one or more configs failed validate (see above)." >&2
  echo "Check the reported configuration and endpoint settings before retrying." >&2
  exit 1
fi

# -- 5. skip already-complete cells, unless --force ----------------------
# The completeness check only reads local files (no vLLM call), so it also
# runs under --dry-run for a non-smoke tag (its config path is a real,
# already-written configs/agents/<tag>/*.json file either way) -- letting
# --dry-run report which cells would be skipped as already complete. Under
# --smoke --dry-run the smoke config is never written to disk (see
# resolve_run_cfg above), so the check is skipped there to avoid failing on
# a missing file.
TO_RUN=()
for pair in "${SELECTED[@]}"; do
  cell="${pair%%:*}"; policy="${pair##*:}"
  run_cfg="${RUN_CFG[$pair]}"
  run_id="${RUN_ID[$pair]}"
  merged_dir="$RUN_ROOT/$run_id"
  if [[ $FORCE -eq 0 ]] \
      && { [[ $SMOKE -eq 0 ]] || [[ $DRY_RUN -eq 0 ]]; } \
      && run_is_complete "$merged_dir" "$run_cfg"; then
    echo "skip (already complete): $pair -> $merged_dir"
    continue
  fi
  TO_RUN+=("$pair")
done

if [[ ${#TO_RUN[@]} -eq 0 ]]; then
  echo "nothing to run: every selected cell already looks complete (use --force to re-run)."
  exit 0
fi

# -- 6. run shard.py per cell, up to -j concurrently ----------------------
mkdir -p "$LOG_DIR"

SHARD_EXTRA_ARGS=(--split "$SPLIT")
if [[ -n "$CONCURRENCY" ]]; then
  SHARD_EXTRA_ARGS+=(--concurrency "$CONCURRENCY")
fi

run_one() {
  local pair="$1" run_cfg="$2" log="$3" status_file="$4"
  if python -m src.agent_system.shard --config "$run_cfg" --env-file "$ENV_FILE" \
      --workers "$WORKERS" --output-root "$RUN_ROOT" "${SHARD_EXTRA_ARGS[@]}" > "$log" 2>&1; then
    echo 0 > "$status_file"
  else
    echo 1 > "$status_file"
  fi
}

STATUS_FILES=()
for pair in "${TO_RUN[@]}"; do
  cell="${pair%%:*}"; policy="${pair##*:}"
  run_cfg="${RUN_CFG[$pair]}"
  log="$LOG_DIR/${cell}_${policy}.log"
  status_file="$LOG_DIR/${cell}_${policy}.status"
  rm -f "$status_file"
  cmd="python -m src.agent_system.shard --config $run_cfg --env-file $ENV_FILE --workers $WORKERS --output-root $RUN_ROOT ${SHARD_EXTRA_ARGS[*]}"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would run ($cell/$policy, log: $log): $cmd"
    continue
  fi
  echo "launch: $cell/$policy -> $log"
  run_one "$pair" "$run_cfg" "$log" "$status_file" &
  STATUS_FILES+=("$status_file")
  while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do
    wait -n
  done
done
wait

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] no commands were executed."
  exit 0
fi

FAILED=0
echo "== results =="
for pair in "${TO_RUN[@]}"; do
  cell="${pair%%:*}"; policy="${pair##*:}"
  status_file="$LOG_DIR/${cell}_${policy}.status"
  log="$LOG_DIR/${cell}_${policy}.log"
  status=$(cat "$status_file" 2>/dev/null || echo 1)
  if [[ "$status" == "0" ]]; then
    echo "PASS: $cell/$policy (log: $log)"
  else
    echo "FAIL: $cell/$policy (log: $log)"
    tail -n 20 "$log" >&2 || true
    FAILED=1
  fi
done

exit $FAILED
