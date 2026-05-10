#!/usr/bin/env bash
set -euo pipefail

# Runs:
# - Main experiments (two datasets x five models): gcomp, ltmle, DeepACE, DLTMLE_correct_separate, PEQ_Net
# - Functional experiments (two datasets x two models): DLTMLE_correct_separate, PEQ_Net
#
# Usage (from repo root):
#   bash pipelines/pipelines.sh
#
# Customize:
#   N_REPEATS=10 EXP_SEED_START=300 MAX_JOBS=4 N_THREADS=4 bash pipelines/pipelines.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

N_REPEATS="${N_REPEATS:-10}"
EXP_SEED_START="${EXP_SEED_START:-300}"
MAX_JOBS="${MAX_JOBS:-4}"
N_THREADS="${N_THREADS:-4}"

LOGDIR="${LOGDIR:-logs}"
mkdir -p "$LOGDIR"

safe () { echo "$1" | tr '/: =+' '____'; }

# Usage: launch_task <job_name> <threads> <python ...args>
launch_task () {
  local name="$1"; shift
  local threads="$1"; shift

  (
    export OMP_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads"
    export OPENBLAS_NUM_THREADS="$threads"
    export NUMEXPR_NUM_THREADS="$threads"
    export VECLIB_MAXIMUM_THREADS="$threads"

    local logfile="$LOGDIR/$(safe "$name").log"
    echo "[START $(date -Is)] $name threads=$threads cmd: python $*"
    echo "[LOG  $(date -Is)] $logfile"
    python -u "$@" >"$logfile" 2>&1
    echo "[DONE  $(date -Is)] $name"
  ) &

  while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
    wait -n
  done
}

run_main_for_dataset () {
  local ds="$1"
  local seed="$2"

  # launch_task "gcomp_${ds}_seed${seed}" "$N_THREADS" pipelines/train_ltmle.py +dataset="${ds}" +model=gcomp exp.seed="${seed}"
  # launch_task "ltmle_${ds}_seed${seed}" "$N_THREADS" pipelines/train_ltmle.py +dataset="${ds}" +model=ltmle exp.seed="${seed}"
  launch_task "deepace_${ds}_seed${seed}" "$N_THREADS" pipelines/train_deepace.py +dataset="${ds}" +model=deepace exp.seed="${seed}"
  launch_task "sep_${ds}_seed${seed}" "$N_THREADS" pipelines/train_dltmle_correct_separate_01.py +dataset="${ds}" +model=dltmle_correct_separate exp.seed="${seed}"
  launch_task "peq_${ds}_seed${seed}" "$N_THREADS" pipelines/train_peq_net_e2e_01.py +dataset="${ds}" +model=peq_net_e2e_01 exp.seed="${seed}"
}

run_func_for_dataset () {
  local ds="$1"
  local seed="$2"

  launch_task "sep_func_${ds}_seed${seed}" "$N_THREADS" pipelines/train_dltmle_correct_separate_func.py +dataset="${ds}" +model=dltmle_correct_separate exp.seed="${seed}"
  launch_task "peq_func_${ds}_seed${seed}" "$N_THREADS" pipelines/train_peq_net_e2e_func.py +dataset="${ds}" +model=peq_net_e2e_func exp.seed="${seed}"
}

for i in $(seq 0 $((N_REPEATS - 1))); do
  EXP_SEED=$((EXP_SEED_START + i))
  echo "=== Enqueue repeat $((i + 1))/${N_REPEATS}: exp_seed=${EXP_SEED} ==="

  # Main experiments: "Limited" and "Expanded"
  run_main_for_dataset "mimic_extract" "${EXP_SEED}"
  run_main_for_dataset "mimic_extract_complex" "${EXP_SEED}"

  # Functional experiments: "Limited" and "Expanded"
  run_func_for_dataset "mimic_extract_func" "${EXP_SEED}"
  run_func_for_dataset "mimic_extract_complex_func" "${EXP_SEED}"
done

wait
echo "All tasks finished."


