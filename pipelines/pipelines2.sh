#!/usr/bin/env bash
set -euo pipefail
#
# MIMIC-Extract experiment launcher (PyTorch trainers, then train_ltmle_capo / R).
# Repo root:  N_REPEATS=20 EXP_SEED_START=1600 MAX_JOBS=4 N_THREADS=8 bash pipelines/pipelines2.sh
# R stage:     MAX_JOBS_R=20 N_THREADS_R=1 (defaults shown)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

N_REPEATS="${N_REPEATS:-20}"
EXP_SEED_START="${EXP_SEED_START:-1600}"
MAX_JOBS="${MAX_JOBS:-4}"
N_THREADS="${N_THREADS:-8}"
MAX_JOBS_R="${MAX_JOBS_R:-20}"
N_THREADS_R="${N_THREADS_R:-1}"
LOGDIR="${LOGDIR:-logs}"
mkdir -p "$LOGDIR"

safe () { echo "$1" | tr '/: =+' '____'; }

# launch <max_parallel_jobs> <log_basename> <OMP_threads> <python> [args...]
launch () {
  local max_parallel="$1"; shift
  local name="$1"; shift
  local threads="$1"; shift

  (
    export OMP_NUM_THREADS="$threads"
    export MKL_NUM_THREADS="$threads"
    export OPENBLAS_NUM_THREADS="$threads"
    export NUMEXPR_NUM_THREADS="$threads"
    export VECLIB_MAXIMUM_THREADS="$threads"
    echo "[START $(date -Is)] $name"
    # Lightning’s tqdm progress bar uses stderr; keep stdout (Hydra / prints) in the log.
    python -u "$@"
    echo "[DONE  $(date -Is)] $name"
  ) &

  while (( $(jobs -pr | wc -l) >= max_parallel )); do
    wait -n || true
  done
}

for i in $(seq 0 $((N_REPEATS - 1))); do
  s=$((EXP_SEED_START + i))
  echo "=== PyTorch $((i + 1))/${N_REPEATS} seed=${s} ==="

  # Main experiments
  for numz in 0 5; do
    [[ "$numz" -eq 0 ]] && ds=mimic_extract || ds=mimic_extract_complex
    launch "$MAX_JOBS" "dltmle_01_${ds}_nz${numz}_seed${s}" "$N_THREADS" \
      pipelines/train_dltmle_correct_01.py "+dataset=${ds}" "+model=dltmle_correct_01_tuned_numz${numz}" "exp.seed=${s}"
    launch "$MAX_JOBS" "peq_01_${ds}_nz${numz}_seed${s}" "$N_THREADS" \
      pipelines/train_peq_net_01.py "+dataset=${ds}" "+model=peq_net_01_tuned_numz${numz}" "exp.seed=${s}"
  done

  for numz in 0 5; do
    [[ "$numz" -eq 0 ]] && ds=mimic_extract_func_stepwise || ds=mimic_extract_complex_func_stepwise
    launch "$MAX_JOBS" "dltmle_fs_${ds}_nz${numz}_seed${s}" "$N_THREADS" \
      pipelines/train_dltmle_correct_func_stepwise.py "+dataset=${ds}" "+model=dltmle_correct_func_tuned_numz${numz}" "exp.seed=${s}"
    launch "$MAX_JOBS" "peq_fs_${ds}_nz${numz}_seed${s}" "$N_THREADS" \
      pipelines/train_peq_net_func_stepwise.py "+dataset=${ds}" "+model=peq_net_func_tuned_numz${numz}" "exp.seed=${s}"

  done

  for numz in 0 5; do
    [[ "$numz" -eq 0 ]] && ds=mimic_extract_func_stepwise_04_05_06_all || ds=mimic_extract_complex_func_stepwise_04_05_06_all
    launch "$MAX_JOBS" "peq_fs_${ds}_seed${s}" "$N_THREADS" \
      pipelines/train_peq_net_func_stepwise.py "+dataset=${ds}" "+model=peq_net_func_tuned_numz${numz}" "exp.seed=${s}"
  done

  for numz in 0 5; do
    [[ "$numz" -eq 0 ]] && ds=mimic_extract_func_stepwise_04_05_06_all || ds=mimic_extract_complex_func_stepwise_0_04_05_06_1_all
    launch "$MAX_JOBS" "dltmle_fs_${ds}_seed${s}" "$N_THREADS" \
      pipelines/train_dltmle_correct_func_stepwise.py "+dataset=${ds}" "+model=dltmle_correct_func_tuned_numz${numz}" "exp.seed=${s}"
  done

  launch "$MAX_JOBS" "deepace_mimic_extract_nz0_seed${s}" "$N_THREADS" \
    pipelines/train_deepace.py +dataset=mimic_extract +model=deepace_tuned_numz0 "exp.seed=${s}"
  launch "$MAX_JOBS" "deepace_mimic_extract_complex_nz5_seed${s}" "$N_THREADS" \
    pipelines/train_deepace.py +dataset=mimic_extract_complex +model=deepace_tuned_numz5 "exp.seed=${s}"

  # Ablation studies
  launch "$MAX_JOBS" "dltmle_finetune_fs_cpx_nz5_seed${s}" "$N_THREADS" \
    pipelines/train_dltmle_correct_func_stepwise_finetune.py \
    +dataset=mimic_extract_complex_func_stepwise +model=dltmle_correct_finetune_func_numz5 "exp.seed=${s}"
  launch "$MAX_JOBS" "dltmle_multiq_fs_cpx_nz5_seed${s}" "$N_THREADS" \
    pipelines/train_dltmle_correct_multiQhead_stepwise.py \
    +dataset=mimic_extract_complex_func_stepwise +model=dltmle_correct_multiQhead_func_numz5 "exp.seed=${s}"

  launch "$MAX_JOBS" "peq_fs_cpx_0_05_1_seed${s}" "$N_THREADS" \
    pipelines/train_peq_net_func_stepwise.py +dataset=mimic_extract_complex_func_stepwise_0_05_1 +model=peq_net_func_tuned_numz5 "exp.seed=${s}"
  launch "$MAX_JOBS" "peq_fs_cpx_0_04_05_06_1_seed${s}" "$N_THREADS" \
    pipelines/train_peq_net_func_stepwise.py +dataset=mimic_extract_complex_func_stepwise_0_04_05_06_1 +model=peq_net_func_tuned_numz5 "exp.seed=${s}"
  launch "$MAX_JOBS" "dltmle_fs_cpx_0_04_05_06_1_seed${s}" "$N_THREADS" \
    pipelines/train_dltmle_correct_func_stepwise.py +dataset=mimic_extract_complex_func_stepwise_0_04_05_06_1 +model=dltmle_correct_func_tuned_numz5 "exp.seed=${s}"

  launch "$MAX_JOBS" "peq_fs_cpx_0_05_1_all_seed${s}" "$N_THREADS" \
    pipelines/train_peq_net_func_stepwise.py +dataset=mimic_extract_complex_func_stepwise_0_05_1_all +model=peq_net_func_tuned_numz5 "exp.seed=${s}"
  launch "$MAX_JOBS" "peq_fs_cpx_0_04_05_06_1_all_seed${s}" "$N_THREADS" \
    pipelines/train_peq_net_func_stepwise.py +dataset=mimic_extract_complex_func_stepwise_0_04_05_06_1_all +model=peq_net_func_tuned_numz5 "exp.seed=${s}"
  
done

wait || true

for i in $(seq 0 $((N_REPEATS - 1))); do
  s=$((EXP_SEED_START + i))
  echo "=== R (ltmle + gcomp) $((i + 1))/${N_REPEATS} seed=${s} ==="

  launch "$MAX_JOBS_R" "ltmle_mimic_extract_seed${s}" "$N_THREADS_R" \
    pipelines/train_ltmle_capo.py +dataset=mimic_extract +model=ltmle "exp.seed=${s}"
  launch "$MAX_JOBS_R" "ltmle_mimic_extract_complex_seed${s}" "$N_THREADS_R" \
    pipelines/train_ltmle_capo.py +dataset=mimic_extract_complex +model=ltmle "exp.seed=${s}"
  launch "$MAX_JOBS_R" "gcomp_mimic_extract_seed${s}" "$N_THREADS_R" \
    pipelines/train_ltmle_capo.py +dataset=mimic_extract +model=gcomp "exp.seed=${s}"
  launch "$MAX_JOBS_R" "gcomp_mimic_extract_complex_seed${s}" "$N_THREADS_R" \
    pipelines/train_ltmle_capo.py +dataset=mimic_extract_complex +model=gcomp "exp.seed=${s}"
done

wait || true
echo "All tasks finished."
