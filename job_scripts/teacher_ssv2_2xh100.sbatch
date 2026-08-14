#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=salt_teacher_ssv2
#SBATCH --output=/scratch/sd6701/salt/p-salt/logs/teacher_ssv2/teacher_ssv2_%j.log
#SBATCH --error=/scratch/sd6701/salt/p-salt/logs/teacher_ssv2/teacher_ssv2_%j.err
#SBATCH --requeue                # permit self-requeue after a crash (see torchrun wrapper)
#SBATCH --open-mode=append       # requeued attempt appends to the SAME %j log -- required
                                 # for the RESUME_RUN_ID recovery grep below

WORKDIR=/scratch/sd6701/salt/p-salt
NPROC=2   # GPUs per node; keep in sync with --gres above

# On a requeued attempt (same job id; --open-mode=append kept the log), recover
# the run uuid from our own earlier output so training resumes from the last
# checkpoint instead of minting a fresh run dir and restarting at epoch 0.
# An explicitly passed RESUME_RUN_ID always wins.
if [[ -z "${RESUME_RUN_ID:-}" && "${SLURM_RESTART_COUNT:-0}" -gt 0 ]]; then
  RESUME_RUN_ID=$(grep -oE 'run_dir=results/teacher_ssv2/checkpoints/[a-f0-9]{8}' \
    "${WORKDIR}/logs/teacher_ssv2/teacher_ssv2_${SLURM_JOB_ID}.log" 2>/dev/null \
    | tail -1 | awk -F/ '{print $NF}') || true
  echo "requeue restart ${SLURM_RESTART_COUNT:-0}: recovered RESUME_RUN_ID=${RESUME_RUN_ID:-none}"
fi

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate salt

export LD_LIBRARY_PATH=/scratch/sd6701/miniconda3/envs/salt/lib:$LD_LIBRARY_PATH
export WANDB_MODE=online
export WANDB__SERVICE_WAIT=300

mkdir -p "${WORKDIR}/logs/teacher_ssv2"

export VJEPA_DEBUG_MASK_COUNTS=1
export VJEPA_VIDEO_BACKEND=auto

# -- Profiling (TrainProfiler in app/rethinking_jepa/profiling.py).
# PROFILE_MODE is the SINGLE switch and drives both the profiler and the launch:
#   off   -> disabled (default; normal training run, requeue path unchanged)
#   torch -> torch.profiler runs inside the process; trace -> <run_dir>/profiling/
#   nsys  -> external nsys wraps the launch (see the conditional below) + move step
# Enable at submit time: sbatch --export=ALL,PROFILE_MODE=nsys submit.sbatch
export PROFILE_MODE="${PROFILE_MODE:-off}"
export PROFILE_START_EPOCH="${PROFILE_START_EPOCH:-2}"  # 0-based: dormant until epoch idx 2 (the 3rd pass); use 1 for the 2nd
export PROFILE_WAIT="${PROFILE_WAIT:-5}"                # skip first 5 steps of that epoch
export PROFILE_WARMUP="${PROFILE_WARMUP:-3}"            # torch mode only; nsys window opens at step `wait`
export PROFILE_ACTIVE="${PROFILE_ACTIVE:-10}"           # record 10 steps
export PROFILE_EXIT_AFTER_STEPS="${PROFILE_EXIT_AFTER_STEPS:-20}"  # clean sys.exit(0) after capture (>= wait+warmup+active)

cd "${WORKDIR}"

# SSv2 teacher. data.manifest points at the raw Qualcomm labels/train.json;
# video_dataset's .json branch resolves ids to 20bn-something-something-v2/<id>.webm
# at load time, so there is no manifest to generate first.
# 2xH100 run: two ranks via torchrun; teacher_train wraps the model in DDP.
# ipe = floor(168913 / (per_rank_batch * 2)) -- with batch 64/rank: 1319.
# NOTE: 2 ranks doubles the GLOBAL batch vs the 1-GPU run unless the per-rank
# batch in configs/teacher_ssv2.yaml is halved.
#
# RESUME: teacher_train.py writes teacher_train_state.pt EVERY epoch, so a crashed
# or timed-out run continues from where it stopped -- but ONLY if it reuses the
# same run dir (a fresh uuid restarts at epoch 0). The requeue block above
# recovers the uuid automatically; to resume by hand, pass the prior run's uuid
# (printed as `run_dir=...` / `random_run_id=...` in its log):
#
#   sbatch --export=ALL,RESUME_RUN_ID=<uuid> submit.sbatch
#
# Leave RESUME_RUN_ID unset for a fresh run.
LAUNCH_ARGS=(--config configs/teacher_ssv2.yaml)
if [[ -n "${RESUME_RUN_ID:-}" ]]; then
  echo "resuming run_id=${RESUME_RUN_ID} (run_dir=${WORKDIR}/results/teacher_ssv2/checkpoints/${RESUME_RUN_ID})"
  LAUNCH_ARGS+=(--resume-preempt --run-id "${RESUME_RUN_ID}")
fi

if [[ "${PROFILE_MODE}" == "nsys" ]]; then
  # nsys mode: external nsys records ONLY between the profiler's cudaProfilerStart/
  # Stop markers (opened at step `wait` of PROFILE_START_EPOCH), so the .nsys-rep is
  # a bounded steady-state slice. Profiling runs never requeue: exit_after_steps
  # ends the job on purpose, and a requeued profiling job would waste the allocation.
  mkdir -p "${WORKDIR}/logs/nsys"
  # nsys must be on PATH (ships with CUDA). Uncomment if your cluster needs a module:
  # module load nsight-systems 2>/dev/null || module load cuda 2>/dev/null || true
  export PROFILE_RUN_DIR_FILE="${WORKDIR}/logs/nsys/rundir_${SLURM_JOB_ID}.txt"
  NSYS_REPORT="${WORKDIR}/logs/nsys/teacher_ssv2_${SLURM_JOB_ID}.nsys-rep"
  if ! nsys profile \
    -c cudaProfilerApi \
    --trace=cuda,nvtx,cudnn,cublas \
    --force-overwrite=true \
    -o "${WORKDIR}/logs/nsys/teacher_ssv2_${SLURM_JOB_ID}" \
    torchrun --standalone --nproc_per_node="${NPROC}" app/main.py "${LAUNCH_ARGS[@]}"; then
    echo "nsys/torchrun failed; profiling run NOT requeued"
    exit 1
  fi
  # Relocate the finished .nsys-rep into the run_id folder (beside last/), using the
  # sentinel file the profiler wrote (nsys's -o path is fixed before run_id exists).
  if [[ -f "${PROFILE_RUN_DIR_FILE}" && -f "${NSYS_REPORT}" ]]; then
    TARGET_RUN_DIR="$(cat "${PROFILE_RUN_DIR_FILE}")"
    if [[ -d "${TARGET_RUN_DIR}" ]]; then
      mv "${NSYS_REPORT}" "${TARGET_RUN_DIR}/"
      rm -f "${PROFILE_RUN_DIR_FILE}"
      echo "[sbatch] moved nsys report -> ${TARGET_RUN_DIR}/$(basename "${NSYS_REPORT}")"
    else
      echo "[sbatch] run_dir '${TARGET_RUN_DIR}' not found; report left at ${NSYS_REPORT}" >&2
    fi
  else
    echo "[sbatch] sentinel or report missing; report (if any) left at ${NSYS_REPORT}" >&2
  fi
else
  # Full training state is written every epoch, so a requeue re-costs at most ~1
  # epoch, not the whole allocation. Capped at 3 restarts so a deterministic crash
  # (bad config, import error, missing data) cannot requeue-loop forever.
  if ! torchrun --standalone --nproc_per_node="${NPROC}" app/main.py "${LAUNCH_ARGS[@]}"; then
    echo "torchrun failed (restart count ${SLURM_RESTART_COUNT:-0})"
    if [[ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]]; then
      echo "requeueing job ${SLURM_JOB_ID}"
      scontrol requeue "${SLURM_JOB_ID}"
    else
      echo "restart limit reached; NOT requeueing"
    fi
    exit 1
  fi
fi
