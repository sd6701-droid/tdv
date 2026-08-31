#!/bin/bash
# Download the released TDV checkpoints from the Hugging Face Hub onto the HPC.
#
#   bash download_tdv_checkpoints.sh            # download into $DEST
#   bash download_tdv_checkpoints.sh --list     # just list what's in the repo
#   DEST=/path/to/dir bash download_tdv_checkpoints.sh
#
# Checkpoints are multi-GB: keep them on /scratch, never in $HOME (small quota).

set -euo pipefail

REPO_ID="${REPO_ID:-ninaddaithankar/tdv}"
DEST="${DEST:-/scratch/sd6701/tdv/pretrained}"

# Point the HF cache at scratch too -- the default ~/.cache/huggingface will
# blow through a home quota, and snapshot_download stages files there first.
export HF_HOME="${HF_HOME:-/scratch/sd6701/hf_cache}"
mkdir -p "${HF_HOME}"

source /scratch/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate tdv

if [[ "${1:-}" == "--list" ]]; then
    python - "$REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi
repo = sys.argv[1]
files = HfApi().list_repo_files(repo)
ckpts = [f for f in files if f.startswith("checkpoints/")]
print(f"{len(ckpts)} file(s) under checkpoints/ in {repo}:")
for f in sorted(ckpts):
    print("  ", f)
PY
    exit 0
fi

mkdir -p "${DEST}"
echo "repo : ${REPO_ID}"
echo "dest : ${DEST}"
echo "cache: ${HF_HOME}"

python - "$REPO_ID" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo,
    repo_type="model",
    allow_patterns=["checkpoints/**"],   # skip README/configs at the repo root
    local_dir=dest,
    max_workers=8,
    resume_download=True,               # safe to re-run after an interruption
)
print(f"\ndownloaded to: {path}")
PY

echo
echo "--- contents ---"
find "${DEST}/checkpoints" -type f -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'
