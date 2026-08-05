#!/bin/bash
#SBATCH --job-name=w61_tilted
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=w61_tilted_%j.out
#SBATCH --error=w61_tilted_%j.err

set -euo pipefail

# =============================================
# CONFIGURABLE: edit defaults or override via env
# =============================================
Y_COL="${Y_COL:-TRUMPISSUE1_W61}"
N_REPEATS="${N_REPEATS:-10}"

# Four separate true-gamma simulation scenarios.
# Slurm --export uses commas to separate variables, so either:
#   1. export GAMMAS="2 1 0.5 0" before sbatch and pass GAMMAS by name; or
#   2. use a colon-separated one-liner: GAMMAS=2:1:0.5:0
GAMMAS="${GAMMAS:-1 0.5 0.25 0}"
GAMMAS="${GAMMAS//:/ }"

# Artificial missingness design.
MISSING_PROP="${MISSING_PROP:-0.10}"
VALIDATION_PROP="${VALIDATION_PROP:-0.15}"

# Kim-Yu gamma estimation controls. Use "all" for the four-method comparison.
GAMMA_ESTIMATOR="${GAMMA_ESTIMATOR:-all}"
GAMMA_BRACKET="${GAMMA_BRACKET:--3,3}"
H_QUANTILES="${H_QUANTILES:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90}"
OUT_DIR="${OUT_DIR:-}"

echo "Y_COL           = $Y_COL"
echo "N_REPEATS       = $N_REPEATS"
echo "GAMMAS          = $GAMMAS"
echo "MISSING_PROP    = $MISSING_PROP"
echo "VALIDATION_PROP = $VALIDATION_PROP"
echo "GAMMA_ESTIMATOR = $GAMMA_ESTIMATOR"
echo "GAMMA_BRACKET   = $GAMMA_BRACKET"
echo "H_QUANTILES     = $H_QUANTILES"
echo "OUT_DIR         = ${OUT_DIR:-<default>}"

cd /lustre/hdd/LAS/zhuz-lab/siyangl/TabPFN
source .venv/bin/activate

# Force unbuffered Python output so .out file updates in real time.
export PYTHONUNBUFFERED=1

# Optional safety: if TabPFN ever falls back to CPU, allow it.
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

# Reduce CUDA memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# Put caches on Lustre.
export HF_HOME=/lustre/hdd/LAS/zhuz-lab/siyangl/.cache/huggingface
export TABPFN_MODEL_CACHE_DIR=/lustre/hdd/LAS/zhuz-lab/siyangl/.cache/tabpfn
mkdir -p "$HF_HOME" "$TABPFN_MODEL_CACHE_DIR"

# Ensure huggingface_hub CLI exists (hf).
python - << 'EOF'
import importlib.util, sys
ok = importlib.util.find_spec("huggingface_hub") is not None
print("huggingface_hub installed:", ok)
sys.exit(0 if ok else 1)
EOF
if [ $? -ne 0 ]; then
  pip install -U huggingface_hub
fi

# Non-interactive HF login (pass HF_TOKEN via sbatch --export).
if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
  echo "Hugging Face login done (non-interactive)."
else
  echo "[WARNING] HF_TOKEN not set. TabPFN HuggingFace download may fail."
fi

# GPU sanity check.
python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

EXTRA_ARGS=()
if [ -n "$OUT_DIR" ]; then
  EXTRA_ARGS+=(--out_dir "$OUT_DIR")
fi

echo "=============================="
echo "W61 Kim-Yu tilted MNAR benchmark"
echo "=============================="

python w61_MNAR_tilted.py \
  --y_col "$Y_COL" \
  --gammas "$GAMMAS" \
  --n_repeats "$N_REPEATS" \
  --missing_prop "$MISSING_PROP" \
  --validation_prop "$VALIDATION_PROP" \
  --gamma_estimator "$GAMMA_ESTIMATOR" \
  --gamma_bracket="$GAMMA_BRACKET" \
  --h_quantiles "$H_QUANTILES" \
  "${EXTRA_ARGS[@]}"

echo "DONE."

## Example:
## sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",N_REPEATS=10 run_w61_MNAR_tilted.sh
##
## Slurm-safe one-liner for a custom gamma grid:
## sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",N_REPEATS=50,GAMMA_ESTIMATOR=all,GAMMAS=2:1:0.5:0 run_w61_MNAR_tilted.sh
##
## Alternative:
## export GAMMAS="2 1 0.5 0"
## sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",N_REPEATS=50,GAMMA_ESTIMATOR=all,GAMMAS run_w61_MNAR_tilted.sh
