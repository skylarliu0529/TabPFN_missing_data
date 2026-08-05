#!/bin/bash
#SBATCH --job-name=w61_job
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=w61_job_%j.out
#SBATCH --error=w61_job_%j.err

set -euo pipefail

# =============================================
# CONFIGURABLE: edit defaults or override via env
# =============================================
# Python benchmark script to run:
#   w61_bench.py  - MAR/random holdout benchmark
#   w61_MNAR.py   - MNAR masking benchmark
PY_SCRIPT="${PY_SCRIPT:-w61_bench.py}"

# Space-separated list of target columns.
Y_COLS="${Y_COLS:-ELECT_TRUMP_W61}"
N_REPEATS="${N_REPEATS:-10}"

# MNAR-only options, used only when PY_SCRIPT=w61_MNAR.py.
MISSING_PROP="${MISSING_PROP:-0.10}"
MNAR_MECHANISM="${MNAR_MECHANISM:-logistic_xy}"
ALPHA="${ALPHA:-0.5}"
GAMMA="${GAMMA:-}"
INTERACTION_X_COL="${INTERACTION_X_COL:-}"
INTERACTION_Y_SCORE="${INTERACTION_Y_SCORE:-code}"
CLASS_Y_SCORE="${CLASS_Y_SCORE:-code}"
METHODS="${METHODS:-baseline,mice,rf,tabpfn}"

echo "PY_SCRIPT    = $PY_SCRIPT"
echo "Y_COLS       = $Y_COLS"
echo "N_REPEATS    = $N_REPEATS"
if [ "$PY_SCRIPT" = "w61_MNAR.py" ]; then
  echo "MISSING_PROP = $MISSING_PROP"
  echo "ALPHA        = $ALPHA"
  echo "GAMMA        = ${GAMMA:-<alpha>}"
  echo "MNAR_MECH    = $MNAR_MECHANISM"
  echo "X_INTERACT   = ${INTERACTION_X_COL:-standardized_linear_pred}"
  echo "Y_SCORE      = $INTERACTION_Y_SCORE"
  echo "CLASS_Y      = $CLASS_Y_SCORE"
  echo "METHODS      = $METHODS"
fi

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

echo "=============================="
echo "W61 benchmark: $PY_SCRIPT"
echo "=============================="

for YCOL in $Y_COLS; do
  echo ""
  echo "########################################"
  echo "# Y_COL = $YCOL"
  echo "########################################"

  case "$PY_SCRIPT" in
    w61_bench.py)
      python "$PY_SCRIPT" \
        --y_col "$YCOL" \
        --n_repeats "$N_REPEATS"
      ;;

    w61_MNAR.py)
      EXTRA_ARGS=()
      if [ -n "$GAMMA" ]; then
        EXTRA_ARGS+=(--gamma "$GAMMA")
      fi

      python "$PY_SCRIPT" \
        --y_col "$YCOL" \
        --n_repeats "$N_REPEATS" \
        --missing_prop "$MISSING_PROP" \
        --alpha "$ALPHA" \
        --mechanism "$MNAR_MECHANISM" \
        --interaction_x_col "$INTERACTION_X_COL" \
        --interaction_y_score "$INTERACTION_Y_SCORE" \
        --class_y_score "$CLASS_Y_SCORE" \
        --methods "$METHODS" \
        "${EXTRA_ARGS[@]}"
      ;;

    *)
      echo "[ERROR] Unsupported PY_SCRIPT: $PY_SCRIPT"
      echo "Use PY_SCRIPT=w61_bench.py or PY_SCRIPT=w61_MNAR.py"
      exit 2
      ;;
  esac
done

echo "DONE."

## Examples:
## sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",PY_SCRIPT=w61_bench.py,Y_COLS="PPLFAV_c_W61",N_REPEATS=50 run_w61_job.sh
## sbatch --export=ALL,HF_TOKEN="$HF_TOKEN",PY_SCRIPT=w61_MNAR.py,Y_COLS="ELECT_TRUMP_W61",N_REPEATS=10,ALPHA=0.5 run_w61_job.sh
