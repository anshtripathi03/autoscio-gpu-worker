#!/usr/bin/env bash
# Pod entrypoint. Weights download on first boot (10-20 min), then load into the GPU.
set -euo pipefail

: "${MODELS_DIR:=/workspace/models}"
: "${PORT:=8000}"

if [ -z "${WORKER_API_KEY:-}" ] && [ "${ALLOW_INSECURE_NO_AUTH:-0}" != "1" ]; then
  echo "FATAL: WORKER_API_KEY is not set. Add it to the Pod's environment variables." >&2
  exit 1
fi

mkdir -p "$MODELS_DIR/hf" "${WORK_DIR:-/tmp/jobs}"
# Both model venvs share one Hugging Face cache on the persistent volume, so a
# Pod restart skips the download.
export HF_HOME="$MODELS_DIR/hf"
export HF_HUB_DISABLE_TELEMETRY=1

free_gb=$(df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -lt 40 ] && [ ! -d "$HF_HOME/hub/models--Lightricks--LTX-Video" ]; then
  echo "WARNING: only ${free_gb} GB free in $MODELS_DIR; the models need ~35 GB." >&2
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: no GPU visible" >&2
echo "Autoscio GPU worker ${GIT_SHA:-dev} starting on :$PORT (models in $MODELS_DIR)"

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
