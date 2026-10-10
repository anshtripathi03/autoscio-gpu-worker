# Autoscio GPU worker: Chatterbox (voice cloning) + LTX-Video (video) on one GPU.
#
# Model WEIGHTS ARE NOT IN THE IMAGE. start.sh downloads them from Hugging Face on
# first boot into $MODELS_DIR (the Pod's /workspace volume), so the image stays
# ~10 GB and a restarted Pod reuses what it already downloaded.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements/ requirements/

# One venv per model: Chatterbox pins transformers 5.x, LTX-Video needs < 4.52.
# --system-site-packages reuses the base image's torch 2.6.0 instead of
# installing a second ~5 GB copy into each venv.
RUN python -m venv --system-site-packages /opt/venv-tts \
    && /opt/venv-tts/bin/pip install -r requirements/tts.txt

RUN python -m venv --system-site-packages /opt/venv-video \
    && /opt/venv-video/bin/pip install -r requirements/video.txt

# Fail the build (not the Pod) if either model package does not import.
RUN /opt/venv-tts/bin/python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS" \
    && /opt/venv-video/bin/python -c "from ltx_video.inference import create_ltx_video_pipeline"

RUN pip install -r requirements/api.txt

COPY app/ app/
# Import each runner in its own venv (no GPU needed: models load only at runtime),
# so a broken import fails this build instead of the Pod.
RUN /opt/venv-tts/bin/python -c "import app.runners.tts_runner" \
    && /opt/venv-video/bin/python -c "import app.runners.video_runner"
COPY start.sh /start.sh
RUN chmod +x /start.sh

ARG GIT_SHA=dev
# expandable_segments: two model processes share one GPU; this reduces fragmentation.
ENV GIT_SHA=${GIT_SHA} \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MODELS_DIR=/workspace/models \
    WORK_DIR=/tmp/jobs \
    PORT=8000

EXPOSE 8000
CMD ["/start.sh"]
