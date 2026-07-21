# REG2026 — single image, BOTH interfaces.
#   Interface 0 (Visual Grounding): ROI .jpeg -> CONCH feature -> generation model -> answer
#   Interface 1 (Workflow Reasoning): WSI .tiff -> CONCH features -> generation model -> chain-of-thought
#
# Pipeline: CONCH ViT-B-16 is used ONLY for visual feature extraction; answers
# come from the trained DeepSeek-R1-Distill-Qwen-1.5B (LoRA + Perceiver resampler).
#
# Weights are NOT baked into the image. They are mounted at /opt/ml/model from a
# separately uploaded model.tar.gz (see model/README.md). Do NOT add COPY model/.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # No network at runtime — load the bundled base LLM from disk.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # core.py + src/ live at /opt/app; vendored libs under lib/ and lib/clam.
    PYTHONPATH=/opt/app:/opt/app/lib:/opt/app/lib/clam

# System libraries for the WSI pipeline:
#   openslide-tools -> libopenslide (openslide-python reads WSI regions)
#   libvips         -> pyvips (convert non-pyramidal slides to pyramidal)
#   libgl1/glib     -> opencv-headless / matplotlib runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        build-essential pkg-config libffi-dev \
        openslide-tools libvips libvips-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

# Torch / torchvision: CUDA 12.8 wheels, matching the training environment.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchvision==0.23.0

# Remaining Python dependencies.
COPY requirements.txt /opt/app/requirements.txt
RUN python -m pip install -r /opt/app/requirements.txt

# Run as a non-root user (uid 1000), matching Grand Challenge conventions.
RUN groupadd -r user && useradd -m -r -g user -u 1000 user

WORKDIR /opt/app

# Code ONLY (weights arrive via /opt/ml/model at run time).
COPY --chown=user:user core.py inference.py /opt/app/
COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user lib/ /opt/app/lib/

USER user

ENTRYPOINT ["python", "inference.py"]
