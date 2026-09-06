FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git build-essential cmake ninja-build curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build llama-cpp-python with CUDA (cuBLAS) support so inference runs on GPU
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model is NOT baked into the image (27B Q4_K_M = ~16 GB, too slow to push from
# a home connection). Attach a RunPod Network Volume with the GGUF at
# /runpod-volume/model.gguf instead — see runpod-worker/UPLOAD_MODEL.md.
ENV MODEL_PATH="/runpod-volume/model.gguf"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
