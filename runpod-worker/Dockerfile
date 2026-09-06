FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Prebuilt CUDA wheel from the llama-cpp-python project — no compiler needed,
# no CUDA driver required at build time (that's what broke the from-source
# build on GPU-less CI runners: linking llava-cli needs libcuda.so.1).
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    --only-binary llama-cpp-python

# Model is NOT baked into the image (27B Q4_K_M = ~16 GB, too slow to push from
# a home connection). Attach a RunPod Network Volume with the GGUF at
# /runpod-volume/model.gguf instead.
ENV MODEL_PATH="/runpod-volume/model.gguf"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
