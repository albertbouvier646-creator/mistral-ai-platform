FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git build-essential cmake ninja-build curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build llama-cpp-python with CUDA (cuBLAS) support so inference runs on GPU.
# CI runners have no GPU/driver, so provide the CUDA driver *stub* lib (ships in
# the devel image for exactly this cross-compile case) so the linker can
# resolve libcuda symbols for the llava example binaries.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}"
ENV CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF"
ENV FORCE_CMAKE=1

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model is NOT baked into the image (27B Q4_K_M = ~16 GB, too slow to push from
# a home connection). Attach a RunPod Network Volume with the GGUF at
# /runpod-volume/model.gguf instead — see runpod-worker/UPLOAD_MODEL.md.
ENV MODEL_PATH="/runpod-volume/model.gguf"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
