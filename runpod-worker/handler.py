"""
RunPod Serverless worker — GGUF-agnostic (Mistral-Nemo-12B or Qwen-27B).

Loads the GGUF once per cold start from a RunPod Network Volume
(llama.cpp on GPU via cuBLAS), then answers RunPod jobs. Supports streaming
(generator handler) so the FastAPI backend can relay tokens over SSE.

Job input shape:
{
  "input": {
    "messages": [{"role": "system|user|assistant", "content": "..."}],
    "max_tokens": 1024,
    "temperature": 0.7,
    "stream": true
  }
}
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # avoid xet's transient 2x-disk-space download

import runpod
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/model.gguf")
HF_REPO = os.environ.get("HF_REPO", "0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF")
HF_FILE = os.environ.get("HF_FILE", "RVN-Q4_K_M-mtp.gguf")
N_CTX = int(os.environ.get("N_CTX", "8192"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))  # -1 = offload all layers
# Leave unset to use the chat template embedded in the GGUF metadata (works for
# both the Mistral and Qwen models). Override via env only if a model ships
# without one and replies look wrong.
CHAT_FORMAT = os.environ.get("CHAT_FORMAT") or None

# The network volume is empty on a fresh endpoint — pull the GGUF straight from
# Hugging Face into it on first cold start, then every later cold start reuses
# the cached file instead of re-downloading.
if not os.path.exists(MODEL_PATH):
    print(f"[worker] {MODEL_PATH} missing, downloading {HF_REPO}/{HF_FILE} ...", flush=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    downloaded_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, local_dir=os.path.dirname(MODEL_PATH))
    if downloaded_path != MODEL_PATH:
        os.replace(downloaded_path, MODEL_PATH)
    print("[worker] model download complete", flush=True)

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    chat_format=CHAT_FORMAT,
)


def handler(job):
    job_input = job["input"]
    messages = job_input["messages"]
    max_tokens = job_input.get("max_tokens", 1024)
    temperature = job_input.get("temperature", 0.7)
    stream = job_input.get("stream", True)

    completion = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )

    if not stream:
        yield completion
        return

    for chunk in completion:
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            yield delta


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
