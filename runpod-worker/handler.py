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
import runpod
from llama_cpp import Llama

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/model.gguf")
N_CTX = int(os.environ.get("N_CTX", "8192"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))  # -1 = offload all layers
# Leave unset to use the chat template embedded in the GGUF metadata (works for
# both the Mistral and Qwen models). Override via env only if a model ships
# without one and replies look wrong.
CHAT_FORMAT = os.environ.get("CHAT_FORMAT") or None

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
