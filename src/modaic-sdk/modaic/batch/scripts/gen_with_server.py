from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

APP_NAME = "modaic-gpt-oss-server-test"
MODEL_ID = "openai/gpt-oss-120b"
REASONING_PARSER = "openai_gptoss"
GPU_CONFIG = "H200:4"
SERVER_PORT = 8000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

REQUEST_BODY = {
    "model": MODEL_ID,
    "messages": [
        {
            "role": "user",
            "content": "What is 17 times 19? Think carefully, then give the final answer.",
        }
    ],
    "max_completion_tokens": 512,
    "temperature": 0.0,
    "include_reasoning": True,
    "chat_template_kwargs": {"enable_thinking": True},
}

app = modal.App(APP_NAME)

image = modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12").uv_pip_install(
    "vllm>=0.17.0",
    "flashinfer-python",
    "torch>=2.10,<2.11",
    "datasets",
    "huggingface-hub",
    "hf_transfer",
    "hf-xet>=1.1.7",
    "tqdm",
    "transformers>=4.50,<5",
    "python-dotenv",
    "pandas",
)


def _wait_for_server(timeout_seconds: int = 900) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Timed out waiting for vLLM server readiness")


def _post_json(path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{SERVER_URL}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


@app.function(gpu=GPU_CONFIG, image=image, timeout=60 * 60 * 2)
def run_server_test() -> None:
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--reasoning-parser",
        REASONING_PARSER,
    ]
    print("Starting:", " ".join(cmd), flush=True)
    server = subprocess.Popen(cmd, env=env)

    try:
        _wait_for_server()
        print("\n=== request ===", flush=True)
        print(json.dumps(REQUEST_BODY, indent=2), flush=True)

        models_response = json.loads(
            urllib.request.urlopen(f"{SERVER_URL}/v1/models", timeout=30).read().decode("utf-8")
        )
        print("\n=== /v1/models ===", flush=True)
        print(json.dumps(models_response, indent=2), flush=True)

        chat_response = _post_json("/v1/chat/completions", REQUEST_BODY)
        print("\n=== /v1/chat/completions ===", flush=True)
        print(json.dumps(chat_response, indent=2), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=30)


@app.local_entrypoint()
def main() -> None:
    run_server_test.remote()
