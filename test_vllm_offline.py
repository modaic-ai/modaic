"""Light test: vLLM offline batch generation on Modal with Qwen3.5-4B + reasoning parsing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import modal

MINUTES = 60

app = modal.App("test-vllm-offline")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install("vllm>=0.18.0", "huggingface-hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

GPU = "h100"
MODEL_ID = "Qwen/Qwen3.5-4B"

vllm_throughput_kwargs = {
    "max_model_len": 4096 * 4,
    "attention_backend": "flashinfer",
    "async_scheduling": True,
}

# -- Hardcoded test messages ------------------------------------------------

TEST_MESSAGES = [
    [{"role": "user", "content": "What is 25 * 37? Think step by step."}],
    [{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
    [
        {"role": "user", "content": "Write a Python function that checks if a string is a palindrome."},
    ],
    [{"role": "user", "content": "/no_think What is the capital of France?"}],
    [
        {
            "role": "user",
            "content": (
                "A train travels 120 km in 1.5 hours. "
                "It then travels another 80 km in 1 hour. "
                "What is the average speed for the entire journey?"
            ),
        }
    ],
]


@app.cls(
    image=vllm_image,
    gpu=GPU,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class Vllm:
    @modal.enter()
    def start(self):
        from vllm import LLM
        from vllm.reasoning import ReasoningParserManager

        self.llm = LLM(model=MODEL_ID, **vllm_throughput_kwargs)
        self.sampling_params = self.llm.get_default_sampling_params()
        self.sampling_params.max_tokens = 2048
        self.sampling_params.chat_template_kwargs = {"enable_thinking": True}

        parser_cls = ReasoningParserManager.get_reasoning_parser("qwen3")
        self.reasoning_parser = parser_cls(tokenizer=self.llm.get_tokenizer())

        # warm-up
        self.llm.chat([{"role": "user", "content": "Is this thing on?"}])

    @modal.method()
    def generate(self, messages_batch: list[list[dict]]) -> list[dict]:
        start = time.time()
        responses = self.llm.chat(messages_batch, sampling_params=self.sampling_params)
        duration_s = time.time() - start

        in_tokens = sum(len(r.prompt_token_ids) for r in responses)
        out_tokens = sum(len(r.outputs[0].token_ids) for r in responses)

        print(f"\n{'='*60}")
        print(f"Processed {in_tokens} prompt tokens in {duration_s:.1f}s")
        print(f"Generated {out_tokens} output tokens in {duration_s:.1f}s")
        print(f"Throughput: {out_tokens / duration_s:.0f} tok/s")
        print(f"{'='*60}\n")

        results = []
        for i, response in enumerate(responses):
            output = response.outputs[0]
            reasoning, content = self.reasoning_parser.extract_reasoning(
                output.text, request=None
            )
            result = {
                "index": i,
                "text": content,
                "reasoning": reasoning,
                "finish_reason": output.finish_reason,
                "prompt_tokens": len(response.prompt_token_ids),
                "output_tokens": len(output.token_ids),
            }
            results.append(result)
        return results

    @modal.exit()
    def stop(self):
        del self.llm


@app.local_entrypoint()
def main():
    vllm_instance = Vllm()
    results = vllm_instance.generate.remote(TEST_MESSAGES)

    for r in results:
        print(f"\n--- Message {r['index']} ---")
        print(f"Prompt tokens: {r['prompt_tokens']} | Output tokens: {r['output_tokens']} | Finish: {r['finish_reason']}")
        if r["reasoning"]:
            preview = r["reasoning"][:300]
            print(f"[Reasoning] {preview}{'...' if len(r['reasoning']) > 300 else ''}")
        print(f"[Response]  {r['text'][:500]}")

    print(f"\n✅ Done — {len(results)} responses generated.")
