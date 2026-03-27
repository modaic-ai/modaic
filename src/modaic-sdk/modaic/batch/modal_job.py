"""Modal-based vLLM offline batch generation using LLM.chat()."""

import os
import time
from typing import Any, Optional

import modal

APP_NAME = "modaic-generate-responses"
VOLUME_ROOT = "/cache"
INPUT_FILENAME = "input.parquet"
OUTPUT_FILENAME = "output.parquet"
SHARED_VOLUME_NAME = "modaic-generate-responses-cache"

app = modal.App(APP_NAME)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
cache_volume = modal.Volume.from_name(SHARED_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install(
        "vllm>=0.18.0",
        "huggingface-hub",
        "hf_transfer",
        "hf-xet>=1.1.7",
        "pandas",
        "datasets",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/.cache/vllm": vllm_cache_vol,
    VOLUME_ROOT: cache_volume,
}

# Sentinel for "not set" since modal.parameter() doesn't support Optional
_UNSET_INT = -1


@app.cls(
    gpu="H100",
    image=image,
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
)
class ResponseGenerator:
    # modal.parameter() only supports str, int, bool, bytes
    model_id: str = modal.parameter(default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    reasoning_parser: str = modal.parameter(default="")
    enforce_eager: bool = modal.parameter(default=False)
    max_model_len: int = modal.parameter(default=_UNSET_INT)
    # gpu_memory_utilization as int percentage (90 = 0.90)
    gpu_memory_utilization_pct: int = modal.parameter(default=90)
    tensor_parallel_size: int = modal.parameter(default=_UNSET_INT)
    language_model_only: bool = modal.parameter(default=False)

    @modal.enter()
    def start(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        from vllm import LLM
        from vllm import SamplingParams

        print(
            f"ResponseGenerator.start begin: model={self.model_id}, reasoning_parser={self.reasoning_parser or 'none'}, enforce_eager={self.enforce_eager}, tensor_parallel_size={self.tensor_parallel_size}, language_model_only={self.language_model_only}"
        )
        llm_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "enforce_eager": self.enforce_eager,
            "gpu_memory_utilization": self.gpu_memory_utilization_pct / 100.0,
        }
        if self.max_model_len != _UNSET_INT:
            llm_kwargs["max_model_len"] = self.max_model_len
        if self.tensor_parallel_size != _UNSET_INT:
            llm_kwargs["tensor_parallel_size"] = self.tensor_parallel_size
        if self.language_model_only:
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
        if self.model_id.lower() == "qwen/qwen3.5-4b":
            llm_kwargs["enable_chunked_prefill"] = False

        print(f"Initializing vLLM with kwargs={llm_kwargs}")
        self.llm = LLM(**llm_kwargs)
        print("vLLM engine constructed")

        self._reasoning_parser = None
        if self.reasoning_parser:
            print(f"Initializing reasoning parser={self.reasoning_parser}")
            from vllm.reasoning import ReasoningParserManager

            parser_cls = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
            self._reasoning_parser = parser_cls(tokenizer=self.llm.get_tokenizer())
            print(f"Reasoning parser ready={parser_cls.__name__}")

        # Keep warm-up bounded so startup cannot stall on an unbounded decode.
        print(
            f"Starting warmup: model={self.model_id}, reasoning_parser={self.reasoning_parser or 'none'}"
        )
        self.llm.chat(
            [[{"role": "user", "content": "warmup"}]],
            sampling_params=SamplingParams(max_tokens=1),
            chat_template_kwargs={"enable_thinking": False},
        )
        print(f"Engine ready: model={self.model_id}, reasoning_parser={self.reasoning_parser or 'none'}")

    @modal.method()
    def generate(
        self,
        input_path: str,
        output_path: str,
        messages_column: str = "messages",
        hf_dataset_id: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> dict[str, Any]:
        import json

        import pandas as pd
        from vllm import SamplingParams

        # --- Load input ---
        cache_volume.reload()

        if hf_dataset_id:
            from datasets import load_dataset

            print(f"Loading dataset from HuggingFace Hub: {hf_dataset_id}")
            ds = load_dataset(hf_dataset_id, split="train")
            messages_batch = list(ds[messages_column])
        else:
            print(f"Loading dataset from volume: {input_path}")
            df = pd.read_parquet(input_path)
            messages_batch = df[messages_column].tolist()

        print(f"Loaded {len(messages_batch)} conversations")

        # --- Build sampling params ---
        sp_kwargs: dict[str, Any] = {}
        if temperature is not None:
            sp_kwargs["temperature"] = temperature
        if top_p is not None:
            sp_kwargs["top_p"] = top_p
        if top_k is not None:
            sp_kwargs["top_k"] = top_k
        if min_p is not None:
            sp_kwargs["min_p"] = min_p
        if max_tokens is not None:
            sp_kwargs["max_tokens"] = max_tokens
        if repetition_penalty is not None:
            sp_kwargs["repetition_penalty"] = repetition_penalty

        chat_template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
        if thinking_budget is not None and enable_thinking:
            chat_template_kwargs["thinking_budget"] = thinking_budget

        sampling_params = SamplingParams(**sp_kwargs)

        # --- Generate ---
        print(f"Generating {len(messages_batch)} responses...")
        start = time.time()
        responses = self.llm.chat(
            messages_batch,
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )
        duration_s = time.time() - start

        in_tokens = sum(len(r.prompt_token_ids) for r in responses)
        out_tokens = sum(len(r.outputs[0].token_ids) for r in responses)

        print(f"Processed {in_tokens} prompt tokens in {duration_s:.1f}s")
        print(f"Generated {out_tokens} output tokens in {duration_s:.1f}s")
        if duration_s > 0:
            print(f"Throughput: {out_tokens / duration_s:.0f} tok/s")

        # --- Build output ---
        output_responses = []
        output_errors = []
        failures = 0
        reasoning_rows = 0

        for i, response in enumerate(responses):
            output = response.outputs[0]
            text = output.text
            reasoning_content = None
            error = None

            print(f"[DEBUG] Row {i}: finish_reason={output.finish_reason}, "
                  f"output_tokens={len(output.token_ids)}, "
                  f"raw_text_type={type(text).__name__}, "
                  f"raw_text_len={len(text) if text else 0}, "
                  f"raw_text_preview={repr(text[:200]) if text else repr(text)}")

            try:
                if self._reasoning_parser is not None and enable_thinking:
                    reasoning_content, text = self._reasoning_parser.extract_reasoning(
                        text, request=None
                    )
                    print(f"[DEBUG] Row {i} after parsing: "
                          f"reasoning_len={len(reasoning_content) if reasoning_content else 0}, "
                          f"text_type={type(text).__name__}, "
                          f"text_preview={repr(text[:200]) if text else repr(text)}")
                    text = text or ""
            except Exception as exc:
                error = f"Reasoning parse error: {exc}"
                print(f"[DEBUG] Row {i} parse error: {exc}")

            if error:
                failures += 1
            if reasoning_content:
                reasoning_rows += 1

            final_json = json.dumps({"text": text, "reasoning_content": reasoning_content})
            print(f"[DEBUG] Row {i}: final_json_preview={final_json[:300]}")
            output_responses.append(final_json)
            output_errors.append(error)

        # --- Write output parquet ---
        out_df = pd.DataFrame({
            "response": output_responses,
            "response_error": output_errors,
        })
        out_df.to_parquet(output_path, index=False)
        cache_volume.commit()
        print(f"Output written to {output_path}")

        return {
            "total": len(responses),
            "failures": failures,
            "reasoning_rows": reasoning_rows,
            "prompt_tokens": in_tokens,
            "output_tokens": out_tokens,
            "duration_s": duration_s,
        }

    @modal.exit()
    def stop(self):
        del self.llm
