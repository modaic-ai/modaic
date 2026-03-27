from __future__ import annotations

import os
import time
from typing import Any, Optional


class LocalVLLMRunner:
    def __init__(
        self,
        model_id: str,
        reasoning_parser: str = "",
        enforce_eager: bool = False,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: Optional[int] = None,
    ) -> None:
        try:
            from vllm import LLM
            from vllm import SamplingParams
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'modaic.batch requires vLLM for local vLLM batch jobs. Install it with `uv add "modaic[vllm]"`.'
            ) from exc

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        llm_kwargs: dict[str, Any] = {
            "model": model_id,
            "enforce_eager": enforce_eager,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if tensor_parallel_size is not None:
            llm_kwargs["tensor_parallel_size"] = tensor_parallel_size

        self.llm = LLM(**llm_kwargs)
        self._reasoning_parser = None
        if reasoning_parser:
            from vllm.reasoning import ReasoningParserManager

            parser_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
            self._reasoning_parser = parser_cls(tokenizer=self.llm.get_tokenizer())

        self.llm.chat(
            [[{"role": "user", "content": "warmup"}]],
            sampling_params=SamplingParams(max_tokens=1),
            chat_template_kwargs={"enable_thinking": False},
        )

    def generate(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from vllm import SamplingParams

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
        if enable_thinking and thinking_budget is not None:
            chat_template_kwargs["thinking_budget"] = thinking_budget

        sampling_params = SamplingParams(**sp_kwargs)

        start = time.time()
        responses = self.llm.chat(
            messages_batch,
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )
        duration_s = time.time() - start

        prompt_tokens = sum(len(response.prompt_token_ids) for response in responses)
        output_tokens = sum(len(response.outputs[0].token_ids) for response in responses)

        output_rows: list[dict[str, Any]] = []
        failures = 0
        reasoning_rows = 0

        for response in responses:
            output = response.outputs[0]
            text = output.text
            reasoning_content = None
            error = None

            try:
                if self._reasoning_parser is not None and enable_thinking:
                    reasoning_content, text = self._reasoning_parser.extract_reasoning(text, request=None)
                    text = text or ""
            except Exception as exc:
                error = f"Reasoning parse error: {exc}"

            if error is not None:
                failures += 1
            if reasoning_content:
                reasoning_rows += 1

            output_rows.append(
                {
                    "response": {"text": text or "", "reasoning_content": reasoning_content},
                    "error": error,
                }
            )

        return output_rows, {
            "total": len(responses),
            "failures": failures,
            "reasoning_rows": reasoning_rows,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "duration_s": duration_s,
        }

    def close(self) -> None:
        del self.llm
        self._reasoning_parser = None
