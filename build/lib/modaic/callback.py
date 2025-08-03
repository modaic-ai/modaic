import json
import time
from pathlib import Path
from dspy.utils.callback import BaseCallback


class JsonTraceCallback(BaseCallback):
    def __init__(self, out_path: str = "traces.jsonl"):
        self.out_path = Path(out_path)
        self._stack = []  # parent‑child nesting

    # ---------- helper ----------
    def _write(self, record: dict):
        record["ts"] = time.time()
        self.out_path.write_text(
            (self.out_path.read_text() if self.out_path.exists() else "")
            + json.dumps(record)
            + "\n"
        )

    # ---------- hooks ----------
    def on_module_start(self, call_id, instance, inputs):
        parent = self._stack[-1] if self._stack else None
        self._stack.append(call_id)
        self._write(
            {
                "event": "module_start",
                "call_id": call_id,
                "parent_id": parent,
                "module": instance.__class__.__name__,
                "inputs": inputs,
            }
        )

    def on_module_end(self, call_id, outputs, exception):
        self._write(
            {
                "event": "module_end",
                "call_id": call_id,
                "outputs": outputs,
                "exception": str(exception) if exception else None,
            }
        )
        self._stack.pop()

    # You can add on_lm_start/end, on_tool_start/end, etc.
