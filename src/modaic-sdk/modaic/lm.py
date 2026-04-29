import copy
import uuid
from contextvars import ContextVar
from typing import Any

import dspy
from dspy.dsp.utils import settings
from modaic_client import settings as modaic_settings


class LM(dspy.LM):
    def __init__(self, *args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else None)
        if model.startswith("modaic/"):
            kwargs["model"] = "openai/" + kwargs["model"].removeprefix("modaic/")
            kwargs["api_base"] = f"{modaic_settings.modaic_api_url}/api/v1"

        api_base = kwargs.get("api_base") or ""
        if (
            modaic_settings.modaic_api_url
            and api_base.startswith(modaic_settings.modaic_api_url)
            and not kwargs.get("api_key")
        ):
            kwargs["api_key"] = modaic_settings.modaic_token

        super().__init__(*args, **kwargs)

        # unique per instance
        self._local_history: ContextVar[list | None] = ContextVar(
            f"local_history_{uuid.uuid4().hex}",
            default=None,
        )

    @property
    def local_history(self) -> list:
        lst = self._local_history.get()
        if lst is None:
            lst = []
            self._local_history.set(lst)
        return lst

    def reset_local_history(self) -> None:
        self._local_history.set([])

    def update_history(self, entry: Any):
        super().update_history(entry)
        if len(self.local_history) >= settings.max_history_size:
            self.local_history.pop(0)
        self.local_history.append(entry)

    @classmethod
    def from_lm(cls, lm: dspy.LM) -> "LM":
        return cls(
            model=lm.model,
            model_type=lm.model_type,
            cache=lm.cache,
            callbacks=lm.callbacks,
            num_retries=lm.num_retries,
            provider=lm.provider,
            finetuning_model=lm.finetuning_model,
            launch_kwargs=lm.launch_kwargs,
            train_kwargs=lm.train_kwargs,
            use_developer_role=lm.use_developer_role,
            **lm.kwargs,
        )

    def __deepcopy__(self, memo: Any):
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj

        for k, v in self.__dict__.items():
            if isinstance(v, ContextVar):
                setattr(new_obj, k, v)  # shallow copy
            else:
                setattr(new_obj, k, copy.deepcopy(v, memo))

        return new_obj
