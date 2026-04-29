"""Register `modaic/<model>` with litellm as an OpenAI-compatible provider.

Imported (and run) by `modaic/__init__.py` so `dspy.LM("modaic/<model>")` routes
to the Modaic API with no client-side model-string surgery.
"""

import litellm
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig
from modaic_client import settings as modaic_settings


class _ModaicProviderConfig(SimpleProviderConfig):
    """SimpleProviderConfig whose base_url is read live from modaic_settings.

    Lets `modaic.configure(modaic_api_url=...)` after import still affect routing.
    """

    def __init__(self):
        super().__init__(
            "modaic",
            {
                "base_url": "",  # overridden by the property below
                "api_key_env": "MODAIC_TOKEN",
                "base_class": "openai_gpt",
            },
        )

    @property  # type: ignore[override]
    def base_url(self) -> str:
        return f"{modaic_settings.modaic_api_url}/api/v1"

    @base_url.setter
    def base_url(self, _: str) -> None:
        # absorb the assignment SimpleProviderConfig.__init__ does
        pass


def register_modaic_provider() -> None:
    """Idempotently register the `modaic` provider with litellm."""
    JSONProviderRegistry._providers["modaic"] = _ModaicProviderConfig()
    if "modaic" not in litellm.provider_list:
        litellm.provider_list.append("modaic")


register_modaic_provider()
