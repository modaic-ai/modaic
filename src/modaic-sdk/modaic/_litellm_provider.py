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


# Models served through an OpenAI-compatible gateway (e.g. Vercel AI Gateway)
# whose litellm entry does not advertise structured-output support, but which
# DO accept an OpenAI `json_schema` response_format. Without this, dspy's
# JSONAdapter sees ``supports_response_schema == False`` and falls back to
# ``response_format={"type": "json_object"}`` — which the Vercel AI Gateway
# rejects for these models with ``400 "Invalid input"``. Registering them as
# ``supports_response_schema`` keeps dspy on the structured (json_schema) path.
#
# Key is the *bare* model id (no provider prefix): litellm strips the gateway
# provider before looking the model up in its registry, so registering the
# fully-qualified ``vercel_ai_gateway/openai/gpt-oss-120b`` would not match.
_GATEWAY_STRUCTURED_OUTPUT_MODELS = ("openai/gpt-oss-120b",)


def _register_gateway_structured_output_support() -> None:
    """Tell litellm the gateway-served models below accept ``json_schema``.

    Idempotent: ``register_model`` merges into ``litellm.model_cost``.
    """
    for model in _GATEWAY_STRUCTURED_OUTPUT_MODELS:
        litellm.register_model(
            {
                model: {
                    "supports_response_schema": True,
                    "litellm_provider": "vercel_ai_gateway",
                    "mode": "chat",
                }
            }
        )


def register_modaic_provider() -> None:
    """Idempotently register the `modaic` provider with litellm."""
    JSONProviderRegistry._providers["modaic"] = _ModaicProviderConfig()
    if "modaic" not in litellm.provider_list:
        litellm.provider_list.append("modaic")


register_modaic_provider()
_register_gateway_structured_output_support()
