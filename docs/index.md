# Welcome to Modaic Docs

## Getting Started

Install Modaic

```bash
uv install modaic
```

## Modaic Principles

In Modaic there are two types of context. `Molecular` and `Atomic`. Molecular context is

## Create a Simple RAG Framework

Lets create a simple agent that can answer questions about the weather.

```python
from modaic import PrecompiledAgent, PrecompiledConfig

class WeatherConfig(PrecompiledConfig):
    agent_type = "WeatherAgent" # !! This is super important so you can load the agent later!!

class WeatherAgent(PrecompiledAgent):
    config_class = WeatherConfig # !! This is super important to link the agent to the config!!
    def __init__(self, config: WeatherConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, query: str) -> str:
        return self.get_weather(query)

    def get_weather(self, city: str) -> str:
        return f"The weather in {city} is sunny."

agent = WeatherAgent(PrecompiledConfig())
agent.forward("What is the weather in Tokyo?")
```

## Defining your own context
