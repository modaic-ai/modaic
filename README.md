<p align="center">
  <a href="https://docs.modaic.dev">
    <img alt="Docs" src="https://img.shields.io/badge/docs-available-brightgreen.svg" />
  </a>
  <a href="https://pypi.org/project/modaic/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/modaic" />
  </a>
  <a href="https://deepwiki.com/modaic-ai/modaic">
    <img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg" />
  </a>
</p>

<p align="center">
  <img alt="Mo!" src="docs/images/name-logo-dark.png" width="400" />
</p>

# Modaic 🐙 = Modular + Mosaic

Modaic is a framework for quickly deploying optimized language models for repeatable, non-verifiable tasks, such as evaluation. We build on the “Declarative AI Programming” standard introduced by [DSPy](https://dspy.ai) and leverage frontier research in interpretability for uncertainty estimation.

## Key Features

- **Hub Support**: Package DSPy programs to re-use across projects and teams in one line of code.
- **Reliable Confidence Estimation**: Use dedicated probes to extract calibrated confidence scores from your models when they generate decisions or labels (for discrete decision tasks like classification).
- **Automatic Alignment**: Set ground truth + feedback to automatically align your models with your decision rules and industry expertise.

## Installation

### Using uv (recommended)

```bash
uv add modaic
```

Optional (for hub operations):

```bash
export MODAIC_TOKEN="<your-token>"
```

### Using pip
Please note that you will not be able to push DSPY programs to the Modaic Hub with pip.
```bash
pip install modaic
```
## Quick Start

### Creating a Simple Program

```python
from modaic import PrecompiledProgram, PrecompiledConfig

class WeatherConfig(PrecompiledConfig):
    weather: str = "sunny"

class WeatherProgram(PrecompiledProgram):
    config: WeatherConfig

    def __init__(self, config: WeatherConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, query: str) -> str:
        return f"The weather in {query} is {self.config.weather}."

weather_program = WeatherProgram(WeatherConfig())
print(weather_program(query="Tokyo"))
weather_program.push_to_hub("me/my-weather-program")
```

Save and load locally:

```python
weather_program.save_precompiled("./my-weather")

from modaic import AutoProgram, AutoConfig

cfg = AutoConfig.from_precompiled("./my-weather", local=True)
loaded = AutoProgram.from_precompiled("./my-weather", local=True)
print(loaded(query="Kyoto"))
```

from hub:

```python
from modaic import AutoProgram, AutoConfig

loaded = AutoProgram.from_precompiled("me/my-weather-program", rev="v2.0.0")
print(loaded(query="Kyoto"))
```

## Architecture
### Program Types

1. **PrecompiledProgram**: Statically defined programs with explicit configuration
2. **AutoProgram**: Dynamically loaded programs from Modaic Hub or local repositories
## Support

For issues and questions:
- GitHub Issues: `https://github.com/modaic-ai/modaic/issues`
- Docs: `https://docs.modaic.dev`

## Development

install development dependencies:
```bash
uv sync --all-extras
```
