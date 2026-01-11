[![Docs](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://docs.modaic.dev)
[![PyPI](https://img.shields.io/pypi/v/modaic)](https://pypi.org/project/modaic/)


# Modaic 🐙
Modular + Mosaic, a Python framework for composing and maintaining DSPy applications.

## Key Features

- **Hub Support**: Load and share precompiled DSPY programs from Modaic Hub
- **Program Framework**: Precompiled and auto-loading DSPY programs
- **Automated LM Judge Alignment**: Continuously align your LM judges to your preferences while staying at the pareto frontier!

Never lose your progress again.
Save everything you need to compare and reproduce optimization runs with GEPA, MIPROv2, etc. — architecture, hyperparameters, precompiled prompts, predictions, git commits, and even datasets — in 5 minutes. Modaic is free for personal use and academic projects, and it's easy to get started.

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
```

Save and load locally:

```python
weather_program.save_precompiled("./my-weather")

from modaic import AutoProgram, AutoConfig

cfg = AutoConfig.from_precompiled("./my-weather", local=True)
loaded = AutoProgram.from_precompiled("./my-weather", local=True)
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
