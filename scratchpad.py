from modaic import PrecompiledAgent, PrecompiledConfig


class WeatherConfig(PrecompiledConfig):
    weather: str = "sunny"


class WeatherAgent(PrecompiledAgent):
    config: WeatherConfig

    def __init__(self, config: WeatherConfig, **kwargs):
        super().__init__(config, **kwargs)

    def forward(self, query: str) -> str:
        return f"The weather in {query} is {self.config.weather}."


agent = WeatherAgent(WeatherConfig())
print(agent(query="Tokyo"))
