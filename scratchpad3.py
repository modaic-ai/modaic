import dspy
from pydantic import BaseModel
from typing import List


def get_weather(city: str) -> str:
    """
    Get the weather in a city.
    """
    return f"The weather in {city} is sunny."


agent = dspy.ReAct("question->answer", tools=[get_weather])
agent.set_lm(dspy.LM(model="openai/gpt-4o-mini"))


response = agent(question="What is the weather in Tokyo?")
print(response)
