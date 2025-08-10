import dspy
from pydantic import BaseModel
from typing import List


class UserProfile(BaseModel):
    name: str
    age: int
    description: str
    email: str


def get_user_profiles(description: str) -> List[UserProfile]:
    """
    Get a user profile based on a description.
    """
    return [
        UserProfile(
            name="John Doe",
            age=30,
            description="John is a designer",
            email="john.doe@example.com",
        ),
        UserProfile(
            name="Jane Doe",
            age=25,
            description="Jane is a software engineer",
            email="jane.doe@example.com",
        ),
    ]


def email_user(email: str, message: str) -> str:
    """
    Email a user.
    """
    return f"Email sent to {email} with message: {message}"


agent = dspy.ReAct(
    signature="description->response", tools=[get_user_profiles, email_user]
)
agent.set_lm(dspy.LM(model="openai/gpt-4o-mini"))


response = agent(
    description="Can you email a designer in my network? I need someone to design the UI for my website."
)
print(response)
