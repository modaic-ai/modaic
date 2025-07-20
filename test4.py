import os
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

# Instantiate client (it picks up OPENAI_API_KEY by default)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_gpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    # Use attribute access instead of dict indexing
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print(chat_gpt("Hello, world!"))
