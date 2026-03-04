"""
Example: Using return_messages with modaic.Predict to get raw messages and outputs.

When `return_messages=True`, the prediction gains two extra attributes:
  - `_messages`: The request messages sent to the LM (system + user prompts).
  - `_outputs`: A dict with "text" (the assistant's response) and optionally
    "reasoning_content" (chain-of-thought from reasoning models).
"""

import dspy
from modaic import Predict
from modaic.safe_lm import SafeLM

# SafeLM wraps dspy.LM with thread-safe local history tracking,
# which is required for return_messages to work.
lm = SafeLM(model="together_ai/tytodd/openai/gpt-oss-120b-e13758ee")
dspy.configure(lm=lm)

# Define a simple signature
summarize = Predict("text -> summary")

# Call with return_messages=True
prediction = summarize(text="Is africa a country? Think very hard step by step", return_messages=True)

# The normal prediction output
print("Summary:", prediction.summary)

# The request messages sent to the LM
print("\n--- Messages ---")
for msg in prediction._messages:
    print(f"[{msg['role']}] {msg['content'][:100]}...")

# The assistant's output
print("\n--- Outputs ---")
print("Text:", prediction._outputs["text"])

# reasoning_content is only present if the model supports it
if "reasoning_content" in prediction._outputs:
    print("Reasoning:", prediction._outputs["reasoning_content"])
