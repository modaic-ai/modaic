"""Load a deployed Arbiter from Modaic Hub and run it.

Run:
    uv run python run_arbiter.py

Requirements:
    - MODAIC_TOKEN set locally (so the SDK can authenticate to Modaic Hub).
    - The provider's API key for the Arbiter's configured model set as an
      Environment Variable on Modaic Hub. Without that, the server-side
      run will fail.
"""

from modaic import Arbiter

# Loads metadata from `your-org/code-completions` on Modaic Hub.
# Pass `revision="v1"` to pin a specific tag/branch/commit.
arbiter = Arbiter("your-org/code-completions")

# kwargs MUST match the input fields declared on the dspy.Signature.
# arbiter(...) and arbiter.predict(...) are equivalent — both run the
# arbiter on the Modaic backend and log the run on Modaic Hub.
result = arbiter(
    prompt="Print hello world",
    completion="print('Hello World')",
)

# result.output has fields matching the signature's output fields.
print("quality:    ", result.output.quality)
print("reasoning:  ", result.reasoning)
print("confidence: ", result.confidence)  # lazy — fetched on first access
