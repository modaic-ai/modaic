import dspy
from dspy import InputField, OutputField

class SummarizeSignature(dspy.Signature):
    """Summarize the given text into a concise summary."""
    text: str = dspy.InputField(desc="The text to summarize")
    summary: str = dspy.OutputField(desc="A concise summary of the text")

# Serialize and deserialize
schema = SummarizeSignature.model_json_schema()
print("Original schema:", schema)
print("\nOriginal class:", SummarizeSignature)
print("Original class name:", SummarizeSignature.__name__)

# Now see if we can compare schemas
from dspy import make_signature

fields = {}
properties = schema.get("properties", {})
for name, field in properties.items():
    field_kwargs = {k: v for k, v in field.items() if k in ['desc', 'prefix']}
    InputOrOutputField = InputField if field.get("__dspy_field_type") == "input" else OutputField
    if default := field.get("default"):
        fields[name] = (str, InputOrOutputField(default=default, **field_kwargs))
    else:
        fields[name] = (str, InputOrOutputField(**field_kwargs))

recreated = make_signature(
    signature=fields,
    instructions=schema.get("description"),
    signature_name=schema.get("title"),
)

print("\nRecreated class:", recreated)
print("Recreated class name:", recreated.__name__)
print("\nAre they equal (==)?", recreated == SummarizeSignature)
print("Are they the same (is)?", recreated is SummarizeSignature)
print("\nOriginal schema:", SummarizeSignature.model_json_schema())
print("\nRecreated schema:", recreated.model_json_schema())
print("\nSchemas equal?", SummarizeSignature.model_json_schema() == recreated.model_json_schema())
