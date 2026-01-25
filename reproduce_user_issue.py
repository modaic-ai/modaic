import dspy

class EvalSignature(dspy.Signature):
    """Signature for evaluating two responses."""
    reasoning = dspy.OutputField()
    label = dspy.OutputField()

def reproduce():
    adapter = dspy.ChatAdapter()
    
    class MockLM:
        def __init__(self):
            self.model = "together_ai/Qwen/Qwen3-VL-32B-Instruct"
    
    lm = MockLM()
    inputs = {}
    processed_signature = adapter._call_preprocess(lm, {}, EvalSignature, inputs)

    print("Testing truncated content...")
    content = "[[ ## reasoning ## ]]\nThis is some reasoning... [[ ## label ## ]]\nA>B"
    # Notice missing [[ ## completed ## ]]
    
    output = {"text": content}
    try:
        parsed_outputs = adapter._call_postprocess(processed_signature, EvalSignature, [output], lm)
        print(f"SUCCESS: Parsed without completed marker: {parsed_outputs}")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    reproduce()
