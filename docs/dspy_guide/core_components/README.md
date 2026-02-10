# Core Components Section
This section covers the core components of DSPy. These are the building blocks of DSPy programs. For concepts to be covered in this section it must be the case that it is impossible to build a complete dspy program without them.
```python
    [
    {"role": "system", "content": """
    Your input fields are:  
    1. `question` (str):  
    Your output fields are:  1. `answer` (str):  All interactions will be structured in the following way, with the appropriate values filled in.  [[ ## question ## ]]  {question}  [[ ## answer ## ]]  {answer}  [[ ## completed ## ]]  In adhering to this structure, your objective is:   Given the fields `question`, produce the fields `answer`."""}
    ]
    ```