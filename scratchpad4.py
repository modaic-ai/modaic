from modaic.context import Context, Text

class CustomContext(Context):
    name: str
    age: int
    t_context: Text
    

class CustomText(Text):
    some_more_text: str

c = CustomContext(name="John", age=30, t_context=CustomText(text="Hello, world!", some_more_text="This is some more text"))
print(c.model_dump(serialize_as_any=True))