from pydantic import BaseModel, Field
from modaic.context import Context
class InnerContext(Context):
    name: str # [x]
    age: int # [x]
    password: str = Field(hidden=True) # [x]


class SingleNestedContext(Context):
    link: str # [x]
    private: str = Field(hidden=True) # [x]
    inner_context: InnerContext # [x]


class BaseContext(Context):
    state: str # [x]
    inherited1: str  = "inherited1"
    inherited2: str = "inherited2"
    inherited3: str = "inherited3"
    weight: int = Field(hidden=True) # [x]


class InheritedContext(BaseContext):
    occupation: str # [x]
    ssn: str = Field(hidden=True) # [x]


class DoubleInheritedContext(InheritedContext):
    favorite_artist: str # [x]
    pin: str = Field(hidden=True) # [x]
    single_nested_context: SingleNestedContext # [x]
    

include_hidden = False
serialize_as_any = False
i = InnerContext(name="John", age=30, password="this should be hidden")
s = SingleNestedContext(link="https://www.google.com", private="this is private", inner_context=i)
d = DoubleInheritedContext(
    state="CA",
    weight=251,
    occupation="freelance furry",
    ssn="123-45-6789",
    favorite_artist="Sabrina Carpenter",
    pin="1234",
    single_nested_context=s,
)
dump = d.model_dump(include_hidden=include_hidden, serialize_as_any=serialize_as_any)
print(dump)