from modaic.types import Schema
from modaic.context import Context


json_schema = Context.model_json_schema()
# print(json_schema)
d = Schema.from_json_schema(json_schema)
# print(d)
print(d.as_dict())