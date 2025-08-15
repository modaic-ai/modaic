from modaic.context import TextSchema, LongText
from collections.abc import Mapping
from modaic.context.query_language import Filter
import json

query = (TextSchema.text == "Hello, world!") & (TextSchema.metadata["doc"] == 1)

# print(query)
# print(isinstance(query, Mapping))


# def recursive_print(v):
#     if isinstance(v, list):
#         for item in v:
#             recursive_print(item)
#     elif isinstance(v, Mapping):
#         for k, v in v.items():
#             print(k)
#             recursive_print(v)
#     else:
#         print("type", type(v))
#         print(v)
#         print()


# recursive_print(query)

print(json.dumps(Filter(query), indent=2))
