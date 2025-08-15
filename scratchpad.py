# from modaic.context import TextSchema, LongText
# from collections.abc import Mapping
# from modaic.context.query_language import Filter
import json

# query = (TextSchema.text == "Hello, world!") & (TextSchema.metadata["doc"] == 1)

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

# print(json.dumps(Filter(query), indent=2))

import re

list1 = [
    "aenum>=3.1.16",
    "dspy>=2.6.27",
    "duckdb>=1.3.2",
    "gitpython>=3.1.45",
    "langchain-text-splitters>=0.3.9",
    "openpyxl>=3.1.5",
    "pillow>=11.3.0",
    "pinecone>=7.3.0",
    "pymilvus>=2.5.14",
    "tomlkit>=0.13.3",
]

list2 = ["dspy", "pillow", "pymilvus", "duckdb"]

# Make one combined whole-word regex
pattern = re.compile(r"\b(" + "|".join(map(re.escape, list2)) + r")\b")

filtered_list = [pkg for pkg in list1 if not pattern.search(pkg)]
print(filtered_list)
