from modaic.context import Context, Text, parse_modaic_filter
from langchain_community.query_constructors.milvus import MilvusTranslator

class CustomContext(Context):
    name: str
    age: int
    t_context: Text
    

filt = (CustomContext.age < 30) & (CustomContext.age > 12)
print(str(filt))
print(parse_modaic_filter(MilvusTranslator(), filt))
# converted = MilvusTranslator().visit_operation(filt.query)
# converted = MilvusTranslator().visit_comparison(filt.query)
# print(converted)