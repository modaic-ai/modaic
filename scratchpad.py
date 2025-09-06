from pymilvus import MilvusClient, DataType
from modaic.indexing import DummyEmbedder
import os

embedder = DummyEmbedder()
client = MilvusClient(uri="index2.db")

embedding = embedder("hello").tolist()
print(embedding)

data = [
    {
        "id": 0,
        "vector": embedding,
        "color": "pink_8682",
    },
    {
        "id": 1,
        "vector": embedding,
        "color": "red_7025",
    },
    {
        "id": 2,
        "vector": embedding,
        "color": "orange_6781",
    },
    {
        "id": 3,
        "vector": embedding,
        "color": "pink_9298",
    },
    {
        "id": 4,
        "vector": embedding,
        "color": "red_4794",
    },
    {
        "id": 5,
        "vector": embedding,
        "color": "yellow_4222",
    },
    {
        "id": 6,
        "vector": embedding,
        "color": "red_9392",
    },
    {
        "id": 7,
        "vector": embedding,
        "color": "grey_8510",
    },
    {
        "id": 8,
        "vector": embedding,
        "color": "white_9381",
    },
    {
        "id": 9,
        "vector": embedding,
        "color": "purple_4976",
    },
]
try:
    milvus_schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    milvus_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    milvus_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
    milvus_schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=100)

    client.create_collection(collection_name="quick_setup", schema=milvus_schema)
    res = client.insert(collection_name="quick_setup", data=data)

    print(res)
except Exception as e:
    raise e
finally:
    os.remove("index2.db")
