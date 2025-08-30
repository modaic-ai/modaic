from modaic.context import TextSchema, ContextSchema, Relation
from gqlalchemy import Node, Memgraph, Relationship
from pydantic import BaseModel, Field
from pydantic.v1 import Field as V1Field
from modaic.databases.graph_database import GraphDatabase, MemgraphConfig

config = MemgraphConfig()

db = GraphDatabase(config)

# start_node = 29
# result = db.execute_and_fetch(f"MATCH (n) WHERE id(n) = {start_node} RETURN n")


# print(next(result))
# class ChatsWith(Relationship, type="CHATS_WITH"):
#     last_chatted: str


# print("ChatsWith._type", ChatsWith._type)

# field = V1Field()
# print(field)
# print(field.default)
# db = Memgraph()
# class Uses(Relationship, type="USES_TEST"):
#     x: str
#     source: dict


# rel = Uses(x="test", source={"x": "test"})
# print(rel.x)
# print(rel.source)

# rel = Uses(x="test")

# print(Uses.type)
# print(rel._type)

t1 = TextSchema(text="test text1")
t2 = TextSchema(text="test text2")
# print(type(t1.to_gqlalchemy(db)))
# print(type(t2.to_gqlalchemy(db)))
# t1.save(db)
# t2.save(db)
x = t1 >> Relation(x="test", _type="TEST_REL") >> t2
x.save(db)
print(x)
print(x.start_node_gql_id)
print(x.end_node_gql_id)
# print(x.to_gqlalchemy(db))


# field_annotations = get_annotations(cls)
# field_defaults = get_defaults(cls)
# gqlalchemy_class = type(
#     f"{cls.__name__}Node",
#     (gqlalchemy.Relationship,),
#     {
#         "__annotations__": {**field_annotations, "modaic_id": str},
#         "modaic_id": V1Field(unique=True, db=db),
#         **field_defaults,
#     },
#     type=cls._type,
# )
# cls._gqlalchemy_class_registry[cls._type] = gqlalchemy_class
# # Return a new GQLAlchemy Node object
# gqlalchemy_class = cls._gqlalchemy_class_registry[cls._type]


# if self._gqlalchemy_id is None:
#     return gqlalchemy_class(
#         _labels=set(self._labels),
#         modaic_id=self.id,
#         **self.model_dump(exclude={"id"}),
#     )
# else:
#     return gqlalchemy_class(
#         _labels=set(self._labels),
#         modaic_id=self.id,
#         _id=self._gqlalchemy_id,
#         **self.model_dump(exclude={"id"}),
#     )
