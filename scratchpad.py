from modaic.context import Text, Context
from gqlalchemy import Node, Memgraph
from pydantic import BaseModel, Field
from pydantic.v1 import Field as V1Field
from typing import (
    Any,
    get_origin,
    get_args,
    Annotated,
    Union,
    Literal,
    Final,
    ClassVar,
    Type,
    List,
)
from types import UnionType
from modaic.context.base import get_annotations, get_defaults

# config = MemgraphConfig()

# db = GraphDatabase(config)
db = Memgraph()


# class SpecialText(Text):
#     count: int = 1


# class EvenMoreSpecialText(SpecialText):
#     x: List[int] = Field(default_factory=lambda: [1, 2, 3])


# text = Text(text="test")
# text_node = text.to_gqlalchemy(db)
# print("TEXT", text_node)

# special_text = SpecialText(text="test", count=2)
# special_text_node = special_text.to_gqlalchemy(db)
# print("SPECIAL TEXT", special_text_node)

# even_more_special_text = EvenMoreSpecialText(text="test")
# even_more_special_text_node = even_more_special_text.to_gqlalchemy(db)
# print("EVEN MORE SPECIAL TEXT", even_more_special_text_node)


# text2 = Text(text="test2")
# text2_node = text2.to_gqlalchemy(db)
# print("TEXT2", text2_node)

# def cast_type_if_base_model(field_type):
#     """
#     If field_type is a typing construct, reconstruct it from origin/args.
#     If it's a Pydantic BaseModel subclass, map it to `dict`.
#     Otherwise return the type itself.
#     """
#     origin = get_origin(field_type)

#     # Non-typing constructs
#     if origin is None:
#         # Only call issubclass on real classes
#         if isinstance(field_type, type) and issubclass(field_type, BaseModel):
#             return dict
#         return field_type

#     args = get_args(field_type)

#     # Annotated[T, m1, m2, ...]
#     if origin is Annotated:
#         base, *meta = args
#         # Annotated allows multiple args; pass a tuple to __class_getitem__
#         return Annotated.__class_getitem__((cast_type_if_base_model(base), *meta))

#     # Unions: typing.Union[...] or PEP 604 (A | B)
#     if origin in (Union, UnionType):
#         return Union[tuple(cast_type_if_base_model(a) for a in args)]

#     # Literal / Final / ClassVar accept tuple args via typing protocol
#     if origin in (Literal, Final, ClassVar):
#         return origin.__getitem__([cast_type_if_base_model(a) for a in args])

#     # Builtin generics (PEP 585): list[T], dict[K, V], set[T], tuple[...]
#     if origin in (list, set, frozenset):
#         (T,) = args
#         return origin[cast_type_if_base_model(T)]
#     if origin is dict:
#         K, V = args
#         return dict[cast_type_if_base_model(K), cast_type_if_base_model(V)]
#     if origin is tuple:
#         # tuple[int, ...] vs tuple[int, str]
#         if len(args) == 2 and args[1] is Ellipsis:
#             return tuple[cast_type_if_base_model(args[0]), ...]
#         return tuple[
#             tuple([cast_type_if_base_model(a) for a in args])
#         ]  # tuple[(A, B, C)]

#     # ABC generics (e.g., Mapping, Sequence, Iterable, etc.) usually accept tuple args
#     try:
#         return origin.__class_getitem__([cast_type_if_base_model(a) for a in args])
#     except Exception:
#         # Last resort: try simple unpack for 1–2 arity generics
#         if len(args) == 1:
#             return origin[cast_type_if_base_model(args[0])]
#         elif len(args) == 2:
#             return origin[
#                 cast_type_if_base_model(args[0]), cast_type_if_base_model(args[1])
#             ]
#         raise


# def get_annotations(cls: Type):
#     if not issubclass(cls, Context):
#         return {}
#     elif cls is Context:
#         exclude = ["id", "_gqlalchemy_id", "_type_registry", "_labels"]
#         res = {
#             k: cast_type_if_base_model(v)
#             for k, v in cls.__annotations__.items()
#             if k not in exclude
#         }
#         return res
#     else:
#         annotations = {}
#         for base in cls.__bases__:
#             annotations.update(get_annotations(base))
#         annotations.update(
#             {
#                 k: cast_type_if_base_model(v)
#                 for k, v in cls.__annotations__.items()
#                 if k != "id"
#             }
#         )
#         return annotations


# def cast_if_base_model(field_default):
#     if isinstance(field_default, BaseModel):
#         return field_default.model_dump()
#     return field_default


# def get_defaults(cls: Type[Context]):
#     defaults: dict[str, Any] = {}
#     for name, v2_field in cls.model_fields.items():
#         if name == "id" or v2_field.is_required():
#             continue
#         kwargs = {}
#         if extra_kwargs := getattr(v2_field, "json_schema_extra", None):
#             kwargs.update(extra_kwargs)

#         factory = v2_field.default_factory
#         if factory is not None:
#             kwargs["default_factory"] = lambda f=factory: cast_if_base_model(f())
#         else:
#             kwargs["default"] = cast_if_base_model(v2_field.default)
#         v1_field = V1Field(**kwargs)
#         defaults[name] = v1_field

#     return defaults


class Example(Context):
    a: List[int] = Field(default_factory=lambda: [1, 2, 3])
    b: Text = Field(default_factory=lambda: Text(text="test"))
    c: int = Field(default=1, unique=True, db=db)
    d: int = Field(default=1, index=True, db=db)
    e: int = Field(default=1, exists=True, db=db)


field_annotations = get_annotations(Example)
field_defaults = get_defaults(Example)

print("ANNOTATIONS")
print(field_annotations)
print("DEFAULTS")
print(field_defaults)

DynamicNode = type(
    "DynamicNode",
    (Node,),
    {
        "__annotations__": {**field_annotations, "modaic_id": str},
        "modaic_id": V1Field(unique=True, db=db),
        # Defaults for optional fields
        **field_defaults,
    },
    # optional: set label used in DB
    label="NEW_LABEL",
    # type="LABEL1",
    # _labels=["LABEL1", "LABEL2", "LABEL3"],
)

DynamicNode.modaic_class = Example

print("DYNAMIC NODE", DynamicNode.modaic_class)

# # text_node = DynamicNode(text="test")

# node = DynamicNode(
#     _labels=["NEW_LABEL", "NEW_LABEL2", "NEW_LABEL3"], text="test", modaic_id="123"
# )
# print(node)
# node.save(db)
# print(node)
# res = node.save(Memgraph())
# print(res)
# print(text.model_dump())

# from gqlalchemy import create, Memgraph

# result = create().node(labels="Person", name="Alice", age=30).return_().execute()

# for x in result:
#     print(x)
