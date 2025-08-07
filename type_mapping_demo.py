#!/usr/bin/env python3
"""
Demonstration of the enhanced type mapping functionality for Milvus integration.
This shows how Vector types from modaic.context.typing are properly mapped to Milvus DataTypes.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from typing import List

# Import the custom types directly
from modaic.types import Vector, float32, int8, float16, bfloat16


# Mock the pymilvus DataType for demonstration
class MockDataType:
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT16 = "FLOAT16"
    BFLOAT16 = "BFLOAT16"
    FLOAT32 = "FLOAT32"
    FLOAT64 = "FLOAT64"
    FLOAT = "FLOAT"
    VARCHAR = "VARCHAR"
    BOOL = "BOOL"
    JSON = "JSON"
    ARRAY = "ARRAY"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    FLOAT16_VECTOR = "FLOAT16_VECTOR"
    BFLOAT16_VECTOR = "BFLOAT16_VECTOR"
    BINARY_VECTOR = "BINARY_VECTOR"


# Patch the DataType in the milvus module
import modaic.databases.integrations.milvus as milvus_module

milvus_module.DataType = MockDataType

from modaic.databases.integrations.milvus import (
    _map_modaic_type_to_milvus,
    _is_vector_type,
    _extract_vector_info,
)


def demo_type_mapping():
    """
    Demonstrate the enhanced type mapping functionality.
    """
    print("=== Enhanced Type Mapping Demo ===\n")

    # Test Vector types
    print("1. Vector Types:")
    vector_float32_128 = Vector[float32, 128]
    vector_float16_256 = Vector[float16, 256]
    vector_bfloat16_512 = Vector[bfloat16, 512]

    print(
        f"Vector[float32, 128]: {_map_modaic_type_to_milvus(vector_float32_128, 'embedding')}"
    )
    print(
        f"Vector[float16, 256]: {_map_modaic_type_to_milvus(vector_float16_256, 'features')}"
    )
    print(
        f"Vector[bfloat16, 512]: {_map_modaic_type_to_milvus(vector_bfloat16_512, 'vector')}"
    )

    # Test vector detection
    print(f"\nVector detection:")
    print(f"Is Vector[float32, 128] a vector? {_is_vector_type(vector_float32_128)}")
    print(f"Is List[float] a vector? {_is_vector_type(List[float])}")

    # Test vector info extraction
    print(f"\nVector info extraction:")
    dtype, dim = _extract_vector_info(vector_float32_128)
    print(f"Vector[float32, 128] -> dtype: {dtype}, dimension: {dim}")

    # Test modaic scalar types
    print(f"\n2. Modaic Scalar Types:")
    print(f"float32: {_map_modaic_type_to_milvus(float32, 'score')}")
    print(f"int8: {_map_modaic_type_to_milvus(int8, 'category')}")
    print(f"float16: {_map_modaic_type_to_milvus(float16, 'weight')}")

    # Test List with modaic types
    print(f"\n3. Lists with Modaic Types:")
    print(f"List[float32]: {_map_modaic_type_to_milvus(List[float32], 'embeddings')}")
    print(f"List[int8]: {_map_modaic_type_to_milvus(List[int8], 'indices')}")

    # Test standard types
    print(f"\n4. Standard Python Types:")
    print(f"str: {_map_modaic_type_to_milvus(str, 'text')}")
    print(f"int: {_map_modaic_type_to_milvus(int, 'count')}")
    print(f"float: {_map_modaic_type_to_milvus(float, 'value')}")
    print(f"bool: {_map_modaic_type_to_milvus(bool, 'active')}")


if __name__ == "__main__":
    demo_type_mapping()
