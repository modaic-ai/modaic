from aenum import NamedConstant


class Dense(NamedConstant):
    DENSE = 1
    SPARSE = 2


class VectorType(NamedConstant):
    class FLOAT(NamedConstant):
        DENSE = Dense.DENSE
        SPARSE = Dense.SPARSE

    class INT(NamedConstant):
        DENSE = Dense.DENSE
        SPARSE = Dense.SPARSE


# ✅ Dot-chaining + identity to the base constants
assert VectorType.FLOAT.DENSE is Dense.DENSE
assert VectorType.INT.SPARSE is Dense.SPARSE
