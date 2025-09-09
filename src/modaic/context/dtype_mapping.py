import numpy as np

INTEGER_DTYPE_MAPPING = {
    np.int8: "TINYINT",
    np.int16: "SMALLINT",
    np.int32: "INT",
    np.int64: "BIGINT",
}

SPECIAL_INTEGER_DTYPE_MAPPING = {"Int64": "BIGINT", "UInt64": "BIGINT UNSIGNED"}

FLOAT_DTYPE_MAPPING = {
    np.float16: "FLOAT",
    np.float32: "FLOAT",
    np.float64: "DOUBLE",
}

OTHER_DTYPE_MAPPING = {
    "boolean": "BOOLEAN",
    "datetime": "DATETIME",
    "timedelta": "TIME",
    "string": "VARCHAR(255)",
    "category": "VARCHAR(255)",
    "default": "TEXT",
}
