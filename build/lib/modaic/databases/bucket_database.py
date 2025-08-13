from .database import ContextDatabase

class BucketDatabase(ContextDatabase):
    """
    A database that stores context objects in a bucket like S3.
    """