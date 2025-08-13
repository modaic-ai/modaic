from .database import ContextDatabase

class DirectoryDatabase(ContextDatabase):
    """
    A database that stores context objects in a local file system directory. Not to be confused with the BucketDatabase in local mode.
    This database is designed to be used in-place and in tandem with a user's local folder.
    """
    raise NotImplementedError("Not implemented")