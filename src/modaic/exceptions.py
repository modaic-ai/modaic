class ModaicError(Exception):
    pass


class ModaicHubError(ModaicError):
    """Base class for all hub-related errors."""

    pass


class RepositoryExistsError(ModaicHubError):
    """Raised when repository already exists"""

    pass


class AuthenticationError(ModaicHubError):
    """Raised when authentication fails"""

    pass


class RepositoryNotFoundError(ModaicHubError):
    """Raised when repository does not exist"""

    pass


class SchemaError(ModaicError):
    """Raised when a schema is invalid"""

    pass


class BackendCompatibilityError(ModaicError):
    """Raised when a feature is not supported by a backend"""

    pass


class MissingSecretError(AuthenticationError):
    """Raised when a secret is missing"""

    def __init__(self, message: str, secret_name: str):
        self.message = message
        self.secret_name = secret_name
        super().__init__(message)
