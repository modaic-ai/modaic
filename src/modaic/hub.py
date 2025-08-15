from pathlib import Path
from git import Repo


class HubError(Exception):
    """Base class for all hub-related errors."""


class RepoExistsError(HubError):
    """Raised when a repository already exists on the hub."""

    def __init__(self, repo_name: str):
        super().__init__(f"Repository '{repo_name}' already exists.")


class AuthenticationError(HubError):
    """Raised when authentication fails or access is denied."""

    def __init__(self, owner: str):
        super().__init__(f"Authentication failed for '{owner}'.")


def create_remote_repo(repo_id: str) -> None:
    """
    Creates a remote repository in modaic hub on the given repo_id.

    Args:
        repo_id: The path on Modaic hub to create the remote repository.

    Raises:
        RepoExistsError: If the repository already exists on the hub.
        AuthenticationError: If authentication fails or access is denied.
    """
    # TODO: Implement
