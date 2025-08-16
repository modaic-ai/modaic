import requests
from typing import Optional


class RepoExistsError(Exception):
    """Raised when repository already exists"""

    pass


class AuthenticationError(Exception):
    """Raised when authentication fails"""

    pass


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


def create_remote_repo(
    repo_id: str, access_token: str, base_url: Optional[str] = "https://git.modaic.dev"
) -> None:
    """
    Creates a remote repository in modaic hub on the given repo_id.

    Args:
        repo_id: The path on Modaic hub to create the remote repository.
        access_token: User's access token for authentication.
        base_url: Base URL of the Gitea instance (optional, defaults to settings if available).

    Raises:
        RepoExistsError: If the repository already exists on the hub.
        AuthenticationError: If authentication fails or access is denied.
        ValueError: If inputs are invalid.
    """
    if not repo_id or not repo_id.strip():
        raise ValueError("Repository ID cannot be empty")

    if not access_token or not access_token.strip():
        raise ValueError("Access token cannot be empty")

    repo_name = repo_id.strip().split("/")[-1]

    if len(repo_name) > 100:
        raise ValueError("Repository name too long (max 100 characters)")

    if not base_url:
        base_url = "https://git.modaic.dev"

    api_url = f"{base_url.rstrip('/')}/api/v1/user/repos"

    headers = {
        "Authorization": f"token {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ModaicClient/1.0",
    }

    repo_data = {
        "name": repo_name,
        "description": "",
        "private": False,
        "auto_init": True,
        "default_branch": "main",
        "trust_model": "default",
    }

    try:
        response = requests.post(api_url, json=repo_data, headers=headers, timeout=30)

        if response.status_code == 201:
            return

        error_data = {}
        try:
            error_data = response.json()
        except Exception:
            pass

        error_message = error_data.get("message", f"HTTP {response.status_code}")

        if response.status_code == 409 or "already exists" in error_message.lower():
            raise RepoExistsError(f"Repository '{repo_name}' already exists")
        elif response.status_code == 401:
            raise AuthenticationError("Invalid access token or authentication failed")
        elif response.status_code == 403:
            raise AuthenticationError("Access denied - insufficient permissions")
        else:
            raise Exception(f"Failed to create repository: {error_message}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")
