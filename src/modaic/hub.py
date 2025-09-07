import os
from typing import Any, Dict, Optional

import git
import requests
from dotenv import load_dotenv

from .exceptions import AuthenticationError, ModaicHubError, RepositoryExistsError, RepositoryNotFoundError

load_dotenv()

MODAIC_TOKEN = os.getenv("MODAIC_TOKEN")
MODAIC_HUB_URL = os.getenv("MODAIC_HUB_URL", "git.modaic.dev").replace("https://", "").rstrip("/")

USE_GITHUB = "github.com" in MODAIC_HUB_URL

user_info = None


def create_remote_repo(repo_path: str, access_token: str, exist_ok=False) -> None:
    """
    Creates a remote repository in modaic hub on the given repo_path. e.g. "user/repo"

    Args:
        repo_path: The path on Modaic hub to create the remote repository.
        access_token: User's access token for authentication.


    Raises:
        AlreadyExists: If the repository already exists on the hub.
        AuthenticationError: If authentication fails or access is denied.
        ValueError: If inputs are invalid.
    """
    if not repo_path or not repo_path.strip():
        raise ValueError("Repository ID cannot be empty")

    repo_name = repo_path.strip().split("/")[-1]

    if len(repo_name) > 100:
        raise ValueError("Repository name too long (max 100 characters)")

    api_url = get_repos_endpoint()

    headers = get_headers(access_token)

    payload = get_repo_payload(repo_name)
    # TODO: Implement orgs path. Also switch to using gitea's push-to-create

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        # print(response.json())
        # print(response.status_code)

        if response.status_code == 201:
            return

        error_data = {}
        try:
            error_data = response.json()
        except Exception:
            pass

        error_message = error_data.get("message", f"HTTP {response.status_code}")

        if response.status_code == 409 or response.status_code == 422 or "already exists" in error_message.lower():
            if exist_ok:
                return
            else:
                raise RepositoryExistsError(f"Repository '{repo_name}' already exists")
        elif response.status_code == 401:
            raise AuthenticationError("Invalid access token or authentication failed")
        elif response.status_code == 403:
            raise AuthenticationError("Access denied - insufficient permissions")
        else:
            raise ModaicHubError(f"Failed to create repository: {error_message}")

    except requests.exceptions.RequestException as e:
        raise ModaicHubError(f"Request failed: {str(e)}")


def push_folder_to_hub(
    folder,
    repo_path,
    access_token: Optional[str] = None,
    commit_message="(no commit message)",
):
    """
    Pushes a local directory as a commit to a remote git repository.
    Steps:
    1. If local folder is not a git repository, initialize it.
    2. Checkout to a temporary 'snapshot' branch.
    3. Add and commit all files in the local folder.
    4. Add origin to local repository (if not already added) and fetch it
    5. Switch to the 'main' branch at origin/main
    6. use `git restore --source=snapshot --staged --worktree .` to sync working tree of 'main' to 'snapshot' and stage changes to 'main'
    7. Commit changes to 'main' with custom commit message
    8. Fast forward push to origin/main
    9. Delete the 'snapshot' branch

    Args:
        folder: The local folder to push to the remote repository.
        namespace: The namespace of the remote repository. e.g. "user" or "org"
        repo_name: The name of the remote repository. e.g. "repo"
        access_token: The access token to use for authentication.
        commit_message: The message to use for the commit.

    Warning:
        This is not the standard pull/push workflow. No merging/rebasing is done.
        This simply pushes new changes to make main mirror the local directory.

    Warning:
        Assumes that the remote repository exists
    """
    if not access_token and MODAIC_TOKEN:
        access_token = MODAIC_TOKEN
    elif not access_token and not MODAIC_TOKEN:
        raise AuthenticationError("MODAIC_TOKEN is not set")

    if "/" not in repo_path:
        raise NotImplementedError(
            "Modaic fast paths not yet implemented. Please load agents with 'user/repo' or 'org/repo' format"
        )
    assert repo_path.count("/") <= 1, f"Extra '/' in repo_path: {repo_path}"

    create_remote_repo(repo_path, access_token, exist_ok=True)
    username = get_user_info(access_token)["login"]

    try:
        # 1) If local folder is not a git repository, initialize it.
        local_repo = git.Repo.init(folder)
        # 2) Checkout to a temporary 'snapshot' branch (create or reset if exists).
        local_repo.git.switch("-C", "snapshot")
        # 3) Add and commit all files in the local folder.
        if local_repo.is_dirty(untracked_files=True):
            local_repo.git.add("-A")
            local_repo.git.commit("-m", "Local snapshot before transplant")
        # 4) Add origin to local repository (if not already added) and fetch it
        remote_url = f"https://{username}:{access_token}@{MODAIC_HUB_URL}/{repo_path}.git"
        try:
            local_repo.create_remote("origin", remote_url)
        except git.exc.GitCommandError:
            pass

        try:
            local_repo.git.fetch("origin")
        except git.exc.GitCommandError:
            raise RepositoryNotFoundError(f"Repository '{repo_path}' does not exist")

        # 5) Switch to the 'main' branch at origin/main
        local_repo.git.switch("-C", "main", "origin/main")

        # 4) Make main’s index + working tree EXACTLY match snapshot (incl. deletions)
        local_repo.git.restore("--source=snapshot", "--staged", "--worktree", ".")

        # 5) One commit that transforms remote contents into your local snapshot
        if local_repo.is_dirty(untracked_files=True):
            local_repo.git.commit("-m", commit_message)

        # 6) Fast-forward push: preserves prior remote history + your single commit
        local_repo.git.push("-u", "origin", "main")
    finally:
        # clean up - switch to main and delete snapshot branch
        try:
            local_repo.git.switch("main")
        except git.exc.GitCommandError:
            local_repo.git.switch("-c", "main")
        local_repo.git.branch("-D", "snapshot")


def get_headers(access_token: str) -> Dict[str, str]:
    # print("ACCESS TOKEN", access_token)
    if USE_GITHUB:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {access_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    else:
        return {
            "Authorization": f"token {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ModaicClient/1.0",
        }


def get_repos_endpoint() -> str:
    if USE_GITHUB:
        return "https://api.github.com/user/repos"
    else:
        return f"https://{MODAIC_HUB_URL}/api/v1/user/repos"


def get_repo_payload(repo_name: str) -> Dict[str, Any]:
    payload = {
        "name": repo_name,
        "description": "",
        "private": False,
        "auto_init": True,
        "default_branch": "main",
    }
    if not USE_GITHUB:
        payload["trust_model"] = "default"
    return payload


def get_user_info(access_token: str) -> Dict[str, Any]:
    """
    Returns the user info for the given access token.
    Caches the user info in the global user_info variable.

    Args:
        access_token: The access token to get the user info for.

    Returns:
        {
            "login": str,
            "email": str,
            "avatar_url": str,
            "name": str,
        }
    """
    global user_info
    if user_info:
        return user_info
    if USE_GITHUB:
        response = requests.get("https://api.github.com/user", headers=get_headers(access_token)).json()
        user_info = {
            "login": response["login"],
            "email": response["email"],
            "avatar_url": response["avatar_url"],
            "name": response["name"],
        }
    else:
        response = requests.get(f"https://{MODAIC_HUB_URL}/api/v1/user", headers=get_headers(access_token)).json()
        user_info = {
            "login": response["login"],
            "email": response["email"],
            "avatar_url": response["avatar_url"],
            "name": response["full_name"],
        }
    return user_info


if __name__ == "__main__":
    pass
    # print(get_user_info(MODAIC_TOKEN))
