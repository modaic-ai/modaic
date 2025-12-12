import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import git
import requests
from dotenv import find_dotenv, load_dotenv
from git.repo.fun import BadName, BadObject, name_to_object

from .exceptions import (
    AuthenticationError,
    ModaicError,
    RepositoryExistsError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from .utils import Timer, compute_cache_dir

env_file = find_dotenv(usecwd=True)
load_dotenv(env_file)

from .constants import MODAIC_GIT_URL, MODAIC_TOKEN, PROGRAMS_CACHE, TEMP_DIR, USE_GITHUB

user_info = None


def create_remote_repo(repo_path: str, access_token: str, exist_ok: bool = False, private: bool = False) -> None:
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

    payload = get_repo_payload(repo_name, private=private)
    # TODO: Implement orgs path. Also switch to using gitea's push-to-create

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)

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
            raise Exception(f"Failed to create repository: {error_message}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}") from e


def sync_and_push(
    sync_dir: Path,
    repo_path: str,
    access_token: Optional[str] = None,
    commit_message: str = "(no commit message)",
    private: bool = False,
    branch: str = "main",
    tag: str = None,
):
    """
    1. Syncs a non-git repository to a git repository.
    2. Pushes the git repository to modaic hub.

    Args:
        sync_dir: The 'sync' directory containing the desired layout of symlinks to the source code files.
        repo_path: The path on Modaic hub to create the remote repository. e.g. "user/repo"
        access_token: The access token to use for authentication.
        commit_message: The message to use for the commit.
        private: Whether the repository should be private. Defaults to False.
        branch: The branch to push to. Defaults to "main".
        tag: The tag to push to. Defaults to None.
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

    if "/" in branch:
        raise ModaicError(
            f"Branch name '{branch}' is invalid. Must be a single branch name without any remote prefix (e.g., 'main', not 'origin/main')"
        )

    if "/" not in repo_path:
        raise NotImplementedError(
            "Modaic fast paths not yet implemented. Please load programs with 'user/repo' or 'org/repo' format"
        )
    assert repo_path.count("/") <= 1, f"Extra '/' in repo_path: {repo_path}"
    # TODO: try pushing first and on error create the repo. create_remote_repo currently takes ~1.5 seconds to run
    remote_repo_timer = Timer("create_remote_repo")
    create_remote_repo(repo_path, access_token, exist_ok=True, private=private)
    remote_repo_timer.done()
    username_timer = Timer("get_username")
    username = get_user_info(access_token)["login"]
    username_timer.done()
    repo_dir = TEMP_DIR / repo_path
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Initialize git as git repo if not already initialized.
    init_git_timer = Timer("init_git")
    repo = git.Repo.init(repo_dir)
    remote_url = f"https://{username}:{access_token}@{MODAIC_GIT_URL}/{repo_path}.git"

    if "origin" not in [r.name for r in repo.remotes]:
        repo.create_remote("origin", remote_url)
    else:
        repo.remotes.origin.set_url(remote_url)

    try:
        repo.remotes.origin.fetch()
    except git.exc.GitCommandError:
        raise RepositoryNotFoundError(f"Repository '{repo_path}' does not exist") from None

    # Switch to the branch or create it if it doesn't exist. And ensure it is up to date.
    try:
        repo.git.switch("-C", branch, f"origin/{branch}")
    except git.exc.GitCommandError:
        repo.git.branch("-C", branch)
    init_git_timer.done()

    sync_repo_timer = Timer("sync_repo")
    _sync_repo(sync_dir, repo_dir)
    sync_repo_timer.done()

    commit_and_push_timer = Timer("commit_and_push")
    repo.git.add("-A")
    try:
        repo.git.commit("-m", commit_message)
    except git.exc.GitCommandError:
        return
    if tag:
        repo.git.tag(tag)

    # Handle error when working tree is clean (nothing to push)
    repo.remotes.origin.push()
    commit_and_push_timer.done()


def get_headers(access_token: str) -> Dict[str, str]:
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
        return f"https://{MODAIC_GIT_URL}/api/v1/user/repos"


def get_repo_payload(repo_name: str, private: bool = False) -> Dict[str, Any]:
    payload = {
        "name": repo_name,
        "description": "",
        "private": private,
        "auto_init": True,
        "default_branch": "main",
    }
    if not USE_GITHUB:
        payload["trust_model"] = "default"
    return payload


# TODO: add persistent filesystem based cache mapping access_token to user_info. Currently takes ~1 second
def get_user_info(access_token: str) -> Dict[str, Any]:
    """
    Returns the user info for the given access token.
    Caches the user info in the global user_info variable.

    Args:
        access_token: The access token to get the user info for.

    Returns:
    ```python
        {
            "login": str,
            "email": str,
            "avatar_url": str,
            "name": str,
        }
    ```
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
        response = requests.get(f"https://{MODAIC_GIT_URL}/api/v1/user", headers=get_headers(access_token)).json()
        user_info = {
            "login": response["login"],
            "email": response["email"],
            "avatar_url": response["avatar_url"],
            "name": response["full_name"],
        }
    return user_info


# TODO:
def git_snapshot(
    repo_path: str,
    *,
    rev: str = "main",
    access_token: Optional[str] = None,
) -> Path:
    """
    Ensure a local cached checkout of a hub repository and return its path.

    Args:
      repo_path: Hub path ("user/repo").
      rev: Branch, tag, or full commit SHA to checkout; defaults to "main".

    Returns:
      Absolute path to the local cached repository under PROGRAMS_CACHE/repo_path.
    """

    if access_token is None and MODAIC_TOKEN is not None:
        access_token = MODAIC_TOKEN
    elif access_token is None:
        raise ValueError("Access token is required")

    program_dir = Path(PROGRAMS_CACHE) / repo_path
    main_dir = program_dir / "main"

    username = get_user_info(access_token)["login"]
    try:
        main_dir.parent.mkdir(parents=True, exist_ok=True)

        remote_url = f"https://{username}:{access_token}@{MODAIC_GIT_URL}/{repo_path}.git"

        # Ensure we have a main checkout at program_dir/main
        if not main_dir.exists():
            git.Repo.clone_from(remote_url, main_dir, branch="main")

        # Attatch origin
        main_repo = git.Repo(main_dir)
        if "origin" not in [r.name for r in main_repo.remotes]:
            main_repo.create_remote("origin", remote_url)
        else:
            main_repo.remotes.origin.set_url(remote_url)

        main_repo.remotes.origin.fetch()

        revision = resolve_revision(main_dir, rev)

        if revision.type == "commit" or revision.type == "tag":
            rev_dir = program_dir / revision.sha

            if not rev_dir.exists():
                rev_dir.git.worktree("add", str(rev_dir.resolve()), revision.sha)

            shortcut_dir = program_dir / revision.name
            shortcut_dir.unlink(missing_ok=True)
            shortcut_dir.symlink_to(rev_dir, target_is_directory=True)

        elif revision.type == "branch":
            rev_dir = program_dir / revision.name

            if not rev_dir.exists():
                rev_dir.git.worktree("add", str(rev_dir.resolve()), f"origin/{revision.name}")
            else:
                repo = git.Repo(rev_dir)
                repo.remotes.origin.pull()

    except Exception as e:
        shutil.rmtree(program_dir)
        raise e


def _move_to_commit_sha_folder(repo: git.Repo) -> git.Repo:
    """
    Moves the repo to a new path based on the commit SHA. (Unused for now)
    Args:
        repo: The git.Repo object.

    Returns:
        The new git.Repo object.
    """
    commit = repo.head.commit
    repo_dir = Path(repo.working_dir)
    new_path = repo_dir / commit.hexsha
    repo_dir.rename(new_path)
    return git.Repo(new_path)


def load_repo(repo_path: str, is_local: bool = False, rev: str = "main") -> Path:
    if is_local:
        path = Path(repo_path)
        if not path.exists():
            raise FileNotFoundError(f"Local repo path {repo_path} does not exist")
        return path
    else:
        return git_snapshot(repo_path, rev=rev)


@dataclass
class Revision:
    """
    Represents a revision of a git repository.
    Args:
        type: The type of the revision. e.g. "branch", "tag", "commit"
        name: The name of the revision. e.g. "main", "v1.0.0", "1234567"
        sha: Full commit SHA of the revision. e.g. "1234567890abcdef1234567890abcdef12345678" (None for branches)
    """

    type: Literal["branch", "tag", "commit"]
    name: str
    sha: Optional[str] = None


def resolve_revision(repo: git.Repo, rev: str) -> Revision:
    """
    Resolves the revision to a branch, tag, or commit SHA.
    Args:
        repo: The git.Repo object.
        rev: The revision to resolve.

    Returns:
        Revision dataclass where:
          - type ∈ {"branch", "tag", "commit"}
          - name is the normalized name:
              - branch: branch name without any remote prefix (e.g., "main", not "origin/main")
              - tag: tag name (e.g., "v1.0.0")
              - commit: full commit SHA
          - sha is the target commit SHA for branch/tag, or the commit SHA itself for commit
    Raises:
        ValueError: If the revision is not a valid branch, tag, or commit SHA.

    Example:
        >>> resolve_revision(repo, "main")
        Revision(type="branch", name="main", sha="<sha>")
        >>> resolve_revision(repo, "v1.0.0")
        Revision(type="tag", name="v1.0.0", sha="<sha>")
        >>> resolve_revision(repo, "1234567890")
        Revision(type="commit", name="<sha>", sha="<sha>")
    """
    repo.remotes.origin.fetch()

    # Fast validation of rev; if not found, try origin/<rev> for branches existing only on remote
    try:
        ref = repo.rev_parse(rev)
    except BadName:
        try:
            ref = repo.rev_parse(f"origin/{rev}")
        except BadName:
            raise RevisionNotFoundError(
                f"Revision '{rev}' is not a valid branch, tag, or commit SHA", rev=rev
            ) from None
        else:
            rev = f"origin/{rev}"

    if not isinstance(ref, git.objects.Commit):
        raise RevisionNotFoundError(f"Revision '{rev}' is not a valid branch, tag, or commit SHA", rev=rev) from None

    # Try to resolve to a reference where possible (branch/tag), else fallback to commit
    try:
        ref = name_to_object(repo, rev, return_ref=True)
    except BadObject:
        pass

    # Commit SHA case
    if isinstance(ref, git.objects.Commit):
        full_sha = ref.hexsha
        return Revision(type="commit", name=full_sha[:7], sha=full_sha)

    # refs/tags/<tag>
    m_tag = re.match(r"^refs/tags/(?P<tag>.+)$", ref.name)
    if m_tag:
        tag_name = m_tag.group("tag")
        commit_sha = ref.commit.hexsha  # TagReference.commit returns the peeled commit
        return Revision(type="tag", name=tag_name, sha=commit_sha)

    # refs/heads/<branch>
    m_head = re.match(r"^refs/heads/(?P<branch>.+)$", ref.name)
    if m_head:
        branch_name = m_head.group("branch")
        commit_sha = ref.commit.hexsha
        return Revision(type="branch", name=branch_name, sha=None)

    # refs/remotes/<remote>/<branch> (normalize branch name without remote, e.g., drop 'origin/')
    m_remote = re.match(r"^refs/remotes/(?P<remote>[^/]+)/(?P<branch>.+)$", ref.name)
    if m_remote:
        branch_name = m_remote.group("branch")
        commit_sha = ref.commit.hexsha
        return Revision(type="branch", name=branch_name, sha=None)

    # Some refs may present as "<remote>/<branch>" or just "<branch>" in name; handle common forms
    m_remote_simple = re.match(r"^(?P<remote>[^/]+)/(?P<branch>.+)$", ref.name)
    if m_remote_simple:
        branch_name = m_remote_simple.group("branch")
        commit_sha = ref.commit.hexsha
        return Revision(type="branch", name=branch_name, sha=commit_sha)

    # If we still haven't matched, attempt to treat as a tag/branch name directly
    # Try heads/<name>
    try:
        possible_ref = name_to_object(repo, f"refs/heads/{ref.name}", return_ref=True)
        commit_sha = possible_ref.commit.hexsha
        return Revision(type="branch", name=ref.name, sha=commit_sha)
    except Exception:
        pass
    # Try tags/<name>
    try:
        possible_ref = name_to_object(repo, f"refs/tags/{ref.name}", return_ref=True)
        commit_sha = possible_ref.commit.hexsha
        return Revision(type="tag", name=ref.name, sha=commit_sha)
    except Exception:
        pass

    # As a last resort, if it peels to a commit, return commit
    try:
        commit_obj = repo.commit(ref.name)
        full_sha = commit_obj.hexsha
        return Revision(type="commit", name=full_sha, sha=full_sha)
    except Exception:
        raise RevisionNotFoundError(f"Revision '{rev}' is not a valid branch, tag, or commit SHA", rev=rev) from None


def _sync_repo(sync_dir: Path, repo_dir: Path) -> None:
    """Syncs a 'sync' directory containing the a desired layout of symlinks to the source code files to the 'repo' directory a git repository tracked by modaic hub"""
    if sys.platform.startswith("win"):
        subprocess.run(["robocopy", str(sync_dir.resolve()), str(repo_dir.resolve()), "/MIR"])
    else:
        subprocess.run(["rsync", "-aL", "--delete", str(sync_dir.resolve()), str(repo_dir.resolve())])
