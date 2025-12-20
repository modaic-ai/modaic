import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Union

import git
import requests
from dotenv import find_dotenv, load_dotenv
from git.repo.fun import BadName, BadObject, name_to_object

from .constants import MODAIC_CACHE, MODAIC_GIT_URL, MODAIC_TOKEN, PROGRAMS_CACHE, TEMP_DIR, USE_GITHUB
from .exceptions import (
    AuthenticationError,
    ModaicError,
    RepositoryExistsError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from .module_utils import copy_update_from, copy_update_program_dir, create_sync_dir, smart_link, sync_dir_from
from .utils import aggresive_rmtree

if TYPE_CHECKING:
    from .precompiled import PrecompiledProgram, Retriever

env_file = find_dotenv(usecwd=True)
load_dotenv(env_file)

user_info = None


@dataclass
class Commit:
    """
    Represents a commit in a git repository.
    Args:
        repo: The path to the git repository.
        sha: The full commit SHA.
    """

    repo: str
    sha: str

    def __repr__(self):
        return f"{self.repo}@{self.sha}"

    def __str__(self):
        return f"{self.repo}@{self.sha}"


def create_remote_repo(repo_path: str, access_token: str, exist_ok: bool = False, private: bool = False) -> bool:
    """
    Creates a remote repository in modaic hub on the given repo_path. e.g. "user/repo"

    Args:
        repo_path: The path on Modaic hub to create the remote repository.
        access_token: User's access token for authentication.


    Raises:
        AlreadyExists: If the repository already exists on the hub.
        AuthenticationError: If authentication fails or access is denied.
        ValueError: If inputs are invalid.

    Returns:
        True if the a new repository was created, False if it already existed.
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
            return True

        error_data = {}
        try:
            error_data = response.json()
        except Exception:
            pass

        error_message = error_data.get("message", f"HTTP {response.status_code}")

        if response.status_code == 409 or response.status_code == 422 or "already exists" in error_message.lower():
            if exist_ok:
                return False
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


def _has_ref(repo: git.Repo, ref: str) -> bool:
    try:
        repo.rev_parse(ref)
        return True
    except BadName:
        return False


def _attempt_push(repo: git.Repo, branch: str, tag: Optional[str] = None) -> None:
    refs = [branch]
    if tag:
        try:
            repo.git.tag(tag)
        except git.exc.GitCommandError:
            raise ModaicError(f"tag: {tag} already exists") from None
        refs.append(tag)

    try:
        repo.remotes.origin.push(refs)
    except git.exc.GitCommandError as e:  # handle nothing to push error
        raise ModaicError(f"Git push failed: {e}") from None


def sync_and_push(
    module: Union["PrecompiledProgram", "Retriever"],
    repo_path: str,
    access_token: Optional[str] = None,
    commit_message: str = "(no commit message)",
    private: bool = False,
    branch: str = "main",
    tag: str = None,
    with_code: bool = False,
) -> Commit:
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
    # First create the sync directory which will be used to update the git repository.
    if module._source is None:
        sync_dir = create_sync_dir(repo_path, with_code=with_code)
    else:
        sync_dir = sync_dir_from(module._source)
    save_auto_json = with_code and not module._source
    module.save_precompiled(sync_dir, _with_auto_classes=save_auto_json)

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
    create_remote_repo(repo_path, access_token, exist_ok=True, private=private)
    username = get_user_info(access_token)["login"]
    repo_dir = TEMP_DIR / repo_path
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Initialize git as git repo if not already initialized.
    repo = git.Repo.init(repo_dir)
    remote_url = f"https://{username}:{access_token}@{MODAIC_GIT_URL}/{repo_path}.git"
    try:
        if "origin" not in [r.name for r in repo.remotes]:
            repo.create_remote("origin", remote_url)
        else:
            repo.remotes.origin.set_url(remote_url)

        try:
            repo.remotes.origin.fetch()
        except git.exc.GitCommandError:
            raise RepositoryNotFoundError(f"Repository '{repo_path}' does not exist") from None

        # Handle main branch separately. Get latest version of main, add changes, and push.
        if branch == "main":
            try:
                repo.git.switch("-C", "main", "origin/main")
            except git.exc.GitCommandError:
                pass
            _sync_repo(sync_dir, repo_dir)
            repo.git.add("-A")
            # git commit exits non-zero when there is nothing to commit (clean tree).
            # Treat that as a no-op, but bubble up unexpected commit errors.
            _smart_commit(repo, commit_message)
            _attempt_push(repo, "main", tag)
            return Commit(repo_path, repo.head.commit.hexsha)

        # Ensure existence of main branch.
        # first attempt to sync main branch with origin
        try:
            repo.git.switch("-C", "main", "origin/main")
        # if that fails we must add changes to main and push.
        except git.exc.GitCommandError:
            _sync_repo(sync_dir, repo_dir)
            repo.git.add("-A")
            _smart_commit(repo, commit_message)
            repo.remotes.origin.push("main")

        # Now that main exists, switch to target branch and sync.
        # Switch to the branch or create it if it doesn't exist. And ensure it is up to date.
        try:
            repo.git.switch("-C", branch, f"origin/{branch}")
        except git.exc.GitCommandError:
            # if origin/branch does not exist this is a new branch
            # if source_commit is provided, start the new branch there
            if module._source_commit and _has_ref(repo, module._source_commit.sha):
                repo.git.switch("-C", branch, module._source_commit.sha)
            # otherwise start the new branch from main
            else:
                repo.git.switch("-C", branch)

        _sync_repo(sync_dir, repo_dir)
        repo.git.add("-A")

        # Handle error when working tree is clean (nothing to commit)
        _smart_commit(repo, commit_message)
        _attempt_push(repo, branch, tag)
        return Commit(repo_path, repo.head.commit.hexsha)
    except Exception as e:
        try:
            aggresive_rmtree(repo_dir)
        except Exception:
            raise ModaicError(
                f"Failed to cleanup MODAIC_CACHE after a failed operation. We recommend manually deleting your modaic cache as it may be corrupted. Your cache is located at {MODAIC_CACHE}"
            ) from e
        raise e


def _smart_commit(repo: git.Repo, commit_message: str) -> None:
    user_info = get_user_info(MODAIC_TOKEN)
    repo.git.config("user.email", user_info["email"])
    repo.git.config("user.name", user_info["name"])
    try:
        repo.git.commit("-m", commit_message)
    except git.exc.GitCommandError as e:
        if "nothing to commit" in str(e).lower():
            raise ModaicError("Nothing to commit") from e
        raise ModaicError("Git commit failed") from e


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
) -> Tuple[Path, Optional[Commit]]:
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
        if not (main_dir / ".git").exists():
            shutil.rmtree(main_dir, ignore_errors=True)
            git.Repo.clone_from(remote_url, main_dir, multi_options=["--branch", "main"])

        # Attatch origin
        main_repo = git.Repo(main_dir)
        if "origin" not in [r.name for r in main_repo.remotes]:
            main_repo.create_remote("origin", remote_url)
        else:
            main_repo.remotes.origin.set_url(remote_url)

        main_repo.remotes.origin.fetch()

        revision = resolve_revision(main_repo, rev)

        if revision.type == "commit" or revision.type == "tag":
            rev_dir = program_dir / revision.sha

            if not rev_dir.exists():
                main_repo.git.worktree("add", str(rev_dir.resolve()), revision.sha)

            shortcut_dir = program_dir / revision.name
            shortcut_dir.unlink(missing_ok=True)
            smart_link(shortcut_dir, rev_dir)

        elif revision.type == "branch":
            rev_dir = program_dir / revision.name

            if not rev_dir.exists():
                main_repo.git.worktree("add", str(rev_dir.resolve()), f"origin/{revision.name}")
            else:
                repo = git.Repo(rev_dir)
                repo.remotes.origin.pull(revision.name)

            # get the up to date sha for the branch
            revision = resolve_revision(main_repo, f"origin/{revision.name}")

        return rev_dir, Commit(repo_path, revision.sha)

    except Exception as e:
        try:
            aggresive_rmtree(program_dir)
        except Exception:
            raise ModaicError(
                f"Failed to cleanup MODAIC_CACHE after a failed operation. We recommend manually deleting your modaic cache as it may be corrupted. Your cache is located at {MODAIC_CACHE}"
            ) from e
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


def load_repo(repo_path: str, is_local: bool = False, rev: str = "main") -> Tuple[Path, Optional[Commit]]:
    if is_local:
        path = Path(repo_path)
        if not path.exists():
            raise FileNotFoundError(f"Local repo path {repo_path} does not exist")
        return path, None
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
    sha: str


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
        return Revision(type="branch", name=branch_name, sha=commit_sha)

    # refs/remotes/<remote>/<branch> (normalize branch name without remote, e.g., drop 'origin/')
    m_remote = re.match(r"^refs/remotes/(?P<remote>[^/]+)/(?P<branch>.+)$", ref.name)
    if m_remote:
        branch_name = m_remote.group("branch")
        commit_sha = ref.commit.hexsha
        return Revision(type="branch", name=branch_name, sha=commit_sha)

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


# Not in use currently
def _update_staging_dir(
    module: Union["PrecompiledProgram", "Retriever"],
    repo_dir: Path,
    repo_path: str,
    with_code: bool = False,
    source: Optional[Path] = None,
):
    # if source is not None then module was loaded with AutoProgram/AutoRetriever, we will use its source repo from MODAIC_CACHE/programs to update the repo_dir
    if source and sys.platform.startswith("win"):
        # Windows - source provided: Copy code from source into repo_dir
        copy_update_from(repo_dir, source)
    elif source and not sys.platform.startswith("win"):
        # Linux/Unix - source provided: Sync code from source into repo_dir (uses symlinks)
        sync_dir = sync_dir_from(source)
        _sync_repo(sync_dir, repo_dir)
    elif not source and sys.platform.startswith("win"):
        # Windows - no source provided: Copy code from workspace into repo_dir
        copy_update_program_dir(repo_dir, repo_path, with_code=with_code)
    elif not source and not sys.platform.startswith("win"):
        # Linux/Unix - no source provided: Sync code from workspace into repo_dir (uses symlinks)
        sync_dir = create_sync_dir(repo_path, with_code=with_code)
        _sync_repo(sync_dir, repo_dir)

    # save auto_classes.json only if we are saving the code and not using a source repo
    save_auto_json = with_code and not source
    module.save_precompiled(repo_dir, _with_auto_classes=save_auto_json)


def _sync_repo(sync_dir: Path, repo_dir: Path) -> None:
    """Syncs a 'sync' directory containing the a desired layout of symlinks to the source code files to the 'repo' directory a git repository tracked by modaic hub"""
    if sys.platform.startswith("win"):
        subprocess.run(
            [
                "robocopy",
                f"{sync_dir.resolve()}/",
                f"{repo_dir.resolve()}/",
                "/MIR",
                "/XD",
                ".git",  # make sure .git is not deleted
            ],
        )
    else:
        subprocess.run(
            [
                "rsync",
                "-aL",
                "--delete",
                "--ignore-times",  # rsync usually looks at edit times to determine if it should skip a file.  Disabling this behavior is useful for our pytest-suite.
                f"{sync_dir.resolve()}/",
                f"{repo_dir.resolve()}/",
                "--exclude",
                ".git",  # make sure .git is not deleted
            ],
        )
