# FIXME: make faster. Currently takes ~9 seconds
def push_folder_to_hub(
    folder: str,
    repo_path: str,
    access_token: Optional[str] = None,
    commit_message: str = "(no commit message)",
    private: bool = False,
    branch: str = "main",
    tag: str = None,
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

    if "/" not in repo_path:
        raise NotImplementedError(
            "Modaic fast paths not yet implemented. Please load programs with 'user/repo' or 'org/repo' format"
        )
    assert repo_path.count("/") <= 1, f"Extra '/' in repo_path: {repo_path}"
    # TODO: try pushing first and on error create the repo. create_remote_repo currently takes ~1.5 seconds to run
    create_remote_repo(repo_path, access_token, exist_ok=True, private=private)
    username = get_user_info(access_token)["login"]

    # FIXME: takes 6 seconds
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
        remote_url = f"https://{username}:{access_token}@{MODAIC_GIT_URL}/{repo_path}.git"
        try:
            local_repo.create_remote("origin", remote_url)
        except git.exc.GitCommandError:
            pass

        try:
            local_repo.git.fetch("origin")
        except git.exc.GitCommandError:
            raise RepositoryNotFoundError(f"Repository '{repo_path}' does not exist") from None

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
