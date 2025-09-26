from git import Git
import os
from pathlib import Path
from git import Repo

here = Path(__file__).parent
repo = Repo(here)

# Current commit object
commit = repo.head.commit

# Full commit SHA
print(commit.hexsha)