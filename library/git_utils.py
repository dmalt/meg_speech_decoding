import os


def is_repo_clean() -> bool:
    return not os.popen("git status --porcelain").read()


def get_latest_commit_hash() -> str:
    return os.popen("git rev-parse HEAD").read().strip()
