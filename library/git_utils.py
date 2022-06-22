import logging
import os
import sys

log = logging.getLogger(__name__)


def is_repo_clean() -> bool:
    return not os.popen("git status --porcelain").read()


def get_latest_commit_hash() -> str:
    return os.popen("git rev-parse HEAD").read().strip()


def prompt_proceeding_with_dirty_repo() -> None:
    log.warning("Git repository is not clean. Continue? (y/n)")
    while (ans := input("-> ")).lower() not in ("y", "n"):
        print("Please input 'y' or 'n'")
    log.info(f"Answer: {ans}")
    if ans == "n":
        sys.exit(0)


def dump_commit_hash(debug: bool) -> None:
    if is_repo_clean():
        log.info("Git repository is clean. Dumping commit hash.")
        with open("commit_hash", "w") as f:
            f.write(get_latest_commit_hash())
    elif not debug:
        prompt_proceeding_with_dirty_repo()
    else:
        log.info("Git repository is not clean, but we're debugging. Skipping hash dump.")
