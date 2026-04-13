import os
import runpy
import sys


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    runpy.run_path(os.path.join(repo_root, "scripts", "run_iql.py"), run_name="__main__")


if __name__ == "__main__":
    main()

