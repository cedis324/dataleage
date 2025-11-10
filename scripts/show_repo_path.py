"""Utility script to print the absolute path of the repository root."""
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    print(repo_root)


if __name__ == "__main__":
    main()
