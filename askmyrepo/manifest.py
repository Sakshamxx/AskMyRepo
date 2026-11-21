"""Read corpus.toml and list the files AskMyRepo should index."""

import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = {".py": "python", ".md": "markdown", ".sql": "sql", ".json": "json",
             ".toml": "toml"}


# One file to index, with its repository, version and commit.
@dataclass
class CorpusFile:
    """One file to index, and where it came from."""

    repo: str        # the repository's name in corpus.toml, e.g. "shoplite"
    path: str        # the path inside the repository, e.g. "shoplite/services/stock.py"
    language: str    # python, markdown, sql, json, toml or text
    commit: str      # the last commit that changed the file (short hash)
    text: str        # the file's contents at that version
    version: str = ""     # the release it belongs to, e.g. "2.0"
    latest: bool = True   # True for the newest version listed for its repository


# Makes corpus.toml's glob patterns (*, **) usable as regexes.
def pattern_to_regex(pattern: str) -> re.Pattern:
    """Turn a pattern such as docs/** or **/*.py into a regular expression."""
    regex = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            regex += "(?:.*/)?"  # any number of folders, including none
            i += 3
        elif pattern.startswith("**", i):
            regex += ".*"
            i += 2
        elif pattern[i] == "*":
            regex += "[^/]*"     # anything inside one folder
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return re.compile(regex + "$")


# True if a path matches any include or exclude pattern.
def matches_any(path: str, patterns: list[str]) -> bool:
    """True if the path matches at least one pattern."""
    return any(pattern_to_regex(pattern).match(path) for pattern in patterns)


# Thin wrapper to run git in a repository.
def git(repo_dir: Path, *args: str) -> str:
    """Run a git command in a repository and return what it prints."""
    return subprocess.run(["git", "-C", str(repo_dir), *args], check=True,
                          capture_output=True, text=True).stdout


# Lists files Git tracks at a ref, so junk like .venv never gets in.
def tracked_files(repo_dir: Path, ref: str) -> list[str]:
    """Every file Git tracks at this ref (a branch, tag or commit), sorted."""
    return sorted(git(repo_dir, "ls-tree", "-r", "--name-only", ref).splitlines())


# Records which commit last changed a file (kept as chunk metadata).
def last_commit(repo_dir: Path, ref: str, path: str) -> str:
    """The short hash of the last commit up to ref that changed this file."""
    return git(repo_dir, "log", "-1", "--format=%h", ref, "--", path).strip()


# Reads the repository blocks from corpus.toml.
def load_manifest(manifest_path: Path) -> list[dict]:
    """Read the [[repository]] blocks from corpus.toml."""
    with open(manifest_path, "rb") as f:
        return tomllib.load(f)["repository"]


# Applies the manifest: the full list of files to index.
def list_corpus_files(manifest_path: Path) -> list[CorpusFile]:
    """Apply the manifest: every included, not excluded file of every repository and version."""
    files = []
    for repo in load_manifest(manifest_path):
        repo_dir = (manifest_path.parent / repo["path"]).resolve()
        if not (repo_dir / ".git").exists():
            raise FileNotFoundError(f"{repo['name']}: no Git repository at {repo_dir}")
        # The first version listed is the newest. No list means the checked-out commit.
        versions = repo.get("versions", [{"ref": "HEAD", "version": ""}])
        for n, release in enumerate(versions):
            ref = release["ref"]
            for path in tracked_files(repo_dir, ref):
                excluded = matches_any(path, repo.get("exclude", []))
                if matches_any(path, repo["include"]) and not excluded:
                    files.append(CorpusFile(
                        repo["name"], path, LANGUAGES.get(Path(path).suffix, "text"),
                        last_commit(repo_dir, ref, path), git(repo_dir, "show", f"{ref}:{path}"),
                        release["version"], latest=(n == 0)))
    return files
