"""Tests for the corpus manifest."""

import subprocess

import pytest

from askmyrepo.manifest import list_corpus_files, matches_any


def make_repo(folder, files: dict) -> None:
    """Create a Git repository with these files committed."""
    folder.mkdir()
    for path, text in files.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_text(text)
    for command in (["init", "-q"], ["add", "."],
                    ["-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(folder), *command], check=True)


@pytest.mark.parametrize("path, pattern, expected", [
    ("app/models/user.py", "**/*.py", True),
    ("setup.py", "**/*.py", True),             # ** also matches no folders at all
    ("docs/openapi.json", "docs/**", True),
    ("docs/api.md", "*.md", False),            # * stays inside one folder
    ("tests/test_a.py", "tests/**", True),
    ("app/tests_helper.py", "tests/**", False),
])
def test_patterns_match_like_globs(path, pattern, expected):
    assert matches_any(path, [pattern]) is expected


def test_manifest_keeps_included_tracked_files_and_drops_excluded_ones(tmp_path):
    make_repo(tmp_path / "shop", {"app/cart.py": "x = 1\n", "tests/test_cart.py": "x = 1\n",
                                  "README.md": "# Shop\n", "data.csv": "a,b\n"})
    (tmp_path / "shop" / "untracked.py").write_text("x = 1\n")
    manifest = tmp_path / "work" / "corpus.toml"
    manifest.parent.mkdir()
    manifest.write_text('[[repository]]\nname = "shop"\npath = "../shop"\n'
                        'include = ["**/*.py", "**/*.md"]\nexclude = ["tests/**"]\n')

    files = list_corpus_files(manifest)

    assert [(f.path, f.language) for f in files] == [("README.md", "markdown"),
                                                     ("app/cart.py", "python")]
    assert all(f.repo == "shop" and len(f.commit) >= 7 for f in files)
    assert files[1].text == "x = 1\n" and files[1].latest


def test_each_listed_version_is_read_from_git(tmp_path):
    make_repo(tmp_path / "shop", {"app/cart.py": "OLD = 1\n"})
    git = ["git", "-C", str(tmp_path / "shop")]
    subprocess.run([*git, "tag", "v1"], check=True)
    (tmp_path / "shop" / "app" / "cart.py").write_text("NEW = 2\n")
    subprocess.run([*git, "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "commit", "-qam", "y"], check=True)
    manifest = tmp_path / "corpus.toml"
    manifest.write_text('[[repository]]\nname = "shop"\npath = "shop"\ninclude = ["**/*.py"]\n'
                        'versions = [{ref = "HEAD", version = "2"}, {ref = "v1", version = "1"}]\n')

    files = list_corpus_files(manifest)

    assert [(f.version, f.latest, f.text) for f in files] == [
        ("2", True, "NEW = 2\n"), ("1", False, "OLD = 1\n")]


def test_a_missing_repository_is_a_clear_error(tmp_path):
    manifest = tmp_path / "corpus.toml"
    manifest.write_text('[[repository]]\nname = "shop"\npath = "../nowhere"\ninclude = ["**"]\n')
    with pytest.raises(FileNotFoundError, match="shop: no Git repository"):
        list_corpus_files(manifest)
