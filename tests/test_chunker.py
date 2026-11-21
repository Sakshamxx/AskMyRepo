"""Tests for cutting files into chunks."""

from pathlib import Path

import pytest

from askmyrepo.chunker import chunk_files, syntax_chunks, whole_file_chunks, window_chunks
from askmyrepo.manifest import CorpusFile

PYTHON = '''"""Stock rules."""

from datetime import timedelta

TTL = timedelta(minutes=15)


# Two checkouts can both pass this check.
def reserve(order):
    """Hold stock."""
    return order


@decorator
def release(entry):
    """Give stock back."""
    return entry
'''


def corpus_file(tmp_path: Path, name: str, text: str, language: str) -> CorpusFile:
    """Write a file and describe it as part of the corpus."""
    return CorpusFile("shop", name, language, "abc1234", text, "2.0")


def test_whole_file_is_one_chunk_covering_every_line(tmp_path):
    chunks = whole_file_chunks(corpus_file(tmp_path, "stock.py", PYTHON, "python"))
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, PYTHON.count("\n"))
    assert chunks[0].id == f"shop@2.0:stock.py:1-{PYTHON.count(chr(10))}"


def test_python_is_cut_on_functions_with_their_comments_and_imports(tmp_path):
    chunks = syntax_chunks(corpus_file(tmp_path, "stock.py", PYTHON, "python"))

    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        ("", 1, 5), ("reserve", 8, 11), ("release", 14, 17)]
    reserve = chunks[1]
    assert "# Two checkouts can both pass this check." in reserve.text  # the comment travels
    assert "from datetime import timedelta" in reserve.text             # so do the imports
    assert reserve.text.startswith("# File: shop/stock.py (version 2.0)\n# Symbol: reserve\n")
    assert "@decorator" in chunks[2].text


def test_a_long_class_is_cut_into_its_methods(tmp_path):
    methods = "".join(f"    def m{i}(self):\n" + '        """Doc."""\n' + "        pass\n" * 8
                      for i in range(10))
    text = f"class Big:\n    \"\"\"A big class.\"\"\"\n\n{methods}"
    chunks = syntax_chunks(corpus_file(tmp_path, "big.py", text, "python"))
    assert chunks[0].symbol == "Big"
    assert [c.symbol for c in chunks[1:]] == [f"Big.m{i}" for i in range(10)]


def test_markdown_is_cut_at_headings(tmp_path):
    text = "Intro line\n# Title\ntext\n## Part\nmore\n"
    chunks = syntax_chunks(corpus_file(tmp_path, "doc.md", text, "markdown"))
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        ("", 1, 1), ("Title", 2, 3), ("Part", 4, 5)]


def test_a_minified_file_is_one_chunk_however_long(tmp_path):
    text = "{" + ",".join(f'"k{i}":{i}' for i in range(5000)) + "}\n"
    chunks = syntax_chunks(corpus_file(tmp_path, "spec.json", text, "json"))
    assert len(chunks) == 1
    assert len(chunks[0].text) > 50_000


def test_other_files_are_cut_every_sixty_lines(tmp_path):
    text = "".join(f"SELECT {i};\n" for i in range(130))
    chunks = syntax_chunks(corpus_file(tmp_path, "views.sql", text, "sql"))
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (61, 120), (121, 130)]


def test_chunk_files_uses_the_chosen_mode(tmp_path):
    file = corpus_file(tmp_path, "stock.py", PYTHON, "python")
    assert len(chunk_files([file], "files")) == 1
    assert len(chunk_files([file], "syntax")) == 3
    assert len(chunk_files([file], "fixed", size=5)) == 4


def test_fixed_windows_can_overlap(tmp_path):
    text = "".join(f"line {i}\n" for i in range(1, 51))
    file = corpus_file(tmp_path, "notes.txt", text, "text")
    assert [(c.start_line, c.end_line) for c in window_chunks(file, size=20, overlap=5)] == [
        (1, 20), (16, 35), (31, 50)]
    assert [(c.start_line, c.end_line) for c in window_chunks(file, size=50)] == [(1, 50)]


def test_overlap_must_be_smaller_than_the_window(tmp_path):
    file = corpus_file(tmp_path, "notes.txt", "a\n", "text")
    with pytest.raises(ValueError, match="overlap"):
        window_chunks(file, size=10, overlap=10)


def test_chunks_carry_their_version_and_folders(tmp_path):
    file = CorpusFile("shop", "shoplite/services/stock.py", "python", "abc1234", PYTHON, "1.1",
                      latest=False)
    chunk = syntax_chunks(file)[1]
    assert chunk.metadata()["folders"] == ["shoplite", "shoplite/services"]
    assert (chunk.metadata()["version"], chunk.metadata()["latest"]) == ("1.1", False)
    assert chunk.id.startswith("shop@1.1:")
