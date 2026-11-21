"""Cut files into passages (chunks): whole files, or on function and class boundaries."""

import re
from dataclasses import asdict, dataclass

from tree_sitter_language_pack import get_parser

from askmyrepo.manifest import CorpusFile

DEFINITIONS = {"function_definition", "class_definition", "decorated_definition"}
IMPORTS = {"import_statement", "import_from_statement", "future_import_statement"}
MAX_CLASS_LINES = 80  # a longer class is cut into its methods
WINDOW_LINES = 60     # files with no parser (SQL, JSON...) are cut every 60 lines
HEADING = re.compile(r"^#{1,3} ")


# The unit that gets embedded, indexed and cited.
@dataclass
class Chunk:
    """One passage of one file, with the metadata a search can filter on."""

    repo: str
    path: str
    language: str
    symbol: str      # the function, class or heading it holds; "" for none
    start_line: int  # first line in the file, counting from 1
    end_line: int    # last line, inclusive
    commit: str
    text: str        # what is embedded and shown: a short header, then the lines
    version: str = ""     # the release, e.g. "2.0"
    latest: bool = True   # from the newest version of its repository

    # Stable id used as the vector key in Pinecone and the keyword index.
    @property
    def id(self) -> str:
        """A unique id for the vector store, e.g. shoplite@2.0:shoplite/utils/ids.py:18-20."""
        return f"{self.repo}@{self.version}:{self.path}:{self.start_line}-{self.end_line}"

    # Folder list that lets searches filter by folder.
    @property
    def folders(self) -> list[str]:
        """Every folder the file sits in, e.g. ["shoplite", "shoplite/services"]."""
        parts = self.path.split("/")[:-1]
        return ["/".join(parts[:n]) for n in range(1, len(parts) + 1)]

    # Fields stored next to the vector in Pinecone.
    def metadata(self) -> dict:
        """Everything except the id, for storing next to the vector (folders for filtering)."""
        return {**asdict(self), "folders": self.folders}


# Text prepended to a chunk so it carries its file, symbol and imports.
def header(file: CorpusFile, symbol: str, imports: str = "") -> str:
    """The context that travels with a chunk: its file, its symbol and the file's imports."""
    version = f" (version {file.version})" if file.version else ""
    lines = [f"# File: {file.repo}/{file.path}{version}"]
    if symbol:
        lines.append(f"# Symbol: {symbol}")
    if imports:
        lines.append(imports)
    return "\n".join(lines) + "\n\n"


# Builds a Chunk from a line range of one file.
def make_chunk(file: CorpusFile, lines: list[str], start: int, end: int, symbol: str = "",
               imports: str = "") -> Chunk:
    """Build a chunk from lines start..end (counting from 1, inclusive)."""
    body = "\n".join(lines[start - 1:end])
    return Chunk(file.repo, file.path, file.language, symbol, start, end, file.commit,
                 header(file, symbol, imports) + body, file.version, file.latest)


# Simplest strategy: one chunk per file.
def whole_file_chunks(file: CorpusFile) -> list[Chunk]:
    """The whole file as one chunk (the simplest thing that works)."""
    lines = file.text.splitlines()
    return [make_chunk(file, lines, 1, max(len(lines), 1))]


# Reads the name of a function or class syntax node.
def definition_name(node) -> str:
    """The name of a function or class node (looking inside decorators)."""
    if node.type == "decorated_definition":
        node = node.child_by_field_name("definition")
    return node.child_by_field_name("name").text.decode()


# Distinguishes classes from functions, decorated or not.
def is_class(node) -> bool:
    """True for a class definition, with or without decorators."""
    if node.type == "decorated_definition":
        node = node.child_by_field_name("definition")
    return node.type == "class_definition"


# Pulls comments directly above a definition into its chunk.
def start_with_comments(node) -> int:
    """The node's first row, moved up over the comment lines directly above it."""
    start = node.start_point.row
    previous = node.prev_sibling
    while (previous is not None and previous.type == "comment"
           and previous.end_point.row == start - 1):
        start = previous.start_point.row
        previous = previous.prev_sibling
    return start


# Splits an oversized class into its head and one chunk per method.
def class_body_chunks(file, lines, node, start_row, imports) -> list[Chunk]:
    """Cut a long class into its opening part and one chunk per method."""
    class_node = node
    if node.type == "decorated_definition":
        class_node = node.child_by_field_name("definition")
    class_name = definition_name(node)
    methods = [child for child in class_node.child_by_field_name("body").children
               if child.type in DEFINITIONS]
    chunks = []
    first_method_row = start_with_comments(methods[0]) if methods else node.end_point.row + 1
    chunks.append(make_chunk(file, lines, start_row + 1, first_method_row, class_name, imports))
    for method in methods:
        name = f"{class_name}.{definition_name(method)}"
        chunks.append(make_chunk(file, lines, start_with_comments(method) + 1,
                                 method.end_point.row + 1, name, imports))
    return chunks


# Python strategy: one chunk per function or class, plus loose top-level code.
def python_chunks(file: CorpusFile) -> list[Chunk]:
    """Cut Python on function and class boundaries, keeping each one's comments and imports."""
    source = file.text.encode("utf-8")
    lines = file.text.splitlines()
    root = get_parser("python").parse(source).root_node
    imports = "\n".join(child.text.decode() for child in root.children if child.type in IMPORTS)

    chunks, loose = [], []  # loose: top-level statements that aren't definitions
    for child in root.children:
        if child.type not in DEFINITIONS:
            loose.append(child)
            continue
        start_row = start_with_comments(child)
        loose = [n for n in loose if n.end_point.row < start_row]  # its comments belong to it
        if loose:
            chunks.append(loose_chunk(file, lines, loose, imports))
            loose = []
        end_row = child.end_point.row
        if end_row - start_row + 1 > MAX_CLASS_LINES and is_class(child):
            chunks.extend(class_body_chunks(file, lines, child, start_row, imports))
        else:
            chunks.append(make_chunk(file, lines, start_row + 1, end_row + 1,
                                     definition_name(child), imports))
    if loose:
        chunks.append(loose_chunk(file, lines, loose, imports))
    return [chunk for chunk in chunks if chunk is not None]


# Groups top-level statements (docstring, constants) into one chunk.
def loose_chunk(file, lines, nodes, imports) -> Chunk | None:
    """One chunk for a run of top-level statements (docstring, constants), skipping bare imports."""
    if all(node.type in IMPORTS or node.type == "comment" for node in nodes):
        return None
    return make_chunk(file, lines, nodes[0].start_point.row + 1, nodes[-1].end_point.row + 1,
                      "", imports)


# Markdown strategy: one chunk per heading section.
def markdown_chunks(file: CorpusFile) -> list[Chunk]:
    """Cut Markdown at its headings, one chunk per section."""
    lines = file.text.splitlines()
    starts = [i for i, line in enumerate(lines) if HEADING.match(line)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    chunks = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        if any(line.strip() for line in lines[start:end]):
            title = lines[start].lstrip("# ").strip() if HEADING.match(lines[start]) else ""
            chunks.append(make_chunk(file, lines, start + 1, end, title))
    return chunks


# Fixed-size strategy: every N lines, with optional overlap.
def window_chunks(file: CorpusFile, size: int = WINDOW_LINES, overlap: int = 0) -> list[Chunk]:
    """Cut a file every `size` lines, each window repeating the last `overlap` lines of the one
    before (a minified file is one long line, so one chunk). The naive splitter for any file."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    lines = file.text.splitlines()
    step = size - overlap
    chunks = []
    for start in range(0, max(len(lines), 1), step):
        chunks.append(make_chunk(file, lines, start + 1, min(start + size, max(len(lines), 1))))
        if start + size >= len(lines):
            break
    return chunks


# Picks the right syntax-aware strategy for the file's language.
def syntax_chunks(file: CorpusFile) -> list[Chunk]:
    """Cut a file the way its language allows."""
    if file.language == "python":
        return python_chunks(file)
    if file.language == "markdown":
        return markdown_chunks(file)
    return window_chunks(file)


# Entry point: chunk every corpus file with the chosen strategy.
def chunk_files(files: list[CorpusFile], mode: str, size: int = WINDOW_LINES,
                overlap: int = 0) -> list[Chunk]:
    """Chunk every file: "files" (whole files), "syntax" (on boundaries) or "fixed" (every
    `size` lines with `overlap`, the naive splitter, kept for comparison)."""
    cut = {"files": whole_file_chunks, "syntax": syntax_chunks,
           "fixed": lambda file: window_chunks(file, size, overlap)}[mode]
    return [chunk for file in files for chunk in cut(file)]
