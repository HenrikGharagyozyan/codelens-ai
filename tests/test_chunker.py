"""Tests for SemanticChunker: turning parsed symbols back into source slices."""

import pytest

from codelens.indexer.chunker import Chunk, SemanticChunker
from codelens.parser.models import Class, Function, Symbol
from codelens.parser.python_parser import parse_python_file

SOURCE = '''class Service:
    """Talks to the database."""

    def run(self):
        return connect()

    def stop(self):
        return None


def helper(a, b):
    """Adds numbers."""
    return a + b
'''


@pytest.fixture
def repo(tmp_path):
    """A one-file repository plus its parsed symbols."""
    path = tmp_path / "app.py"
    path.write_text(SOURCE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def parsed(repo):
    """Symbols for repo/app.py, with file paths made relative like the indexer does."""
    classes, functions, _ = parse_python_file(repo / "app.py")
    symbols: list[Symbol] = []
    for cls in classes:
        cls.file_path = "app.py"
        for method in cls.methods:
            method.file_path = "app.py"
        symbols.append(cls)
        symbols.extend(cls.methods)
    for func in functions:
        func.file_path = "app.py"
        symbols.append(func)
    return symbols


def by_name(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.symbol_name: c for c in chunks}


class TestChunkContent:
    def test_creates_one_chunk_per_symbol(self, repo, parsed):
        chunks = SemanticChunker(repo).create_chunks(parsed)

        assert sorted(c.symbol_name for c in chunks) == ["Service", "helper", "run", "stop"]

    def test_a_function_chunk_holds_its_whole_body(self, repo, parsed):
        chunk = by_name(SemanticChunker(repo).create_chunks(parsed))["helper"]

        assert chunk.content.startswith("def helper(a, b):")
        assert "return a + b" in chunk.content

    def test_a_class_chunk_stops_before_its_first_method(self, repo, parsed):
        chunk = by_name(SemanticChunker(repo).create_chunks(parsed))["Service"]

        assert chunk.content.startswith("class Service:")
        assert "return connect()" not in chunk.content

    def test_a_class_chunk_lists_its_method_names(self, repo, parsed):
        chunk = by_name(SemanticChunker(repo).create_chunks(parsed))["Service"]

        assert "# Methods: run, stop" in chunk.content

    def test_a_class_chunk_keeps_its_docstring(self, repo, parsed):
        chunk = by_name(SemanticChunker(repo).create_chunks(parsed))["Service"]

        assert "Talks to the database." in chunk.content

    def test_a_docstring_outside_the_header_slice_is_appended(self, tmp_path):
        """A class whose first method precedes the docstring slice still keeps the docs."""
        (tmp_path / "a.py").write_text("class A:\n    def m(self):\n        pass\n", encoding="utf-8")
        cls = Class(
            name="A",
            file_path="a.py",
            line_number=1,
            end_line_number=3,
            docstring="Injected docs.",
            methods=[Function(name="m", file_path="a.py", line_number=2, end_line_number=3)],
        )

        chunk = SemanticChunker(tmp_path).create_chunks([cls])[0]

        assert "Injected docs." in chunk.content

    def test_a_class_without_methods_spans_to_its_end_line(self, tmp_path):
        (tmp_path / "a.py").write_text("class Empty:\n    x = 1\n    y = 2\n", encoding="utf-8")
        cls = Class(name="Empty", file_path="a.py", line_number=1, end_line_number=3)

        chunk = SemanticChunker(tmp_path).create_chunks([cls])[0]

        assert "x = 1" in chunk.content
        assert "y = 2" in chunk.content

    def test_content_is_stripped_of_surrounding_blank_space(self, repo, parsed):
        chunks = SemanticChunker(repo).create_chunks(parsed)

        assert all(c.content == c.content.strip() for c in chunks)


class TestChunkMetadata:
    def test_chunk_ids_are_unique_and_carry_the_line_number(self, repo, parsed):
        chunks = SemanticChunker(repo).create_chunks(parsed)
        ids = [c.chunk_id for c in chunks]

        assert len(ids) == len(set(ids))
        assert by_name(chunks)["run"].chunk_id == "app.py::run:4"

    def test_line_numbers_match_the_source(self, repo, parsed):
        chunks = by_name(SemanticChunker(repo).create_chunks(parsed))

        assert chunks["Service"].start_line == 1
        assert chunks["run"].start_line == 4
        assert chunks["helper"].start_line == 11

    def test_a_class_chunk_ends_before_its_first_method(self, repo, parsed):
        chunks = by_name(SemanticChunker(repo).create_chunks(parsed))

        assert chunks["Service"].end_line == 3

    def test_symbol_types_are_labelled(self, repo, parsed):
        chunks = by_name(SemanticChunker(repo).create_chunks(parsed))

        assert chunks["Service"].symbol_type == "class"
        assert chunks["run"].symbol_type == "function"
        assert chunks["helper"].symbol_type == "function"

    def test_a_plain_symbol_is_labelled_symbol(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        sym = Symbol(name="x", file_path="a.py", line_number=1, end_line_number=1)

        assert SemanticChunker(tmp_path).create_chunks([sym])[0].symbol_type == "symbol"

    def test_a_symbol_without_an_end_line_runs_to_the_end_of_file(self, tmp_path):
        (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        func = Function(name="f", file_path="a.py", line_number=1, end_line_number=None)

        chunk = SemanticChunker(tmp_path).create_chunks([func])[0]

        assert "return 1" in chunk.content
        assert chunk.end_line == 2


class TestPathHandling:
    def test_relative_symbol_paths_are_resolved_against_the_root(self, repo, parsed):
        chunks = SemanticChunker(repo).create_chunks(parsed)

        assert all(c.file_path == "app.py" for c in chunks)

    def test_absolute_symbol_paths_are_stored_relative_to_the_root(self, repo):
        classes, _, _ = parse_python_file(repo / "app.py")

        chunks = SemanticChunker(repo).create_chunks([classes[0]])

        assert chunks[0].file_path == "app.py"

    def test_a_path_outside_the_root_falls_back_to_the_file_name(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "far.py").write_text("def f():\n    pass\n", encoding="utf-8")
        root = tmp_path / "root"
        root.mkdir()
        func = Function(name="f", file_path=str(outside / "far.py"), line_number=1, end_line_number=2)

        chunk = SemanticChunker(root).create_chunks([func])[0]

        assert chunk.file_path == "far.py"

    def test_missing_files_are_skipped_silently(self, tmp_path):
        func = Function(name="ghost", file_path="gone.py", line_number=1, end_line_number=2)

        assert SemanticChunker(tmp_path).create_chunks([func]) == []

    def test_each_file_is_read_only_once_for_all_its_symbols(self, repo, parsed, monkeypatch):
        opened: list = []
        real_open = open

        def counting_open(file, *args, **kwargs):
            opened.append(file)
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr("builtins.open", counting_open)
        SemanticChunker(repo).create_chunks(parsed)

        assert len(opened) == 1


class TestEmptyInput:
    def test_no_symbols_produces_no_chunks(self, tmp_path):
        assert SemanticChunker(tmp_path).create_chunks([]) == []

    def test_accepts_a_string_root(self, repo, parsed):
        assert SemanticChunker(str(repo)).create_chunks(parsed)
