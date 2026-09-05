"""Tests for ContextRetriever: hybrid ranking and the prompt context it builds."""

import pytest

from codelens.context.retriever import ContextRetriever
from tests.conftest import FakeVectorStore


def vector_hit(chunk_id, document="code", **meta):
    metadata = {
        "chunk_id": chunk_id,
        "symbol_name": meta.get("symbol_name", "run"),
        "symbol_type": meta.get("symbol_type", "function"),
        "file_path": meta.get("file_path", "src/app.py"),
        "start_line": meta.get("start_line", 10),
        "end_line": meta.get("end_line", 14),
    }
    return {"id": chunk_id, "document": document, "metadata": metadata, "distance": 0.1}


@pytest.fixture
def retriever(populated_db):
    """Retriever over the shared index, with a vector store we drive by hand."""
    return ContextRetriever(populated_db, FakeVectorStore())


class TestLineNumbering:
    def test_numbers_start_at_the_symbols_real_line(self, retriever):
        numbered = retriever._number_lines("def run(self):\n    return 1", start_line=10)

        assert numbered.splitlines()[0].startswith("10 | def run")
        assert numbered.splitlines()[1].startswith("11 |")

    def test_numbering_width_stays_aligned_across_a_digit_boundary(self, retriever):
        numbered = retriever._number_lines("a\n" * 3 + "b", start_line=98)

        prefixes = [line.split("|")[0] for line in numbered.splitlines()]
        assert len({len(p) for p in prefixes}) == 1, "line-number column must be padded"

    def test_a_single_line_is_numbered(self, retriever):
        assert retriever._number_lines("x = 1", start_line=7) == "7 | x = 1"

    def test_every_source_line_is_preserved(self, retriever):
        code = "a\nb\nc"

        assert len(retriever._number_lines(code, 1).splitlines()) == 3


class TestFormatRelated:
    def test_related_symbols_carry_verified_locations(self, retriever):
        assert "src/db.py:42" in retriever._format_related("**calls:**", ["connect"])

    def test_unindexed_symbols_are_marked_external(self, retriever):
        """A stdlib call must never be given a fabricated location."""
        rendered = retriever._format_related("**calls:**", ["enumerate"])

        assert "external, no location" in rendered
        assert ":" not in rendered.replace("**calls:**", "")

    def test_all_definitions_of_an_ambiguous_name_are_listed(self, populated_db):
        populated_db.insert_symbol(
            "src/other.py::connect", "connect", "function", "src/other.py", 7
        )
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        rendered = retriever._format_related("**calls:**", ["connect"])

        assert "src/db.py:42" in rendered
        assert "src/other.py:7" in rendered

    def test_the_label_is_kept_verbatim(self, retriever):
        assert retriever._format_related("**What calls `run`:**", ["connect"]).startswith(
            "**What calls `run`:**"
        )


class TestExactSymbolId:
    def test_resolves_a_name_and_file_to_its_id(self, retriever):
        assert retriever._get_exact_symbol_id("run", "src/app.py") == "src/app.py::Service.run"

    def test_returns_none_for_the_wrong_file(self, retriever):
        assert retriever._get_exact_symbol_id("run", "src/db.py") is None

    @pytest.mark.parametrize(("name", "path"), [("", "src/app.py"), ("run", ""), (None, None)])
    def test_returns_none_on_missing_input(self, retriever, name, path):
        assert retriever._get_exact_symbol_id(name, path) is None


class TestHybridSearch:
    def test_merges_results_from_both_backends(self, populated_db):
        store = FakeVectorStore([vector_hit("vector-only", symbol_name="other")])
        retriever = ContextRetriever(populated_db, store)

        results = retriever._hybrid_search("Service", limit=10)
        ids = [r["chunk_id"] for r in results]

        assert "vector-only" in ids
        assert "src/app.py::Service:5" in ids

    def test_a_chunk_found_by_both_backends_outranks_one_found_by_either(self, populated_db):
        shared = "src/app.py::Service:5"
        store = FakeVectorStore([vector_hit("vector-only"), vector_hit(shared)])
        retriever = ContextRetriever(populated_db, store)

        results = retriever._hybrid_search("Service", limit=10)

        assert results[0]["chunk_id"] == shared

    def test_keyword_only_hits_are_given_a_full_metadata_dictionary(self, populated_db):
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        result = retriever._hybrid_search("Service", limit=5)[0]

        assert result["document"] == "class Service:\n    pass"
        assert result["metadata"]["file_path"] == "src/app.py"
        assert result["metadata"]["start_line"] == 5
        assert result["metadata"]["end_line"] == 18

    def test_honours_the_limit(self, populated_db):
        store = FakeVectorStore([vector_hit(f"c{i}") for i in range(5)])
        retriever = ContextRetriever(populated_db, store)

        assert len(retriever._hybrid_search("Service", limit=2)) == 2

    def test_returns_nothing_when_both_backends_are_empty(self, populated_db):
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        assert retriever._hybrid_search("kubernetes") == []

    def test_asks_both_backends_for_more_candidates_than_the_limit(self, populated_db):
        store = FakeVectorStore()
        retriever = ContextRetriever(populated_db, store)

        retriever._hybrid_search("Service", limit=2)

        assert store.queries == [("Service", 10)]


class TestBuildContext:
    def test_returns_none_when_nothing_is_retrieved(self, populated_db):
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        assert retriever.build_context("kubernetes") is None

    def test_states_the_line_number_contract_up_front(self, populated_db):
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        context = retriever.build_context("Service")

        assert "<line> | <code>" in context
        assert "Never compute, guess" in context

    def test_code_is_rendered_with_real_line_numbers(self, populated_db):
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        context = retriever.build_context("Service")

        assert "5 | class Service:" in context
        assert "6 |     pass" in context

    def test_the_header_names_the_citation_to_use(self, populated_db):
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        context = retriever.build_context("Service")

        assert "cite as src/app.py:5" in context
        assert "lines 5-18" in context

    def test_nested_definitions_are_listed_with_exact_lines(self, populated_db):
        """Without this the model guesses where a method starts inside a class chunk."""
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        context = retriever.build_context("Service")

        assert "Definitions inside this chunk:" in context
        assert "`run` -> src/app.py:10" in context

    def test_callers_of_the_symbol_are_included(self, populated_db):
        store = FakeVectorStore(
            [
                vector_hit(
                    "src/db.py::connect:42",
                    "def connect(): ...",
                    symbol_name="connect",
                    file_path="src/db.py",
                    start_line=42,
                    end_line=44,
                )
            ]
        )
        retriever = ContextRetriever(populated_db, store)

        context = retriever.build_context("connect")

        assert "What calls `connect`:" in context
        assert "`run` (src/app.py:10)" in context

    def test_callees_of_the_symbol_are_included(self, populated_db):
        store = FakeVectorStore(
            [
                vector_hit(
                    "src/app.py::Service.run:10",
                    "def run(self): ...",
                    symbol_name="run",
                    file_path="src/app.py",
                    start_line=10,
                    end_line=14,
                )
            ]
        )
        retriever = ContextRetriever(populated_db, store)

        context = retriever.build_context("run")

        assert "What `run` calls:" in context
        assert "`connect` (src/db.py:42)" in context
        assert "`print` (external, no location)" in context

    def test_a_chunk_without_a_trustworthy_anchor_is_not_numbered(self, populated_db):
        store = FakeVectorStore([vector_hit("no-anchor", "def mystery(): ...", start_line=-1)])
        retriever = ContextRetriever(populated_db, store)

        context = retriever.build_context("mystery")

        assert "line numbers unavailable" in context
        assert "-1 |" not in context

    def test_the_global_pseudo_symbol_gets_no_call_graph(self, populated_db):
        store = FakeVectorStore([vector_hit("mod", "import os", symbol_name="global")])
        retriever = ContextRetriever(populated_db, store)

        context = retriever.build_context("imports")

        assert "What calls" not in context

    def test_multiple_chunks_are_separated_and_numbered(self, populated_db):
        store = FakeVectorStore([vector_hit("c1"), vector_hit("c2", symbol_name="other")])
        retriever = ContextRetriever(populated_db, store)

        context = retriever.build_context("run", limit=2)

        assert "### Chunk 1:" in context
        assert "### Chunk 2:" in context
        assert "\n---\n" in context


class TestRenderHelpers:
    def test_render_code_numbers_the_body_and_names_the_citation(self, retriever):
        meta = {"symbol_name": "run", "file_path": "src/app.py", "start_line": 10, "end_line": 11}

        header, body = retriever._render_code("def run():\n    pass", 1, meta)

        assert "cite as src/app.py:10" in header
        assert "10 | def run():" in body
        assert "11 |     pass" in body

    def test_render_code_refuses_to_number_an_unanchored_chunk(self, retriever):
        meta = {"symbol_name": "run", "file_path": "src/app.py", "start_line": -1, "end_line": -1}

        header, body = retriever._render_code("def run(): ...", 2, meta)

        assert "line numbers unavailable" in header
        assert "|" not in body

    def test_nested_definitions_are_listed_with_exact_lines(self, retriever):
        rendered = retriever._render_nested_definitions("src/app.py", 5, 18)

        assert "`run` -> src/app.py:10" in rendered

    def test_nested_definitions_returns_none_outside_any_range(self, retriever):
        assert retriever._render_nested_definitions("src/app.py", 100, 200) is None

    @pytest.mark.parametrize(
        ("path", "start", "end"),
        [(None, 1, 2), ("src/app.py", None, 2), ("src/app.py", 1, None)],
    )
    def test_nested_definitions_returns_none_without_an_anchor(self, retriever, path, start, end):
        assert retriever._render_nested_definitions(path, start, end) is None

    def test_call_graph_reports_callers_and_callees(self, retriever):
        sections = retriever._render_call_graph("run", "src/app.py", 10, 14)
        joined = "\n".join(sections)

        assert "What `run` calls:" in joined
        assert "`connect` (src/db.py:42)" in joined
        assert "`print` (external, no location)" in joined

    def test_call_graph_is_empty_for_an_isolated_symbol(self, populated_db):
        populated_db.insert_symbol("src/lone.py::lone", "lone", "function", "src/lone.py", 1)
        retriever = ContextRetriever(populated_db, FakeVectorStore())

        assert retriever._render_call_graph("lone", "src/lone.py", 1, 3) == []
