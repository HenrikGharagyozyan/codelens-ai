"""Regression tests for citation correctness.

These lock in the fix for the original failure: the LLM produced correct line
numbers for symbols that appeared in the context and invented line numbers for
symbols that did not.
"""

import pytest

from codelens.context.citations import CitationVerifier
from codelens.context.retriever import ContextRetriever
from codelens.repository.db import DatabaseManager


@pytest.fixture
def db(tmp_path):
    manager = DatabaseManager(tmp_path / "test.db")

    manager.insert_file("src/app.py", "py", 100, 20)
    manager.insert_file("src/db.py", "py", 100, 20)

    manager.insert_symbol("src/app.py::Service", "Service", "class", "src/app.py", 5)
    manager.insert_symbol("src/app.py::Service.run", "run", "method", "src/app.py", 10)
    manager.insert_symbol("src/db.py::connect", "connect", "function", "src/db.py", 42)

    manager.conn.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("src/app.py::Service:5", "src/app.py", "Service", "class", 5, 18, "class Service:\n    pass"),
    )
    manager.conn.commit()

    yield manager
    manager.close()


class TestLineNumbering:
    def test_numbers_start_at_the_symbols_real_line(self, db):
        retriever = ContextRetriever(db, vector_store=None)
        numbered = retriever._number_lines("def run(self):\n    return 1", start_line=10)

        assert numbered.split("\n")[0].startswith("10 | def run")
        assert numbered.split("\n")[1].startswith("11 |")

    def test_numbering_width_stays_aligned(self, db):
        retriever = ContextRetriever(db, vector_store=None)
        numbered = retriever._number_lines("a\n" * 3 + "b", start_line=98)

        prefixes = [line.split("|")[0] for line in numbered.split("\n")]
        assert len({len(p) for p in prefixes}) == 1, "line-number column must be padded"

    def test_related_symbols_carry_verified_locations(self, db):
        retriever = ContextRetriever(db, vector_store=None)
        rendered = retriever._format_related("**calls:**", ["connect"])

        assert "src/db.py:42" in rendered

    def test_unindexed_symbols_are_marked_external(self, db):
        """A stdlib call must never be given a fabricated location."""
        retriever = ContextRetriever(db, vector_store=None)
        rendered = retriever._format_related("**calls:**", ["enumerate"])

        assert "external, no location" in rendered
        assert ":" not in rendered.replace("**calls:**", "")


class TestCitationVerifier:
    def test_accepts_a_correct_definition_line(self, db):
        check = CitationVerifier(db).check("src/db.py", 42)
        assert check.is_valid
        assert check.symbol == "connect"

    def test_accepts_a_line_inside_a_known_chunk(self, db):
        """Citing a line in a function body, not just its `def`, is legitimate."""
        assert CitationVerifier(db).check("src/app.py", 12).is_valid

    def test_rejects_a_line_that_belongs_to_no_symbol(self, db):
        assert CitationVerifier(db).check("src/db.py", 999).status == "no_symbol"

    def test_rejects_an_unknown_file(self, db):
        assert CitationVerifier(db).check("src/ghost.py", 1).status == "unknown_file"

    def test_corrects_a_wrong_line_for_a_named_symbol(self, db):
        """The original bug: `connect` cited at 137 when it lives at 42."""
        check = CitationVerifier(db).check_named("connect", "src/db.py", 137)

        assert check.status == "corrected"
        assert check.corrected_line == 42


class TestRepair:
    def test_rewrites_a_hallucinated_line_to_the_real_one(self, db):
        answer = "The pool is opened by `connect` in src/db.py:137 during startup."
        repaired, checks = CitationVerifier(db).repair(answer)

        assert "src/db.py:42" in repaired
        assert "src/db.py:137" not in repaired
        assert checks[0].status == "corrected"

    def test_leaves_a_correct_citation_untouched(self, db):
        answer = "See `connect` in src/db.py:42."
        repaired, _ = CitationVerifier(db).repair(answer)

        assert repaired == answer

    def test_strips_the_line_when_it_cannot_be_resolved(self, db):
        answer = "Defined in src/ghost.py:12 somewhere."
        repaired, _ = CitationVerifier(db).repair(answer)

        assert "src/ghost.py:12" not in repaired
        assert "src/ghost.py" in repaired

    def test_ignores_numbers_that_are_not_citations(self, db):
        """`rrf_k = 60` and similar prose must not be treated as a citation."""
        answer = "The RRF constant is k:60 and the ratio is 3:4."
        repaired, checks = CitationVerifier(db).repair(answer)

        assert checks == []
        assert repaired == answer

    def test_handles_several_citations_in_one_answer(self, db):
        answer = (
            "`connect` is at src/db.py:137, "
            "`run` is at src/app.py:10, "
            "and `Service` is at src/app.py:5."
        )
        repaired, checks = CitationVerifier(db).repair(answer)

        assert len(checks) == 3
        assert "src/db.py:42" in repaired
        assert "src/app.py:10" in repaired
        assert "src/app.py:5" in repaired
