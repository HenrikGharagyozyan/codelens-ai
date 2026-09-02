"""Regression tests for citation correctness.

These lock in the fix for the original failure: the LLM produced correct line
numbers for symbols that appeared in the context and invented line numbers for
symbols that did not.

Line-numbering of the context itself is covered in test_retriever.py.
"""

import pytest

from codelens.context.citations import CITATION_RE, CitationCheck, CitationVerifier


@pytest.fixture
def verifier(populated_db):
    return CitationVerifier(populated_db)


class TestCitationPattern:
    @pytest.mark.parametrize(
        "text",
        [
            "src/db.py:42",
            "db.py:42",
            "src/pkg/mod.pyi:1",
            "main.go:7",
            "app/component.tsx:120",
            "lib/util.rs:3",
            "Main.java:99",
        ],
    )
    def test_matches_a_path_and_line(self, text):
        assert CITATION_RE.fullmatch(text) is not None

    def test_captures_an_optional_line_range(self):
        match = CITATION_RE.fullmatch("src/db.py:42-58")

        assert match.group("line") == "42"
        assert match.group("end") == "58"

    @pytest.mark.parametrize("text", ["ratio:60", "k:60", "http://x:80", "notes.md:12"])
    def test_ignores_prose_that_merely_contains_a_colon(self, text):
        assert CITATION_RE.search(text) is None


class TestCitationCheck:
    def test_only_the_ok_status_is_valid(self):
        assert CitationCheck("a.py", 1, "ok").is_valid
        assert not CitationCheck("a.py", 1, "corrected", corrected_line=2).is_valid
        assert not CitationCheck("a.py", 1, "no_symbol").is_valid
        assert not CitationCheck("a.py", 1, "unknown_file").is_valid


class TestCheck:
    def test_accepts_a_correct_definition_line(self, verifier):
        check = verifier.check("src/db.py", 42)

        assert check.is_valid
        assert check.symbol == "connect"

    def test_accepts_a_line_inside_a_known_chunk(self, verifier):
        """Citing a line in a function body, not just its `def`, is legitimate."""
        check = verifier.check("src/app.py", 12)

        assert check.is_valid
        assert check.symbol == "Service"

    def test_accepts_the_chunk_boundaries_themselves(self, verifier):
        assert verifier.check("src/app.py", 5).is_valid
        assert verifier.check("src/app.py", 18).is_valid

    def test_rejects_a_line_that_belongs_to_no_symbol(self, verifier):
        assert verifier.check("src/db.py", 999).status == "no_symbol"

    def test_rejects_an_unknown_file(self, verifier):
        assert verifier.check("src/ghost.py", 1).status == "unknown_file"

    def test_an_unknown_file_is_reported_before_any_line_lookup(self, verifier):
        check = verifier.check("src/ghost.py", 42)

        assert check.status == "unknown_file"
        assert check.symbol is None


class TestCheckNamed:
    def test_accepts_a_symbol_cited_at_its_real_location(self, verifier):
        check = verifier.check_named("connect", "src/db.py", 42)

        assert check.is_valid
        assert check.symbol == "connect"

    def test_corrects_a_wrong_line_for_a_named_symbol(self, verifier):
        """The original bug: `connect` cited at 137 when it lives at 42."""
        check = verifier.check_named("connect", "src/db.py", 137)

        assert check.status == "corrected"
        assert check.corrected_line == 42

    def test_falls_back_to_a_plain_check_for_the_wrong_file(self, verifier):
        check = verifier.check_named("connect", "src/app.py", 12)

        assert check.is_valid
        assert check.symbol == "Service"

    def test_falls_back_to_a_plain_check_for_an_unknown_symbol(self, verifier):
        assert verifier.check_named("ghost", "src/ghost.py", 1).status == "unknown_file"


class TestVerify:
    def test_returns_one_check_per_citation(self, verifier):
        checks = verifier.verify("See src/db.py:42 and src/app.py:10.")

        assert len(checks) == 2

    def test_uses_a_nearby_backticked_symbol_to_repair_the_line(self, verifier):
        checks = verifier.verify("`connect` is defined in src/db.py:137.")

        assert checks[0].status == "corrected"

    def test_resolves_a_dotted_symbol_to_its_last_segment(self, verifier):
        checks = verifier.verify("`Service.run` lives at src/app.py:99.")

        assert checks[0].status == "corrected"
        assert checks[0].corrected_line == 10

    def test_ignores_a_symbol_name_too_far_from_the_citation(self, verifier):
        answer = "`connect` " + "x" * 300 + " see src/db.py:137."

        assert verifier.verify(answer)[0].status == "no_symbol"

    def test_returns_nothing_for_an_answer_without_citations(self, verifier):
        assert verifier.verify("There is no code here.") == []


class TestRepair:
    def test_rewrites_a_hallucinated_line_to_the_real_one(self, verifier):
        answer = "The pool is opened by `connect` in src/db.py:137 during startup."

        repaired, checks = verifier.repair(answer)

        assert "src/db.py:42" in repaired
        assert "src/db.py:137" not in repaired
        assert checks[0].status == "corrected"

    def test_leaves_a_correct_citation_untouched(self, verifier):
        answer = "See `connect` in src/db.py:42."

        assert verifier.repair(answer)[0] == answer

    def test_strips_the_line_when_it_cannot_be_resolved(self, verifier):
        answer = "Defined in src/ghost.py:12 somewhere."

        repaired, _ = verifier.repair(answer)

        assert "src/ghost.py:12" not in repaired
        assert "src/ghost.py" in repaired

    def test_strips_a_line_that_belongs_to_no_symbol(self, verifier):
        repaired, _ = verifier.repair("Look at src/db.py:999 for details.")

        assert "src/db.py:999" not in repaired
        assert "src/db.py" in repaired

    def test_ignores_numbers_that_are_not_citations(self, verifier):
        """`rrf_k = 60` and similar prose must not be treated as a citation."""
        answer = "The RRF constant is k:60 and the ratio is 3:4."

        repaired, checks = verifier.repair(answer)

        assert checks == []
        assert repaired == answer

    def test_handles_several_citations_in_one_answer(self, verifier):
        answer = (
            "`connect` is at src/db.py:137, "
            "`run` is at src/app.py:10, "
            "and `Service` is at src/app.py:5."
        )

        repaired, checks = verifier.repair(answer)

        assert len(checks) == 3
        assert "src/db.py:42" in repaired
        assert "src/app.py:10" in repaired
        assert "src/app.py:5" in repaired

    def test_preserves_the_surrounding_prose_exactly(self, verifier):
        answer = "Before. `connect` at src/db.py:137. After."

        repaired, _ = verifier.repair(answer)

        assert repaired.startswith("Before. ")
        assert repaired.endswith(". After.")

    def test_an_answer_without_citations_is_returned_unchanged(self, verifier):
        answer = "I could not find that in the retrieved context."

        assert verifier.repair(answer) == (answer, [])
