"""Tests for building FTS5 MATCH expressions from free-form queries.

These lock in the fix for a silent failure: wrapping the whole query in quotes
made every search an exact-phrase search, so a natural-language question matched
nothing and hybrid retrieval quietly degraded to vector-only.
"""

import pytest

from codelens.repository.fts import build_match_query, extract_terms, split_identifier


class TestSplitIdentifier:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("search_chunks_keyword", ["search", "chunks", "keyword"]),
            ("HttpClient", ["Http", "Client"]),
            ("HTTPServer", ["HTTP", "Server"]),
            ("parse", ["parse"]),
            ("DB_PATH", ["DB", "PATH"]),
        ],
    )
    def test_splits_snake_and_camel_case(self, token, expected):
        assert split_identifier(token) == expected


class TestExtractTerms:
    def test_drops_stopwords(self):
        terms = extract_terms("Where is the parser?")

        assert "where" not in terms
        assert "the" not in terms
        assert "parser" in terms

    def test_keeps_the_whole_identifier_and_its_parts(self):
        """The exact form should rank a direct hit highest; the parts prevent a miss."""
        terms = extract_terms("search_chunks_keyword")

        assert terms[0] == "search_chunks_keyword"
        assert {"search", "chunks", "keyword"} <= set(terms)

    def test_deduplicates_repeated_terms(self):
        assert extract_terms("parser parser PARSER") == ["parser"]

    def test_drops_noise_fragments_from_acronyms(self):
        """Splitting `SQLite` yields a two-letter `SQ` that only adds noise."""
        terms = extract_terms("SQLite")

        assert "sq" not in terms
        assert "sqlite" in terms

    def test_keeps_short_whole_tokens(self):
        """`db` on its own is meaningful even though it is only two characters."""
        assert "db" in extract_terms("db path")

    def test_returns_nothing_for_a_query_of_only_stopwords(self):
        assert extract_terms("where is the") == []


class TestBuildMatchQuery:
    def test_joins_terms_with_or(self):
        assert build_match_query("virtual table") == '"virtual" OR "table"'

    def test_quotes_every_term(self):
        """Unquoted, FTS5 operators inside a query are syntax, not search terms."""
        query = build_match_query("NOT OR NEAR parser")

        assert query is not None
        for term in query.split(" OR "):
            assert term.startswith('"') and term.endswith('"')

    def test_returns_none_when_nothing_is_searchable(self):
        assert build_match_query("where is the") is None
        assert build_match_query("???") is None
        assert build_match_query("") is None

    @pytest.mark.parametrize("query", ['a "quoted" phrase', "call()", "x * y", "a^b", "-flag"])
    def test_survives_characters_that_are_fts5_syntax(self, query, populated_db):
        """A query full of punctuation must search, not raise a syntax error."""
        populated_db.search_chunks_keyword(query)


class TestKeywordSearchIntegration:
    def test_a_natural_language_question_still_matches(self, populated_db):
        """The original bug: a phrase search made every question return nothing."""
        rows = populated_db.search_chunks_keyword("How does the service connect?")

        assert rows != []

    def test_matches_on_partial_term_overlap(self, populated_db):
        """Terms are ORed: one hit is enough, the whole phrase need not match."""
        rows = populated_db.search_chunks_keyword("Service alongside a nonexistent elephant")

        assert rows != []

    def test_returns_nothing_when_no_term_matches(self, populated_db):
        assert populated_db.search_chunks_keyword("nonexistent elephant") == []

    def test_returns_nothing_for_an_unsearchable_query(self, populated_db):
        """No terms survive, so we must skip the query rather than send empty SQL."""
        assert populated_db.search_chunks_keyword("where is the") == []
