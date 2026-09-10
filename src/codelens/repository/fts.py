"""Turning a user's question into an FTS5 MATCH expression.

FTS5 has its own query syntax, so a raw question cannot be passed through. The
naive escape — wrapping the whole string in quotes — is safe but turns every
query into an exact phrase search, which matches nothing for a natural-language
question and silently reduces hybrid search to vector-only.

Splitting the query into individually quoted terms joined by OR restores what
BM25 is actually for: ranking by how many query terms a chunk contains and how
rare those terms are.
"""

import re

# Words that appear in almost every question and in a lot of code. Left in the
# query they add no ranking signal but do pull in unrelated chunks.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "get",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "use",
        "used",
        "using",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)

# FTS5's unicode61 tokenizer splits on anything that is not a letter or digit,
# so we tokenize the query the same way it tokenized the documents.
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

MIN_TERM_LENGTH = 2

# A fragment produced by splitting an identifier needs to be longer to be worth
# keeping: `db` or `id` written as a whole token is meaningful, but the `SQ` that
# falls out of splitting `SQLite` is pure noise.
MIN_FRAGMENT_LENGTH = 3


def split_identifier(token: str) -> list[str]:
    """Splits `search_chunks_keyword` and `HttpClient` into their parts.

    A question rarely spells an identifier exactly as the code does, so indexing
    the parts as well lets "keyword search" match `search_chunks_keyword`.
    """
    parts = [p for p in token.split("_") if p]

    expanded = []
    for part in parts:
        # HttpClient -> Http, Client;  HTTPServer -> HTTP, Server
        expanded.extend(re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", part))

    return expanded


def extract_terms(query: str) -> list[str]:
    """Extracts the searchable terms from a query, in order, without duplicates."""
    terms: list[str] = []
    seen: set[str] = set()

    for token in TOKEN_RE.findall(query):
        # Keep the whole identifier AND its parts: the full form ranks an exact
        # match highest, the parts keep the query from missing entirely.
        fragments = split_identifier(token) if ("_" in token or not token.islower()) else []
        candidates = [(token, MIN_TERM_LENGTH)]
        candidates += [(fragment, MIN_FRAGMENT_LENGTH) for fragment in fragments]

        for candidate, min_length in candidates:
            lowered = candidate.lower()
            if len(lowered) < min_length or lowered in STOPWORDS or lowered in seen:
                continue
            seen.add(lowered)
            terms.append(lowered)

    return terms


def build_match_query(query: str) -> str | None:
    """
    Builds an FTS5 MATCH expression from a free-form query.

    Returns None when nothing searchable survives, so the caller can skip the
    query instead of raising an FTS5 syntax error on an empty expression.
    """
    terms = extract_terms(query)
    if not terms:
        return None

    # Each term is quoted so that FTS5 treats it as a literal, never as syntax
    # (NOT, OR, NEAR, *, ^ and friends all lose their meaning inside quotes).
    return " OR ".join(f'"{term}"' for term in terms)
