"""Tests for the Typer CLI surface.

Commands are exercised through Typer's CliRunner with the application context
replaced by fakes, so no database, vector store or LLM is touched.
"""

import pytest
from typer.testing import CliRunner

from codelens.cli import commands_index, commands_search
from codelens.cli import main as main_module
from codelens.cli.context import AppContext
from codelens.cli.main import app
from tests.conftest import FakeVectorStore


class FakeGemini:
    def __init__(self, answer="`connect` lives in src/db.py:137."):
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    def ask(self, context, question):
        self.asked.append((context, question))
        return self.answer


class FakeRetriever:
    def __init__(self, context="CONTEXT"):
        self.context = context
        self.calls: list[tuple[str, int]] = []

    def build_context(self, query, limit=4):
        self.calls.append((query, limit))
        return self.context


class FakeAppContext:
    """Stands in for AppContext with every heavy dependency pre-wired."""

    def __init__(self, db=None, retriever=None, gemini=None, vector_store=None, verifier=None):
        self.db = db
        self.retriever = retriever
        self.gemini = gemini
        self.vector_store = vector_store
        self.verifier = verifier


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli(monkeypatch, populated_db):
    """Installs a FakeAppContext (over the populated index) as the CLI context."""
    from codelens.context.citations import CitationVerifier

    ctx = FakeAppContext(
        db=populated_db,
        retriever=FakeRetriever(),
        gemini=FakeGemini(),
        vector_store=FakeVectorStore(),
        verifier=CitationVerifier(populated_db),
    )
    monkeypatch.setattr(main_module, "AppContext", lambda: ctx)
    return ctx


class TestApplicationWiring:
    def test_help_lists_every_command(self, runner):
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        for command in ("index", "init", "inspect", "search", "ask", "chat", "graph"):
            assert command in result.stdout

    def test_no_arguments_shows_help_instead_of_failing(self, runner):
        result = runner.invoke(app, [])

        assert "Usage" in result.stdout

    def test_an_unknown_command_exits_non_zero(self, runner):
        assert runner.invoke(app, ["nonexistent"]).exit_code != 0


class TestInitAndIndex:
    def test_init_reports_success(self, runner, cli):
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "initialized successfully" in result.stdout

    def test_index_reports_the_indexer_summary(self, runner, cli, monkeypatch, tmp_path):
        class FakeIndexer:
            def __init__(self, path):
                self.path = path

            def run(self):
                return 7, 13, tmp_path / ".codelens.db"

        monkeypatch.setattr(commands_index, "CodebaseIndexer", FakeIndexer)

        result = runner.invoke(app, ["index", str(tmp_path)])

        assert result.exit_code == 0
        assert "Files indexed: 7" in result.stdout
        assert "Symbols extracted: 13" in result.stdout

    def test_index_defaults_to_the_current_directory(self, runner, cli, monkeypatch):
        seen = {}

        class FakeIndexer:
            def __init__(self, path):
                seen["path"] = path

            def run(self):
                return 0, 0, "db"

        monkeypatch.setattr(commands_index, "CodebaseIndexer", FakeIndexer)
        runner.invoke(app, ["index"])

        assert seen["path"] == "."


class TestInspect:
    def test_prints_classes_methods_and_functions(self, runner, cli, tmp_path):
        source = tmp_path / "sample.py"
        source.write_text(
            "class Service(Base):\n"
            "    def run(self, x):\n"
            "        pass\n"
            "\n"
            "def helper(a):\n"
            "    pass\n",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["inspect", str(source)])

        assert result.exit_code == 0
        assert "Service(Base)" in result.stdout
        assert "run" in result.stdout
        assert "helper" in result.stdout

    def test_rejects_a_missing_file(self, runner, cli, tmp_path):
        result = runner.invoke(app, ["inspect", str(tmp_path / "ghost.py")])

        assert result.exit_code == 1
        assert "valid Python file" in result.stdout

    def test_rejects_a_non_python_file(self, runner, cli, tmp_path):
        notes = tmp_path / "notes.md"
        notes.write_text("# hi\n", encoding="utf-8")

        result = runner.invoke(app, ["inspect", str(notes)])

        assert result.exit_code == 1


class TestSearch:
    def test_lists_matching_symbols_with_their_ids(self, runner, cli):
        result = runner.invoke(app, ["search", "connect"])

        assert result.exit_code == 0
        assert "connect" in result.stdout
        assert "src/db.py::connect" in result.stdout

    def test_reports_when_nothing_matches(self, runner, cli):
        result = runner.invoke(app, ["search", "kubernetes"])

        assert result.exit_code == 0
        assert "No symbols found" in result.stdout

    def test_semantic_search_renders_scores_and_previews(self, runner, cli):
        cli.vector_store.results = [
            {
                "id": "src/db.py::connect:42",
                "document": "def connect():\n    return 1\n    # tail",
                "metadata": {"symbol_name": "connect", "file_path": "src/db.py"},
                "distance": 0.1234,
            }
        ]

        result = runner.invoke(app, ["search-semantic", "database connection"])

        assert result.exit_code == 0
        assert "connect" in result.stdout
        assert "0.1234" in result.stdout

    def test_semantic_search_reports_an_empty_index(self, runner, cli):
        result = runner.invoke(app, ["search-semantic", "anything"])

        assert "No relevant code found" in result.stdout


class TestAsk:
    def test_passes_the_retrieved_context_to_the_model(self, runner, cli):
        runner.invoke(app, ["ask", "How does connect work?"])

        assert cli.retriever.calls == [("How does connect work?", 5)]
        assert cli.gemini.asked[0][0] == "CONTEXT"

    def test_repairs_a_hallucinated_citation_before_printing(self, runner, cli):
        """The model's answer cites src/db.py:137; the index says 42."""
        result = runner.invoke(app, ["ask", "where is connect?"])

        assert "src/db.py:42" in result.stdout
        assert "137" not in result.stdout

    def test_reports_the_citation_audit(self, runner, cli):
        result = runner.invoke(app, ["ask", "where is connect?"])

        assert "Citations:" in result.stdout
        assert "fixed:" in result.stdout

    def test_stops_when_no_context_is_found(self, runner, cli):
        cli.retriever.context = None

        result = runner.invoke(app, ["ask", "anything"])

        assert "No relevant context found" in result.stdout
        assert cli.gemini.asked == []

    def test_reports_an_llm_failure_without_crashing(self, runner, cli):
        def boom(context, question):
            raise RuntimeError("quota exceeded")

        cli.gemini.ask = boom

        result = runner.invoke(app, ["ask", "anything"])

        assert result.exit_code == 0
        assert "Error communicating with LLM" in result.stdout
        assert "quota exceeded" in result.stdout

    def test_reports_an_empty_model_response(self, runner, cli):
        cli.gemini.answer = ""

        result = runner.invoke(app, ["ask", "anything"])

        assert "empty response" in result.stdout


class TestCitationReport:
    def test_prints_nothing_without_citations(self, capsys):
        commands_search._report_citations([])

        assert capsys.readouterr().out == ""

    def test_counts_verified_citations(self, capsys, populated_db):
        from codelens.context.citations import CitationVerifier

        checks = CitationVerifier(populated_db).verify("`connect` is in src/db.py:42.")
        commands_search._report_citations(checks)

        assert "1/1 verified" in capsys.readouterr().out

    def test_flags_dropped_citations(self, capsys, populated_db):
        from codelens.context.citations import CitationVerifier

        checks = CitationVerifier(populated_db).verify("See src/ghost.py:9.")
        commands_search._report_citations(checks)

        assert "unverifiable line dropped" in capsys.readouterr().out


class TestGraph:
    def test_lists_the_symbols_outgoing_calls(self, runner, cli):
        result = runner.invoke(app, ["graph", "run"])

        assert result.exit_code == 0
        assert "connect()" in result.stdout
        assert "print()" in result.stdout

    def test_reports_an_unknown_symbol(self, runner, cli):
        result = runner.invoke(app, ["graph", "kubernetes"])

        assert "not found in index" in result.stdout

    def test_reports_a_leaf_symbol(self, runner, cli):
        result = runner.invoke(app, ["graph", "connect"])

        assert "doesn't call any other known functions" in result.stdout

    def test_prefers_an_exact_name_match_over_a_substring(self, runner, cli):
        cli.db.insert_symbol("src/app.py::runner", "runner", "function", "src/app.py", 30)

        result = runner.invoke(app, ["graph", "run"])

        assert "src/app.py::Service.run" in result.stdout

    def test_falls_back_to_the_first_partial_match(self, runner, cli):
        result = runner.invoke(app, ["graph", "conn"])

        assert "src/db.py::connect" in result.stdout


class TestInspectChunks:
    def test_prints_stored_chunks(self, runner, cli):
        result = runner.invoke(app, ["inspect-chunks"])

        assert result.exit_code == 0
        assert "src/app.py::Service:5" in result.stdout
        assert "class Service:" in result.stdout

    def test_honours_the_limit(self, runner, cli):
        cli.db.conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("second", "src/db.py", "connect", "function", 42, 44, "def connect(): ..."),
        )
        cli.db.conn.commit()

        result = runner.invoke(app, ["inspect-chunks", "--limit", "1"])

        assert "second" not in result.stdout

    def test_tells_the_user_to_index_first(self, runner, cli):
        cli.db.conn.execute("DELETE FROM chunks")
        cli.db.conn.commit()

        result = runner.invoke(app, ["inspect-chunks"])

        assert "No chunks found" in result.stdout


class TestAppContext:
    def test_dependencies_are_created_lazily_and_memoised(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        ctx = AppContext()

        assert ctx._db is None
        first = ctx.db

        assert ctx.db is first
        first.close()

    def test_the_retriever_reuses_the_contexts_db_and_store(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        ctx = AppContext()
        monkeypatch.setattr(
            "codelens.indexer.vector_store.VectorStore.__init__",
            lambda self, db_path=".codelens_vector": None,
        )

        retriever = ctx.retriever

        assert retriever.db is ctx.db
        assert retriever.vector_store is ctx.vector_store
        ctx.db.close()

    def test_the_verifier_is_bound_to_the_contexts_db(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        ctx = AppContext()

        assert ctx.verifier.db is ctx.db
        ctx.db.close()