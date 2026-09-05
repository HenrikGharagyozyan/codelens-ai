"""Tests for the interactive chat command and its persistence layer.

`chat` is a REPL, so it is driven here by scripting `Prompt.ask` and
`IntPrompt.ask` rather than by typing into a terminal.
"""

import pytest
from typer.testing import CliRunner

from codelens.cli import commands_chat
from codelens.cli import main as main_module
from codelens.cli.main import app
from codelens.repository.chat import ChatRepository


class FakeChatSession:
    def __init__(self):
        self.sent: list[str] = []


class FakeGemini:
    """A Gemini client whose stream is scripted per turn."""

    def __init__(self, replies=None):
        self.replies = list(replies or ["`connect` lives in src/db.py:137."])
        self.sessions: list[FakeChatSession] = []
        self.history_dicts = None
        self.tool = None
        self.fail_on_start = False

    def start_chat_with_tools(self, search_tool_fn, history_dicts=None):
        if self.fail_on_start:
            raise RuntimeError("no API key")
        self.tool = search_tool_fn
        self.history_dicts = history_dicts
        session = FakeChatSession()
        self.sessions.append(session)
        return session

    def send_chat_message_stream(self, chat_session, message):
        chat_session.sent.append(message)
        reply = self.replies.pop(0) if self.replies else ""
        yield reply


class FakeRetriever:
    def __init__(self, context="CONTEXT"):
        self.context = context
        self.calls: list[tuple[str, int]] = []

    def build_context(self, query, limit=4):
        self.calls.append((query, limit))
        return self.context


class FakeAppContext:
    def __init__(self, db, retriever, gemini, verifier):
        self.db = db
        self.retriever = retriever
        self.gemini = gemini
        self.verifier = verifier


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli(monkeypatch, populated_db):
    from codelens.context.citations import CitationVerifier

    ctx = FakeAppContext(
        db=populated_db,
        retriever=FakeRetriever(),
        gemini=FakeGemini(),
        verifier=CitationVerifier(populated_db),
    )
    monkeypatch.setattr(main_module, "AppContext", lambda: ctx)
    return ctx


@pytest.fixture
def answers(monkeypatch):
    """Scripts the interactive prompts; returns the list to fill in per test."""
    script = {"session": 0, "questions": []}

    monkeypatch.setattr(
        commands_chat.IntPrompt, "ask", staticmethod(lambda *a, **kw: script["session"])
    )

    def next_question(*a, **kw):
        if not script["questions"]:
            raise EOFError
        return script["questions"].pop(0)

    monkeypatch.setattr(commands_chat.Prompt, "ask", staticmethod(next_question))
    return script


class TestChatRepository:
    def test_messages_come_back_in_chronological_order(self, db):
        db.chat.create_session("s1", title="First chat")
        db.chat.add_message("s1", "user", "hello")
        db.chat.add_message("s1", "model", "hi there")

        history = db.chat.get_history("s1")

        assert [(row["role"], row["content"]) for row in history] == [
            ("user", "hello"),
            ("model", "hi there"),
        ]

    def test_is_reachable_from_the_database_manager(self, db):
        assert isinstance(db.chat, ChatRepository)
        assert db.chat.conn is db.conn

    def test_can_be_constructed_directly_over_a_connection(self, db):
        repo = ChatRepository(db.conn)
        repo.create_session("s1")

        assert len(repo.get_recent_sessions()) == 1

    def test_recent_sessions_are_newest_first(self, db):
        db.chat.create_session("old", title="Old")
        db.conn.execute("UPDATE chat_sessions SET created_at = '2020-01-01' WHERE id = 'old'")
        db.chat.create_session("new", title="New")

        assert [s["id"] for s in db.chat.get_recent_sessions()] == ["new", "old"]


class TestSessionSelection:
    def test_choosing_zero_starts_a_new_session_with_no_history(self, cli, answers):
        answers["session"] = 0

        session_id, history, created = commands_chat._select_session(cli)

        assert history is None
        assert created is False
        assert len(session_id) == 36  # a UUID

    def test_choosing_an_existing_session_loads_its_history(self, cli, answers):
        cli.db.chat.create_session("s1", title="Earlier chat")
        cli.db.chat.add_message("s1", "user", "what is RRF?")
        cli.db.chat.add_message("s1", "model", "rank fusion")
        answers["session"] = 1

        session_id, history, created = commands_chat._select_session(cli)

        assert session_id == "s1"
        assert created is True
        assert history == [
            {"role": "user", "content": "what is RRF?"},
            {"role": "model", "content": "rank fusion"},
        ]


class TestSearchTool:
    def test_the_tool_returns_retrieved_context(self, cli):
        tool = commands_chat._make_search_tool(cli)

        assert tool("hybrid search") == "CONTEXT"
        assert cli.retriever.calls == [("hybrid search", 4)]

    def test_the_tool_reports_an_empty_index_instead_of_none(self, cli):
        cli.retriever.context = None
        tool = commands_chat._make_search_tool(cli)

        assert tool("anything") == "No relevant code found."


class TestChatTurn:
    def test_persists_the_question_and_the_repaired_answer(self, cli):
        session = FakeChatSession()
        cli.db.chat.create_session("s1")

        commands_chat._chat_turn(cli, session, "s1", "where is connect?")
        history = cli.db.chat.get_history("s1")

        assert history[0]["content"] == "where is connect?"
        # The model said src/db.py:137; the index says 42.
        assert "src/db.py:42" in history[1]["content"]
        assert "137" not in history[1]["content"]

    def test_a_streaming_failure_is_reported_not_raised(self, cli, capsys):
        def boom(chat_session, message):
            raise RuntimeError("quota exceeded")

        cli.gemini.send_chat_message_stream = boom
        cli.db.chat.create_session("s1")

        commands_chat._chat_turn(cli, FakeChatSession(), "s1", "anything")

        assert "Error communicating with LLM" in capsys.readouterr().out


class TestChatCommand:
    def test_exit_ends_the_session(self, runner, cli, answers):
        answers["questions"] = ["exit"]

        result = runner.invoke(app, ["chat"])

        assert result.exit_code == 0
        assert "Goodbye!" in result.stdout

    def test_the_first_question_creates_the_session_record(self, runner, cli, answers):
        answers["questions"] = ["how does indexing work?", "exit"]

        runner.invoke(app, ["chat"])
        sessions = cli.db.chat.get_recent_sessions()

        assert len(sessions) == 1
        assert sessions[0]["title"] == "how does indexing work?"

    def test_a_long_question_is_truncated_into_the_title(self, runner, cli, answers):
        answers["questions"] = ["x" * 60, "exit"]

        runner.invoke(app, ["chat"])

        assert cli.db.chat.get_recent_sessions()[0]["title"] == "x" * 40 + "..."

    def test_resuming_a_session_does_not_create_another(self, runner, cli, answers):
        cli.db.chat.create_session("s1", title="Earlier chat")
        answers["session"] = 1
        answers["questions"] = ["a follow-up", "exit"]

        runner.invoke(app, ["chat"])

        assert len(cli.db.chat.get_recent_sessions()) == 1

    def test_resumed_history_is_handed_to_the_model(self, runner, cli, answers):
        cli.db.chat.create_session("s1", title="Earlier chat")
        cli.db.chat.add_message("s1", "user", "earlier question")
        answers["session"] = 1
        answers["questions"] = ["exit"]

        runner.invoke(app, ["chat"])

        assert cli.gemini.history_dicts == [{"role": "user", "content": "earlier question"}]

    def test_blank_input_is_ignored(self, runner, cli, answers):
        answers["questions"] = ["   ", "exit"]

        runner.invoke(app, ["chat"])

        assert cli.db.chat.get_recent_sessions() == []

    def test_a_client_that_cannot_start_is_reported(self, runner, cli, answers):
        cli.gemini.fail_on_start = True

        result = runner.invoke(app, ["chat"])

        assert "Error initializing Gemini client" in result.stdout

    def test_ctrl_d_leaves_the_repl(self, runner, cli, answers):
        answers["questions"] = []  # next_question raises EOFError

        result = runner.invoke(app, ["chat"])

        assert result.exit_code == 0
        assert "Goodbye!" in result.stdout
