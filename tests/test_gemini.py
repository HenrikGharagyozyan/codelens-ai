"""Tests for GeminiClient, with the google-genai SDK replaced by fakes.

No network call is ever made, and `load_dotenv` is neutralised so the
developer's real `.env` cannot leak into the suite.
"""

import pytest

from codelens.llm import gemini as gemini_module
from codelens.llm.gemini import SYSTEM_PROMPT, GeminiClient


class FakeResponse:
    def __init__(self, text="an answer"):
        self.text = text


class FakeChat:
    def __init__(self, model=None, config=None, history=None):
        self.model = model
        self.config = config
        self.history = history
        self.sent: list[str] = []
        self.stream_chunks: list = []

    def send_message(self, message):
        self.sent.append(message)
        return FakeResponse("chat answer")

    def send_message_stream(self, message):
        self.sent.append(message)
        return iter(self.stream_chunks)


class FakeModels:
    def __init__(self):
        self.calls: list[dict] = []
        self.response = FakeResponse()

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.response


class FakeChats:
    def __init__(self):
        self.created: list[FakeChat] = []

    def create(self, model=None, config=None, history=None):
        chat = FakeChat(model=model, config=config, history=history)
        self.created.append(chat)
        return chat


class FakeGenaiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.models = FakeModels()
        self.chats = FakeChats()


@pytest.fixture
def genai_client(monkeypatch):
    """Installs the fake SDK client and a valid API key."""
    created: list[FakeGenaiClient] = []

    def _factory(api_key=None):
        client = FakeGenaiClient(api_key=api_key)
        created.append(client)
        return client

    monkeypatch.setattr(gemini_module, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setattr(gemini_module.genai, "Client", _factory)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return created


@pytest.fixture
def client(genai_client):
    return GeminiClient()


class TestInitialisation:
    def test_passes_the_api_key_from_the_environment(self, client, genai_client):
        assert genai_client[0].api_key == "test-key"

    def test_raises_a_clear_error_without_an_api_key(self, monkeypatch):
        monkeypatch.setattr(gemini_module, "load_dotenv", lambda *a, **kw: None)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiClient()

    def test_raises_on_an_empty_api_key(self, monkeypatch):
        monkeypatch.setattr(gemini_module, "load_dotenv", lambda *a, **kw: None)
        monkeypatch.setenv("GEMINI_API_KEY", "")

        with pytest.raises(ValueError):
            GeminiClient()

    def test_uses_a_low_temperature_and_the_system_prompt(self, client):
        assert client.config.temperature == 0.2
        assert client.config.system_instruction == SYSTEM_PROMPT

    def test_disables_tokenizer_parallelism(self, client):
        import os

        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


class TestSystemPrompt:
    def test_demands_terminal_clickable_citations(self):
        assert "path/to/file.py:line_number" in SYSTEM_PROMPT

    def test_forbids_inventing_line_numbers(self):
        assert "NEVER compute, infer, offset, or estimate a line number" in SYSTEM_PROMPT

    def test_explains_the_numbered_line_format(self):
        assert "`<line> | <code>`" in SYSTEM_PROMPT

    def test_tells_the_model_what_to_do_with_external_symbols(self):
        assert "(external, no location)" in SYSTEM_PROMPT


class TestAsk:
    def test_embeds_the_context_and_the_question_in_the_prompt(self, client):
        client.ask("CONTEXT-BLOCK", "How does indexing work?")
        prompt = client.client.models.calls[0]["contents"]

        assert "CONTEXT-BLOCK" in prompt
        assert "How does indexing work?" in prompt

    def test_marks_the_boundaries_of_the_context(self, client):
        client.ask("ctx", "q")
        prompt = client.client.models.calls[0]["contents"]

        assert "--- CODE CONTEXT START ---" in prompt
        assert "--- CODE CONTEXT END ---" in prompt

    def test_sends_the_configured_model_and_config(self, client):
        client.ask("ctx", "q")
        call = client.client.models.calls[0]

        assert call["model"] == client.model_name
        assert call["config"] is client.config

    def test_returns_the_response_text(self, client):
        client.client.models.response = FakeResponse("42")

        assert client.ask("ctx", "q") == "42"


class TestChatSessions:
    def test_start_chat_uses_the_shared_config(self, client):
        chat = client.start_chat()

        assert chat.model == client.model_name
        assert chat.config is client.config

    def test_send_chat_message_returns_the_text(self, client):
        chat = client.start_chat()

        assert client.send_chat_message(chat, "hello") == "chat answer"
        assert chat.sent == ["hello"]

    def test_tools_are_registered_on_the_session(self, client):
        def search(query: str) -> str:
            return "result"

        chat = client.start_chat_with_tools(search)

        assert chat.config.tools == [search]
        assert chat.config.system_instruction == SYSTEM_PROMPT

    def test_a_tool_session_starts_empty_without_history(self, client):
        chat = client.start_chat_with_tools(lambda q: "")

        assert chat.history is None

    def test_stored_history_is_converted_to_sdk_content(self, client):
        history = [
            {"role": "user", "content": "what is CodeLens?"},
            {"role": "model", "content": "a code assistant"},
        ]

        chat = client.start_chat_with_tools(lambda q: "", history_dicts=history)

        assert [c.role for c in chat.history] == ["user", "model"]
        assert chat.history[0].parts[0].text == "what is CodeLens?"


class TestStreaming:
    @staticmethod
    def _chunk(*texts):
        class Part:
            def __init__(self, text):
                self.text = text

        class Content:
            def __init__(self, parts):
                self.parts = parts

        class Candidate:
            def __init__(self, parts):
                self.content = Content(parts)

        class Chunk:
            def __init__(self, parts):
                self.candidates = [Candidate(parts)]

        return Chunk([Part(t) for t in texts])

    def test_yields_the_text_of_every_part_in_order(self, client):
        chat = client.start_chat()
        chat.stream_chunks = [self._chunk("Hello "), self._chunk("world", "!")]

        assert list(client.send_chat_message_stream(chat, "hi")) == ["Hello ", "world", "!"]

    def test_skips_non_text_parts(self, client):
        """Function-call parts have no `.text` and must not break the stream."""

        class ToolPart:
            text = None

        chat = client.start_chat()
        chunk = self._chunk("visible")
        chunk.candidates[0].content.parts.insert(0, ToolPart())
        chat.stream_chunks = [chunk]

        assert list(client.send_chat_message_stream(chat, "hi")) == ["visible"]

    def test_skips_chunks_without_candidates(self, client):
        class Empty:
            candidates = None

        chat = client.start_chat()
        chat.stream_chunks = [Empty(), self._chunk("text")]

        assert list(client.send_chat_message_stream(chat, "hi")) == ["text"]

    def test_an_empty_stream_yields_nothing(self, client):
        chat = client.start_chat()

        assert list(client.send_chat_message_stream(chat, "hi")) == []
