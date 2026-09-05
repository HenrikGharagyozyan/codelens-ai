import os
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

from codelens.llm.prompts import SYSTEM_PROMPT

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _iter_text_parts(chunk):
    """Flattens the deeply nested object structure of a Gemini stream chunk."""
    if not getattr(chunk, "candidates", None):
        return

    for candidate in chunk.candidates:
        content = getattr(candidate, "content", None)
        if not content or not getattr(content, "parts", None):
            continue

        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                yield text


class GeminiClient:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in .env file. Please create one.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.6-flash"

        # Configure generation parameters and the system prompt
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,  # Low temperature for accurate code citations
        )

    def ask(self, context_chunks: str, question: str) -> str:
        """Builds the system prompt and sends a request to the AI."""

        prompt = f"""
        Below is some relevant context extracted from the user's repository via AST
        parsing and dependency tracking:
        
        --- CODE CONTEXT START ---
        {context_chunks}
        --- CODE CONTEXT END ---
        
        Based ONLY on the provided context, answer the following question. 
        Include code snippets in your answer where helpful.
        If the context does not contain enough information to answer fully, say so, but
        provide the best possible insight based on what is available.
        
        User's question: {question}
        """

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt, config=self.config
        )

        return response.text

    def start_chat(self):
        """Creates and returns a Gemini chat session object."""
        return self.client.chats.create(model=self.model_name, config=self.config)

    def send_chat_message(self, chat_session, message: str) -> str:
        """Sends a message to the current chat session and returns the full response."""
        # The chat_session object stores the message history itself
        response = chat_session.send_message(message)
        return response.text

    def start_chat_with_tools(
        self, search_tool_fn: Callable[[str], str], history_dicts: list[dict] = None
    ):
        """
        Creates a chat session with a connected code-search tool.
        `search_tool_fn` is a Python function that Gemini can call itself.
        """
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            tools=[search_tool_fn],  # Pass the function as a tool
        )

        # Convert our dictionaries to types.Content objects for the SDK
        history = None
        if history_dicts:
            history = []
            for msg in history_dicts:
                history.append(
                    types.Content(
                        role=msg["role"], parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

        return self.client.chats.create(model=self.model_name, config=config, history=history)

    def send_chat_message_stream(self, chat_session, message: str):
        """Sends a message and returns a generator for streaming output."""
        response_stream = chat_session.send_message_stream(message)
        for chunk in response_stream:
            yield from _iter_text_parts(chunk)
