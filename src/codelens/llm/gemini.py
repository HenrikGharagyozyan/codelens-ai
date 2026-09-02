import os
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Strict LLM instructions: respond so that paths are clickable in the terminal
SYSTEM_PROMPT = """You are CodeLens, an expert AI engineering assistant helping a developer
navigate their local codebase.
You have access to the codebase structure, AST metadata, and file contents.

CRITICAL REQUIREMENT FOR CITATIONS AND LINKS:
Whenever you refer to a specific file, class, function, or line of code, you MUST provide
a direct, terminal-clickable citation.
Format the citation strictly as: `path/to/file.py:line_number`
Do NOT use Markdown links (e.g., [file](path/to/file.py)).
Do NOT wrap the path in backticks if it prevents the terminal emulator from making it
clickable (plain text is preferred for paths).

Example of correct formatting:
The database connection is initialized in src/codelens/repository/db.py:24 inside the
__init__ method.

Example of INCORRECT formatting:
The database connection is initialized in [db.py](src/codelens/repository/db.py)

LINE NUMBERS ARE FACTS, NOT ESTIMATES. Obey these rules without exception:

1. Every code block in the context is rendered as `<line> | <code>`, where
   `<line>` is the REAL line number in the file. Read the number off the line
   you are citing and copy it verbatim.
2. NEVER compute, infer, offset, or estimate a line number. Do not reason like
   "the class starts at 5 and this method looks like the third one, so ~73".
   If you did not read the number, you do not know it.
3. The context also lists verified locations for related symbols, in the forms
   "**Definitions inside this chunk:** `name` -> path:line" and
   "`name` (path:line)". These are authoritative. Use them.
4. If a symbol is marked "(external, no location)", it is not part of this
   repository. Name it, but give NO citation for it.
5. If you want to mention a symbol whose line number appears nowhere in the
   context, cite the FILE ONLY, with no `:line` suffix, and say the exact line
   is not in the retrieved context. Guessing is a factual error, and a wrong
   line is far worse for the user than a missing one.
6. Never cite a file that does not appear in the context.
"""


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
        """Sends a message and returns a generator for streaming output.

        The caller prints the yielded fragments as they arrive, character by character.
        """
        response_stream = chat_session.send_message_stream(message)
        for chunk in response_stream:
            # Work around the `non-text parts` warning by reading only text parts manually.
            if getattr(chunk, "candidates", None):
                for candidate in chunk.candidates:
                    if getattr(candidate, "content", None) and getattr(
                        candidate.content, "parts", None
                    ):
                        for part in candidate.content.parts:
                            # If a response part contains text, return it
                            if getattr(part, "text", None):
                                yield part.text
