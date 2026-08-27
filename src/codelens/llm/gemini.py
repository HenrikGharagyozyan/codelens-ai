import os
from typing import Callable
from google import genai
from google.genai import types
from dotenv import load_dotenv


# Strict LLM instructions: respond so that paths are clickable in the terminal
SYSTEM_PROMPT = """You are CodeLens, an expert AI engineering assistant helping a developer navigate their local codebase.
You have access to the codebase structure, AST metadata, and file contents.

CRITICAL REQUIREMENT FOR CITATIONS AND LINKS:
Whenever you refer to a specific file, class, function, or line of code, you MUST provide a direct, terminal-clickable citation.
Format the citation strictly as: `path/to/file.py:line_number`
Do NOT use Markdown links (e.g., [file](path/to/file.py)).
Do NOT wrap the path in backticks if it prevents the terminal emulator from making it clickable (plain text is preferred for paths).

Example of correct formatting:
The database connection is initialized in src/codelens/repository/db.py:24 inside the __init__ method.

Example of INCORRECT formatting:
The database connection is initialized in [db.py](src/codelens/repository/db.py)

Always verify that the file paths are relative to the project root and include the exact line number where possible.
"""

class GeminiClient:
    def __init__(self):
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in .env file. Please create one.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-3.6-flash'

        # Configure generation parameters and the system prompt
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2, # Low temperature for accurate code citations
        )


    def ask(self, context_chunks: str, question: str) -> str:
        """Builds the system prompt and sends a request to the AI."""

        prompt = f"""
        Below is some relevant context extracted from the user's repository via AST parsing and dependency tracking:
        
        --- CODE CONTEXT START ---
        {context_chunks}
        --- CODE CONTEXT END ---
        
        Based ONLY on the provided context, answer the following question. 
        Include code snippets in your answer where helpful.
        If the context does not contain enough information to answer fully, say so, but provide the best possible insight based on what is available.
        
        User's question: {question}
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.config
        )
        
        return response.text

    def start_chat(self):
        """Creates and returns a Gemini chat session object."""
        return self.client.chats.create(
            model=self.model_name,
            config=self.config
        )

    def send_chat_message(self, chat_session, message: str) -> str:
        """Sends a message to the current chat session and returns the full response."""
        # The chat_session object stores the message history itself
        response = chat_session.send_message(message)
        return response.text

    def start_chat_with_tools(self, search_tool_fn: Callable[[str], str]):
        """
        Creates a chat session with a connected code-search tool.
        `search_tool_fn` is a Python function that Gemini can call itself.
        """
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            tools=[search_tool_fn], # Pass the function as a tool
        )

        return self.client.chats.create(
            model=self.model_name,
            config=config
        )

    def send_chat_message_stream(self, chat_session, message: str):
        """Sends a message and returns a generator for streaming output (prints character by character)."""
        response_stream = chat_session.send_message_stream(message)
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text