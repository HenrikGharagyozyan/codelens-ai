import os
from google import genai
from dotenv import load_dotenv


class GeminiClient:
    def __init__(self):
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in .env file. Please create one.")
        
        self.client = genai.Client(api_key=api_key)


    def ask(self, context_chunks: str, question: str) -> str:
        """Builds the system prompt and sends a request to the AI."""

        # Build a clean context string from the matching code chunks
        context_str = ""
        for chunk in context_chunks:
            meta = chunk["metadata"]
            context_str += f"--- File: {meta['file_path']} | Symbol: {meta['symbol_name']} ---\n"
            context_str += f"{chunk['document']}\n\n"
        
        prompt = f"""
        You are CodeLens, an expert AI assistant designed to help software engineers navigate and understand their local codebase.
        
        Below is some relevant context extracted from the user's repository via AST parsing and dependency tracking:
        
        --- CODE CONTEXT START ---
        {context_str}
        --- CODE CONTEXT END ---
        
        Based ONLY on the provided context, answer the following question. 
        Include code snippets in your answer where helpful.
        If the context does not contain enough information to answer fully, say so, but provide the best possible insight based on what is available.
        
        User's question: {question}
        """
        
        response = self.client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        
        return response.text