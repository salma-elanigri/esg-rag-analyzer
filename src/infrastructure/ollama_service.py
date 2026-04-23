from src.core.interfaces import ILLMService
import ollama
from typing import Generator


class OllamaService(ILLMService):
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name

    def generate(self, prompt: str) -> Generator:
        # make ollama stream the response
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        # Extract the text content from the response
        for chunk in response:
            yield chunk["message"]["content"]
