import anthropic

from src.core.interfaces import ILLMService
from typing import Generator
from dotenv import load_dotenv

load_dotenv()

class AnthropicService(ILLMService):
    def __init__(self, model_name: str = "claude-haiku-4-5-20251001", max_tokens: int = 1000):
        self.model_name = model_name
        self.client = anthropic.Anthropic()
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> Generator:
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text