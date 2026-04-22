import anthropic

from src.core.interfaces import ILLMService
from typing import Generator


class AnthropicService(ILLMService):
    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    def generate(self, prompt: str) -> Generator:
        message = anthropic.Anthropic().messages.create(
            model="claude-opus-4-7",
            max_tokens=1000,
            messages=[
                prompt],
        )
        return message