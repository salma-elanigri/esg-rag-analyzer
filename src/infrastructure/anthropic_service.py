from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from src.core.interfaces import ILLMService
from typing import Generator
from dotenv import load_dotenv

load_dotenv(override=True)

class AnthropicService(ILLMService):
    def __init__(self, model_name: str = "claude-haiku-4-5-20251001", max_tokens: int = 1000):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.llm = ChatAnthropic(model=self.model_name, max_tokens=self.max_tokens)

    def get_llm(self) -> BaseChatModel:
        return self.llm

    def generate(self, prompt: str) -> Generator:
        for chunk in self.llm.stream(prompt):
            yield chunk.text