from src.core.interfaces import ILLMService
import ollama


class OllamaService(ILLMService):
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )

        # Extract the text content from the response
        return response['message']['content']
