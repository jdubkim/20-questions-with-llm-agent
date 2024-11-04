from abc import ABC, abstractmethod
import re

from openai import OpenAI


class ModelWrapper(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text based on the prompt."""
        pass


class OpenAIModelWrapper(ModelWrapper):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI()
        self.MAX_RETRIES = 3

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return response.choices[0].message.content


class VLLMModelWrapper(ModelWrapper):
    def __init__(self, server_url: str):
        self.server_url = server_url

    def generate(self, prompt: str, **kwargs) -> str:
        import requests

        response = requests.post(self.server_url, json={"prompt": prompt, **kwargs})
        return response.json()["response"].strip()


class DummyModelWrapper(ModelWrapper):
    def generate(self, prompt: str, **kwargs) -> str:
        return "Dummy response"
