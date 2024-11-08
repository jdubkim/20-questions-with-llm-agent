from abc import ABC, abstractmethod
import asyncio
from openai import AsyncOpenAI

from src.exceptions import APIError


RETRY_WAIT_TIME = 1.0


class ModelWrapper(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        pass


class OpenAIModelWrapper(ModelWrapper):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.client = AsyncOpenAI()
        self.max_retries = max_retries

    async def generate(self, prompts: list[dict[str, str]], **kwargs) -> str:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name, messages=prompts, **kwargs
                )

                if not response.choices:
                    raise APIError("No response choices returned.")

                return response.choices[0].message.content

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    asyncio.sleep(RETRY_WAIT_TIME)

        raise APIError("Failed to generate response.")


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
