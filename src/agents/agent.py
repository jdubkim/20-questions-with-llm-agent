from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, api_endpoint: str):
        self.api_endpoint = api_endpoint

    @abstractmethod
    async def act(self, state: dict) -> dict:
        pass
