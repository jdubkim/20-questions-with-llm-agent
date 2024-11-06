from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import json

from src.env import TURN_TYPE


@dataclass
class ModelConfig:
    name: str = "gpt-4o-mini"
    max_retries: int = 3
    temperature: float = 0.7


@dataclass
class EnvConfig:
    max_turns: int = 20
    debug: bool = False
    knowledge_base: list[str] = field(
        default_factory=lambda: ["dog", "cat", "chicken", "car", "plane"]
    )


@dataclass
class PromptConfig:
    host_system: str
    guesser_system: str
    templates: dict[str, dict[str, str]]

    def encode(self) -> dict:
        return {
            "host_system": self.host_system,
            "guesser_system": self.guesser_system,
            "templates": {
                k.value if isinstance(k, Enum) else k: v
                for k, v in self.templates.items()
            },
        }

    @classmethod
    def decode(cls, data: dict):
        templates = {
            TURN_TYPE(k) if k in [e.value for e in TURN_TYPE] else k: v
            for k, v in data["templates"].items()
        }
        return cls(
            host_system=data["host_system"],
            guesser_system=data["guesser_system"],
            templates=templates,
        )


@dataclass
class Config:
    model: ModelConfig
    env: EnvConfig
    prompts: PromptConfig
    run_id: str
    n_games: int = 1

    def save(self, path: Path):
        data = {
            "model": asdict(self.model),
            "env": asdict(self.env),
            "prompts": self.prompts.encode(),
            "n_games": self.n_games,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path) as f:
            data = json.load(f)
            return cls(
                model=ModelConfig(**data["model"]),
                env=EnvConfig(**data["env"]),
                prompts=PromptConfig.decode(data["prompts"]),
                n_games=data["n_games"],
            )
