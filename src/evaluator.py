from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import Config
from src.env import Failure


@dataclass
class Result:
    topic: str
    n_topics: int
    num_turns: int
    success: bool
    history: list[dict]
    timestamp: str
    failure: Optional[Failure] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if data["failure"]:
            data["failure"] = data["failure"].value

        return data


class Evaluator:
    def __init__(self, config: Config, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.config = config

        # Save config for tracking/experimenting purposes
        self.config.save(self.log_dir / "config.json")

    def log_game(self, result: Result):
        """Log the result of a game."""
        result.timestamp = datetime.now().isoformat()
        self.results.append(result)

        log_file = self.log_dir / f"game_{result.timestamp}.json"
        with open(log_file, "w") as f:
            json.dump(result.to_dict(), f)

    def calculate_metrics(self) -> dict:
        """Calculate metrics for the game agents.
        1. Guess Success Rate: % of games where the guesser correctly guessed the topic.
        2. Average Turns: Average number of turns taken to guess the topic.
        3. Failure Counts: Count of each type of failure, to track agent's potential issues.
        4. # topics: Difficulty of the game.

        Returns:
            dict: _description_
        """
        df = pd.DataFrame([result.to_dict() for result in self.results])

        metrics = {
            "guess_success_rate": df.success.mean(),
            "average_turns": df.num_turns.mean(),
            "failure_counts": df.failure.value_counts().to_dict(),
            "num_topics": df.n_topics.mean(),
        }

        return metrics
