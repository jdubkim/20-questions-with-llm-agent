from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import Config


@dataclass
class Result:
    topic: str
    num_turns: int
    success: bool
    history: list[dict]
    timestamp: str
    failure: Optional[str] = None


class Evaluator:
    def __init__(self, config: Config, log_dir: str = "logs"):
        self.log_dir = Path(log_dir) / config.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.config = config
        self.config.save(self.log_dir / "config.json")

    def log_game(self, result: Result):
        """Log the result of a game."""
        result.timestamp = datetime.now().isoformat()
        self.results.append(result)

        log_file = self.log_dir / f"game_{result.timestamp}.json"
        with open(log_file, "w") as f:
            json.dump(asdict(result), f)

    def calculate_metrics(self) -> dict:
        """Calculate metrics for the game agents.
        1. Guess Success Rate: % of games where the guesser correctly guessed the topic.
        2. Average Turns: Average number of turns taken to guess the topic.
        3. Failure Counts: Count of each type of failure, to track agent's potential issues.
        4. # topics: Difficulty of the game.

        Returns:
            dict: _description_
        """
        if not self.results:
            return {}

        df = pd.DataFrame([asdict(result) for result in self.results])

        metrics = {
            "total games": len(df),
            "guess_success_rate": df.success.mean(),
            "average_turns": df.num_turns.mean(),
            "failure_counts": df.failure.value_counts().to_dict(),
            "num_topics": df.topic.value_counts().to_dict(),
        }
        return metrics
