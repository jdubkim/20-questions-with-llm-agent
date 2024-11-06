# tests/test_evaluator.py
import pytest
from datetime import datetime
import json
from pathlib import Path

from src.evaluator import Result, Evaluator
from src.config import Config, ModelConfig, EnvConfig, PromptConfig


@pytest.fixture
def sample_config():
    return Config(
        model=ModelConfig(name="test-model"),
        env=EnvConfig(max_turns=5),
        prompts=PromptConfig(
            host_system="test host",
            guesser_system="test guesser",
            templates={"test": {"role": "user", "content": "test"}},
        ),
        run_id="test-run",
    )


@pytest.fixture
def sample_history():
    return [
        {"turn": 0, "question": "Is it alive?", "answer": "no", "guess": "rock"},
        {"turn": 1, "question": "Is it man-made?", "answer": "yes", "guess": "car"},
    ]


@pytest.fixture
def temp_log_dir(tmp_path):
    return tmp_path / "test_logs"


class TestEvaluator:
    def test_evaluator_initialization(self, sample_config, temp_log_dir):
        evaluator = Evaluator(sample_config, log_dir=str(temp_log_dir))
        assert evaluator.config == sample_config
        assert (temp_log_dir / sample_config.run_id / "config.json").exists()

    def test_log_game(self, sample_config, temp_log_dir, sample_history):
        evaluator = Evaluator(sample_config, log_dir=str(temp_log_dir))
        result = Result(
            topic="car",
            num_turns=2,
            success=True,
            history=sample_history,
            timestamp=datetime.now().isoformat(),
        )
        evaluator.log_game(result)

        # Check result was added to evaluator
        assert len(evaluator.results) == 1

        # Check game log file exists
        game_logs = list(Path(evaluator.log_dir).glob("game_*.json"))
        assert len(game_logs) == 1

        # Verify log content
        with open(game_logs[0]) as f:
            log_data = json.load(f)
            assert log_data["topic"] == "car"
            assert log_data["num_turns"] == 2
            assert log_data["success"] is True

    def test_calculate_metrics(self, sample_config, temp_log_dir, sample_history):
        evaluator = Evaluator(sample_config, log_dir=str(temp_log_dir))

        # Add mix of successful and failed games
        results = [
            Result(
                topic="car",
                num_turns=2,
                success=True,
                history=sample_history,
                timestamp=datetime.now().isoformat(),
            ),
            Result(
                topic="dog",
                num_turns=5,
                success=False,
                failure="",
                history=sample_history,
                timestamp=datetime.now().isoformat(),
            ),
        ]

        for result in results:
            evaluator.log_game(result)

        metrics = evaluator.calculate_metrics()
        assert metrics["guess_success_rate"] == 0.5
        assert metrics["average_turns"] == 3.5
        assert len(metrics["failure_counts"]) == 1
