# tests/test_evaluator.py
import pytest
from pathlib import Path
import json
from datetime import datetime
from src.evaluator import Result, Evaluator, Failure
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


class TestResult:
    def test_result_initialization(self, sample_history):
        result = Result(
            topic="car",
            n_topics=5,
            num_turns=2,
            success=True,
            history=sample_history,
            timestamp=datetime.now().isoformat(),
        )
        assert result.topic == "car"
        assert result.num_turns == 2
        assert result.success
        assert result.history == sample_history

    def test_result_with_failure(self, sample_history):
        result = Result(
            topic="car",
            n_topics=10,
            num_turns=2,
            success=False,
            failure=Failure.MAX_TURNS_EXCEEDED,
            history=sample_history,
            timestamp=datetime.now().isoformat(),
        )
        assert result.failure == Failure.MAX_TURNS_EXCEEDED


class TestEvaluator:
    def test_evaluator_initialization(self, sample_config, temp_log_dir):
        evaluator = Evaluator(sample_config, log_dir=str(temp_log_dir))
        assert evaluator.config == sample_config
        assert (temp_log_dir / "config.json").exists()

    def test_log_game(self, sample_config, temp_log_dir, sample_history):
        evaluator = Evaluator(sample_config, log_dir=str(temp_log_dir))
        result = Result(
            topic="car",
            n_topics=5,
            num_turns=2,
            success=True,
            history=sample_history,
            timestamp=datetime.now().isoformat(),
        )
        evaluator.log_game(result)
        assert len(evaluator.results) == 1
        assert len(list(temp_log_dir.glob("game_*.json"))) == 1

    def test_calculate_metrics(self, sample_config, temp_log_dir, sample_history):
        evaluator = Evaluator(sample_config, log_dir=str(temp_log_dir))

        # Add mix of successful and failed games
        results = [
            Result(
                topic="car",
                n_topics=5,
                num_turns=2,
                success=True,
                history=sample_history,
                timestamp=datetime.now().isoformat(),
            ),
            Result(
                topic="dog",
                n_topics=5,
                num_turns=5,
                success=False,
                failure=Failure.MAX_TURNS_EXCEEDED,
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
