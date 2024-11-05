# tests/test_env.py
import pytest
from typing import Dict
from unittest.mock import Mock

from src.env import Game20QEnv, TURN_TYPE, AGENT_ROLE


class MockAgent:
    def __init__(self, responses: Dict):
        self.ask_question = Mock(return_value=responses.get("question", "Is it alive?"))
        self.respond = Mock(return_value=responses.get("answer", "yes"))
        self.make_guess = Mock(return_value=responses.get("guess", "chicken"))
        # self.choose_topic = Mock(return_value=responses.get("topic", "chicken"))


@pytest.fixture
def mock_host():
    return MockAgent({"topic": "chicken", "answer": "yes"})


@pytest.fixture
def mock_guesser():
    return MockAgent({"question": "Is it alive?", "guess": "chicken"})


@pytest.fixture
def env(mock_host, mock_guesser):
    return Game20QEnv(mock_host, mock_guesser, True)


def test_reset(env):
    obs = env.reset()

    assert isinstance(obs, list)
    host_obs = obs[AGENT_ROLE.HOST.value]
    guesser_obs = obs[AGENT_ROLE.GUESSER.value]

    assert host_obs.turn == 0
    assert host_obs.turn_type == TURN_TYPE.ASK_QUESTION
    assert not host_obs.active

    assert guesser_obs.turn == 0
    assert guesser_obs.turn_type == TURN_TYPE.ASK_QUESTION
    assert guesser_obs.active


def test_ask_question_step(env):
    env.reset()
    obs, rewards, dones, info = env.step()

    assert env.current_type == TURN_TYPE.ANSWER_QUESTION
    assert env.current_question == "Is it alive?"
    assert info["action"] == "question_asked"

    host_obs = obs[AGENT_ROLE.HOST.value]
    guesser_obs = obs[AGENT_ROLE.GUESSER.value]

    assert host_obs.active
    assert not guesser_obs.active


def test_answer_question_step(env):
    env.reset()
    env.step()  # Ask question
    obs, rewards, dones, info = env.step()  # Answer question

    assert env.current_type == TURN_TYPE.MAKE_GUESS
    assert env.current_answer == "yes"
    assert info["action"] == "question_answered"

    host_obs = obs[AGENT_ROLE.HOST.value]
    guesser_obs = obs[AGENT_ROLE.GUESSER.value]

    assert not host_obs.active
    assert guesser_obs.active


def test_make_guess_step(env):
    env.reset()
    env.topic = "chicken"
    env.step()  # Ask question
    env.step()  # Answer question
    obs, rewards, dones, info = env.step()  # Make guess

    assert info.get("action") == "end_game"
    assert info.get("reason") == "correct_guess"
    assert len(env.history) == 1
    assert env.history[-1]["guess"] == "chicken"

    # Correct guess should end game
    assert dones == [True, True]
    assert rewards == [0.0, 1.0]


def test_max_turns(env):
    env.turn = env.max_turns
    obs, rewards, dones, info = env.step()

    assert info["reason"] == "max_turns"
    assert all(dones)
    assert rewards == [1.0, 0.0]


# Fix fixture usage
def test_wrong_guess(env, mock_host, mock_guesser):  # Use fixtures as parameters
    # Override guesser's guess
    mock_guesser.make_guess = Mock(return_value="dog")
    env = Game20QEnv(mock_host, mock_guesser)

    env.reset()
    env.step()  # Ask question
    env.step()  # Answer question
    obs, rewards, dones, info = env.step()  # Make wrong guess

    assert not any(dones)
    assert all(r == 0.0 for r in rewards)
    assert env.turn == 1
    assert env.current_type == TURN_TYPE.ASK_QUESTION


def test_invalid_responses(env):
    env.reset()

    # Test invalid question
    env.guesser.ask_question = Mock(return_value=123)
    with pytest.raises(ValueError, match="Question must be string"):
        env.step()

    # Test invalid answer
    env.guesser.ask_question = Mock(return_value="Is it alive?")
    env.host.respond = Mock(return_value="maybe")
    env.step()  # Ask question
    with pytest.raises(ValueError, match="Answer must be 'yes' or 'no'"):
        env.step()
