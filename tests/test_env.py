# tests/test_env.py
import pytest
import asyncio
from typing import Dict
from unittest.mock import AsyncMock, Mock

from src.env import Game20QEnv, TURN_TYPE, AGENT_ROLE
from src.exceptions import InvalidQuestionError, InvalidAnswerError, InvalidGuessError
from src.main import KNOWLEDGE_BASE


class MockAgent:
    def __init__(self, responses: Dict):
        self.ask_question = AsyncMock(
            return_value=responses.get("question", "Is it alive?")
        )
        self.respond = AsyncMock(return_value=responses.get("answer", "yes"))
        self.make_guess = AsyncMock(return_value=responses.get("guess", "chicken"))


@pytest.fixture
def mock_host():
    return MockAgent({"topic": "chicken", "answer": "yes"})


@pytest.fixture
def mock_guesser():
    return MockAgent({"question": "Is it alive?", "guess": "chicken"})


@pytest.fixture
def env(mock_host, mock_guesser):
    return Game20QEnv(mock_host, mock_guesser, KNOWLEDGE_BASE, True)


def test_reset(env):
    obs = env.reset()

    assert isinstance(obs, list)
    host_obs = obs[AGENT_ROLE.HOST.value]
    guesser_obs = obs[AGENT_ROLE.GUESSER.value]

    assert host_obs.turn == 1
    assert host_obs.turn_type == TURN_TYPE.ASK_QUESTION
    assert not host_obs.active
    assert guesser_obs.active


@pytest.mark.asyncio
async def test_ask_question_step(env):
    env.reset()
    obs, rewards, dones, info = await env.step()

    assert env.current_type == TURN_TYPE.ANSWER_QUESTION
    assert env.current_question == "Is it alive?"
    assert info["action"] == "question_asked"

    host_obs = obs[AGENT_ROLE.HOST.value]
    guesser_obs = obs[AGENT_ROLE.GUESSER.value]

    assert host_obs.active
    assert not guesser_obs.active


@pytest.mark.asyncio
async def test_answer_question_step(env):
    env.reset()
    await env.step()  # Ask question
    obs, rewards, dones, info = await env.step()  # Answer question

    assert env.current_type == TURN_TYPE.MAKE_GUESS
    assert env.current_answer == "yes"
    assert info["action"] == "question_answered"

    host_obs = obs[AGENT_ROLE.HOST.value]
    guesser_obs = obs[AGENT_ROLE.GUESSER.value]

    assert not host_obs.active
    assert guesser_obs.active


@pytest.mark.asyncio
async def test_make_guess_step(env):
    env.reset()
    env.topic = "chicken"
    await env.step()  # Ask question
    await env.step()  # Answer question
    obs, rewards, dones, info = await env.step()  # Make guess

    assert info.get("action") == "end_game"
    assert info.get("reason") == "correct_guess"
    assert len(env.history) == 1
    assert env.history[-1]["guess"] == "chicken"

    # Correct guess should end game
    assert dones == [True, True]
    assert rewards == [0.0, 1.0]


@pytest.mark.asyncio
async def test_max_turns(env):
    env.turn = env.max_turns + 1
    obs, rewards, dones, info = await env.step()

    assert info["reason"] == "max_turns"
    assert all(dones)
    assert rewards == [1.0, 0.0]


# Fix fixture usage
@pytest.mark.asyncio
async def test_wrong_guess(env, mock_host, mock_guesser):  # Use fixtures as parameters
    # Override guesser's guess
    mock_guesser.make_guess = Mock(return_value=asyncio.Future())
    mock_guesser.make_guess.return_value.set_result("dog")
    env = Game20QEnv(mock_host, mock_guesser, KNOWLEDGE_BASE)

    env.reset()
    await env.step()  # Ask question
    await env.step()  # Answer question
    obs, rewards, dones, info = await env.step()  # Make wrong guess

    assert not any(dones)
    assert all(r == 0.0 for r in rewards)
    assert env.turn == 2
    assert env.current_type == TURN_TYPE.ASK_QUESTION


@pytest.mark.asyncio
async def test_invalid_responses(env):
    env.reset()

    # Test invalid question
    env.guesser.ask_question = Mock(return_value=asyncio.Future())
    env.guesser.ask_question.return_value.set_result(123)
    with pytest.raises(InvalidQuestionError):
        await env.step()

    # Test invalid answer
    env.guesser.ask_question = Mock(return_value=asyncio.Future())
    env.guesser.ask_question.return_value.set_result("Is it alive?")
    env.host.respond = Mock(return_value=asyncio.Future())
    env.host.respond.return_value.set_result("maybe")
    await env.step()  # Ask question
    with pytest.raises(InvalidAnswerError):
        await env.step()

    # Test invalid guess
    env.reset()
    env.guesser.ask_question = Mock(return_value=asyncio.Future())
    env.guesser.ask_question.return_value.set_result("Is it alive?")
    await env.step()  # Ask question
    env.host.respond = Mock(return_value=asyncio.Future())
    env.host.respond.return_value.set_result("yes")
    await env.step()  # Answer question
    env.guesser.make_guess = Mock(
        return_value=asyncio.Future()
    )  # Changed from host to guesser
    env.guesser.make_guess.return_value.set_result(123)
    with pytest.raises(InvalidGuessError):
        await env.step()
