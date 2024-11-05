import pytest
from unittest.mock import Mock, patch
from src.env import Observation, TURN_TYPE, AGENT_ROLE
from src.agents.agent import BaseAgent, HostAgent, GuesserAgent
from src.utils import PromptManager
from src.agents.model import ModelWrapper


@pytest.fixture
def mock_model():
    model = Mock(spec=ModelWrapper)
    model.generate.return_value = "test response"
    return model


@pytest.fixture
def mock_prompt_manager():
    pm = Mock(spec=PromptManager)
    pm.build_agent_prompt.return_value = [{"role": "user", "content": "test prompt"}]
    return pm


@pytest.fixture
def base_observation():
    return Observation(
        turn=1,
        history=[],
        turn_type=TURN_TYPE.ASK_QUESTION,
        active=True,
        role=AGENT_ROLE.GUESSER,
        remaining_turns=19,
        current_question=None,
        current_answer=None,
    )


class TestHostAgent:
    def test_init(self, mock_model, mock_prompt_manager):
        agent = HostAgent(mock_model, mock_prompt_manager)
        assert agent.model == mock_model
        assert agent.prompt_manager == mock_prompt_manager

    # def test_choose_topic(self, mock_model, mock_prompt_manager):
    #     agent = HostAgent(mock_model, mock_prompt_manager)
    #     mock_model.generate.return_value = "cat"
    #     topic = agent.choose_topic(base_observation)
    #     assert topic == "cat"
    #     assert mock_model.generate.called

    def test_respond_invalid_turn(
        self, mock_model, mock_prompt_manager, base_observation
    ):
        agent = HostAgent(mock_model, mock_prompt_manager)
        obs = base_observation._replace(turn_type=TURN_TYPE.ASK_QUESTION)
        with pytest.raises(ValueError):
            agent.respond(obs)

    def test_respond_valid(self, mock_model, mock_prompt_manager, base_observation):
        agent = HostAgent(mock_model, mock_prompt_manager)
        mock_model.generate.return_value = "yes"
        obs = base_observation._replace(turn_type=TURN_TYPE.ANSWER_QUESTION)
        response = agent.respond(obs)
        assert response == "yes"


class TestGuesserAgent:
    def test_init(self, mock_model, mock_prompt_manager):
        agent = GuesserAgent(mock_model, mock_prompt_manager)
        assert agent.model == mock_model
        assert agent.prompt_manager == mock_prompt_manager

    def test_ask_question_invalid_turn(
        self, mock_model, mock_prompt_manager, base_observation
    ):
        agent = GuesserAgent(mock_model, mock_prompt_manager)
        obs = base_observation._replace(turn_type=TURN_TYPE.ANSWER_QUESTION)
        with pytest.raises(ValueError):
            agent.ask_question(obs)

    def test_ask_question_valid(
        self, mock_model, mock_prompt_manager, base_observation
    ):
        agent = GuesserAgent(mock_model, mock_prompt_manager)
        mock_model.generate.return_value = "Is it an animal?"
        obs = base_observation._replace(turn_type=TURN_TYPE.ASK_QUESTION)
        question = agent.ask_question(obs)
        assert question == "Is it an animal?"

    def test_make_guess_invalid_turn(
        self, mock_model, mock_prompt_manager, base_observation
    ):
        agent = GuesserAgent(mock_model, mock_prompt_manager)
        obs = base_observation._replace(turn_type=TURN_TYPE.ASK_QUESTION)
        with pytest.raises(ValueError):
            agent.make_guess(obs)

    def test_make_guess_valid(self, mock_model, mock_prompt_manager, base_observation):
        agent = GuesserAgent(mock_model, mock_prompt_manager)
        mock_model.generate.return_value = "cat"
        obs = base_observation._replace(turn_type=TURN_TYPE.MAKE_GUESS)
        guess = agent.make_guess(obs)
        assert guess == "cat"
