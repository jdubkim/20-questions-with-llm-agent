from abc import ABC

from src.env import Observation, TURN_TYPE
from src.model import ModelWrapper
from src.utils import PromptManager
import src.utils as utils


class BaseAgent(ABC):
    def __init__(
        self,
        model: ModelWrapper,
        prompt_manager: PromptManager,
    ):
        self.model = model
        self.prompt_manager = prompt_manager

    def act(self, observation: Observation) -> str:
        """Generate an action based on the observation."""
        messages = self.prompt_manager.build_agent_prompt(observation)
        response = self.model.generate(messages)
        self.prompt_manager.add_assistant_message(response)
        return response

    def _parse_response(self, response: str) -> str:
        """Parse the LLM response to extract the action."""
        # Implement parsing logic specific to the agent
        return response.strip()


class HostAgent(BaseAgent):
    # def choose_topic(self, observation: Observation) -> str:
    #     """Host chooses a topic to start the game."""
    #     response = self.act(observation)
    #     # TODO: Change calling KNOWLEDGE_BASE directly
    #     response = utils.parse_check_valid_topic(response, KNOWLEDGE_BASE)

    #     return response

    def respond(self, observation: Observation) -> str:
        """Respond to the guesser's question."""
        if observation.turn_type != TURN_TYPE.ANSWER_QUESTION:
            raise ValueError("Host can only respond to questions.")
        response = self.act(observation)
        response = utils.check_valid_response(response)

        return response


class GuesserAgent(BaseAgent):
    def ask_question(self, observation: Observation) -> str:
        """Ask a question to narrow down the topic."""
        if observation.turn_type != TURN_TYPE.ASK_QUESTION:
            raise ValueError("Guesser can only ask questions.")
        response = self.act(observation)
        response = utils.parse_check_question(response)

        return response

    def make_guess(self, observation: Observation) -> str:
        """Make a guess about the topic."""
        if observation.turn_type != TURN_TYPE.MAKE_GUESS:
            raise ValueError("Guesser can only make guesses.")
        response = self.act(observation)
        response = utils.parse_check_guess(response)

        return response
