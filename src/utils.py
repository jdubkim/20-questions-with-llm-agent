import re
from typing import Optional
from src.env import Observation


class PromptManager:
    def __init__(
        self, prompt_templates: dict[str], system_prompt: Optional[str] = None
    ):
        self.messages = []
        self.prompt_templates = prompt_templates
        if system_prompt:
            self.add_message("system", system_prompt)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def add_user_message(self, message: str):
        self.add_message("user", message)

    def add_assistant_message(self, message: str):
        self.add_message("assistant", message)

    def format_observation(self, obs: Observation) -> str:
        """Format the observation into a prompt."""
        prompt_template = self.prompt_templates[obs.turn_type]
        prompt = prompt_template.format(
            **{k: v for k, v in obs.__dict__.items() if f"{{{k}}}" in prompt_template}
        )
        return prompt

    def build_agent_prompt(self, obs: Observation) -> list[dict[str, str]]:
        """Build the prompt for the agent based on the observation."""
        context = self.format_observation(obs)

        # Add context as user message
        self.add_user_message(context)

        return self.get_messages()

    def get_messages(self) -> list[dict[str, str]]:
        return self.messages

    def clear(self):
        """Clear conversation history."""
        self.messages = []


def parse_check_valid_topic(response: str, knowledge_base: list) -> str:
    """Check if the response is a valid topic."""
    if not response:
        raise ValueError("Host must choose a valid topic.")

    if response not in knowledge_base:
        return None

    return response


def check_valid_response(response: str) -> str:
    """Check if the response is valid."""
    if not response:
        raise ValueError("Agent must provide a valid response.")

    if "yes" in response:
        return "yes"
    elif "no" in response:
        return "no"


def parse_check_question(response: str) -> str:
    """Check if the response is a valid question."""
    if not response:
        raise ValueError("Guesser must ask a valid question.")

    pattern = r"([^\n]*\?)"
    questions = re.findall(pattern, response)

    if len(questions) == 0:
        return None

    return questions[0]


def parse_check_guess(response: str) -> str:
    """Check if the response is a valid guess."""
    if not response:
        raise ValueError("Guesser must make a valid guess.")

    return response.replace("*", "").replace('"', "").split("\n")[0]
