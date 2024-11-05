import re
from typing import Optional
from src.env import Observation, KNOWLEDGE_BASE


class PromptManager:
    def __init__(
        self, prompt_templates: dict[str], system_prompt: Optional[str] = None
    ):
        self.messages = []
        self.prompt_templates = prompt_templates
        if system_prompt:
            self.add_system_message(system_prompt)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def add_system_message(self, message: str):
        self.add_message("system", message)

    def add_user_message(self, message: str):
        self.add_message("user", message)

    def add_assistant_message(self, message: str):
        self.add_message("assistant", message)

    def format_observation(self, obs: Observation) -> str:
        """Format the observation into a prompt."""
        prompt_template = self.prompt_templates[obs.turn_type]
        if isinstance(prompt_template, tuple):
            prompt_template = "".join(prompt_template)

        template_vars = obs._asdict()
        # Add global variables
        template_vars.update(
            {
                "KNOWLEDGE_BASE": ", ".join(KNOWLEDGE_BASE),
                # Add other global vars here if needed
            }
        )

        formatted_content = prompt_template.format(
            **{k: v for k, v in template_vars.items() if f"{{{k}}}" in prompt_template}
        )

        return formatted_content

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

    for knowledge_base_item in knowledge_base:
        if knowledge_base_item in response:
            return knowledge_base_item

    return ""


def check_valid_response(response: str) -> str:
    """Check if the response is valid."""
    if not response:
        raise ValueError("Agent must provide a valid response.")

    response = response.lower().strip()

    if "yes" in response:
        return "yes"
    elif "no" in response:
        return "no"
    else:
        return ""


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
