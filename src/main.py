import argparse
import time

from src.env import Game20QEnv, TURN_TYPE, AGENT_ROLE
from src.agents.agent import HostAgent, GuesserAgent
from src.agents.model import OpenAIModelWrapper
from src.utils import PromptManager


HOST_SYSTEM_PROMPT = (
    "You are hosting a game of 20 questions game. Your role is to:\n"
    "1. understand the topic given by the enviornment\n"
    "2. Answer questions about the topic truthfully with only 'yes' or 'no'\n"
    "3. Never reveal the topic directly\n"
    "4. Be consisent with your answers throughout the game\n"
)

GUESSER_SYSTEM_PROMPT = (
    "You are playing a game of 20 questions game as the guesser. Your role is to:\n"
    "1. Ask clear yes/no questions to narrow down the topic\n"
    "2. Track previous questions and answers\n"
    "3. Make eudcated guesses based on the information gathered\n"
    "4. Try to identify the topic within the allowed number of turns\n"
)

PROMPT_TEMPLATES = {
    TURN_TYPE.ASK_QUESTION: (
        "Current game state:\n"
        "Turn: {turn}\n"
        "Was last guess correct: No\n"
        "Ask a single yes/no question to help identify the topic."
    ),
    TURN_TYPE.ANSWER_QUESTION: (
        "Current game state:\n"
        "Turn: {turn}\n"
        "Your chosen topic is: {topic}\n"
        "question: {current_question}\n"
        "Answer only with 'yes' or 'no'."
    ),
    TURN_TYPE.MAKE_GUESS: (
        "Current game state:\n"
        "Turn: {turn}\n"
        "Answer: {current_answer}\n"
        "Make your best guess at the topic. Respond with only your guess in a single or a few words."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a game of 20 questions between two AI agents."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="The OpenAI model to use for generating responses.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=5, help="Maximum number of turns(questions)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information.",
    )

    return parser.parse_args()


def main():
    """Run two agents playing a game of 20 questions."""
    args = parse_args()

    # Initialize model and prompt managers
    model = OpenAIModelWrapper(model_name=args.model, max_retries=1)
    host_prompts = PromptManager(PROMPT_TEMPLATES, HOST_SYSTEM_PROMPT)
    guesser_prompts = PromptManager(PROMPT_TEMPLATES, GUESSER_SYSTEM_PROMPT)

    host = HostAgent(model, host_prompts)
    guesser = GuesserAgent(model, guesser_prompts)

    env = Game20QEnv(host, guesser, debug=args.debug, max_turns=args.max_turns)

    # Run the game
    observations = env.reset()
    done = False

    while not done:
        observations, rewards, dones, info = env.step()
        time.sleep(1)

        if any(dones):
            done = True
            print(f"Game over! The topic was: {env.topic}")
            print(f"Final Rewards - Host: {rewards[0]}, Guesser: {rewards[1]}")
            print(f"Game History:")
            for turn in env.history:
                print(
                    f"Turn {turn['turn']}: Q: {turn['question']} -> A: {turn['answer']} -> Guess: {turn['guess']}"
                )


if __name__ == "__main__":
    main()
