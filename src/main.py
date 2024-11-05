import argparse
import json
import time
from typing import Optional

from src.env import Game20QEnv, TURN_TYPE, KNOWLEDGE_BASE
from src.agents.agent import HostAgent, GuesserAgent
from src.agents.model import OpenAIModelWrapper
from src.config import Config, EnvConfig, ModelConfig, PromptConfig
from src.env import Game20QEnv, Failure
from src.evaluator import Evaluator, Result
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
    f"The topic is chosen from the following list: {', '.join(KNOWLEDGE_BASE)}\n"
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
        "--run-type",
        type=str,
        default="play",
        choices=["play", "eval"],
        help="The type of run to execute: play or eval.",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=5,
        help="Number of games to evaluate the agents on.",
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


def log_failure(env: Game20QEnv, failure: Failure, evaluator: Evaluator):
    """Log the failure and return the failure message."""
    result = Result(
        topic=env.topic,
        n_topics=env.n_topics,
        num_turns=env.turn,
        success=False,
        failure=failure,
        history=env.history,
        timestamp=time.time(),
    )
    evaluator.log_game(result)
    return failure


def print_result(env: Game20QEnv, result: Result):
    print(f"Game over! The topic was: {env.topic}")
    print("Game History:")
    for turn in env.history:
        print(
            f"Turn {turn['turn']}: Q: {turn['question']} -> A: {turn['answer']} -> Guess: {turn['guess']}"
        )


def run_play(
    config, evaluator: Optional[Evaluator] = None, verbose: int = 0
) -> Optional[Result]:
    """Run a single game"""
    # Initialize model and prompt managers
    model = OpenAIModelWrapper(
        model_name=config.model.name,
        max_retries=config.model.max_retries,
    )
    host_prompts = PromptManager(
        config.prompts.templates,
        config.prompts.host_system,
    )
    guesser_prompts = PromptManager(
        config.prompts.templates,
        config.prompts.guesser_system,
    )

    host = HostAgent(model, host_prompts)
    guesser = GuesserAgent(model, guesser_prompts)

    env = Game20QEnv(
        host,
        guesser,
        debug=config.env.debug,
        max_turns=config.env.max_turns,
        knowledge_base=config.env.knowledge_base,
    )

    # Run the game
    observations = env.reset()
    done = False

    while not done:
        try:
            observations, rewards, dones, info = env.step()

            if any(dones):
                done = True
        except Exception as e:
            print(f"An error occurred: {e}")
            failure = e.args[0]
            if evaluator is not None:
                return log_failure(env, failure, evaluator)

    success = info.get("reason") == "correct_guess"
    result = Result(
        topic=env.topic,
        n_topics=env.n_topics,
        num_turns=env.turn,
        success=success,
        history=env.history,
        failure=Failure.MAX_TURNS_EXCEEDED if not success else None,
        timestamp=time.time(),
    )

    if evaluator:
        evaluator.log_game(result)

    if verbose:
        print_result(env, result)

    return result


def run_eval(config: Config):
    """Run multiple games to evaluate the agents."""
    evaluator = Evaluator(config)

    for _ in range(config.n_games):
        run_play(config, evaluator, verbose=1)

    metrics = evaluator.calculate_metrics()
    print(f"Metrics for {config.n_games} games:")
    print(json.dumps(metrics))


def main():
    """Run two agents playing a game of 20 questions."""
    args = parse_args()

    config = Config(
        model=ModelConfig(
            name=args.model,
            max_retries=1,
        ),
        env=EnvConfig(
            max_turns=args.max_turns,
            debug=args.debug,
        ),
        prompts=PromptConfig(
            host_system=HOST_SYSTEM_PROMPT,
            guesser_system=GUESSER_SYSTEM_PROMPT,
            templates=PROMPT_TEMPLATES,
        ),
        n_games=args.n_games,
    )

    if args.run_type == "play":
        run_play(config)

    if args.run_type == "eval":
        run_eval(config)


if __name__ == "__main__":
    main()
