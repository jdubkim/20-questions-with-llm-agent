import argparse
import asyncio
from datetime import datetime
import json
import time
from typing import Optional

from src.env import Game20QEnv, TURN_TYPE
from src.agent import HostAgent, GuesserAgent
from src.model import OpenAIModelWrapper
from src.utils import PromptManager
from src.config import Config, ModelConfig, EnvConfig, PromptConfig
from src.evaluator import Evaluator, Result
from src.exceptions import (
    InvalidQuestionError,
    InvalidAnswerError,
    InvalidGuessError,
    APIError,
)


KNOWLEDGE_BASE = [
    "dog",
    "cat",
    "chicken",
    "car",
    "plane",
]

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
        "--run-id",
        type=str,
        default=str(int(time.time())),
        help="Unique identifier for the run.",
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


def exception_to_failure(e: Exception) -> str:
    if isinstance(e, InvalidQuestionError):
        return "Invalid question"
    elif isinstance(e, InvalidAnswerError):
        return "Invalid answer"
    elif isinstance(e, InvalidGuessError):
        return "Invalid guess"
    elif isinstance(e, APIError):
        return "API error"
    else:
        return "Unknown error"


def print_result(env: Game20QEnv, result: Result):
    print(f"Game over! The topic was: {env.topic}")
    print("Game History:")
    for turn in env.history:
        print(
            f"Turn {turn['turn']}: Q: {turn['question']} -> A: {turn['answer']} -> Guess: {turn['guess']}"
        )


async def run_play(config) -> Optional[Result]:
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
        knowledge_base=config.env.knowledge_base,
        debug=config.env.debug,
        max_turns=config.env.max_turns,
    )

    # Run the game
    observations = env.reset()
    done = False

    try:
        while not done:
            observations, rewards, dones, info = await env.step()
            if any(dones):
                done = True
    except Exception as e:
        failure_reason = exception_to_failure(e)
        result = Result(
            topic=env.topic,
            num_turns=env.turn,
            success=False,
            history=env.history,
            failure=failure_reason,
            timestamp=datetime.now().isoformat(),
        )
        return result

    success = info.get("reason") == "correct_guess"
    result = Result(
        topic=env.topic,
        num_turns=env.turn,
        success=success,
        history=env.history,
        failure=None if success else "Max turns exceeded",
        timestamp=datetime.now().isoformat(),
    )
    return result


async def run_eval(config: Config):
    """Run multiple games to evaluate the agents."""
    # TODO: Change evaluator to run in parallel
    evaluator = Evaluator(config)
    tasks = []

    for game in range(config.n_games):
        task = asyncio.create_task(run_play(config))
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    for result in results:
        evaluator.log_game(result)

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
            knowledge_base=KNOWLEDGE_BASE,
        ),
        prompts=PromptConfig(
            host_system=HOST_SYSTEM_PROMPT,
            guesser_system=GUESSER_SYSTEM_PROMPT,
            templates=PROMPT_TEMPLATES,
        ),
        run_id=args.run_id,
        n_games=args.n_games,
    )

    if args.run_type == "play":
        asyncio.run(run_play(config))
    elif args.run_type == "eval":
        asyncio.run(run_eval(config))
    else:
        print(f"Incorrect run type: {args.run_type}. Choose 'play' or 'eval'.")


if __name__ == "__main__":
    main()
