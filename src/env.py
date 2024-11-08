from enum import Enum
from typing import NamedTuple, Optional
import random
from src.exceptions import (
    InvalidQuestionError,
    InvalidGuessError,
    InvalidAnswerError,
)


class TURN_TYPE(Enum):
    ASK_QUESTION = "ask_question"
    ANSWER_QUESTION = "answer_question"
    MAKE_GUESS = "make_guess"
    WAIT = "wait"


class AGENT_ROLE(Enum):
    HOST = 0
    GUESSER = 1


class Observation(NamedTuple):
    turn: int
    history: list[dict]
    turn_type: TURN_TYPE
    active: bool
    role: AGENT_ROLE
    remaining_turns: int
    current_question: Optional[str] = None
    current_answer: Optional[str] = None
    topic: Optional[str] = None
    knowledge_base: Optional[list[str]] = None


class StepResult(NamedTuple):
    observations: list[Observation]
    rewards: list[float]
    dones: list[bool]
    info: dict


class Game20QEnv:
    def __init__(
        self,
        host_agent: any,
        guesser_agent: any,
        knowledge_base: list[str],
        debug: bool = False,
        max_turns: int = 20,
    ):
        self.host = host_agent
        self.guesser = guesser_agent
        self.debug = debug
        self.max_turns = max_turns

        # State variables
        self.turn = 1
        self.topic = None
        self.knowledge_base = knowledge_base
        self.history = []
        self.current_question = None
        self.current_answer = None
        self.current_type = TURN_TYPE.ASK_QUESTION

    def reset(self) -> list[Observation]:
        """Reset environment"""
        self.turn = 1
        self.history = []
        self.topic = random.choice(self.knowledge_base)

        self.current_type = TURN_TYPE.ASK_QUESTION
        self.current_question = None
        self.current_answer = None

        if self.debug:
            print(f"[DEBUG] Topic: {self.topic}")

        return self._get_observations()

    async def step(self) -> StepResult:
        """Execute one step of the environment"""
        if self.turn > self.max_turns:
            return self._end_game("max_turns")

        if self.current_type == TURN_TYPE.ASK_QUESTION:
            return await self._handle_ask_question()
        elif self.current_type == TURN_TYPE.ANSWER_QUESTION:
            return await self._handle_answer_question()
        elif self.current_type == TURN_TYPE.MAKE_GUESS:
            return await self._handle_make_guess()

    async def _handle_ask_question(self) -> StepResult:
        """Handle guesser asking question"""
        question = await self.guesser.ask_question(
            self._get_observations()[AGENT_ROLE.GUESSER.value]
        )
        if not isinstance(question, str):
            raise InvalidQuestionError("Guesser must ask a valid question.")

        self.current_question = question
        self.current_type = TURN_TYPE.ANSWER_QUESTION

        return StepResult(
            self._get_observations(),
            [0.0, 0.0],
            [False, False],
            {"action": "question_asked", "question": question},
        )

    async def _handle_answer_question(self) -> StepResult:
        """Handle host answering question"""
        answer = await self.host.respond(
            self._get_observations()[AGENT_ROLE.HOST.value]
        )
        answer = answer.lower().strip()

        if answer not in ["yes", "no"]:
            raise InvalidAnswerError("Host must respond with 'yes' or 'no'.")

        self.current_answer = answer
        self.current_type = TURN_TYPE.MAKE_GUESS

        return StepResult(
            self._get_observations(),
            [0.0, 0.0],
            [False, False],
            {"action": "question_answered", "answer": answer},
        )

    async def _handle_make_guess(self) -> StepResult:
        """Handle guesser making guess"""
        guess = await self.guesser.make_guess(
            self._get_observations()[AGENT_ROLE.GUESSER.value]
        )
        if not isinstance(guess, str):
            raise InvalidGuessError("Guesser must make a valid guess.")

        # Record completed turn
        turn_info = {
            "turn": self.turn,
            "question": self.current_question,
            "answer": self.current_answer,
            "guess": guess,
        }
        self.history.append(turn_info)

        is_correct = self._check_guess(guess)
        if is_correct:
            return self._end_game("correct_guess")

        # Prepare for next turn
        self.turn += 1
        self.current_type = TURN_TYPE.ASK_QUESTION
        self.current_question = None
        self.current_answer = None

        return StepResult(
            self._get_observations(),
            [0.0, 0.0],
            [False, False],
            {"action": "guess_made", "turn_info": turn_info},
        )

    def _get_observations(self) -> list[Observation]:
        """Get current observations for both agents"""
        return [
            Observation(
                turn=self.turn,
                history=self.history,
                turn_type=self.current_type,
                active=self.current_type == TURN_TYPE.ANSWER_QUESTION,
                role=AGENT_ROLE.HOST,
                remaining_turns=self.max_turns - self.turn,
                current_question=self.current_question,
                current_answer=self.current_answer,
                topic=self.topic,
            ),
            Observation(
                turn=self.turn,
                history=self.history,
                turn_type=self.current_type,
                active=self.current_type
                in [TURN_TYPE.ASK_QUESTION, TURN_TYPE.MAKE_GUESS],
                role=AGENT_ROLE.GUESSER,
                remaining_turns=self.max_turns - self.turn,
                current_question=self.current_question,
                current_answer=self.current_answer,
                knowledge_base=self.knowledge_base,
            ),
        ]

    def _check_guess(self, guess: str) -> bool:
        """Check if guess is correct"""

        guess = guess.lower().strip()
        topic = self.topic.lower().strip()

        return topic in guess

    def _end_game(self, reason: str) -> StepResult:
        """End game due to max turns or correct guess"""
        if reason == "max_turns":
            rewards = [1.0, 0.0]
        elif reason == "correct_guess":
            rewards = [0.0, 1.0]
        else:
            rewards = [-1.0, -1.0]

        dones = [True, True]

        if self.debug:
            print(f"[DEBUG] Game ended: {reason}")
            print(f"[DEBUG] Rewards: {rewards}")

        return StepResult(
            self._get_observations(),
            rewards,
            dones,
            {"action": "end_game", "reason": reason},
        )
