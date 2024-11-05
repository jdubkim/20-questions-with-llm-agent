# 20 Questions with LLM Agents

An implementation of the classic 20 Questions game using Large Language Model agents.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/20-questions-with-llm-agent.git
cd 20-questions-with-llm-agent
```

2. Install dependencies (python 3.10):

```
pip install -r requirements.txt
```

3. Set up OpenAI API key:

```
export OPENAI_API_KEY="your-api-key"
```

## Project Structure

```
.
├── src/
│   ├── agents/
│   │   ├── agent.py     # Agent implementations
│   │   └── model.py     # LLM wrapper
│   ├── env.py           # Game environment
│   ├── utils.py         # Utilities
│   └── main.py          # Entry point
├── tests/               # Unit tests
└── README.md
```

## Usage

Run the agent playing with default settings:

```
python -m src.main --run-type play
```

Run evaluation with multiple games:
```
python -m src.main --run-type eval --n-games 5
```


## Game Settings
Knowledge Base: Limited to 5 predefined topics for consistent evaluation
Topic Selection: Random selection from knowledge base (list of topic candidates)
Maximum Turns: 5 turns per game
Guesser Information: Complete knowledge base  visible to guesser


## Example Game

```Game
Game over! The topic was: plane
Final Rewards - Host: 1.0, Guesser: 0.0
Game History:
Turn 0: Q: Is the topic a living thing? -> A: no -> Guess: A rock.
Turn 1: Q: Is the topic a man-made object? -> A: yes -> Guess: A smartphone.
Turn 2: Q: Is the topic primarily used for entertainment? -> A: no -> Guess: A kitchen appliance.
Turn 3: Q: Is the topic used for construction or repair purposes? -> A: no -> Guess: A piece of furniture.
Turn 4: Q: Is the topic an electronic device? -> A: no -> Guess: A book
```

## Evaluation
Run evaluation mode to collect metrics:

* Success Rate
* Average Turns
* Failure Analysis

Results are saved in logs/ directory with:

* Individual game logs
* Configuration details


## TODO
* Implement more sophisticated agents - ReAct (browse Wikipedia for factual checks)
* Add knowledge library for guesser agents. Prepare a list of candidate topics and binary questions and create a table so that the agent can reduce the search space drastically.
* Implement a multi-threading environment for parallel game simulations (multithreading because agents are called via APIs)

