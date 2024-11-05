# 20 Questions with LLM Agents

An implementation of the classic 20 Questions game using Large Language Model agents.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/20-questions-with-llm-agent.git
cd 20-questions-with-llm-agent
```

2. Install dependencies:

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
python -m src.main
```

With debug mode:

```
python -m src.main --debug
```

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



## Additional Settings to Simplify the Tasks

* Limited topics - the topic is randomly selected from 5 words, instead of letting host agent choose it (gpt-4o-mini found this task difficult, not sharing the chosen topic).
* Max Turns: By default, decreased the max turns to 5 - this will save # tokens..
