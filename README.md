# 20-questions-with-llm-agent

## Objective

* Build agents to play the game of 20 Questions.
  * Agent must be a combination of two sub-agents: host, guesser.
    * Host: Must choose hard "topic" of the game of the other agent in competitive setting. For co-operative setting, choose easy "topic". Also, correctly answer "Yes" or "No" to the questions asked by the guesser.
    * Guesser: Choose a question each turn that can decrease the search space significantly. Also, guess the topic that is most likely to be the answer.
  * Example step would be:
    1. [Turn 0] Environment: Host, choose the topic.
    2. [Turn 0] Host: Chicken (should be an object or living thing).
    3. [Turn 1] Guesser: Question - Is it alive?
    4. [Turn 1] Host: Yes.
    5. [Turn 1] Guesser: Guess - human.
    6. [Turn 1] Environment: Wrong.
    7. [Turn 2] Guesser: Question - Is it an animal?
    8. [Turn 2] Host: Yes.
    9. [Turn 2] Guesser: Guess - bear.
    10. [Turn 2] Environment: Wrong.
    11. [Turn 3] Guesser: Question - Is it a bird?
    12. [Turn 3] Host: Yes.
    13. [Turn 3] Guesser: Quess - Chicken.
    14. [Turn 3] Environment: Correct. End the game.
  * Context provided to the agent
    * Naive method: we can provide the history of messages.
    *
