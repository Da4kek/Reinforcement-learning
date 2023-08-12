# Reinforcement-learning:

> Reinfocement learning is a field of Artificial Intelligence in which you build an intelligent system that learns from its environment through interaction and evaluates what it learns in real-time.

## Fundamental concepts:
1. Agent - the program controlling the object of concern.
2. Environment - this defines the outside world programmatically. Everything the agent interacts with is part of the environment.
3. Rewards - this gives us a score of how the algorithm performs with respect to the environment. In other words, rewards represent gains and losses.
4. Policy - the algorithm used by the agent to decide its actions.

## Markov decision processes:

MDP is a process with a fixed number of states, and it randomly evolves from one state to another at each step. A lot of RL problems with discrete actions are modeled as MDPs, with the agent having no initial clue on the next transition state.     
The agent also has no idea on the rewarding principle, so it has to explore all possible states to begin to decode how to adjust to a perfect rewarding system. This is called the *Q learning*.

The Q-learning algorithm is adapted from the Q-value iteration algorithm, in a situation where the agent has no idea of the transition probabilities and rewards. The agent has to learn the Q-values through exploration and exploitation of the environment. The Q-values are updated at each step of the process, and the agent learns the optimal policy from the Q-values.

## Difference between model-based and model-free RL:

Model-based, as it sounds, has an agent trying to understand its environment and creating a model for it based on its interactions with this environment. In such a system, preferences take priority over the consequences of the actions. The agent is more concerned about the rewards it gets from the environment than the environment itself.

On the other hand, model-free algorithms seek to learn the consequences of their actions through experience via algorithms such as Policy Gradient, Q-learning etc. In other words, such algorithms will carry out an action multiple times and will adjust the policy based on the outcomes.

## Reinforcement learning algorithms:

Among RL's model-free methods is temporal difference (TD) learning, with SARSA and Q-learning being two of the most used algorithms.

### Temporal Difference Learning:
One of the problems with the environment is that rewards usually are not immediately observable. TD learning is an unsupervised technique to predict a variable's expected value in a sequence of states. TD uses a mathematical trick to replace complex reasoning about the future with a simple learning procedure that can produce the same results. Instead of calculating the total future reward, TD tries to predict the combination of immediate reward and its own reward prediction at the next moment in time.  
TD error is the difference between the ultimate correct reward and our current prediction. As similar to other optimization methods, the current value will be updated by its value + learning_rate * error.