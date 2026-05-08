# Lab 09 — Reinforcement Learning Foundations: Teach an Agent to Balance a Pole

**Course:** ITAI 2376 — Module 09 Hands-On Lab
**Author:** Jiri Musil
**Environment:** [Gymnasium CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## Problem Statement

Reinforcement learning is the loop that powers everything from game-playing AI to the RLHF training that shaped the LLMs at the heart of modern agent systems. This lab gives hands-on experience with the loop in a classic, simple environment: keep a pole balanced upright on a sliding cart by pushing left or right. The lab deliberately stops short of implementing Deep Q-Networks from scratch (which would take 10–15 hours of debugging that is better spent on the final project) and instead uses tabular Q-Learning to make the core ideas concrete.

## Approach

### Part 1 — Random agent baseline
Run 10 episodes of CartPole with a fully random policy (`env.action_space.sample()` at every step). The episode terminates when the pole falls past 15° or the cart hits the edge. Maximum possible score is 500.

### Part 2 — Q-Learning agent
Implement tabular Q-Learning end-to-end:

- **State discretization.** CartPole has a continuous 4D state space (cart position, cart velocity, pole angle, pole angular velocity). Discretize each dimension into 10 bins, yielding a Q-table of shape `(10, 10, 10, 10, 2)`.
- **Epsilon-greedy action selection.** Start with epsilon=1.0 (fully random) and decay by 0.995 per episode down to a minimum of 0.01. This balances exploration (try new things) with exploitation (use what you have learned).
- **Q-update rule.** Standard Q-learning update: move the current Q-value toward `reward + discount * max(Q[next_state])`. Discount factor 0.99, learning rate 0.1.
- **Train for 500 episodes** and observe scores rising from random-agent levels to genuine competence.

### Part 3 — Reflection: connecting RL to my final project
Connect the CartPole loop (state → action → reward → next state) to the same loop used in RLHF training of LLMs. The final project agent uses tools and an explicit reasoning loop instead of a learned Q-table, but the conceptual structure is the same.

## Files

- `L09_Jiri_Musil_ITAI_2376.ipynb` — full lab notebook with random agent baseline, Q-Learning implementation, training run, and reflection answers

## Results

| Agent | Average Score (10 episodes) | Best | Worst |
|-------|----------------------------|------|-------|
| Random agent | **18.5** | 30 | 12 |
| Q-Learning agent (after 500 episodes) | Substantially higher than random — see the training curve in the notebook |

Random agent rarely makes it past 30 timesteps before the pole falls. Q-Learning visibly improves over training as exploration decreases and the Q-table converges toward useful estimates.

## Learning Outcomes

- **The RL loop is everywhere.** Once you have actually written `state → action → reward → next_state → update`, every other RL algorithm (DQN, PPO, RLHF) becomes a variation of the same template. The agent does not understand physics — it just learns a lookup table over discretized states.
- **Exploration vs exploitation matters more than algorithm choice.** With epsilon=0 from the start, the agent gets locked into whatever it tried first. With epsilon=1 forever, it never converges. The decay schedule is the actual interesting design choice.
- **State discretization is fragile.** CartPole's continuous state has to be binned to fit in a Q-table, and the bin edges directly affect what the agent can learn. This is why DQN exists — it replaces the table with a neural network that handles continuous states natively.
- **The RLHF connection is real.** GPT-4o, Claude, and Llama were trained with the same loop running over human-labeled preference data instead of CartPole rewards. Understanding the toy version makes the production version far less mysterious.

## Requirements

```bash
pip install gymnasium matplotlib numpy
```

No GPU required. The lab runs comfortably on free Colab CPU in under 5 minutes.

## How to Run

```bash
jupyter lab L09_Jiri_Musil_ITAI_2376.ipynb
```

Run cells top to bottom. The notebook installs Gymnasium silently, runs the random-agent baseline, then trains and evaluates the Q-Learning agent over 500 episodes.

## Notes

The lab intentionally does not implement DQN, Policy Gradient, Actor-Critic, or PPO — those are described in the module booklet but the implementation work is reserved for the term projects, where it is more useful. If you want to push further, swap the Q-table for a small MLP and you have a DQN. The rest of the loop stays the same.
