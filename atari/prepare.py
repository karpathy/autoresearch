"""
Fixed evaluation harness for Atari autoresearch experiments.
DO NOT MODIFY — this is the ground truth metric.

Provides:
  - evaluate_agent(): runs an agent for N episodes, returns mean reward
  - make_env(): creates a standard ALE environment with fixed settings
  - Constants: GAME, NUM_EVAL_EPISODES, MAX_STEPS_PER_EPISODE, TIME_BUDGET

The agent module (agent.py) is the only file the AI researcher modifies.

Usage:
    python prepare.py          # verify environment works
    python prepare.py --list   # list all available games
"""

import argparse
import time

import numpy as np

import gymnasium as gym
import ale_py  # noqa: F401 — registers ALE environments

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

GAME = "ALE/Breakout-v5"          # the game to optimize
NUM_EVAL_EPISODES = 30            # episodes per evaluation (enough for stable mean)
MAX_STEPS_PER_EPISODE = 10_000    # cap per episode to prevent infinite loops
TIME_BUDGET = 300                 # 5-minute budget for agent.py's train() function
SEED = 42                         # fixed seed for reproducible evaluation

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(render_mode=None):
    """Create a standard ALE environment with fixed settings."""
    env = gym.make(
        GAME,
        render_mode=render_mode,
        frameskip=1,           # no extra frameskip (ALE default already applies 4)
        repeat_action_probability=0.25,  # standard stochasticity
    )
    return env

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_agent(agent, num_episodes=NUM_EVAL_EPISODES, seed=SEED):
    """
    Run the agent for num_episodes episodes and return evaluation stats.

    The agent must implement:
        agent.act(observation) -> action (int)
        agent.reset() -> None  (called at start of each episode, optional)

    Returns dict with:
        mean_reward, std_reward, min_reward, max_reward,
        mean_steps, total_steps, episodes
    """
    env = make_env(render_mode=None)
    rewards = []
    steps_list = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(agent, 'reset'):
            agent.reset()

        total_reward = 0.0
        steps = 0

        while True:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated or steps >= MAX_STEPS_PER_EPISODE:
                break

        rewards.append(total_reward)
        steps_list.append(steps)

    env.close()

    rewards = np.array(rewards)
    steps_arr = np.array(steps_list)

    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "mean_steps": float(steps_arr.mean()),
        "total_steps": int(steps_arr.sum()),
        "episodes": num_episodes,
    }

# ---------------------------------------------------------------------------
# Main — verify setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atari autoresearch: verify environment")
    parser.add_argument("--list", action="store_true", help="List all available ALE games")
    args = parser.parse_args()

    if args.list:
        all_envs = sorted(e for e in gym.envs.registry if e.startswith("ALE/"))
        print(f"Available ALE games ({len(all_envs)}):")
        for e in all_envs:
            print(f"  {e}")
        exit(0)

    # Verify environment works
    print(f"Game: {GAME}")
    print(f"Eval episodes: {NUM_EVAL_EPISODES}")
    print(f"Max steps/episode: {MAX_STEPS_PER_EPISODE}")
    print(f"Time budget: {TIME_BUDGET}s")
    print()

    env = make_env()
    obs, info = env.reset(seed=SEED)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Action meanings: {env.unwrapped.get_action_meanings()}")

    # Quick random sanity check (3 episodes)
    total = 0.0
    for ep in range(3):
        obs, _ = env.reset(seed=SEED + ep)
        ep_reward = 0.0
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1
            if terminated or truncated or steps >= 1000:
                break
        total += ep_reward
        print(f"  Sanity check ep {ep+1}: reward={ep_reward:.0f}, steps={steps}")

    env.close()
    print(f"\nRandom baseline ~{total/3:.1f} reward/episode")
    print("Setup OK! Ready to experiment.")
