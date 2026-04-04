"""
Atari Game Player using Gymnasium ALE.

Plays Atari games with different strategies:
  - random: uniformly random actions
  - greedy: simple heuristic-based play (Breakout-specific: tracks ball, moves paddle)
  - human:  keyboard-controlled via render window

Usage:
    python atari_player.py                          # random agent on Breakout
    python atari_player.py --game Pong              # random agent on Pong
    python atari_player.py --strategy greedy         # heuristic agent
    python atari_player.py --strategy human          # keyboard play
    python atari_player.py --episodes 5 --record     # record 5 episodes to video/
"""

import argparse
import os
import sys
import time

import numpy as np

import gymnasium as gym
import ale_py  # noqa: F401 — registers ALE environments


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class RandomAgent:
    """Picks a uniformly random action every step."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


class GreedyBreakoutAgent:
    """
    Simple heuristic for Breakout: move the paddle toward the ball's x position.
    Falls back to random actions for non-Breakout games.
    """

    PADDLE_Y_RANGE = (189, 194)   # approximate paddle row in 210×160 frame

    def __init__(self, action_space):
        self.action_space = action_space
        # Breakout actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        self.noop, self.fire, self.right, self.left = 0, 1, 2, 3
        self._fire_countdown = 5  # fire for first N frames to launch ball
        self._steps = 0

    def _find_ball_x(self, obs):
        """Return the ball's approximate x-coordinate, or None."""
        # Look between bricks (end ~93) and paddle (~189) for the small ball
        # The ball is typically a 2x4 or 4x2 bright pixel cluster
        field = obs[100:185, 8:152, :]  # playing field only
        # The ball in Breakout is bright (R=200, white-ish)
        bright = np.max(field, axis=2) > 180
        coords = np.argwhere(bright)
        if len(coords) == 0:
            return None
        # Ball is small — if we see too many bright pixels it's not just the ball
        if len(coords) > 50:
            return None
        return float(np.mean(coords[:, 1])) + 8  # offset for the crop

    def _find_paddle_x(self, obs):
        """Return the paddle's approximate x-center."""
        paddle_strip = obs[self.PADDLE_Y_RANGE[0]:self.PADDLE_Y_RANGE[1], :, :]
        bright = np.max(paddle_strip, axis=2) > 180
        coords = np.argwhere(bright)
        if len(coords) == 0:
            return 80.0  # default center
        return float(np.mean(coords[:, 1]))

    def act(self, obs):
        self._steps += 1
        # Fire at the start to launch the ball
        if self._fire_countdown > 0:
            self._fire_countdown -= 1
            return self.fire

        ball_x = self._find_ball_x(obs)
        if ball_x is None:
            # Ball not visible — probably need to fire a new ball
            self._fire_countdown = 5
            return self.fire

        paddle_x = self._find_paddle_x(obs)
        if ball_x < paddle_x - 2:
            return self.left
        elif ball_x > paddle_x + 2:
            return self.right
        else:
            return self.noop


# ---------------------------------------------------------------------------
# Game name resolution
# ---------------------------------------------------------------------------

POPULAR_GAMES = [
    "Breakout", "Pong", "SpaceInvaders", "MsPacman", "Qbert",
    "Asteroids", "Centipede", "Enduro", "Seaquest", "BeamRider",
]


def resolve_env_id(game_name: str) -> str:
    """Turn a short name like 'Breakout' into 'ALE/Breakout-v5'."""
    if game_name.startswith("ALE/"):
        return game_name
    return f"ALE/{game_name}-v5"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def play(game: str, strategy: str, episodes: int, record: bool, max_steps: int):
    env_id = resolve_env_id(game)
    if strategy == "human":
        render_mode = "human"
    elif record:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = gym.make(env_id, render_mode=render_mode)
    if record:
        video_dir = os.path.join(os.path.dirname(__file__), "videos")
        try:
            env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda _: True)
            print(f"Recording episodes to {video_dir}/")
        except gym.error.DependencyNotInstalled:
            print("Warning: moviepy not installed, skipping recording. Install with: pip install moviepy")
            record = False

    # Pick agent
    if strategy == "random":
        agent = RandomAgent(env.action_space)
    elif strategy == "greedy":
        agent = GreedyBreakoutAgent(env.action_space)
    elif strategy == "human":
        agent = RandomAgent(env.action_space)  # human mode uses render_mode="human"
        print("Human mode: watch the game in the render window.")
        print("(Actions are random — for true keyboard control, use gymnasium's play utility.)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total_reward_all = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated or steps >= max_steps:
                break

        total_reward_all.append(total_reward)
        lives = info.get("lives", "?")
        print(f"Episode {ep:3d} | Reward: {total_reward:7.1f} | Steps: {steps:5d} | Lives left: {lives}")

    env.close()

    # Summary
    rewards = np.array(total_reward_all)
    print(f"\n{'='*50}")
    print(f"Game:     {env_id}")
    print(f"Strategy: {strategy}")
    print(f"Episodes: {episodes}")
    print(f"Mean reward:   {rewards.mean():.2f}")
    print(f"Std reward:    {rewards.std():.2f}")
    print(f"Min reward:    {rewards.min():.2f}")
    print(f"Max reward:    {rewards.max():.2f}")
    print(f"{'='*50}")

    return rewards


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Play Atari games with Gymnasium ALE")
    parser.add_argument("--game", default="Breakout", help=f"Game name (default: Breakout). Popular: {', '.join(POPULAR_GAMES)}")
    parser.add_argument("--strategy", choices=["random", "greedy", "human"], default="random", help="Agent strategy (default: random)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play (default: 3)")
    parser.add_argument("--record", action="store_true", help="Record episodes to video/ directory")
    parser.add_argument("--max-steps", type=int, default=10_000, help="Max steps per episode (default: 10000)")
    parser.add_argument("--list-games", action="store_true", help="List all available ALE games and exit")
    args = parser.parse_args()

    if args.list_games:
        all_envs = [e for e in gym.envs.registry if e.startswith("ALE/")]
        print(f"Available ALE games ({len(all_envs)}):")
        for e in sorted(all_envs):
            print(f"  {e}")
        return

    play(args.game, args.strategy, args.episodes, args.record, args.max_steps)


if __name__ == "__main__":
    main()
