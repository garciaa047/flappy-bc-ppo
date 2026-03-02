import pygame

from stable_baselines3 import PPO

from flappy.env import FlappyBirdEnv
from flappy.game import CONFIG

import argparse


def evaluate(model, episodes=10, fps=None, headless=False, reward_limit=float("inf")):
    if headless:
        env = FlappyBirdEnv()
    else:
        env = FlappyBirdEnv("human")

    model = PPO.load(model)

    clock = pygame.time.Clock() if fps is not None else None

    try:
        for ep in range(1, episodes+ 1):
            obs, info = env.reset()

            ep_reward = 0
            done = False
            truncated = False
            while not done and not truncated and ep_reward < reward_limit:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                env.render()
                if clock is not None:
                    clock.tick(fps)

            print(f"Episode {ep} reward: {ep_reward}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Flappy Bird Policy")
    parser.add_argument("--model", type=str, default="models/flappyPPO_best_model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--fps", type=int, default=None,
        help=f"Render speed in frames per second. Use {CONFIG['FPS']} to match the playable game speed. Default is uncapped."
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--reward_limit", type=int, default=float("inf"))
    args = parser.parse_args()

    evaluate(model=args.model, episodes=args.episodes, fps=args.fps, headless=args.headless, reward_limit=args.reward_limit)
