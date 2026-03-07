"""
train_ppo.py - Train a Flappy Bird PPO agent from scratch.

Hyperparameters:
    learning_rate : 1e-4
    ent_coef      : 0.01
    n_epochs      : 10
    clip_range    : 0.2

Saves:
    models/flappyPPO_latest      - saved on completion or Ctrl+C
    models/flappyPPO_best_model  - highest mean eval reward seen during training

Usage:
    python train_ppo.py
    python train_ppo.py --total_timesteps 500_000
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from flappy.env import FlappyBirdEnv

SAVE_NAME           = "models/flappyPPO"
TENSORBOARD_LOG_DIR = "./tensorboard_logs/PPO_log/"

# --- Shared hyperparameters ---
# Keep these identical to finetune_ppo.py so the only variable is the BC init.
LEARNING_RATE = 1e-4
ENT_COEF      = 0.01
N_EPOCHS      = 10
CLIP_RANGE    = 0.2


# --- Callback ---

class BestModelCallback(BaseCallback):
    """Evaluates periodically and saves the model whenever a new best mean reward is reached."""

    def __init__(self, eval_env, eval_freq=50_000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env         = eval_env
        self.eval_freq        = eval_freq
        self.n_eval_episodes  = n_eval_episodes
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            if self.verbose:
                print(f"  [Eval] mean_reward={mean_reward:.2f}  best={self.best_mean_reward:.2f}")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"{SAVE_NAME}_best_model")
                if self.verbose:
                    print(f"  [Eval] New best! Saved → {SAVE_NAME}_best_model")
        return True


# --- Training ---

def train(total_timesteps):
    os.makedirs("models", exist_ok=True)

    env      = FlappyBirdEnv()
    eval_env = FlappyBirdEnv(render_mode=None)

    model = PPO(
        "MlpPolicy", env,
        learning_rate = LEARNING_RATE,
        ent_coef      = ENT_COEF,
        n_epochs      = N_EPOCHS,
        clip_range    = CLIP_RANGE,
        verbose       = 1,
        tensorboard_log = TENSORBOARD_LOG_DIR,
    )

    callback = BestModelCallback(eval_env=eval_env)

    print(f"Training PPO from scratch")
    print(f"  learning_rate={LEARNING_RATE}  ent_coef={ENT_COEF}  "
          f"n_epochs={N_EPOCHS}  clip_range={CLIP_RANGE}  steps={total_timesteps:,}")
    print(f"To view TensorBoard logs run: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    print("Press Ctrl+C at any time to stop and save the latest model.\n")

    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        model.save(f"{SAVE_NAME}_latest")
        print(f"Saved → {SAVE_NAME}_latest")
        eval_env.close()
        env.close()


# --- Entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Flappy Bird PPO agent from scratch")
    parser.add_argument(
        "--total_timesteps", type=int, default=1_000_000,
        help="Total training steps (default: 1_000_000)",
    )
    args = parser.parse_args()

    train(total_timesteps=args.total_timesteps)
