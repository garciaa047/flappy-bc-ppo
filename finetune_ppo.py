"""
finetune_ppo.py — Fine-tune a BC-trained Flappy Bird policy with PPO.

Loads a BC-trained model (default: flappyPPO_bc) and runs PPO on top with a
reduced learning rate to preserve the BC initialisation rather than overwriting
it with large early gradient steps.

Saves:
    flappyPPO_finetuned_latest      — saved on completion or Ctrl+C
    flappyPPO_finetuned_best_model  — highest mean eval reward seen during training

Usage:
    python finetune_ppo.py
    python finetune_ppo.py --bc_model flappyPPO_bc --total_timesteps 500_000
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from flappy.env import FlappyBirdEnv

SAVE_NAME       = "models/flappyPPO_finetuned"
TENSORBOARD_LOG = "./tensorboard_logs/finetuned_PPO_log/"


# ── Callback ──────────────────────────────────────────────────────────────────
# Note: cannot import this from Base_PPO.py because that file has no
# __main__ guard and would execute the full training loop on import.

class BestModelCallback(BaseCallback):
    """Evaluates periodically and saves the model whenever a new best mean reward is reached."""

    def __init__(self, eval_env, eval_freq=10_000, n_eval_episodes=5, verbose=1):
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


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(bc_model, total_timesteps, learning_rate, ent_coef):
    os.makedirs("models", exist_ok=True)

    env      = FlappyBirdEnv()
    eval_env = FlappyBirdEnv(render_mode=None)

    # Load the BC model and override hyperparameters for fine-tuning.
    # custom_objects replaces the values that were saved with the model
    # without touching the policy weights — the BC initialisation is preserved.
    model = PPO.load(
        bc_model,
        env=env,
        custom_objects={
            "learning_rate": learning_rate,
            "ent_coef":      ent_coef,
        },
    )

    # Set after load — PPO.load kwargs vary across SB3 versions.
    model.verbose         = 1
    model.tensorboard_log = TENSORBOARD_LOG

    callback = BestModelCallback(eval_env=eval_env)

    print(f"Fine-tuning from '{bc_model}'")
    print(f"  learning_rate={learning_rate}  ent_coef={ent_coef}  steps={total_timesteps:,}")
    print(f"To view TensorBoard logs run: tensorboard --logdir {TENSORBOARD_LOG}")
    print("Press Ctrl+C at any time to stop and save the latest model.\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted.")
    finally:
        model.save(f"{SAVE_NAME}_latest")
        print(f"Saved → {SAVE_NAME}_latest")
        eval_env.close()
        env.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a BC Flappy Bird policy with PPO"
    )
    parser.add_argument(
        "--bc_model", type=str, default="models/flappyPPO_bc",
        help="BC model to load as starting point (default: flappyPPO_bc)",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=1_000_000,
        help="Total PPO fine-tuning steps (default: 1_000_000)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate — lower than Base_PPO to preserve BC weights (default: 1e-4)",
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01,
        help="Entropy coefficient — small bonus encourages exploration around the BC policy (default: 0.01)",
    )
    args = parser.parse_args()

    finetune(
        bc_model=args.bc_model,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
    )
