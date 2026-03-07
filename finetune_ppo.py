"""
finetune_ppo.py - Fine-tune a BC-trained Flappy Bird policy with PPO.

Loads a BC-trained model (default: flappyPPO_bc) and runs PPO on top.

Uses almost the same hyperparameters as finetune_ppo.py so that the main 
difference between the two runs is the policy initialisation (random vs BC). 

Shared hyperparameters: (both scripts):
    learning_rate : 1e-4
    ent_coef      : 0.01

Different hyperparameters:
    n_epochs      : 3     (vs 10)   - fewer gradient steps per rollout batch;
                          n_epochs=10 causes rapid entropy decay when the
                          policy starts from a good BC initialisation.
    clip_range    : 0.1   (vs 0.2)  - tighter trust region for the same reason.

BC+PPO specific addition:
    CriticWarmupCallback -> freezes the actor for the first WARMUP_STEPS
    timesteps so the critic can wamrup to BC weights before it starts any
    actor updates. Direct consequence of the cold start problem.
    Increased the time to collapse from ~100k timesteps to ~600k timesteps.

Saves:
    flappyPPO_finetuned_latest      - saved on completion or Ctrl+C
    flappyPPO_finetuned_best_model  - highest mean eval reward seen during training

Example Usage:
    python finetune_ppo.py
    python finetune_ppo.py --bc_model flappyPPO_bc --total_timesteps 500_000
    python finetune_ppo.py --warmup_steps 100_000
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from flappy.env import FlappyBirdEnv

SAVE_NAME       = "models/flappyPPO_finetuned"
TENSORBOARD_LOG = "./tensorboard_logs/finetuned_PPO_log/"

LEARNING_RATE = 1e-4
ENT_COEF = 0.01

# Conservative update settings to make sure PPO doesn't lose BC headstart.
N_EPOCHS   = 3
CLIP_RANGE = 0.1


# --- CALLBACKS ---

class CriticWarmupCallback(BaseCallback):
    """
    Freeze the actor for the first `warmup_steps` timesteps.

    During warm up the critic trains freely on BC policy and builds
    accurate estimates before it is allowed to cause updates.
    Once `warmup_steps` is reached the actor is unfrozen and normal PPO
    begins.

    Actor parameters frozen:
        policy.mlp_extractor.policy_net   - hidden layers for the actor
        policy.action_net                 - final action-logit layer
    """

    def __init__(self, warmup_steps=75_000, verbose=1):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self._unfrozen    = False

    def _freeze_actor(self):
        for param in self.model.policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = False
        for param in self.model.policy.action_net.parameters():
            param.requires_grad = False

    def _unfreeze_actor(self):
        for param in self.model.policy.mlp_extractor.policy_net.parameters():
            param.requires_grad = True
        for param in self.model.policy.action_net.parameters():
            param.requires_grad = True

    def _on_training_start(self):
        self._freeze_actor()
        if self.verbose:
            print(f"  [Warmup] Actor frozen for first {self.warmup_steps:,} steps "
                  f"- critic bootstrapping.")

    def _on_step(self) -> bool:
        if not self._unfrozen and self.num_timesteps >= self.warmup_steps:
            self._unfreeze_actor()
            self._unfrozen = True
            if self.verbose:
                print(f"\n  [Warmup] Actor unfrozen at {self.num_timesteps:,} steps "
                      f"- joint training begins.")
        return True


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


# --- Fine Tuning ---

def finetune(bc_model, total_timesteps, warmup_steps):
    os.makedirs("models", exist_ok=True)

    env      = FlappyBirdEnv()
    eval_env = FlappyBirdEnv(render_mode=None)

    # Load the BC model and override hyperparameters to match train_ppo.py.
    # custom_objects replaces the values saved with the model without touching
    # the policy weights, the BC initialisation is preserved.
    model = PPO.load(
        bc_model,
        env=env,
        custom_objects={
            "learning_rate": LEARNING_RATE,
            "ent_coef":      ENT_COEF,
            "n_epochs":      N_EPOCHS,
            "clip_range":    CLIP_RANGE,
        },
    )

    # Set after load as PPO.load kwargs vary across SB3 versions.
    model.verbose         = 1
    model.tensorboard_log = TENSORBOARD_LOG

    callback = CallbackList([
        CriticWarmupCallback(warmup_steps=warmup_steps),
        BestModelCallback(eval_env=eval_env),
    ])

    print(f"Fine-tuning from '{bc_model}'")
    print(f"  learning_rate={LEARNING_RATE}  ent_coef={ENT_COEF}  steps={total_timesteps:,}")
    print(f"  n_epochs={N_EPOCHS}  clip_range={CLIP_RANGE}  warmup_steps={warmup_steps:,}")
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
        print("\nFine tuning interrupted.")
    finally:
        model.save(f"{SAVE_NAME}_latest")
        print(f"Saved → {SAVE_NAME}_latest")
        eval_env.close()
        env.close()


# --- Main Guard ---

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
        "--warmup_steps", type=int, default=75_000,
        help="Steps to freeze the actor while the critic bootstraps (default: 75_000)",
    )
    args = parser.parse_args()

    finetune(
        bc_model=args.bc_model,
        total_timesteps=args.total_timesteps,
        warmup_steps=args.warmup_steps,
    )
