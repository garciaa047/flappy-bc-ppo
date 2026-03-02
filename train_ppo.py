import os

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from flappy.env import FlappyBirdEnv

total_timesteps     = 1_000_000
tensorboard_log_dir = "./tensorboard_logs/PPO_log/"
SAVE_NAME           = "models/flappyPPO"


class BestModelCallback(BaseCallback):
    """Evaluates periodically and saves the model whenever a new best mean reward is reached."""

    def __init__(self, eval_env, eval_freq=10_000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
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


os.makedirs("models", exist_ok=True)

env      = FlappyBirdEnv()
eval_env = FlappyBirdEnv(render_mode=None)

# Parallel environments
# vec_env = make_vec_env(FlappyBirdEnv, n_envs=4, env_kwargs={"render_mode": None})

model    = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)
callback = BestModelCallback(eval_env=eval_env)

print(f"To view Tensorboard Logs run: tensorboard --logdir {tensorboard_log_dir}")
print("Press Ctrl+C at any time to stop training and save the latest model.\n")

try:
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
except KeyboardInterrupt:
    print("\nTraining interrupted.")
finally:
    model.save(f"{SAVE_NAME}_latest")
    print(f"Saved → {SAVE_NAME}_latest")
    eval_env.close()

del model

model = PPO.load(f"{SAVE_NAME}_latest")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    env.render()
