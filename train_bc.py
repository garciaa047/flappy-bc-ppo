"""
train_bc.py - Train a Behavioural Cloning policy on collected human demonstrations.

Loads all episode_XXXX.npz files from human_data/, trains a BC policy using the
imitation library, then saves a PPO-compatible model as flappyPPO_bc.

The saved model can be loaded directly with PPO.load() for evaluation or PPO
fine-tuning without any conversion step.

Usage:
    python train_bc.py
    python train_bc.py --n_epochs 100 --batch_size 128 --save_name flappyPPO_bc
"""

import os
import glob
import logging
import argparse

import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import bc
from imitation.data.types import Transitions

from flappy.env import FlappyBirdEnv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --- Data loading ---

def load_transitions(data_dir: str) -> Transitions:
    """
    Load all episode_XXXX.npz files from data_dir and return a single
    Transitions object ready for the imitation BC trainer.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No episode files found in '{data_dir}/'. "
            "Run collect_human_data.py first."
        )

    obs_list      = []
    acts_list     = []
    next_obs_list = []
    dones_list    = []
    scores        = []

    for path in files:
        data = np.load(path)
        obs_list.append(data["obs"])
        acts_list.append(data["acts"])
        next_obs_list.append(data["next_obs"])
        dones_list.append(data["dones"])
        scores.append(int(np.sum(data["rewards"] == 1.0)))

    obs      = np.concatenate(obs_list,      axis=0)
    acts     = np.concatenate(acts_list,     axis=0)
    next_obs = np.concatenate(next_obs_list, axis=0)
    dones    = np.concatenate(dones_list,    axis=0)
    # imitation requires an infos array - unused by BC but must be present
    infos    = np.array([{} for _ in range(len(obs))])

    logger.info(f"Loaded {len(files)} episode(s) | {len(obs):,} transitions")
    logger.info(
        f"Score - min: {min(scores)}  max: {max(scores)}  "
        f"mean: {np.mean(scores):.1f}  median: {np.median(scores):.1f}"
    )

    return Transitions(obs=obs, acts=acts, next_obs=next_obs, dones=dones, infos=infos)


# --- Evaluation ---

def evaluate(model, n_episodes: int) -> float:
    """Run n_episodes headless and return mean reward."""
    env = FlappyBirdEnv(render_mode=None)
    rewards = []
    try:
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = truncated = False
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)
    finally:
        env.close()

    logger.info(
        f"Eval ({n_episodes} eps) - "
        f"mean: {np.mean(rewards):.2f}  "
        f"min: {min(rewards):.0f}  "
        f"max: {max(rewards):.0f}"
    )
    return float(np.mean(rewards))


# --- Training ---

def train(data_dir, n_epochs, batch_size, l2_weight, save_name, eval_episodes):
    transitions = load_transitions(data_dir)

    env = FlappyBirdEnv(render_mode=None)

    # Create a PPO model for its MlpPolicy architecture only.
    # BC trains the policy weights in-place via the shared policy object, so
    # saving ppo_model afterwards produces a file that PPO.load() accepts
    # directly - no conversion needed for later fine-tuning.
    ppo_model = PPO("MlpPolicy", env, verbose=0)

    epoch_counter = [0]

    def on_epoch_end():
        epoch_counter[0] += 1
        if epoch_counter[0] % 10 == 0 or epoch_counter[0] == n_epochs:
            logger.info(f"  Epoch {epoch_counter[0]}/{n_epochs}")

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=ppo_model.policy,
        rng=np.random.default_rng(),
        batch_size=batch_size,
        l2_weight=l2_weight,
    )

    logger.info(
        f"\nTraining BC - epochs: {n_epochs}  "
        f"batch_size: {batch_size}  l2_weight: {l2_weight}\n"
    )

    bc_trainer.train(n_epochs=n_epochs, on_epoch_end=on_epoch_end)

    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    ppo_model.save(save_name)
    logger.info(f"\nSaved → {save_name}")

    if eval_episodes > 0:
        logger.info(f"\nEvaluating over {eval_episodes} episode(s)...")
        evaluate(ppo_model, eval_episodes)

    env.close()


# --- Entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BC policy on human Flappy Bird demonstrations"
    )
    parser.add_argument("--data_dir",      type=str,   default="human_data",
                        help="Folder containing episode_XXXX.npz files (default: human_data)")
    parser.add_argument("--n_epochs",      type=int,   default=50,
                        help="Number of passes over the dataset (default: 50)")
    parser.add_argument("--batch_size",    type=int,   default=64,
                        help="Minibatch size for BC updates (default: 64)")
    parser.add_argument("--l2_weight",     type=float, default=1e-5,
                        help="L2 regularisation weight to reduce overfitting (default: 1e-5)")
    parser.add_argument("--save_name",     type=str,   default="models/flappyPPO_bc",
                        help="Output model filename (default: models/flappyPPO_bc)")
    parser.add_argument("--eval_episodes", type=int,   default=5,
                        help="Episodes to evaluate after training, 0 to skip (default: 5)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        l2_weight=args.l2_weight,
        save_name=args.save_name,
        eval_episodes=args.eval_episodes,
    )
