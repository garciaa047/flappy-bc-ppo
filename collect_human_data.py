"""
collect_human_data.py - Play Flappy Bird and log transitions for behavioural cloning.

Saves one .npz file per episode to human_data/.
Each file contains:
    obs      - (T, 5) float32   normalized observations  (matches FlappyBirdEnv)
    acts     - (T,)   int64     actions (0 = do nothing, 1 = flap)
    next_obs - (T, 5) float32   next normalized observations
    rewards  - (T,)   float32   rewards (+1 pipe, -1 death, 0 otherwise)
    dones    - (T,)   bool      True on the terminal step of an episode

Controls: SPACE to flap | ESC / close window to quit and save current episode
"""

import os
import argparse

import numpy as np
import pygame

from flappy.game import FlappyBirdGame, CONFIG, PLAYING, GAME_OVER, WAITING
from flappy.env import VEL_MIN, VEL_MAX

SAVE_DIR = "human_data"


# --- Normalization ---

def _normalize_obs(raw: dict) -> np.ndarray:
    w = CONFIG["SCREEN_WIDTH"]
    h = CONFIG["SCREEN_HEIGHT"]
    return np.array([
        np.clip(raw["bird_y"] / h,                                        0.0, 1.0),
        np.clip((raw["bird_velocity"] - VEL_MIN) / (VEL_MAX - VEL_MIN),  0.0, 1.0),
        np.clip(raw["pipe_x"] / w,                                        0.0, 1.0),
        np.clip(raw["pipe_top_y"] / h,                                    0.0, 1.0),
        np.clip(raw["pipe_bottom_y"] / h,                                 0.0, 1.0),
    ], dtype=np.float32)


# --- File helpers ---

def _next_episode_index(save_dir: str) -> int:
    """Return the next episode number based on files already in save_dir."""
    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    ]
    if not existing:
        return 0
    indices = [int(f[len("episode_"):-len(".npz")]) for f in existing]
    return max(indices) + 1


def _save_episode(save_dir, ep_idx, obs, acts, next_obs, rewards, dones) -> str:
    path = os.path.join(save_dir, f"episode_{ep_idx:04d}.npz")
    np.savez_compressed(
        path,
        obs=np.array(obs,      dtype=np.float32),
        acts=np.array(acts,    dtype=np.int64),
        next_obs=np.array(next_obs, dtype=np.float32),
        rewards=np.array(rewards,   dtype=np.float32),
        dones=np.array(dones,       dtype=bool),
    )
    return path


# --- Main collection loop ---

def collect(max_episodes=None):
    os.makedirs(SAVE_DIR, exist_ok=True)

    game  = FlappyBirdGame()
    clock = pygame.time.Clock()

    ep_idx        = _next_episode_index(SAVE_DIR)
    episodes_done = 0

    # Per-episode buffers
    obs_buf      = []
    acts_buf     = []
    next_obs_buf = []
    rew_buf      = []
    done_buf     = []
    prev_score   = 0

    def flush_episode():
        """Save the current episode buffer to disk and reset it."""
        nonlocal ep_idx, episodes_done, prev_score
        if not obs_buf:
            return

        if game.score < 15:
            print(f"Episode not saved (score {game.score} < 15 threshold)")
        else:
            path = _save_episode(
                SAVE_DIR, ep_idx,
                obs_buf, acts_buf, next_obs_buf, rew_buf, done_buf,
            )
            print(f"Episode {ep_idx:04d} | steps: {len(obs_buf):5d} | score: {game.score:3d} | saved → {path}")
            ep_idx += 1
        episodes_done += 1
        obs_buf.clear()
        acts_buf.clear()
        next_obs_buf.clear()
        rew_buf.clear()
        done_buf.clear()
        prev_score = 0

    print(f"Saving demonstrations to '{SAVE_DIR}/'")
    print("SPACE to flap | ESC or close window to quit\n")

    running = True
    try:
        while running and (max_episodes is None or episodes_done < max_episodes):
            clock.tick(CONFIG["FPS"])

            # --- Event handling ---
            flap = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        if game.state == WAITING:
                            game.state = PLAYING
                            flap = True
                        elif game.state == PLAYING:
                            flap = True
                        elif game.state == GAME_OVER:
                            flush_episode()
                            game.reset()

            # --- Step ---
            if game.state == PLAYING:
                obs    = _normalize_obs(game.get_observation())
                action = 1 if flap else 0

                if flap:
                    game.bird.flap()

                game._update(int(1000 / CONFIG["FPS"]))

                next_obs   = _normalize_obs(game.get_observation())
                terminated = game.state == GAME_OVER

                if terminated:
                    reward = -1.0
                elif game.score > prev_score:
                    reward = 1.0
                    prev_score = game.score
                else:
                    reward = 0.0

                obs_buf.append(obs)
                acts_buf.append(action)
                next_obs_buf.append(next_obs)
                rew_buf.append(reward)
                done_buf.append(terminated)

            game._draw()

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        flush_episode()
        pygame.quit()
        print(f"\nDone. {episodes_done} episode(s) saved to '{SAVE_DIR}/'.")


# --- Entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect human Flappy Bird demonstrations")
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Stop automatically after this many episodes. Default: run until ESC or Ctrl+C."
    )
    args = parser.parse_args()

    collect(max_episodes=args.max_episodes)
