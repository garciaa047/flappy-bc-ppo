"""
flappy/env.py — Gymnasium environment wrapping FlappyBirdGame.

Observation space: Box(5,) float32, values normalized to [0, 1]
    [bird_y_norm, bird_velocity_norm, pipe_x_norm, pipe_top_norm, pipe_bottom_norm]

Action space: Discrete(2)
    0 = do nothing
    1 = flap

Reward:
    +1.0  when the bird passes a pipe
    -1.0  on death (collision or out of bounds)
     0.0  otherwise

render_mode:
    "human"     — visible pygame window (evaluation / debugging)
    "rgb_array" — offscreen render, returns (H, W, 3) uint8 numpy array (training)
    None        — no rendering (fastest training)
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Velocity normalization bounds ──────────────────────────────────────────────
# Bird velocity in practice stays well within [-10, +10]; clip for safety.
VEL_MIN = -10.0
VEL_MAX =  10.0


class FlappyBirdEnv(gym.Env):
    """
    Gymnasium wrapper for FlappyBirdGame.

    The game object is owned internally (composition, not inheritance).
    flappy/game.py is left untouched.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()

        assert render_mode in (None, "human", "rgb_array"), (
            f"render_mode must be 'human', 'rgb_array', or None. Got: {render_mode!r}"
        )
        self.render_mode = render_mode

        # Suppress the pygame display window for non-human modes.
        # Must be set BEFORE pygame.init() is called inside FlappyBirdGame.
        if render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        # Import here so the env var is already set before pygame is touched.
        from flappy.game import FlappyBirdGame, CONFIG, PLAYING
        self._Game    = FlappyBirdGame
        self._CONFIG  = CONFIG
        self._PLAYING = PLAYING

        self.game = FlappyBirdGame()

        # Fixed timestep: deterministic and reproducible across rollouts.
        self._fixed_dt = int(1000 / CONFIG["FPS"])  # ~16 ms at 60 FPS

        # ── Spaces ────────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5,  dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = flap

        self._prev_score = 0

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset to initial state. Returns (obs, info)."""
        super().reset(seed=seed)

        self.game.reset()
        # Skip the WAITING state — that exists only for human keyboard input.
        self.game.state = self._PLAYING
        self._prev_score = 0

        obs  = self._get_obs()
        info = {"score": 0}
        return obs, info

    def step(self, action):
        """
        Advance one frame.

        Parameters
        ----------
        action : int  — 0 (do nothing) or 1 (flap)

        Returns
        -------
        obs         : np.ndarray shape (5,)
        reward      : float
        terminated  : bool
        truncated   : bool  (always False — no time limit)
        info        : dict  {"score": int}
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        _, score, terminated = self.game.step(
            action=bool(action), dt=self._fixed_dt
        )

        obs = self._get_obs()

        if terminated:
            reward = -1.0
        elif score > self._prev_score:
            reward = 1.0
            self._prev_score = score
        else:
            reward = 0.0

        info = {"score": score}
        return obs, reward, terminated, False, info

    def render(self):
        """
        Render the current frame.

        "human"     — draws to the pygame window; pumps events to keep it responsive.
        "rgb_array" — draws offscreen, returns (H, W, 3) uint8 array.
        None        — no-op.
        """
        if self.render_mode is None:
            return

        import pygame

        self.game._draw()

        if self.render_mode == "human":
            # Pump events so the OS doesn't mark the window as unresponsive.
            pygame.event.pump()

        elif self.render_mode == "rgb_array":
            # surfarray returns (W, H, 3); gymnasium expects (H, W, 3).
            arr = pygame.surfarray.array3d(self.game.screen)
            return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        """Shut down pygame cleanly."""
        try:
            import pygame
            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_obs(self):
        """
        Pull raw values from the game, normalize to [0, 1], return as float32
        numpy array.
        """
        raw = self.game.get_observation()
        w   = self._CONFIG["SCREEN_WIDTH"]
        h   = self._CONFIG["SCREEN_HEIGHT"]

        bird_y_norm = float(np.clip(raw["bird_y"] / h, 0.0, 1.0))
        vel_norm    = float(np.clip(
            (raw["bird_velocity"] - VEL_MIN) / (VEL_MAX - VEL_MIN), 0.0, 1.0
        ))
        pipe_x_norm = float(np.clip(raw["pipe_x"] / w, 0.0, 1.0))
        top_norm    = float(np.clip(raw["pipe_top_y"] / h, 0.0, 1.0))
        bot_norm    = float(np.clip(raw["pipe_bottom_y"] / h, 0.0, 1.0))

        return np.array(
            [bird_y_norm, vel_norm, pipe_x_norm, top_norm, bot_norm],
            dtype=np.float32,
        )
