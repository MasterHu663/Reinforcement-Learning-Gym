"""Matplotlib renderer for selected environments in a batched CartPole task.

Rendering transfers only selected (x, xdot, theta, thetadot) rows to CPU.
No graphics library is involved during training unless render() is called.
"""

from __future__ import annotations

import numpy as np


class CartPoleRenderer:
    """Visualize one or several CartPole instances in separate subplots."""

    def __init__(self, cfg, watched_envs=(0,)):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        self.cfg = cfg
        self._plt = plt
        self.watched_envs = tuple(watched_envs)
        if not self.watched_envs:
            raise ValueError("At least one index is required in render_envs")
        self.fig, axes = plt.subplots(
            1, len(self.watched_envs), squeeze=False,
            figsize=(5.8 * len(self.watched_envs), 3.9),
        )
        self.axes = axes.ravel()
        self._carts = []
        self._poles = []
        self._joints = []

        for env_id, ax in zip(self.watched_envs, self.axes):
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-cfg.x_threshold * 1.25, cfg.x_threshold * 1.25)
            ax.set_ylim(-0.4, 1.75)
            ax.axhline(0.05, color="0.4", linewidth=2.0)
            ax.axvline(-cfg.x_threshold, color="tomato", linestyle="--", alpha=0.4)
            ax.axvline(cfg.x_threshold, color="tomato", linestyle="--", alpha=0.4)
            ax.set_title(f"CartPole env {env_id}")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            cart = Rectangle((-0.3, 0.05), 0.6, 0.28, facecolor="black")
            pole, = ax.plot([0.0, 0.0], [0.3, 1.3], color="#cb8b51", lw=8,
                            solid_capstyle="round")
            joint = Circle((0.0, 0.3), 0.055, facecolor="#8184cb", zorder=5)
            ax.add_patch(cart)
            ax.add_patch(joint)
            self._carts.append(cart)
            self._poles.append(pole)
            self._joints.append(joint)

        self.fig.tight_layout()

    def update(self, state, *, render_mode: str | None = "human"):
        """Update artists; ``rgb_array`` returns one RGB image per watched env."""
        # Only rendered environments leave their current device (e.g. CUDA).
        selected = state[list(self.watched_envs)].detach().cpu().numpy()
        pole_length = 2.0 * self.cfg.length
        axle_y = 0.33

        for row, cart, pole, joint in zip(
            selected, self._carts, self._poles, self._joints
        ):
            x, _, theta, _ = row
            x = float(x)
            theta = float(theta)
            cart.set_x(x - 0.3)
            pole.set_data(
                [x, x + pole_length * np.sin(theta)],
                [axle_y, axle_y + pole_length * np.cos(theta)],
            )
            joint.center = (x, axle_y)

        if render_mode == "rgb_array":
            # Return an image per environment (not a GPU batch of images).
            self.fig.canvas.draw()
            rgba = np.asarray(self.fig.canvas.buffer_rgba())
            # Separate subplot crops, each with height H and width W / count.
            width_per_env = rgba.shape[1] // len(self.watched_envs)
            return [
                rgba[:, i * width_per_env:(i + 1) * width_per_env, :3].copy()
                for i in range(len(self.watched_envs))
            ]

        self._plt.ion()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)
        return None

    def close(self) -> None:
        self._plt.close(self.fig)
