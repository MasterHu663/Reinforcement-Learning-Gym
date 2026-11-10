"""Play Flappy Bird with the keyboard: python -m environments.flappy_bird.play."""

from __future__ import annotations

import argparse

import torch

from .envConfig import FlappyBirdEnvConfig
from .envForRL import FlappyBirdEnv


class _PlayEnv(FlappyBirdEnv):
    def render(self):
        # step() renders before autoreset: retain that frame for the game-over screen.
        frames = super().render()
        self.frame = frames[0]
        return frames


class HumanGame:
    """Own keyboard events and pacing while the environment supplies native RGB frames."""

    def __init__(self, cfg: FlappyBirdEnvConfig):
        import pygame

        if cfg.num_envs != 1 or cfg.device != "cpu" or cfg.render_mode != "rgb_array":
            raise ValueError("HumanGame requires one CPU environment with render_mode='rgb_array'")
        self.pg = pygame
        self.env = _PlayEnv(cfg)
        self.env.configure_rendering(True)
        self.phase = "ready"
        self.running = True
        self.flap = False
        self.episode_reward = 0.0
        self.status = "Press Space / Up to start"
        self.action = torch.zeros(1, dtype=torch.int64)
        self.env.reset(seed=cfg.seed)

    def handle_event(self, event):
        pg = self.pg
        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
            self.running = False
        elif event.type == pg.KEYDOWN and not getattr(event, "repeat", False):
            if event.key == pg.K_r:
                self.env.reset()
                self.phase = "ready"
                self.flap = False
                self.episode_reward = 0.0
                self.status = "Press Space / Up to start"
            elif event.key == pg.K_p:
                if self.phase == "playing":
                    self.phase = "paused"
                    self.status = "Paused | P to resume"
                elif self.phase == "paused":
                    self.phase = "playing"
                    self.status = "Playing"
                self.flap = False
            elif event.key in (pg.K_SPACE, pg.K_UP) and self.phase != "paused":
                if self.phase in ("ready", "game_over"):
                    # The environment already reset when the previous episode ended.
                    self.episode_reward = 0.0
                self.phase = "playing"
                self.status = "Playing"
                self.flap = True

    def tick(self):
        if not self.running or self.phase != "playing":
            return
        self.action.fill_(int(self.flap))
        self.flap = False
        _, reward, terminated, truncated, info = self.env.step(self.action)
        self.episode_reward += reward.item()
        if terminated.item() or truncated.item():
            self.phase = "game_over"
            reason = "Game over" if terminated.item() else "Limit reached"
            self.status = (
                f"{reason} | Score {info['score'].item()} | Steps {info['episode_length'].item()}"
                " | Space to retry"
            )
            print(f"{self.status} | Reward {self.episode_reward:.1f}")

    def run(self):
        pg, cfg = self.pg, self.env.cfg
        pg.display.init()
        pg.font.init()
        pg.key.set_repeat()  # One key press produces one flap, regardless of OS repeat settings.
        game_size = cfg.screen_width * cfg.render_scale, cfg.screen_height * cfg.render_scale
        screen = pg.display.set_mode((game_size[0], game_size[1] + 48))
        pg.display.set_caption("Flappy Bird | Human control")
        font = pg.font.Font(None, 20)
        clock = pg.time.Clock()
        controls = "Space / Up: flap    P: pause    R: restart    Esc: quit"
        controls_surface = font.render(controls, True, (200, 210, 220))
        previous_status = None
        status_surface = None
        while self.running:
            for event in pg.event.get():
                self.handle_event(event)
                if not self.running:
                    break
            if not self.running:
                break
            self.tick()
            # RGB rendering consumes no events and applies no frame-rate limit.
            surface = pg.surfarray.make_surface(self.env.frame.transpose(1, 0, 2))
            screen.blit(pg.transform.scale(surface, game_size), (0, 0))
            screen.fill((25, 31, 36), (0, game_size[1], game_size[0], 48))
            if self.status != previous_status:
                status_surface = font.render(self.status, True, (255, 255, 235))
                previous_status = self.status
            # Keep both footer lines visible even with a small window scale.
            for index, text_surface in enumerate((status_surface, controls_surface)):
                available_width = game_size[0] - 8
                if text_surface.get_width() > available_width:
                    text_surface = pg.transform.scale(text_surface, (
                        available_width, max(1, round(text_surface.get_height() * available_width / text_surface.get_width())),
                    ))
                screen.blit(text_surface, (4, game_size[1] + 4 + index * 22))
            pg.display.flip()
            clock.tick(cfg.tick_hz)

    def close(self):
        self.env.close()
        self.pg.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible pipe sequence")
    parser.add_argument("--player-width", type=int, default=FlappyBirdEnvConfig.player_width, help="Bird width in native pixels")
    parser.add_argument("--player-height", type=int, default=FlappyBirdEnvConfig.player_height, help="Bird height in native pixels")
    args = parser.parse_args()
    env_kwargs = {
        "num_envs": 1,
        "device": "cpu",
        "max_episode_length": None,
        "score_limit": None,
        "seed": args.seed,
        "flap_velocity": -3,
        "render_mode": "rgb_array",
        "player_width": args.player_width,
        "player_height": args.player_height,

        # Configure shared physics and pipe parameters for manual play and training in envConfig.py.
    }
    cfg = FlappyBirdEnvConfig(**env_kwargs)
    game = HumanGame(cfg)
    try:
        game.run()
    except KeyboardInterrupt:
        pass
    finally:
        game.close()

if __name__ == "__main__":
    main()
