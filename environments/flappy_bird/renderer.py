"""Asset-based pixel rendering at the simulator's native resolution.

RGB output is always native size (128 x 64 by default). Only the human window
is enlarged, using nearest-neighbor scaling. Headless training loads no assets.
"""

from __future__ import annotations

from pathlib import Path

# The supplied assets are up-, middle-, and down-flap frames, respectively.
_BIRD_FILES = (
    "95f2779a5d1f88ae91732fa001cb69b0.png",
    "e634adc55cda8cb31e971733047c233f.png",
    "24e04d16d7cdd944f00d3bf10affddb0.png",
)
_DIGITS = (
    ("111", "101", "101", "101", "111"),
    ("010", "110", "010", "010", "111"),
    ("111", "001", "111", "100", "111"),
    ("111", "001", "111", "001", "111"),
    ("101", "101", "111", "001", "001"),
    ("111", "100", "111", "001", "111"),
    ("111", "100", "111", "101", "111"),
    ("111", "001", "010", "010", "010"),
    ("111", "101", "111", "101", "111"),
    ("111", "101", "111", "001", "111"),
)
_STEPS_LABEL = (
    "111 111 111 110 111",
    "100 010 100 101 100",
    "111 010 110 110 111",
    "001 010 100 100 001",
    "111 010 111 100 111",
)


class FlappyBirdRenderer:
    """Draw selected environments; animation follows physics ticks."""

    SKY = (111, 198, 207)
    PIPE_OUTLINE = (45, 82, 42)
    PIPE_GREEN = (111, 190, 47)
    PIPE_LIGHT = (190, 233, 99)
    PIPE_SHADOW = (72, 137, 41)
    SAND = (226, 206, 147)

    def __init__(self, cfg):
        import pygame

        self.cfg = cfg
        self.watched_envs = tuple(cfg.render_envs)
        if not self.watched_envs:
            raise ValueError("At least one index is required in render_envs")
        self._pygame = pygame
        self._display = None
        self._clock = None
        self.closed = False
        assets = Path(__file__).with_name("assets")
        self._birds = tuple(
            pygame.image.load(str(assets / name)) for name in _BIRD_FILES
        )
        self._rotated_birds = {}
        self._pipe_surfaces = {}
        self._digits = tuple(self._make_glyph(rows) for rows in _DIGITS)
        self._steps_label = self._make_glyph(_STEPS_LABEL)
        self._background = self._make_background()
        self._ground = self._make_ground()

    def _make_background(self):
        pg, cfg = self._pygame, self.cfg
        surface = pg.Surface((cfg.screen_width, cfg.screen_height))
        surface.fill(self.SKY)
        # Low, block-shaped clouds and shrubs leave the playfield readable.
        for x in range(-8, cfg.screen_width, 30):
            pg.draw.rect(surface, (215, 241, 219), (x, cfg.ground_y - 9, 25, 9))
            pg.draw.rect(surface, (215, 241, 219), (x + 4, cfg.ground_y - 12, 12, 3))
            pg.draw.rect(surface, (163, 216, 146), (x + 12, cfg.ground_y - 5, 20, 5))
            pg.draw.rect(surface, (163, 216, 146), (x + 16, cfg.ground_y - 7, 10, 2))
        return surface

    def _make_ground(self):
        pg, cfg = self._pygame, self.cfg
        surface = pg.Surface((cfg.screen_width + 8, max(1, cfg.ground_height)))
        surface.fill(self.SAND)
        pg.draw.rect(surface, self.PIPE_OUTLINE, (0, 0, surface.get_width(), 1))
        pg.draw.rect(surface, self.PIPE_LIGHT, (0, 1, surface.get_width(), 2))
        for x in range(0, surface.get_width(), 4):
            pg.draw.rect(surface, self.PIPE_GREEN, (x, 2, 2, 1))
            pg.draw.rect(surface, (192, 168, 110), (x + 1, 5, 1, 1))
        return surface

    def _make_glyph(self, rows):
        pg = self._pygame
        surface = pg.Surface((len(rows[0]) + 2, len(rows) + 2), pg.SRCALPHA)
        pixels = [(x + 1, y + 1) for y, row in enumerate(rows)
                  for x, value in enumerate(row) if value == "1"]
        for x, y in pixels:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                surface.set_at((x + dx, y + dy), (42, 49, 51))
        for point in pixels:
            surface.set_at(point, (255, 255, 235))
        return surface

    def _pipe(self, height, upper):
        key = height, upper
        if key in self._pipe_surfaces:
            return self._pipe_surfaces[key]
        pg, width = self._pygame, self.cfg.pipe_width
        surface = pg.Surface((width, height))
        surface.fill(self.PIPE_OUTLINE)
        if width > 2:
            pg.draw.rect(surface, self.PIPE_GREEN, (1, 0, width - 2, height))
            pg.draw.rect(surface, self.PIPE_LIGHT, (1, 0, max(1, width // 4), height))
            pg.draw.rect(surface, self.PIPE_SHADOW, (max(1, width - 3), 0, min(2, width - 2), height))
        cap_height = min(4, height)
        cap_y = height - cap_height if upper else 0
        pg.draw.rect(surface, self.PIPE_OUTLINE, (0, cap_y, width, cap_height), 1)
        self._pipe_surfaces[key] = surface
        return surface

    def _bird(self, frame_index, angle):
        # Bound cache size, even if a caller supplies an invalid external state.
        angle = max(-90, min(45, int(angle)))
        key = frame_index, angle
        if key not in self._rotated_birds:
            pg = self._pygame
            rotated = pg.transform.rotate(self._birds[frame_index], angle)
            rotated = rotated.subsurface(rotated.get_bounding_rect())
            # Keep the artwork within the configured physics rectangle.
            # Rotate at source resolution, then downsample only once.
            self._rotated_birds[key] = pg.transform.scale(
                rotated, (self.cfg.player_width, self.cfg.player_height),
            )
        return self._rotated_birds[key]

    def update(self, state, scores, *, episode_lengths=None, render_mode=None):
        cfg, pg = self.cfg, self._pygame
        selected = state[list(self.watched_envs)].detach().cpu().numpy()
        selected_scores = scores[list(self.watched_envs)].detach().cpu().tolist()
        ticks = (episode_lengths[list(self.watched_envs)].detach().cpu().tolist()
                 if episode_lengths is not None else [0] * len(selected))
        surfaces = []
        cycle = (0, 1, 2, 1)
        frame_count = min(cfg.player_sprite_frames, len(self._birds))
        for row, score, tick in zip(selected, selected_scores, ticks):
            surface = self._background.copy()
            for x, gap_y in zip(row[3::2], row[4::2]):
                x, gap_y = int(x), int(gap_y)
                if x >= cfg.screen_width or x + cfg.pipe_width <= 0:
                    continue
                upper_height = max(0, min(cfg.ground_y, gap_y))
                lower_y = max(0, min(cfg.ground_y, gap_y + cfg.pipe_gap))
                if upper_height:
                    surface.blit(self._pipe(upper_height, True), (x, 0))
                if lower_y < cfg.ground_y:
                    surface.blit(self._pipe(cfg.ground_y - lower_y, False), (x, lower_y))
            frame_index = cycle[(tick // cfg.player_anim_ticks) % len(cycle)] % frame_count
            surface.blit(self._bird(frame_index, row[2]), (cfg.player_x, int(row[0])))
            surface.blit(self._ground, (-(tick * cfg.pipe_speed) % 8 - 8, cfg.ground_y))
            if cfg.show_score:
                digits = str(int(score))
                left = (cfg.screen_width - len(digits) * 4 - 1) // 2
                for offset, digit in enumerate(digits):
                    surface.blit(self._digits[int(digit)], (left + offset * 4, 3))
            # Each panel shows its own episode length, using the same ticks as animation.
            label_y = cfg.screen_height - self._steps_label.get_height() - 1
            surface.blit(self._steps_label, (2, label_y))
            number_x = 2 + self._steps_label.get_width() + 2
            for offset, digit in enumerate(str(int(tick))):
                surface.blit(self._digits[int(digit)], (number_x + offset * 4, label_y))
            surfaces.append(surface)

        if render_mode == "rgb_array":
            # Native resolution, independent of the human window's render_scale.
            return [pg.surfarray.array3d(surface).transpose(1, 0, 2).copy() for surface in surfaces]
        if self.closed:
            return None
        if self._display is None:
            pg.display.init()
            self._display = pg.display.set_mode((
                cfg.screen_width * cfg.render_scale * len(surfaces),
                cfg.screen_height * cfg.render_scale,
            ))
            pg.display.set_caption(f"Flappy Bird | {cfg.screen_width} x {cfg.screen_height} | Esc to close")
            self._clock = pg.time.Clock()
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.close()
                return None
        size = cfg.screen_width * cfg.render_scale, cfg.screen_height * cfg.render_scale
        for index, surface in enumerate(surfaces):
            self._display.blit(pg.transform.scale(surface, size), (index * size[0], 0))
        pg.display.flip()
        self._clock.tick(cfg.tick_hz)
        return None

    def close(self):
        if self._display is not None:
            self._pygame.display.quit()
            self._display = None
        self.closed = True
