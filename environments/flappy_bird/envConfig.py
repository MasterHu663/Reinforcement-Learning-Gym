"""Configuration for Flappy Bird on a 128x64 OLED driven by an STM32F427.

Target hardware
    MCU      STM32F427ZGT6: Cortex-M4F, 1 MB Flash, 256 KB SRAM. 64 KB of the
             SRAM is CCM, which the DMA controllers cannot read, so the SPI
             framebuffer must live in the main 192 KB.
    Display  2.42" SSD1309 OLED, 128x64 monochrome, 4-wire SPI
             (SCL, SDA, DC, CS, RES). One frame is 128 * 64 / 8 = 1024 bytes
             in page layout: 8 pages of 128 bytes, each byte holding 8
             vertical pixels with the least significant bit on top.
    Input    Push button (human play) or a trained lookup table (autonomous).

Everything the firmware has to reproduce is an integer so that the Python
simulation and the C game loop step identically:

* Geometry is in OLED pixels; origin top-left, +y points down.
* Vertical position is stored directly in whole pixels. Velocity is pixels per
  tick and gravity is pixels per tick squared (defaults: gravity 1, flap -4,
  max fall 3). A single flap climbs 10 pixels before reaching its apex.
* Horizontal motion (pipes) is whole pixels per tick.
* The planned lookup-table state uses discrete bins; their product determines
  the number of rows of the Q / policy table stored in Flash.

Target simulation contract for envForRL.py and the firmware
    tick    if flap and y > flap_min_y_px: vy = flap_velocity
            else: vy = min(vy + gravity, max_fall_velocity)
            y = min(y + vy, player_max_y_px)                      pixels
            ceiling: if y < 0: y = 0, vy = 0, reward_ceiling; no termination
            every pipe: x -= pipe_speed; when x + pipe_width <= 0 the pipe is
            recycled to max(x of all pipes) + pipe_spacing with a new gap_y
    score   +1 on the tick where pipe_x + pipe_width // 2 passes
            player_x + player_width // 2 (window of pipe_speed pixels)
    crash   y + player_height >= ground_y, or the bird rectangle overlaps a
            pipe rectangle: columns [x, x + pipe_width), upper pipe rows
            [0, gap_y), lower pipe rows [gap_y + pipe_gap, ground_y)
    reward precedence: crash > ceiling > pipe > alive

Discrete state (lookup-table index)
    next pipe = nearest pipe with x + pipe_width > player_x
    dx = clamp(next.x - (player_x + player_width), obs_dx_min_px, obs_dx_max_px)
    dy = clamp(y - next.gap_y, obs_dy_min_px, obs_dy_max_px)
    vy = clamp(vy, flap_velocity, max_fall_velocity)
    i_dx = (dx - obs_dx_min_px) // obs_dx_bin_px
    i_dy = (dy - obs_dy_min_px) // obs_dy_bin_px
    i_vy = (vy - flap_velocity) // obs_vy_bin
    state = (i_dx * n_dy + i_dy) * n_vy + i_vy
    with obs_include_next_gap: dy2 uses the pipe after ``next`` and the index
    becomes state * n_dy2 + i_dy2.
    All operands are non-negative after subtracting the minimum, so C integer
    division and Python ``//`` agree. This lookup-table encoding is a firmware
    target and is not yet exposed by envForRL.py. Current MLP observations are
    normalized (pipe_x, gap_top, gap_bottom) triples sorted by x, followed by
    bird (y, vy, rotation): 3 * pipe_count + 3 floats.

This module has no torch or graphics dependency. ``to_c_header()`` emits the
integer constants as #defines so the firmware and the simulator share one
source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

_Q_DTYPE_BYTES = {"int8": 1, "int16": 2, "float32": 4}
_INT16_MAX = 32767


def _require_int(
    name: str, value, *, minimum: int | None = None, maximum: int | None = None
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


@dataclass
class FlappyBirdEnvConfig:
    # ---- batch / runtime (simulation only) -----------------------------------
    num_envs: int = 1024
    device: str = "cuda:0"  # use "cpu" to debug without CUDA
    seed: int | None = None



    state_mode: Literal["Lidar-on", "Lidar-off"] = "Lidar-off"  # Batched Lidar is not implemented yet.
    score_limit: int | None = None  # Truncate after this many pipes; None = never.
    max_episode_length: int | None = None  # Optional time limit in simulation ticks.
    physics_dtype: Literal["int16", "int32"] = "int32"  # Torch dtype of the integer state.
    validate_actions: bool = False  # True costs a GPU-to-CPU synchronization
    debug: bool = False

    # ---- actions -------------------------------------------------------------
    # Discrete action IDs: 0 = idle, 1 = flap. One scalar action, two choices.
    action_dim: int = field(init=False, default=1)
    num_actions: int = field(init=False, default=2)

    # ---- timing --------------------------------------------------------------
    tick_hz: int = 20  # Shared play/training physics rate: 50 ms per tick.

    # ---- display: SSD1309, 128x64, 1 bit per pixel ---------------------------
    screen_width: int = 128
    screen_height: int = 64

    page_height: int = field(init=False, default=8)  # SSD1309 page: 8 rows, 1 byte per column.
    double_buffer: bool = True  # Draw one frame while SPI DMA sends the other.
    spi_hz: int = 10_000_000  # SSD1309 serial clock limit (100 ns cycle).
    sram_bytes: int = 192 * 1024  # DMA-reachable SRAM; the 64 KB CCM is excluded.
    # Half of the 1 MB Flash for the trained table; the rest holds code, fonts, sprites.
    flash_budget_bytes: int = 512 * 1024

    # ---- geometry (pixels) ---------------------------------------------------
    ground_height: int = 8  # One SSD1309 page at the bottom of the screen.
    player_x: int = 24  # Bird's left edge, fixed like the original (~0.2 * width).
    player_width: int = 9
    player_height: int = 6
    pipe_width: int = 10
    pipe_gap: int = 19  # Vertical opening between the upper and lower pipe.
    pipe_count: int = 4  # Enough pipe pairs to recycle beyond the screen at this spacing.
    pipe_spacing: int = 40  # Left edge to left edge; 2 seconds between pipes.
    pipe_speed: int = 1  # Pixels per tick; 20 pixels/second to the left.
    pipe_first_x: int = 96  # Left edge of the first pipe at reset.
    # Pipe opening top row: sample any integer in [reset_low, reset_high].
    # These bounds apply to pipes, not to the bird's initial state.
    reset_low: int = 5
    reset_high: int = 34

    # ---- vertical kinematics (whole pixels) --------------------------------
    gravity: int = 1  # pixels / tick^2
    flap_velocity: int = -4  # pixels / tick; one flap climbs 4+3+2+1=10 pixels.
    max_fall_velocity: int = 3  # pixels / tick; maximum descent is 60 pixels/second.
    flap_min_y_px: int = 3  # Flap is ignored when the bird's top y <= 3.

    # ---- observation discretization -----------------------------------------
    obs_dx_bin_px: int = 4
    obs_dy_bin_px: int = 2
    obs_vy_bin: int = 1  # pixels / tick per bin
    obs_include_next_gap: bool = False  # Planned table encoding: add the following pipe's gap offset.
    obs_dy2_bin_px: int = 16  # Coarse on purpose: every dy2 bin multiplies the table size.
    q_table_dtype: Literal["int8", "int16", "float32"] = "int16"

    # ---- rewards (training only; the MCU never computes them) ---------------
    reward_alive: float = 0.1
    reward_pipe: float = 1.0
    reward_crash: float = -1.0
    reward_ceiling: float = -0.5

    # ---- rendering: color PC preview at the OLED's native resolution --------
    render_mode: Literal["human", "rgb_array"] | None = None
    render_envs: tuple[int, ...] = (0,)
    render_scale: int = 6  # Window pixels per OLED pixel.
    show_score: bool = True
    player_sprite_frames: int = 3  # Wing frames stored in Flash (up / mid / down).
    player_anim_ticks: int = 3  # Ticks per wing frame (original: 3).
    oled_on_color: tuple[int, int, int] = (80, 170, 255)  # Reserved for a monochrome hardware preview.
    oled_off_color: tuple[int, int, int] = (0, 0, 0)

    # ---- timing ------------------------------------------------------------
    @property
    def tick_ms(self) -> float:
        """Game-loop period in milliseconds."""
        return 1000.0 / self.tick_hz

    @property
    def tick_period_us(self) -> int:
        """Timer period for the MCU game loop, in microseconds."""
        return round(1_000_000 / self.tick_hz)

    # ---- display -----------------------------------------------------------
    @property
    def framebuffer_pages(self) -> int:
        return self.screen_height // self.page_height

    @property
    def framebuffer_bytes(self) -> int:
        """Size of one 1-bpp frame in SSD1309 page layout."""
        return self.screen_width * self.framebuffer_pages

    @property
    def framebuffer_sram_bytes(self) -> int:
        return self.framebuffer_bytes * (2 if self.double_buffer else 1)

    @property
    def frame_transfer_ms(self) -> float:
        """Time to push one full frame over SPI, excluding command bytes."""
        return self.framebuffer_bytes * 8 * 1000.0 / self.spi_hz

    # ---- geometry ----------------------------------------------------------
    @property
    def ground_y(self) -> int:
        """First row of the ground; the playfield is rows [0, ground_y)."""
        return self.screen_height - self.ground_height

    @property
    def player_start_y_px(self) -> int:
        """Bird's top row at reset: vertically centered in the playfield."""
        return (self.ground_y - self.player_height) // 2

    @property
    def player_max_y_px(self) -> int:
        """Top row of the bird when it rests on the ground."""
        return self.ground_y - self.player_height

    @property
    def flap_rise_px(self) -> int:
        """Whole pixels climbed between a flap and the apex."""
        rise, vy = 0, self.flap_velocity
        while vy < 0:
            rise -= vy
            vy += self.gravity
        return rise

    @property
    def player_min_y_px(self) -> int:
        """The bird's top is clamped to the first screen row."""
        return 0

    # ---- observation ranges and table size ---------------------------------
    @property
    def obs_dx_min_px(self) -> int:
        # The next pipe satisfies x + pipe_width > player_x.
        return 1 - (self.pipe_width + self.player_width)

    @property
    def obs_dx_max_px(self) -> int:
        # Right after a pipe is passed the following one is at most this far.
        # The first pipe after reset starts farther away and is clamped to this
        # value; spending table rows on the approach phase buys nothing.
        return self.pipe_spacing - self.pipe_width - self.player_width

    @property
    def obs_dy_min_px(self) -> int:
        return self.player_min_y_px - self.reset_high

    @property
    def obs_dy_max_px(self) -> int:
        return self.player_max_y_px - self.reset_low

    @property
    def n_dx(self) -> int:
        return (self.obs_dx_max_px - self.obs_dx_min_px) // self.obs_dx_bin_px + 1

    @property
    def n_dy(self) -> int:
        return (self.obs_dy_max_px - self.obs_dy_min_px) // self.obs_dy_bin_px + 1

    @property
    def n_vy(self) -> int:
        return (self.max_fall_velocity - self.flap_velocity) // self.obs_vy_bin + 1

    @property
    def n_dy2(self) -> int:
        if not self.obs_include_next_gap:
            return 1
        return (self.obs_dy_max_px - self.obs_dy_min_px) // self.obs_dy2_bin_px + 1

    @property
    def num_discrete_states(self) -> int:
        """Rows of the lookup table; the index formula is in the module docstring."""
        return self.n_dx * self.n_dy * self.n_vy * self.n_dy2

    @property
    def observation_dim(self) -> int:
        """Current MLP input: three features per pipe, then y, vy, rotation."""
        return 3 * self.pipe_count + 3

    @property
    def state_dim(self) -> int:
        """Critic input width; this does not describe the full simulator state."""
        return self.observation_dim

    @property
    def q_table_bytes(self) -> int:
        """Flash needed to store every Q-value."""
        return self.num_discrete_states * self.num_actions * _Q_DTYPE_BYTES[self.q_table_dtype]

    @property
    def policy_table_bytes(self) -> int:
        """Flash needed for the greedy policy only: one bit (flap or not) per state."""
        return math.ceil(self.num_discrete_states / 8)

    # ---- validation --------------------------------------------------------
    def __post_init__(self) -> None:
        for name in (
            "num_envs", "tick_hz", "screen_width", "screen_height", "spi_hz",
            "sram_bytes", "flash_budget_bytes", "player_width", "player_height",
            "pipe_width", "pipe_gap", "pipe_count", "pipe_spacing", "pipe_speed",
            "gravity", "max_fall_velocity", "obs_dx_bin_px",
            "obs_dy_bin_px", "obs_vy_bin", "obs_dy2_bin_px", "render_scale",
            "player_sprite_frames", "player_anim_ticks",
        ):
            _require_int(name, getattr(self, name), minimum=1)
        for name in (
            "ground_height", "player_x", "pipe_first_x", "reset_low",
            "reset_high",
        ):
            _require_int(name, getattr(self, name), minimum=0)
        _require_int("flap_velocity", self.flap_velocity, maximum=-1)
        _require_int("flap_min_y_px",self.flap_min_y_px,minimum=0,maximum=self.player_max_y_px - 1,)
        for name in ("max_episode_length", "score_limit"):
            value = getattr(self, name)
            if value is not None:
                _require_int(name, value, minimum=1)

        if self.physics_dtype not in ("int16", "int32"):
            raise ValueError("physics_dtype must be 'int16' or 'int32'")
        if self.q_table_dtype not in _Q_DTYPE_BYTES:
            raise ValueError("q_table_dtype must be 'int8', 'int16' or 'float32'")
        if self.render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if any(i < 0 or i >= self.num_envs for i in self.render_envs):
            raise ValueError("render_envs contains an invalid environment index")
        if self.render_mode is not None and not self.render_envs:
            raise ValueError("render_envs must not be empty when rendering")
        for name in ("oled_on_color", "oled_off_color"):
            color = getattr(self, name)
            if len(color) != 3 or any(
                isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255
                for v in color
            ):
                raise ValueError(f"{name} must contain three RGB bytes")
        for name in ("reward_alive", "reward_pipe", "reward_crash", "reward_ceiling"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

        # Display and MCU budgets.
        if self.screen_height % self.page_height != 0:
            raise ValueError("screen_height must be a multiple of the 8-row SSD1309 page")
        if self.framebuffer_sram_bytes > self.sram_bytes:
            raise ValueError("framebuffers do not fit in the DMA-reachable SRAM")
        if self.frame_transfer_ms >= self.tick_ms:
            raise ValueError(
                f"a full SPI frame takes {self.frame_transfer_ms:.2f} ms but one tick "
                f"is {self.tick_ms:.2f} ms; lower tick_hz or raise spi_hz"
            )
        if self.screen_height > _INT16_MAX:
            raise ValueError("pixel positions must fit int16_t on the MCU")

        # Playfield layout.
        if self.ground_height >= self.screen_height:
            raise ValueError("ground_height must leave room for the playfield")
        if self.player_height >= self.ground_y:
            raise ValueError("the bird must fit between the ceiling and the ground")
        if self.player_x + self.player_width > self.screen_width:
            raise ValueError("the bird must lie inside the screen horizontally")
        if self.pipe_first_x < self.player_x + self.player_width:
            raise ValueError("the first pipe must start in front of the bird")
        if self.pipe_gap <= self.player_height:
            raise ValueError("pipe_gap must be taller than the bird")
        if self.pipe_spacing <= self.pipe_width + self.player_width:
            raise ValueError(
                "pipe_spacing must exceed pipe_width + player_width so that only "
                "one pipe can overlap the bird at a time"
            )
        if self.pipe_count * self.pipe_spacing < self.screen_width + self.pipe_width:
            raise ValueError(
                "pipe_count * pipe_spacing must cover screen_width + pipe_width so "
                "recycled pipes re-enter from off screen"
            )
        if self.reset_high < self.reset_low:
            raise ValueError("reset_high must be >= reset_low")
        if self.reset_low < 1 or self.reset_high + self.pipe_gap > self.ground_y - 1:
            raise ValueError("every gap must leave at least one pipe row above and below")

        # Table budget.
        if self.q_table_bytes > self.flash_budget_bytes:
            raise ValueError(
                f"Q table needs {self.q_table_bytes} bytes "
                f"({self.num_discrete_states} states) but flash_budget_bytes is "
                f"{self.flash_budget_bytes}; coarsen the bins or use a smaller dtype"
            )

    # ---- firmware export ---------------------------------------------------
    def to_c_header(self, guard: str = "FLAPPY_CONFIG_H", prefix: str = "FB_") -> str:
        """Render the firmware-facing constants as a C header.

        Only integers are emitted. The MCU never evaluates rewards or the float
        observation, and float literals could drift from the simulation.
        """
        rows: list[tuple[str, int, str]] = [
            ("TICK_HZ", self.tick_hz, "game loop rate"),
            ("TICK_PERIOD_US", self.tick_period_us, "timer period"),
            ("SCREEN_WIDTH", self.screen_width, "pixels"),
            ("SCREEN_HEIGHT", self.screen_height, "pixels"),
            ("PAGE_HEIGHT", self.page_height, "SSD1309 rows per page"),
            ("FRAMEBUFFER_PAGES", self.framebuffer_pages, ""),
            ("FRAMEBUFFER_BYTES", self.framebuffer_bytes, "one 1-bpp frame"),
            ("GROUND_Y", self.ground_y, "first ground row"),
            ("GROUND_HEIGHT", self.ground_height, "pixels"),
            ("PLAYER_X", self.player_x, "bird left edge"),
            ("PLAYER_WIDTH", self.player_width, ""),
            ("PLAYER_HEIGHT", self.player_height, ""),
            ("PLAYER_START_Y_PX", self.player_start_y_px, "top row at reset"),
            ("PLAYER_MIN_Y_PX", self.player_min_y_px, "highest reachable top row"),
            ("PLAYER_MAX_Y_PX", self.player_max_y_px, "top row when on the ground"),
            ("PLAYER_SPRITE_FRAMES", self.player_sprite_frames, ""),
            ("PLAYER_ANIM_TICKS", self.player_anim_ticks, "ticks per wing frame"),
            ("PIPE_WIDTH", self.pipe_width, ""),
            ("PIPE_GAP", self.pipe_gap, "opening height"),
            ("PIPE_COUNT", self.pipe_count, "pipe pairs kept alive"),
            ("PIPE_SPACING", self.pipe_spacing, "left edge to left edge"),
            ("PIPE_SPEED", self.pipe_speed, "pixels per tick, leftward"),
            ("PIPE_FIRST_X", self.pipe_first_x, "first pipe at reset"),
            ("PIPE_GAP_Y_MIN", self.reset_low, "inclusive top row of the opening"),
            ("PIPE_GAP_Y_MAX", self.reset_high, "inclusive; sample every integer pixel"),
            ("GRAVITY", self.gravity, "pixels / tick^2"),
            ("FLAP_VELOCITY", self.flap_velocity, "pixels / tick"),
            ("MAX_FALL_VELOCITY", self.max_fall_velocity, "pixels / tick"),
            ("FLAP_MIN_Y_PX", self.flap_min_y_px, "flap ignored at or above"),
            ("OBS_DX_MIN_PX", self.obs_dx_min_px, ""),
            ("OBS_DX_MAX_PX", self.obs_dx_max_px, ""),
            ("OBS_DX_BIN_PX", self.obs_dx_bin_px, ""),
            ("OBS_DY_MIN_PX", self.obs_dy_min_px, ""),
            ("OBS_DY_MAX_PX", self.obs_dy_max_px, ""),
            ("OBS_DY_BIN_PX", self.obs_dy_bin_px, ""),
            ("OBS_VY_MIN", self.flap_velocity, "pixels / tick"),
            ("OBS_VY_MAX", self.max_fall_velocity, "pixels / tick"),
            ("OBS_VY_BIN", self.obs_vy_bin, "pixels / tick per bin"),
            ("OBS_INCLUDE_NEXT_GAP", int(self.obs_include_next_gap), ""),
            ("OBS_DY2_BIN_PX", self.obs_dy2_bin_px, ""),
            ("N_DX", self.n_dx, ""),
            ("N_DY", self.n_dy, ""),
            ("N_VY", self.n_vy, ""),
            ("N_DY2", self.n_dy2, "1 when the 4th feature is off"),
            ("NUM_STATES", self.num_discrete_states, "table rows"),
            ("NUM_ACTIONS", self.num_actions, "0 idle, 1 flap"),
            ("Q_TABLE_BYTES", self.q_table_bytes, f"{self.q_table_dtype} Q-values"),
            ("POLICY_TABLE_BYTES", self.policy_table_bytes, "1 bit per state"),
        ]
        width = max(len(prefix + name) for name, _, _ in rows)
        lines = [
            "/* Generated by FlappyBirdEnvConfig.to_c_header(); edit envConfig.py instead. */",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
        ]
        for name, value, comment in rows:
            literal = f"({value})" if value < 0 else str(value)
            line = f"#define {prefix + name:<{width}} {literal}"
            lines.append(f"{line}  /* {comment} */" if comment else line)
        lines += ["", f"#endif /* {guard} */", ""]
        return "\n".join(lines)
