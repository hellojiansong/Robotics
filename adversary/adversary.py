"""Adversarial terrain agent.

Tracks recent rover failures and proposes terrain difficulty params
that target observed weaknesses, while staying solvable. Difficulty
ramps up over time but backs off when the rover is struggling.
"""

from collections import deque


# Difficulty bounds
DIFFICULTY_MIN = 0.0
DIFFICULTY_MAX = 1.0

# Target success rate window
SUCCESS_TARGET_LOW  = 0.30
SUCCESS_TARGET_HIGH = 0.65

# Per-episode difficulty step.
# STEP_DOWN is deliberately larger: backing off quickly when the rover
# struggles prevents catastrophic forgetting on hard terrain, while slow
# ramp-up avoids premature over-challenge.
STEP_UP   = 0.005
STEP_DOWN = 0.010


class WeaknessTracker:
    """Rolling failure-mode statistics over the last N episodes."""

    def __init__(self, window=50):
        self.window = window
        self.events = deque(maxlen=window)  # each: "success" | "fall" | "timeout"

    def log(self, outcome):
        assert outcome in ("success", "fall", "timeout")
        self.events.append(outcome)

    def rate(self, outcome):
        if not self.events:
            return 0.0
        return sum(1 for e in self.events if e == outcome) / len(self.events)

    @property
    def success_rate(self):
        return self.rate("success")

    @property
    def fall_rate(self):
        return self.rate("fall")

    @property
    def timeout_rate(self):
        return self.rate("timeout")


class AdversarialTerrainAgent:
    """Proposes terrain difficulty params from rover weakness signals."""

    def __init__(self, tracker=None, start_difficulty=0.1):
        self.tracker = tracker or WeaknessTracker()
        self.difficulty = float(start_difficulty)

    def update(self, outcome):
        """Call once per finished episode with 'success'|'fall'|'timeout'."""
        self.tracker.log(outcome)
        sr = self.tracker.success_rate
        if len(self.tracker.events) < 10:
            return  # not enough data

        if sr > SUCCESS_TARGET_HIGH:
            self.difficulty = min(DIFFICULTY_MAX, self.difficulty + STEP_UP)
        elif sr < SUCCESS_TARGET_LOW:
            self.difficulty = max(DIFFICULTY_MIN, self.difficulty - STEP_DOWN)

    def propose(self):
        """Return difficulty params for TerrainGenerator.generate()."""
        return self.build_params(
            self.difficulty,
            self.tracker.fall_rate,
            self.tracker.timeout_rate,
        )

    @staticmethod
    def build_params(difficulty, fall_rate=0.0, timeout_rate=0.0):
        """Convert a difficulty float + weakness rates into a terrain param dict.

        Separated from instance state so training callbacks can compute params
        without owning a tracker (e.g. when pushing difficulty to parallel envs).
        """
        d       = difficulty
        fall    = fall_rate
        timeout = timeout_rate

        # Base scaling — starts easy (0 obstacles, near-flat) and ramps up.
        # num_obstacles reaches 0 below d≈0.15 so early training is obstacle-free.
        num_obstacles = max(0, int(round(-6 + 40 * d)))  # 0..34
        max_size      = 0.20 + 0.35 * d           # 0.20..0.55 m
        max_height    = 0.20 + 0.40 * d           # 0.20..0.60 m
        friction      = 1.0 - 0.5 * d             # 1.0..0.5

        n_hfield_waves   = 4 + int(14 * d)        # 4..18 sine waves
        hfield_frequency = 1.5 + 3.5 * d          # 1.5..5.0 cycles per span
        hfield_amplitude = 0.05 + 1.05 * d        # 0.05..1.10 m (near-flat at d=0)

        n_craters = int(round(4 * d))             # 0..4 Gaussian depressions
        n_mesas   = int(round(3 * d))             # 0..3 logistic plateaux
        n_patches = int(round(4 + 8 * d))         # 4..12 coloured zone decals

        if fall > timeout and fall > 0.2:
            max_height       *= 1.25
            hfield_frequency *= 1.4
            n_mesas           = int(round(n_mesas * 1.5))
        elif timeout > 0.2:
            num_obstacles    = int(round(num_obstacles * 1.3))
            n_hfield_waves   = int(n_hfield_waves * 1.3)
            hfield_frequency *= 1.3
            n_craters        = int(round(n_craters * 1.5))

        return {
            "num_obstacles":    max(0, num_obstacles),
            "max_size":         max_size,
            "max_height":       max_height,
            "friction_scale":   friction,
            "n_hfield_waves":   n_hfield_waves,
            "hfield_frequency": hfield_frequency,
            "hfield_amplitude": hfield_amplitude,
            "n_craters":        max(0, n_craters),  # Bug5 fix: allow 0 at d=0 for flat-start
            "n_mesas":          max(0, n_mesas),    # Bug5 fix: same
            "n_patches":        n_patches,
        }
