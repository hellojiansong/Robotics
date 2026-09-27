import numpy as np


GOAL_REACH_DIST = 0.55  # metres

# Reward scale design — v11
#
# Goal to make ep_rew_mean unambiguous during training:
#   Max non-goal episode (perfect nav, all collectibles, 5000 steps):
#     velocity_to_goal: 2.0 × 2.4 m/s × 5000 = 24 000
#     collectibles:     3 × 8 000              = 24 000
#     progress:         3.0 × 40 m             =    120
#     time_penalty:    -2.0 × 5000             = -10 000
#     ─────────────────────────────────────────  ≈ 38 000  (absolute max)
#
#   Minimum goal episode (reach at step 5000):  100 000
#
#   ep_rew_mean diagnostic thresholds:
#     < -5 000  → rover is falling or going backwards
#     -5 000 to 0 → navigating but not reaching goal
#     > 0       → occasional goals
#     > 50 000  → reliably reaching goal (≥30 % success)
#     > 80 000  → strong policy (≥60 % success)
DEFAULT_WEIGHTS = {
    "goal_bonus":        100_000.0,   # was 25 000 — now clearly above all shaping
    "efficiency_bonus":   15_000.0,   # was 4 000  — strong fast-reach incentive
    "velocity_to_goal":      2.0,     # was 8.0    — pure guidance; max 24 000/ep
    "progress":              3.0,     # was 12.0   — bounded by geography (~120 max)
    "vel_to_collectible":    1.0,     # was 2.0    — small nudge only
    "time_penalty":         -2.0,     # unchanged
    "tilt_penalty":         -1.5,     # unchanged
    "spin_penalty":         -0.5,     # unchanged
    "action_smooth":        -0.3,     # unchanged
    # fall_penalty must be worse than the worst possible timeout so the rover
    # never learns to tip over deliberately to end the episode early.
    # Worst timeout ≈ -10 000; -20 000 is clearly worse.
    "fall_penalty":      -20_000.0,   # was -150 (way too small vs -10 000 timeout)
}


class RewardCalculator:

    TILT_WARN  = 0.45  # rad (~26 deg)
    TILT_FATAL = 0.95  # rad (~54 deg)

    def __init__(self, goal_pos, weights=None, max_episode_steps=5000):
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.w = dict(DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.max_episode_steps = max_episode_steps
        self._prev_dist   = None
        self._prev_action = None

    def reset(self):
        self._prev_dist   = None
        self._prev_action = None

    def compute(self, chassis_pos, chassis_quat, chassis_linvel, chassis_angvel,
                step_count, action=None, nearest_collectible=None):
        info = {"reached_goal": False, "fell_over": False, "reward_components": {}}
        components = info["reward_components"]

        pos_xy  = chassis_pos[:2]
        to_goal = self.goal_pos[:2] - pos_xy
        dist    = float(np.linalg.norm(to_goal))

        # Goal reached
        if dist < GOAL_REACH_DIST:
            goal_r = self.w["goal_bonus"]
            eff_r  = self.w["efficiency_bonus"] * max(0.0, 1.0 - step_count / self.max_episode_steps)
            info["reached_goal"] = True
            components["goal"]       = goal_r
            components["efficiency"] = eff_r
            return goal_r + eff_r, info

        reward = 0.0

        # --- Navigation rewards ---

        # Velocity toward goal — bidirectional: positive when closing, negative when retreating.
        # This directly penalises moving away from the goal, eliminating wandering.
        goal_dir = to_goal / (dist + 1e-6)
        vel_dot  = float(np.dot(chassis_linvel[:2], goal_dir))
        vel_r    = vel_dot * self.w["velocity_to_goal"]
        reward  += vel_r
        components["velocity_to_goal"] = vel_r

        # Progress (delta distance) — also bidirectional; penalises backing up
        if self._prev_dist is not None:
            prog_r = (self._prev_dist - dist) * self.w["progress"]
            reward += prog_r
            components["progress"] = prog_r
        self._prev_dist = dist

        # Time penalty — strong clock to prevent aimless roaming
        time_r = self.w["time_penalty"]
        reward += time_r
        components["time"] = time_r

        # --- Collectible seeking ---
        # Velocity toward the nearest collectible when one is still present.
        # Clipped to 0 so moving away gives 0, not a penalty.
        if nearest_collectible is not None:
            to_col  = nearest_collectible[:2] - pos_xy
            col_d   = float(np.linalg.norm(to_col))
            if col_d > 0.1:
                col_dir    = to_col / col_d
                vel_dot_c  = float(np.dot(chassis_linvel[:2], col_dir))
                col_vel_r  = max(0.0, vel_dot_c) * self.w["vel_to_collectible"]
                reward    += col_vel_r
                components["vel_to_collectible"] = col_vel_r

        # --- Stability penalties ---

        # Spin penalty: yaw rate → spinning in place wastes time and destabilises
        yaw_rate = float(chassis_angvel[2])
        spin_r   = abs(yaw_rate) * self.w["spin_penalty"]   # w is negative → subtracts
        reward  += spin_r
        components["spin"] = spin_r

        # Action smoothness: penalise large control changes between steps
        if action is not None:
            action_arr = np.asarray(action, dtype=np.float32)
            if self._prev_action is not None:
                delta    = float(np.linalg.norm(action_arr - self._prev_action))
                smooth_r = delta * self.w["action_smooth"]   # w is negative → subtracts
                reward  += smooth_r
                components["action_smooth"] = smooth_r
            self._prev_action = action_arr

        # Tilt / fall
        roll, pitch = _quat_to_roll_pitch(chassis_quat)
        max_tilt = max(abs(roll), abs(pitch))

        if max_tilt >= self.TILT_FATAL:
            fall_r = self.w["fall_penalty"]
            reward += fall_r
            info["fell_over"] = True
            components["fell"] = fall_r
            return reward, info

        if max_tilt > self.TILT_WARN:
            severity = (max_tilt - self.TILT_WARN) / (self.TILT_FATAL - self.TILT_WARN)
            tilt_r   = self.w["tilt_penalty"] * severity
            reward  += tilt_r
            components["tilt"] = tilt_r

        return reward, info


def _quat_to_roll_pitch(quat):
    # MuJoCo quat is [w, x, y, z]
    w, x, y, z = (float(v) for v in quat)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    return roll, pitch


