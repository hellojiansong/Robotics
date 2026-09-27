import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from training.reward import RewardCalculator
from env.terrain_generator import TerrainGenerator, COLLECT_DIST, N_COLLECTIBLES
from adversary.adversary import AdversarialTerrainAgent


XML_PATH     = os.path.join(os.path.dirname(__file__), '..', 'robot', 'robot.xml')
DEFAULT_GOAL = np.array([40.0, 0.0, 0.0], dtype=np.float32)

# 4 substeps * 0.002s = 0.008s per RL step
PHYSICS_SUBSTEPS = 4

# Must match ctrlrange in robot.xml
MAX_WHEEL_VEL = 20.0

# Reward for picking up one collectible sphere.
# 8000 makes it worth a ~3m detour at the current shaping scale.
COLLECTIBLE_REWARD = 8000.0

# Normalisation scales for position observations
OBS_DIST_SCALE = 40.0   # world x-span (m)
OBS_Y_SCALE    = 7.0    # corridor half-width (m)
OBS_Z_SCALE    = 2.0    # max expected chassis height (m)

# Lidar raycasts: 9 rays from -90° (right) to +90° (left), horizontal
N_RAYS       = 9
RAY_MAX_DIST = 5.0
RAY_ANGLES   = np.linspace(-np.pi / 2, np.pi / 2, N_RAYS)

# Observation layout (32-D):
# [0:3]   pos_norm         — [x/40, y/7, z/2]
# [3:7]   quat             — [w, x, y, z]
# [7:10]  linvel_body      — linear velocity in body frame
# [10:13] angvel_body      — angular velocity in body frame
# [13:16] goal_vec         — [dx_norm, dy_norm, dist/40]
# [16:21] collectible_info — [dx_norm, dy_norm, dist/40, n_remaining/N, time_frac]
# [21:30] raycasts         — 9 rays normalised to [0, 1]
# [30:32] prev_action      — last applied [left, right] in [-1, 1]
OBS_DIM = 32


class RoverEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        xml_path=XML_PATH,
        goal_pos=DEFAULT_GOAL,
        max_episode_steps=5000,
        reward_weights=None,
        adversary=None,
        enable_obstacles=False,
    ):
        super().__init__()

        self.render_mode       = render_mode
        self.xml_path          = xml_path
        self.goal_pos          = goal_pos.copy()
        self.max_episode_steps = max_episode_steps
        self.enable_obstacles  = enable_obstacles

        self.adversary    = adversary
        self._terrain_gen = TerrainGenerator()

        # Externally-pushed difficulty params from the centralised adversary
        # callback (train.py). Overrides self.adversary when set.
        self._difficulty_params = None

        # Start with the base model; first reset() rebuilds from generated XML
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.reward_calc  = RewardCalculator(
            goal_pos=self.goal_pos,
            weights=reward_weights,
            max_episode_steps=self.max_episode_steps,
        )
        self._collectibles  = []   # list of (geom_idx, np.array([x, y, z]))
        self._terrain_spec  = None
        self._terrain_seed  = None
        self._prev_action_obs = np.zeros(2, dtype=np.float32)
        self._last_raycasts = np.ones(N_RAYS, dtype=np.float32)  # Bug1 fix: cache to avoid double computation

        # Actions in [-1, 1], scaled to rad/s in _apply_action
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )

        self._step_count = 0

    # ------------------------------------------------------------------
    # Public API used by training callbacks
    # ------------------------------------------------------------------

    def set_difficulty_params(self, params):
        """Receive terrain params from the centralised adversary callback."""
        self._difficulty_params = params

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._regenerate_terrain(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._randomize_spawn()
        self._step_count = 0
        self._prev_action_obs = np.zeros(2, dtype=np.float32)
        self.reward_calc.reset()
        mujoco.mj_forward(self.model, self.data)
        self._last_raycasts = self._compute_raycasts()  # Bug1 fix: compute once at reset
        return self._get_obs(), {}

    def _randomize_spawn(self):
        jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        qadr = self.model.jnt_qposadr[jid]
        x = float(self.np_random.uniform(-2.0, 2.0))
        y = float(self.np_random.uniform(-1.0, 1.0))
        self.data.qpos[qadr + 0] = x
        self.data.qpos[qadr + 1] = y
        # Spawn facing toward the goal (±60°) so the policy always starts with the
        # goal roughly ahead. Physical forward = -body_x, so body +x must point
        # opposite to goal, i.e. yaw = goal_dir + π ± noise.
        goal_dir = np.arctan2(self.goal_pos[1] - y, self.goal_pos[0] - x)
        yaw = goal_dir + np.pi + float(self.np_random.uniform(-np.pi / 3, np.pi / 3))
        self.data.qpos[qadr + 3] = float(np.cos(yaw / 2))  # qw
        self.data.qpos[qadr + 4] = 0.0                      # qx
        self.data.qpos[qadr + 5] = 0.0                      # qy
        self.data.qpos[qadr + 6] = float(np.sin(yaw / 2))  # qz

    def _regenerate_terrain(self, seed):
        # Use the env's own RNG to derive a terrain seed when none is supplied.
        # self.np_random is seeded differently per parallel env (via seed+rank in
        # make_env), so episodes across envs will never share terrain seeds.
        if seed is not None:
            terrain_seed = int(seed)
        else:
            terrain_seed = int(self.np_random.integers(0, 2**31 - 1))

        # Priority: externally pushed params > local adversary > defaults
        if self._difficulty_params is not None:
            difficulty = self._difficulty_params
        elif self.adversary is not None:
            difficulty = self.adversary.propose()
        else:
            difficulty = {}

        spec = self._terrain_gen.generate(
            seed=terrain_seed,
            difficulty=difficulty,
            enable_obstacles=self.enable_obstacles,
        )
        self._terrain_spec = spec
        self._terrain_seed = terrain_seed
        xml = self._terrain_gen.build_xml(spec)
        self.model = mujoco.MjModel.from_xml_string(xml)

        if spec.hfield_data is not None:
            hfid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
            if hfid >= 0:
                adr  = int(self.model.hfield_adr[hfid])
                nrow = int(self.model.hfield_nrow[hfid])
                ncol = int(self.model.hfield_ncol[hfid])
                self.model.hfield_data[adr : adr + nrow * ncol] = spec.hfield_data.flatten()

        self.data     = mujoco.MjData(self.model)
        self.goal_pos = spec.goal_pos
        self.reward_calc = RewardCalculator(
            goal_pos=self.goal_pos,
            weights=self.reward_calc.w,
            max_episode_steps=self.max_episode_steps,
        )
        if self._renderer is not None:
            del self._renderer
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Collectibles store 3-D positions [x, y, z]
        self._collectibles = [
            (i, cpos) for i, cpos in enumerate(spec.collectibles)
        ]

    def step(self, action):
        left, right = np.clip(action, -1.0, 1.0)
        self._apply_action(float(left), float(right))

        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # --- Collectible detection (XY-only so it works regardless of terrain z) ---
        collectible_reward = 0.0
        pos_xy = self._sensor("chassis_pos")[:2]
        remaining = []
        for (idx, cpos) in self._collectibles:
            if np.linalg.norm(cpos[:2] - pos_xy) < COLLECT_DIST:
                collectible_reward += COLLECTIBLE_REWARD
                self._hide_collectible(idx)
            else:
                remaining.append((idx, cpos))
        self._collectibles = remaining

        current_raycasts = self._compute_raycasts()
        self._last_raycasts = current_raycasts  # Bug1 fix: cache so _get_obs doesn't recompute
        nearest_col = min(self._collectibles, key=lambda x: np.linalg.norm(x[1][:2] - pos_xy))[1] if self._collectibles else None  # Bug2 fix: use nearest, not first

        reward, reward_info = self.reward_calc.compute(
            chassis_pos=self._sensor("chassis_pos"),
            chassis_quat=self._sensor("chassis_quat"),
            chassis_linvel=self._sensor("chassis_linvel"),
            chassis_angvel=self._sensor("chassis_angvel"),
            step_count=self._step_count,
            action=np.array([left, right], dtype=np.float32),
            nearest_collectible=nearest_col,
        )

        reward += collectible_reward
        if collectible_reward > 0:
            reward_info["reward_components"]["collectible"] = collectible_reward

        terminated = reward_info["fell_over"] or reward_info["reached_goal"]
        truncated  = self._step_count >= self.max_episode_steps

        if self.adversary is not None and (terminated or truncated):
            if reward_info["reached_goal"]:
                outcome = "success"
            elif reward_info["fell_over"]:
                outcome = "fall"
            else:
                outcome = "timeout"
            self.adversary.update(outcome)
            reward_info["adversary_difficulty"] = self.adversary.difficulty

        # Update prev_action for next observation
        self._prev_action_obs = np.array([left, right], dtype=np.float32)

        return self._get_obs(), reward, terminated, truncated, reward_info

    def render(self):
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera="robot_pov")
        return self._renderer.render()

    def render_external(self, distance=5.0, elevation=-28.0, azimuth=125.0):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        cam.distance  = distance
        cam.elevation = elevation
        cam.azimuth   = azimuth

        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    def _apply_action(self, left, right):
        # ctrl order: motor_fl, motor_fr, motor_rl, motor_rr
        # Negated: body +x = physical rear, so -ctrl makes [+1,+1] = physically forward
        self.data.ctrl[0] = -left  * MAX_WHEEL_VEL
        self.data.ctrl[1] = -right * MAX_WHEEL_VEL
        self.data.ctrl[2] = -left  * MAX_WHEEL_VEL
        self.data.ctrl[3] = -right * MAX_WHEEL_VEL

    def _hide_collectible(self, idx):
        """Move a collected geom 10 m underground so it disappears visually."""
        geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, f"collectible_{idx}"
        )
        if geom_id >= 0:
            self.model.geom_pos[geom_id, 2] = -10.0

    def _compute_raycasts(self):
        """Cast N_RAYS horizontal rays using one mj_multiRay call (no Python loop)."""
        quat = self._sensor("chassis_quat")
        pos  = self._sensor("chassis_pos")

        mat9 = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat9, quat.astype(np.float64))
        body_mat = mat9.reshape(3, 3)

        forward = -body_mat[:, 0]   # negate: physical forward = -body_x
        left    =  body_mat[:, 1]

        # Build all (N_RAYS, 3) directions in one vectorised step
        cos_a = np.cos(RAY_ANGLES)
        sin_a = np.sin(RAY_ANGLES)
        vecs  = cos_a[:, None] * forward[None, :] + sin_a[:, None] * left[None, :]
        vecs[:, 2] = 0.0
        xy_norms = np.maximum(np.linalg.norm(vecs[:, :2], axis=1, keepdims=True), 1e-6)
        vecs[:, :2] /= xy_norms

        ray_origin = np.array([pos[0], pos[1], pos[2] + 0.20], dtype=np.float64)
        geomids  = np.full(N_RAYS, -1, dtype=np.int32)
        raw_dist = np.full(N_RAYS, -1.0, dtype=np.float64)

        # Single C-level call for all rays — eliminates the Python-loop overhead
        # normal=None: we don't need hit surface normals
        mujoco.mj_multiRay(
            self.model, self.data,
            ray_origin, vecs.flatten(),
            None, 1, -1,
            geomids, raw_dist, None, N_RAYS, RAY_MAX_DIST,
        )

        return np.where(
            (raw_dist < 0) | (raw_dist > RAY_MAX_DIST),
            1.0, raw_dist / RAY_MAX_DIST,
        ).astype(np.float32)

    def _get_obs(self):
        pos    = self._sensor("chassis_pos")
        quat   = self._sensor("chassis_quat")    # [w, x, y, z]
        linvel = self._sensor("chassis_linvel")  # world frame
        angvel = self._sensor("chassis_angvel")  # world frame

        # --- Position normalised to ~[-1, 1] ---
        pos_norm = np.array([
            pos[0] / OBS_DIST_SCALE,
            pos[1] / OBS_Y_SCALE,
            pos[2] / OBS_Z_SCALE,
        ], dtype=np.float32)

        # --- Body-frame velocities (feature 11) ---
        # body_mat[:, 0] = physical rear of rover (MuJoCo body +x = rear).
        # Negate the x component so that +linvel_body[0] = moving physically forward,
        # consistent with the negated _apply_action and the corrected goal_vec below.
        mat9 = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat9, quat.astype(np.float64))
        body_mat = mat9.reshape(3, 3)
        _lv = body_mat.T @ linvel
        _av = body_mat.T @ angvel
        linvel_body = np.array([-_lv[0],  _lv[1],  _lv[2]], dtype=np.float32)
        angvel_body = np.array([-_av[0],  _av[1],  _av[2]], dtype=np.float32)

        # --- Goal vector in BODY FRAME (physical-forward convention) ---
        # goal_vec[0] > 0: goal ahead   goal_vec[0] < 0: goal behind
        # goal_vec[1] > 0: goal right   goal_vec[1] < 0: goal left
        # (body[:,1] = physical right, so body-frame y > 0 = goal to the right)
        # Negate x because body +x = physical rear; -x = physical forward.
        to_goal    = self.goal_pos[:2] - pos[:2]
        dist       = float(np.linalg.norm(to_goal)) + 1e-6
        to_goal_3d = np.array([to_goal[0], to_goal[1], 0.0], dtype=np.float64)
        goal_body  = body_mat.T @ to_goal_3d
        goal_vec   = np.array([-goal_body[0] / dist, goal_body[1] / dist,
                               dist / OBS_DIST_SCALE], dtype=np.float32)

        # --- Collectible info (direction in body frame, consistent with goal_vec) ---
        time_frac = float(self._step_count) / float(self.max_episode_steps)
        n_remaining_frac = float(len(self._collectibles)) / float(N_COLLECTIBLES)

        if self._collectibles:
            nearest_pos = min(self._collectibles,
                              key=lambda x: np.linalg.norm(x[1][:2] - pos[:2]))[1]
            to_col    = nearest_pos[:2] - pos[:2]
            col_d     = float(np.linalg.norm(to_col)) + 1e-6
            to_col_3d = np.array([to_col[0], to_col[1], 0.0], dtype=np.float64)
            col_body  = body_mat.T @ to_col_3d
            collectible_info = np.array([
                -col_body[0] / col_d,   # negate x: +x = physical forward
                col_body[1] / col_d,
                col_d / OBS_DIST_SCALE,
                n_remaining_frac,
                time_frac,
            ], dtype=np.float32)
        else:
            collectible_info = np.array([
                0.0, 0.0, 1.0,
                0.0,
                time_frac,
            ], dtype=np.float32)

        # --- Raycasts (feature 1) ---
        raycasts = self._last_raycasts  # Bug1 fix: reuse cached value from step()

        return np.concatenate([
            pos_norm,          # 3
            quat,              # 4
            linvel_body,       # 3
            angvel_body,       # 3
            goal_vec,          # 3
            collectible_info,  # 5
            raycasts,          # 9
            self._prev_action_obs,  # 2
        ]).astype(np.float32)  # total 32

    def _sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr : adr + dim].copy()
