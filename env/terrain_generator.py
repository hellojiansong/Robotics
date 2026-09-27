
"""Procedural terrain generator for the rover environment.

Heightfield uses three bands of sine waves (large hills, cross-path ridges,
surface roughness) plus Gaussian crater depressions and logistic-edge mesa
plateaux so the landscape has strong structural variety at all difficulties.

Non-colliding coloured zone patches (mud/ice/sand/rock/moss) give visual
variety on top of the height variation.  Obstacle types include boxes,
boulders, tilted ramps, partial-barrier walls, and tall pillars.
"""

import os
import numpy as np


BASE_XML_PATH = os.path.join(os.path.dirname(__file__), "..", "robot", "robot.xml")

# 40 m world
CORRIDOR_X   = (2.0, 38.0)
CORRIDOR_Y   = (-7.0, 7.0)
START_POS    = np.array([0.0, 0.0])
GOAL_POS     = np.array([40.0, 0.0])  # fallback only; goal is randomised per episode
CLEAR_RADIUS = 2.5

# Randomised goal position range
GOAL_X_RANGE = (36.0, 42.0)
GOAL_Y_RANGE = (-4.0, 4.0)

# Three collectible spheres scattered along the corridor
N_COLLECTIBLES        = 3
COLLECTIBLE_RADIUS    = 0.45
COLLECTIBLE_Z_MIN     = 0.90   # minimum z — raised further if terrain is higher
COLLECTIBLE_Z_MARGIN  = 0.10   # gap between terrain surface and sphere bottom
COLLECT_DIST          = 1.20

# Heightfield covers world X: -2..42, Y: -9..9  (geom centred at x=20)
HFIELD_NROW   = 128
HFIELD_NCOL   = 256
HFIELD_X_HALF = 22.0
HFIELD_Y_HALF = 9.0
HFIELD_GEOM_X = 20.0
HFIELD_BASE   = 0.1

# Coloured visual zone patches (non-colliding ground decals)
PATCH_DEFS = [
    {"name": "mud",  "rgba": "0.42 0.28 0.12 0.90"},
    {"name": "ice",  "rgba": "0.72 0.88 0.96 0.82"},
    {"name": "sand", "rgba": "0.88 0.76 0.44 0.88"},
    {"name": "rock", "rgba": "0.46 0.46 0.50 0.92"},
    {"name": "moss", "rgba": "0.28 0.52 0.22 0.88"},
]


class TerrainSpec:
    """Plain container describing one generated terrain."""

    def __init__(self, obstacles, friction, goal_pos=GOAL_POS, collectibles=None,
                 hfield_data=None, hfield_z_scale=0.0, patches=None, goal_terrain_z=0.0):
        self.obstacles      = obstacles
        self.friction       = friction
        self.goal_pos       = np.array([goal_pos[0], goal_pos[1], 0.0], dtype=np.float32)
        self.collectibles   = collectibles if collectibles is not None else []
        self.hfield_data    = hfield_data      # float32 (nrow, ncol) in [0,1], or None
        self.hfield_z_scale = hfield_z_scale   # metres: data * z_scale = actual height
        self.goal_terrain_z = float(goal_terrain_z)  # terrain height at goal XY
        self.patches        = patches if patches is not None else []


class TerrainGenerator:

    def __init__(self, base_xml_path=BASE_XML_PATH):
        with open(base_xml_path, "r") as f:
            self._base_xml = f.read()

    def generate(self, seed=0, difficulty=None, enable_obstacles=True):
        """Build a TerrainSpec from difficulty params.

        difficulty keys (all optional):
          num_obstacles      int   obstacle count
          max_size           float max obstacle half-extent (m)
          max_height         float max obstacle height (m)
          friction_scale     float global ground friction multiplier [0.3, 1.0]
          n_hfield_waves     int   total sine waves
          hfield_frequency   float frequency scale (cycles per normalised span)
          hfield_amplitude   float max terrain height (m)
          n_craters          int   number of Gaussian crater depressions
          n_mesas            int   number of logistic-edge plateau mesas
          n_patches          int   number of coloured ground-zone decals
        """
        rng = np.random.default_rng(seed)
        d   = self._normalize_difficulty(difficulty or {})

        # Randomise goal position per episode
        goal_x = float(rng.uniform(*GOAL_X_RANGE))
        goal_y = float(rng.uniform(*GOAL_Y_RANGE))
        goal_pos = np.array([goal_x, goal_y])

        obstacles = []
        if enable_obstacles:
            attempts, max_attempts = 0, d["num_obstacles"] * 30
            while len(obstacles) < d["num_obstacles"] and attempts < max_attempts:
                attempts += 1
                cand = self._sample_obstacle(rng, d)
                if self._is_valid(cand, obstacles, goal_pos):
                    obstacles.append(cand)

        friction = float(np.clip(d["friction_scale"], 0.3, 1.0))

        # Generate heightfield BEFORE collectibles so collectible z can be raised
        # above any hills at its XY location.
        hfield_data, hfield_z_scale = None, 0.0
        patches = []
        if enable_obstacles:
            hfield_data, hfield_z_scale = self._generate_hfield(rng, d, goal_x)
            patches = self._generate_patches(rng, d["n_patches"])

        collectibles = self._generate_collectibles(rng, obstacles, hfield_data, hfield_z_scale, goal_pos)

        goal_terrain_z = 0.0
        if hfield_data is not None:
            goal_terrain_z = self._sample_hfield_height(hfield_data, hfield_z_scale,
                                                        goal_pos[0], goal_pos[1])

        return TerrainSpec(
            obstacles=obstacles,
            friction=friction,
            goal_pos=goal_pos,
            collectibles=collectibles,
            hfield_data=hfield_data,
            hfield_z_scale=hfield_z_scale,
            patches=patches,
            goal_terrain_z=goal_terrain_z,
        )

    # ------------------------------------------------------------------
    # Heightfield
    # ------------------------------------------------------------------

    def _generate_hfield(self, rng, d, goal_x=40.0):
        """Three-band sine waves + Gaussian craters + logistic mesas.

        Band 1  large rolling hills    (low freq, weight 1.0)
        Band 2  cross-path ridges      (low X / high Y freq, weight 0.7)
        Band 3  surface roughness      (high freq, weight 0.30)
        Plus:   n_craters Gaussian depressions, n_mesas logistic plateaux.
        """
        n_waves = int(d["n_hfield_waves"])
        freq    = float(d["hfield_frequency"])
        amp     = float(d["hfield_amplitude"])

        world_x = np.linspace(HFIELD_GEOM_X - HFIELD_X_HALF,
                               HFIELD_GEOM_X + HFIELD_X_HALF, HFIELD_NCOL)
        world_y = np.linspace(-HFIELD_Y_HALF, HFIELD_Y_HALF, HFIELD_NROW)
        xn = (world_x - world_x.min()) / (world_x.max() - world_x.min())
        yn = (world_y - world_y.min()) / (world_y.max() - world_y.min())
        XN, YN = np.meshgrid(xn, yn)   # (nrow, ncol)

        raw    = np.zeros((HFIELD_NROW, HFIELD_NCOL), dtype=np.float64)
        n_low  = max(2, n_waves // 3)
        n_rdg  = max(1, n_waves // 5)
        n_high = n_waves - n_low - n_rdg

        for _ in range(n_low):
            fx = rng.uniform(1.0, max(1.1, freq * 0.55))
            fy = rng.uniform(0.6, max(0.7, freq * 0.45))
            px, py = rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
            raw += np.sin(2*np.pi*fx*XN + px) * np.cos(2*np.pi*fy*YN + py)

        for _ in range(n_rdg):
            fx = rng.uniform(0.8, max(1.0, freq * 0.4))
            fy = rng.uniform(freq * 0.6, max(freq * 0.7, freq * 1.2))
            px, py = rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
            raw += 0.7 * np.sin(2*np.pi*fx*XN + px) * np.cos(2*np.pi*fy*YN + py)

        for _ in range(n_high):
            fx = rng.uniform(freq * 0.9, freq * 2.0)
            fy = rng.uniform(freq * 0.4, max(0.5, freq * 0.9))
            px, py = rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)
            raw += 0.30 * np.sin(2*np.pi*fx*XN + px) * np.cos(2*np.pi*fy*YN + py)

        # Gaussian crater depressions
        for _ in range(int(d["n_craters"])):
            cx    = rng.uniform(0.05, 0.95)
            cy    = rng.uniform(0.05, 0.95)
            r     = rng.uniform(0.04, 0.14)
            depth = rng.uniform(2.5, 6.0)
            dist  = np.sqrt((XN - cx)**2 + (YN - cy)**2)
            raw  -= depth * np.exp(-0.5 * (dist / r) ** 2)

        # Logistic-edge mesa plateaux (flat top, steep cliff sides)
        for _ in range(int(d["n_mesas"])):
            cx   = rng.uniform(0.08, 0.92)
            cy   = rng.uniform(0.08, 0.92)
            r    = rng.uniform(0.05, 0.13)
            h    = rng.uniform(2.5, 5.0)
            dist = np.sqrt((XN - cx)**2 + (YN - cy)**2)
            raw += h / (1.0 + np.exp(45.0 * (dist - r)))

        h_min, h_max = raw.min(), raw.max()
        heights = (raw - h_min) / (h_max - h_min) if h_max > h_min else np.full_like(raw, 0.5)

        # Taper flat only within 2 m of spawn and goal — no Y taper.
        tx_start = np.clip(world_x / 2.0, 0.0, 1.0)
        tx_goal  = np.clip((goal_x - world_x) / 2.0, 0.0, 1.0)
        heights *= (tx_start * tx_goal)[np.newaxis, :]

        return heights.astype(np.float32), float(amp)

    # ------------------------------------------------------------------
    # Coloured zone patches
    # ------------------------------------------------------------------

    def _generate_patches(self, rng, n):
        """Return n non-colliding coloured ground-decal slabs."""
        patches = []
        for _ in range(n):
            pdef = PATCH_DEFS[int(rng.integers(len(PATCH_DEFS)))]
            x  = rng.uniform(CORRIDOR_X[0] + 1.5, CORRIDOR_X[1] - 1.5)
            y  = rng.uniform(CORRIDOR_Y[0] + 0.5, CORRIDOR_Y[1] - 0.5)
            rx = rng.uniform(1.0, 3.2)
            ry = rng.uniform(1.0, 3.2)
            ez = rng.uniform(0.0, np.pi / 2)
            patches.append({
                "name":    pdef["name"],
                "rgba":    pdef["rgba"],
                "pos":     (float(x), float(y)),
                "size":    (float(rx), float(ry)),
                "euler_z": float(ez),
            })
        return patches

    # ------------------------------------------------------------------
    # Collectible
    # ------------------------------------------------------------------

    def _generate_collectibles(self, rng, obstacles, hfield_data, hfield_z_scale, goal_pos=None):
        """Place N_COLLECTIBLES spheres above terrain and clear of obstacles."""
        if goal_pos is None:
            goal_pos = GOAL_POS
        mid_x = float(goal_pos[0]) * 0.5
        cols = []
        for _ in range(N_COLLECTIBLES):
            placed = False
            for _ in range(50):
                x = mid_x + rng.uniform(-6.0, 6.0)
                y = rng.uniform(-5.0, 5.0)
                if not self._collectible_blocked(x, y, obstacles):
                    placed = True
                    break
            if not placed:
                # Bug6 fix: all 50 attempts were blocked; fall back to full corridor ignoring obstacles
                x = float(rng.uniform(CORRIDOR_X[0] + 1.0, CORRIDOR_X[1] - 1.0))
                y = float(rng.uniform(CORRIDOR_Y[0] + 1.0, CORRIDOR_Y[1] - 1.0))

            if hfield_data is not None:
                terrain_z = self._sample_hfield_height(hfield_data, hfield_z_scale, x, y)
                z = max(COLLECTIBLE_Z_MIN, terrain_z + COLLECTIBLE_RADIUS + COLLECTIBLE_Z_MARGIN)
            else:
                z = COLLECTIBLE_Z_MIN

            cols.append(np.array([float(x), float(y), float(z)], dtype=np.float32))
        return cols

    def _collectible_blocked(self, x, y, obstacles):
        """True if (x, y) is too close to any obstacle to be reachable."""
        for ob in obstacles:
            ox, oy = ob["pos"][0], ob["pos"][1]
            if np.linalg.norm([x - ox, y - oy]) < self._footprint_radius(ob) + COLLECT_DIST:
                return True
        return False

    def _sample_hfield_height(self, hfield_data, hfield_z_scale, x, y):
        """Bilinear-ish lookup of terrain height (m) at world position (x, y)."""
        x_min = HFIELD_GEOM_X - HFIELD_X_HALF
        x_max = HFIELD_GEOM_X + HFIELD_X_HALF
        xn = float(np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0))
        yn = float(np.clip((y + HFIELD_Y_HALF) / (2.0 * HFIELD_Y_HALF), 0.0, 1.0))
        col = int(xn * (HFIELD_NCOL - 1))
        row = int(yn * (HFIELD_NROW - 1))
        return float(hfield_data[row, col]) * hfield_z_scale

    # ------------------------------------------------------------------
    # XML building
    # ------------------------------------------------------------------

    def build_xml(self, spec):
        """Inject heightfield, patches, obstacles, collectibles, and goal into base XML."""
        geoms = []

        if spec.hfield_data is not None:
            geoms.append(_hfield_geom_xml(spec.friction))

        # Inject goal marker raised above the terrain surface at its XY location
        geoms += _goal_xml(spec.goal_pos, spec.goal_terrain_z)

        for i, patch in enumerate(spec.patches):
            geoms.append(_patch_xml(i, patch))
        for i, ob in enumerate(spec.obstacles):
            geoms.append(_obstacle_xml(i, ob))
        for i, cpos in enumerate(spec.collectibles):
            geoms.append(_collectible_xml(i, cpos))

        ground_friction = f"{spec.friction:.3f} 0.005 0.0001"
        xml = self._base_xml.replace(
            'friction="1.0 0.005 0.0001"',
            f'friction="{ground_friction}"',
            1,
        )

        if spec.hfield_data is not None:
            hfield_asset = (
                f'<hfield name="terrain" '
                f'nrow="{HFIELD_NROW}" ncol="{HFIELD_NCOL}" '
                f'size="{HFIELD_X_HALF:.3f} {HFIELD_Y_HALF:.3f} '
                f'{spec.hfield_z_scale:.3f} {HFIELD_BASE:.3f}"/>'
            )
            xml = xml.replace("</asset>", f"\n    {hfield_asset}\n  </asset>")

        injection = "\n    " + "\n    ".join(geoms) + "\n"
        xml = xml.replace("</worldbody>", injection + "  </worldbody>")
        return xml

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_difficulty(self, d):
        return {
            "num_obstacles":    int(d.get("num_obstacles",    10)),
            "max_size":         float(d.get("max_size",       0.35)),
            "max_height":       float(d.get("max_height",     0.35)),
            "friction_scale":   float(d.get("friction_scale", 1.0)),
            "n_hfield_waves":   int(d.get("n_hfield_waves",   14)),
            "hfield_frequency": float(d.get("hfield_frequency", 4.5)),
            "hfield_amplitude": float(d.get("hfield_amplitude", 0.85)),
            "n_craters":        int(d.get("n_craters",          2)),
            "n_mesas":          int(d.get("n_mesas",            2)),
            "n_patches":        int(d.get("n_patches",          6)),
        }

    def _sample_obstacle(self, rng, d):
        kind   = rng.choice(["box", "sphere", "ramp", "wall", "pillar"])
        x      = rng.uniform(*CORRIDOR_X)
        y      = rng.uniform(*CORRIDOR_Y)
        size   = rng.uniform(0.15, max(0.20, d["max_size"]))
        height = rng.uniform(0.25, max(0.30, d["max_height"]))

        if kind == "sphere":
            return {"type": "sphere", "pos": (x, y, size), "size": (size,)}

        if kind == "ramp":
            return {
                "type":  "box",
                "pos":   (x, y, height / 2),
                "size":  (size * 1.2, size * 1.2, height / 2),
                "euler": (0.0, rng.uniform(-0.40, 0.40), 0.0),
            }

        if kind == "wall":
            # Long thin partial barrier — rover must route around it
            half_len = rng.uniform(1.5, 4.0)
            return {
                "type":  "box",
                "pos":   (x, y, height / 2),
                "size":  (size * 0.22, half_len, height / 2),
                "euler": (0.0, 0.0, rng.uniform(-np.pi / 5, np.pi / 5)),
            }

        if kind == "pillar":
            r = rng.uniform(0.07, max(0.10, size * 0.45))
            h = rng.uniform(0.50, max(0.55, height * 2.5))
            return {"type": "cylinder", "pos": (x, y, h / 2), "size": (r, h / 2)}

        # box
        return {"type": "box", "pos": (x, y, height / 2), "size": (size, size, height / 2)}

    def _footprint_radius(self, ob):
        """Lateral (XY-plane) half-extent of an obstacle for clearance checks."""
        if ob["type"] in ("cylinder", "sphere"):
            return ob["size"][0]  # cylinder/sphere radius
        return max(ob["size"][0], ob["size"][1])  # max XY half-extent for boxes

    def _is_valid(self, cand, existing, goal_pos=None):
        if goal_pos is None:
            goal_pos = GOAL_POS
        cx, cy = cand["pos"][0], cand["pos"][1]
        radius = self._footprint_radius(cand)
        if np.linalg.norm([cx - START_POS[0], cy - START_POS[1]]) < CLEAR_RADIUS + radius:
            return False
        if np.linalg.norm([cx - goal_pos[0],  cy - goal_pos[1]])  < CLEAR_RADIUS + radius:
            return False
        for other in existing:
            ox, oy  = other["pos"][0], other["pos"][1]
            min_sep = radius + self._footprint_radius(other) + 0.20
            if np.linalg.norm([cx - ox, cy - oy]) < min_sep:
                return False
        return True


# ------------------------------------------------------------------
# XML fragment helpers
# ------------------------------------------------------------------

def _goal_xml(goal_pos, terrain_z=0.0):
    """Two stacked cylinders raised above the terrain surface at the goal location."""
    x, y = float(goal_pos[0]), float(goal_pos[1])
    z = float(terrain_z) + 0.013   # sit just above whatever terrain is underneath
    return [
        (f'<geom name="goal_ring" type="cylinder" size="0.55 0.012" '
         f'pos="{x:.3f} {y:.3f} {z:.3f}" '
         f'material="goal_outer_mat" contype="0" conaffinity="0"/>'),
        (f'<geom name="goal_dot" type="cylinder" size="0.12 0.018" '
         f'pos="{x:.3f} {y:.3f} {z:.3f}" '
         f'material="goal_inner_mat" contype="0" conaffinity="0"/>'),
    ]


def _hfield_geom_xml(friction):
    return (
        f'<geom name="hfield_terrain" type="hfield" hfield="terrain" '
        f'pos="{HFIELD_GEOM_X:.3f} 0.000 0.000" '
        f'rgba="0.28 0.52 0.18 1" '
        f'friction="{friction:.3f} 0.005 0.0001"/>'
    )


def _patch_xml(idx, patch):
    x, y   = patch["pos"]
    rx, ry = patch["size"]
    return (
        f'<geom name="patch_{idx}" type="box" '
        f'size="{rx:.3f} {ry:.3f} 0.008" '
        f'pos="{x:.3f} {y:.3f} 0.010" '
        f'euler="0.000 0.000 {patch["euler_z"]:.3f}" '
        f'rgba="{patch["rgba"]}" '
        f'contype="0" conaffinity="0"/>'
    )


def _obstacle_xml(idx, ob):
    pos   = " ".join(f"{v:.3f}" for v in ob["pos"])
    size  = " ".join(f"{v:.3f}" for v in ob["size"])
    euler = ""
    if "euler" in ob:
        euler = f' euler="{ob["euler"][0]:.3f} {ob["euler"][1]:.3f} {ob["euler"][2]:.3f}"'
    return (
        f'<geom name="obstacle_{idx}" type="{ob["type"]}" '
        f'size="{size}" pos="{pos}"{euler} '
        f'rgba="0.90 0.65 0.30 1" friction="1.0 0.005 0.0001"/>'
    )


def _collectible_xml(idx, pos):
    """Single large bright-green sphere; non-colliding, detected by XY distance."""
    return (
        f'<geom name="collectible_{idx}" type="sphere" '
        f'size="{COLLECTIBLE_RADIUS:.3f}" '
        f'pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" '
        f'rgba="0.00 1.00 0.00 1.0" '
        f'contype="0" conaffinity="0"/>'
    )
