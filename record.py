"""Record external + onboard POV GIFs from the rover environment."""

import argparse
import glob
import os
import pickle
import numpy as np
import mujoco
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from training.env_wrapper import RoverEnv, OBS_DIST_SCALE, OBS_Y_SCALE
from adversary.adversary import AdversarialTerrainAgent


OUTPUT_DIR    = "output"
DEFAULT_STEPS = 6000
DEFAULT_FPS   = 24
FRAME_SKIP    = 2      # capture every 2nd step for smoother motion
RES           = (720, 1280)  # (height, width)
MAX_TERRAIN_SEED = np.iinfo(np.int32).max

EXT_DISTANCE  = 18.0   # pulled back for 40 m world
EXT_ELEVATION = -30.0
EXT_AZIMUTH   = 125.0

DEFAULT_EXPERIMENT = "v10_mlp"

# Annotation layout
_FONT      = ImageFont.load_default(size=26)
_FONT_SM   = ImageFont.load_default(size=20)
_PAD       = 10
_LINE_H    = 30
_BOX_COLOR = (0, 0, 0, 170)   # semi-transparent black


def _annotate(frame, ep_num, ep_step, ep_reward, n_collected, n_total):
    """Burn reward HUD onto a raw RGB frame (numpy uint8)."""
    img  = Image.fromarray(frame).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)

    lines = [
        (f"Episode {ep_num}  •  Step {ep_step}", _FONT),
        (f"Reward   {ep_reward:+,.0f}", _FONT),
        (f"Collectibles  {n_collected}/{n_total}", _FONT_SM),
    ]

    box_w = 310
    box_h = len(lines) * _LINE_H + _PAD * 2
    draw.rounded_rectangle([8, 8, 8 + box_w, 8 + box_h], radius=6, fill=_BOX_COLOR)

    y = 8 + _PAD
    for text, font in lines:
        draw.text((_PAD + 10, y), text, fill=(255, 230, 80, 255), font=font)
        y += _LINE_H

    return np.array(Image.alpha_composite(img, over).convert("RGB"), dtype=np.uint8)


def _make_tracking_camera(model):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    cam.distance  = EXT_DISTANCE
    cam.elevation = EXT_ELEVATION
    cam.azimuth   = EXT_AZIMUTH
    return cam


def _autodetect_model(experiment):
    ckpt_dir = os.path.join("experiments", experiment, "checkpoints")
    candidates = [
        os.path.join(ckpt_dir, "best_model.zip"),
        os.path.join(ckpt_dir, "rover_ppo_final.zip"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c[:-4]
    steps = sorted(glob.glob(os.path.join(ckpt_dir, "rover_ppo_*_steps.zip")))
    if steps:
        return steps[-1][:-4]
    return None


def _next_seed(rng):
    return int(rng.integers(0, MAX_TERRAIN_SEED))


def _print_terrain_summary(env, seed):
    spec = getattr(env, "_terrain_spec", None)
    if spec is None:
        return

    print(f"Terrain seed: {seed}")
    hf_str = f"hfield z_scale={spec.hfield_z_scale:.3f}m" if spec.hfield_data is not None else "flat"
    n_pat  = len(getattr(spec, "patches", []))
    print(f"  obstacles: {len(spec.obstacles)} | collectibles: {len(spec.collectibles)} | patches: {n_pat} | friction: {spec.friction:.3f} | {hf_str}")
    if not spec.obstacles and spec.hfield_data is None:
        print("  terrain: flat")
        return

    preview = spec.obstacles[:5]
    for i, ob in enumerate(preview, start=1):
        pos  = tuple(round(float(v), 3) for v in ob["pos"])
        size = tuple(round(float(v), 3) for v in ob["size"])
        print(f"  obstacle {i}: type={ob['type']}, pos={pos}, size={size}")
    if len(spec.obstacles) > len(preview):
        print(f"  ... {len(spec.obstacles) - len(preview)} more obstacle(s)")


def _load_obs_normalizer(experiment, model_path=None):
    """Load VecNormalize obs running stats for manual normalisation in recording.

    Prefers vecnormalize_best.pkl (saved at the same step as best_model.zip) over
    the end-of-run vecnormalize.pkl to avoid obs-normalisation drift when the best
    checkpoint was saved well before training finished.
    """
    candidates = []
    if model_path and "best_model" in os.path.basename(model_path):
        candidates.append(os.path.join("experiments", experiment, "checkpoints", "vecnormalize_best.pkl"))
    candidates.append(os.path.join("experiments", experiment, "vecnormalize.pkl"))

    norm_path = next((p for p in candidates if os.path.isfile(p)), None)
    if norm_path is None:
        return None
    print(f"  Using norm stats: {norm_path}")
    try:
        with open(norm_path, "rb") as f:
            vn = pickle.load(f)
        # VecNormalize stores obs stats in obs_rms (RunningMeanStd)
        if hasattr(vn, "obs_rms") and vn.norm_obs:
            return vn.obs_rms
    except Exception as e:
        print(f"  Warning: could not load VecNormalize stats: {e}")
    return None


def _normalize_obs(obs, obs_rms, clip_obs=10.0):
    """Apply the same normalisation that VecNormalize used during training."""
    if obs_rms is None:
        return obs
    return np.clip(
        (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
        -clip_obs, clip_obs,
    ).astype(np.float32)


def _save_trajectory_plot(all_trajs, output_dir):
    """Save a top-down trajectory map for all recorded episodes."""
    if not all_trajs:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#2a2a2a")
    fig.patch.set_facecolor("#1a1a1a")

    cmap = plt.cm.get_cmap("tab10", len(all_trajs))
    for ep_idx, (traj, goal) in enumerate(all_trajs):
        if len(traj) < 2:
            continue
        xs, ys = zip(*traj)
        color = cmap(ep_idx % 10)
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.85,
                label=f"Ep {ep_idx + 1}")
        ax.plot(xs[0], ys[0], "o", color=color, markersize=5)
        ax.plot(xs[-1], ys[-1], "x", color=color, markersize=7, markeredgewidth=2)
        # Draw goal ring
        circle = plt.Circle((goal[0], goal[1]), 0.55, color="lime",
                             fill=False, linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)

    ax.set_xlim(-5, 48)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_title("Rover Trajectories (top-down)", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555555")
    if len(all_trajs) <= 10:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.5)

    out_path = os.path.join(output_dir, "trajectories.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved trajectory plot -> {out_path}")


def record(model_path=None, n_steps=DEFAULT_STEPS, fps=DEFAULT_FPS,
           adversarial=True, difficulty=1.0, seed=None, enable_obstacles=True,
           experiment=DEFAULT_EXPERIMENT):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seed_rng = np.random.default_rng(seed)

    adversary = None
    if adversarial:
        adversary = AdversarialTerrainAgent(start_difficulty=difficulty)
        adversary.update = lambda outcome: None  # type: ignore  freeze difficulty

    env = RoverEnv(render_mode=None, adversary=adversary, enable_obstacles=enable_obstacles)

    terrain_seed = _next_seed(seed_rng)
    obs, _ = env.reset(seed=terrain_seed)
    _print_terrain_summary(env, terrain_seed)

    ext_renderer = mujoco.Renderer(env.model, height=RES[0], width=RES[1])
    pov_renderer = mujoco.Renderer(env.model, height=RES[0], width=RES[1])
    track_cam    = _make_tracking_camera(env.model)

    if model_path is None:
        model_path = _autodetect_model(experiment)

    policy   = None
    obs_rms  = None
    is_recurrent = False

    if model_path:
        # Try RecurrentPPO first, fall back to PPO
        try:
            from sb3_contrib import RecurrentPPO
            policy = RecurrentPPO.load(model_path, device="cpu")
            is_recurrent = True
            print(f"Loaded RecurrentPPO model from: {model_path}")
        except Exception:
            from stable_baselines3 import PPO
            policy = PPO.load(model_path, device="cpu")
            print(f"Loaded PPO model from: {model_path}")

        obs_rms = _load_obs_normalizer(experiment, model_path)
        if obs_rms is not None:
            print("  Loaded VecNormalize obs stats for recording.")
    else:
        print("WARNING: no model found — recording with random actions")

    # LSTM state for recurrent policy
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)

    ext_frames, pov_frames = [], []

    # Per-episode tracking
    ep_num       = 1
    ep_step      = 0
    ep_reward    = 0.0
    n_collected  = 0
    from env.terrain_generator import N_COLLECTIBLES
    traj_xy      = []              # [(x, y), ...] for current episode
    all_trajs    = []              # [(traj_xy, goal_pos), ...] for all episodes

    print(f"Recording {n_steps} steps (adversarial={adversarial}, difficulty={difficulty}, "
          f"frame_skip={FRAME_SKIP})")

    for step in range(n_steps):
        if policy is not None:
            norm_obs = _normalize_obs(obs, obs_rms)
            norm_obs_batch = norm_obs[np.newaxis]  # add batch dim: (1, obs_dim)
            action, lstm_states = policy.predict(
                norm_obs_batch,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            action = action[0]  # remove batch dim
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_step   += 1

        # Track collectible pickups from reward info
        comp = info.get("reward_components", {})
        if comp.get("collectible", 0) > 0:
            n_collected += int(round(comp["collectible"] / 8000.0))

        # Log chassis XY for trajectory plot
        chassis_pos = env._sensor("chassis_pos")
        traj_xy.append((float(chassis_pos[0]), float(chassis_pos[1])))

        if step % FRAME_SKIP == 0:
            ext_renderer.update_scene(env.data, camera=track_cam)
            raw = ext_renderer.render().astype(np.uint8)
            ext_frames.append(_annotate(raw, ep_num, ep_step, ep_reward, n_collected, N_COLLECTIBLES))

            pov_renderer.update_scene(env.data, camera="robot_pov")
            raw = pov_renderer.render().astype(np.uint8)
            pov_frames.append(_annotate(raw, ep_num, ep_step, ep_reward, n_collected, N_COLLECTIBLES))

        if terminated or truncated:
            outcome = ("goal" if info.get("reached_goal")
                       else "fall" if info.get("fell_over")
                       else "timeout")
            print(f"  Ep {ep_num}: {outcome:7s}  reward={ep_reward:+,.0f}  "
                  f"steps={ep_step}  collected={n_collected}/{N_COLLECTIBLES}")

            all_trajs.append((list(traj_xy), env.goal_pos.copy()))
            traj_xy = []

            ep_num       += 1
            ep_reward     = 0.0
            ep_step       = 0
            n_collected   = 0
            episode_start = np.ones((1,), dtype=bool)
            lstm_states   = None  # reset LSTM state at episode boundary

            terrain_seed = _next_seed(seed_rng)
            obs, _ = env.reset(seed=terrain_seed)
            _print_terrain_summary(env, terrain_seed)
            # Renderers must be rebuilt: terrain regen swaps env.model
            del ext_renderer, pov_renderer
            ext_renderer = mujoco.Renderer(env.model, height=RES[0], width=RES[1])
            pov_renderer = mujoco.Renderer(env.model, height=RES[0], width=RES[1])
            track_cam    = _make_tracking_camera(env.model)

        if (step + 1) % 500 == 0:
            print(f"  {step + 1}/{n_steps} steps  ({len(ext_frames)} frames captured)")

    # Save trajectory plot
    all_trajs.append((list(traj_xy), env.goal_pos.copy()))  # include in-progress episode
    _save_trajectory_plot(all_trajs, OUTPUT_DIR)

    ext_path = os.path.join(OUTPUT_DIR, "external_pov.gif")
    pov_path = os.path.join(OUTPUT_DIR, "robot_pov.gif")

    print(f"\nSaving external POV -> {ext_path}  ({len(ext_frames)} frames)")
    imageio.mimsave(ext_path, ext_frames, fps=fps, loop=0, quantizer="nq")
    print(f"Saving robot POV    -> {pov_path}  ({len(pov_frames)} frames)")
    imageio.mimsave(pov_path, pov_frames, fps=fps, loop=0, quantizer="nq")

    print("\nDone.")
    print(f"  external_pov.gif  {os.path.getsize(ext_path) / 1024:.0f} KB")
    print(f"  robot_pov.gif     {os.path.getsize(pov_path) / 1024:.0f} KB")

    del ext_renderer, pov_renderer
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record dual-POV GIFs from the rover env")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (auto-detected from experiments/ if omitted)")
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT,
                        help="Experiment name for auto-detection and VecNormalize loading")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--fps",   type=int, default=DEFAULT_FPS)
    parser.add_argument("--seed",  type=int, default=None,
                        help="Optional terrain seed for reproducible recording")
    parser.add_argument("--flat",  action="store_true",
                        help="Disable rocky terrain and adversary (record on flat ground)")
    parser.add_argument("--difficulty", type=float, default=0.6,
                        help="Fixed adversarial difficulty in [0,1]")
    args = parser.parse_args()

    record(
        model_path=args.model,
        n_steps=args.steps,
        fps=args.fps,
        adversarial=not args.flat,
        difficulty=args.difficulty,
        seed=args.seed,
        enable_obstacles=not args.flat,
        experiment=args.experiment,
    )
