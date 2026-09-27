"""record2.py — Stochastic best-episode recorder for presentations.

Runs N episodes with stochastic actions (same distribution as training),
keeps only the best (goal-reached > highest reward) and produces:

  best_external_pov.gif   — tracking camera, enhanced HUD
  best_robot_pov.gif      — onboard camera, enhanced HUD
  best_combined.gif       — side-by-side both views in one file
  trajectories.png        — all episodes, best highlighted, obstacles shown
  episode_summary.png     — bar chart of all episode rewards
"""

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
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont

from training.env_wrapper import RoverEnv
from adversary.adversary import AdversarialTerrainAgent
from env.terrain_generator import N_COLLECTIBLES

OUTPUT_DIR       = "output"
DEFAULT_FPS      = 24
FRAME_SKIP       = 6
MAX_RECORD_FRAMES = 2000            # ~83s at 24fps; PIL GIF buffers all frames so cap memory
RES              = (360, 640)       # memory-efficient; keeps 2 episode buffers in RAM
MAX_TERRAIN_SEED = np.iinfo(np.int32).max
GOAL_TOTAL_DIST  = 42.0             # approximate max distance rover ever travels

EXT_DISTANCE  = 18.0
EXT_ELEVATION = -30.0
EXT_AZIMUTH   = 125.0

DEFAULT_EXPERIMENT = "v13_mlp"
DEFAULT_N_EPISODES = 20

_FONT     = ImageFont.load_default(size=18)
_FONT_SM  = ImageFont.load_default(size=14)
_FONT_BIG = ImageFont.load_default(size=28)
_PAD      = 8
_LINE_H   = 22
_BOX_COLOR   = (0, 0, 0, 170)
_BAR_BG      = (60, 60, 60, 200)
_BAR_FG      = (0, 210, 80, 220)
_BAR_WARN    = (255, 160, 0, 220)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

def _draw_progress_bar(draw, x, y, w, h, frac, color):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=3, fill=_BAR_BG)
    fill_w = max(4, int(w * min(frac, 1.0)))
    draw.rounded_rectangle([x, y, x + fill_w, y + h], radius=3, fill=color)


def _annotate(frame, ep_num, ep_step, ep_reward, n_collected, n_total,
              dist_to_goal=None, speed=None, initial_dist=None, goal_flash=False):
    img  = Image.fromarray(frame).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)
    W, H = img.size

    # --- Top-left info box ---
    lines = [
        (f"Ep {ep_num}  Step {ep_step}", _FONT),
        (f"Reward  {ep_reward:+,.0f}",   _FONT),
        (f"Collectibles  {n_collected}/{n_total}", _FONT_SM),
    ]
    if speed is not None:
        lines.append((f"Speed  {speed:.2f} m/s", _FONT_SM))

    box_w = 210
    box_h = len(lines) * _LINE_H + _PAD * 2 + (20 if dist_to_goal is not None else 0)
    draw.rounded_rectangle([6, 6, 6 + box_w, 6 + box_h], radius=5, fill=_BOX_COLOR)
    y = 6 + _PAD
    for text, font in lines:
        draw.text((_PAD + 8, y), text, fill=(255, 230, 80, 255), font=font)
        y += _LINE_H

    # Goal progress bar
    if dist_to_goal is not None and initial_dist is not None and initial_dist > 0:
        progress = 1.0 - dist_to_goal / initial_dist
        bar_color = _BAR_FG if progress > 0.5 else _BAR_WARN
        bx, bw, bh = _PAD + 8, box_w - _PAD * 3, 10
        draw.text((bx, y), "Goal", fill=(200, 200, 200, 220), font=_FONT_SM)
        _draw_progress_bar(draw, bx + 34, y + 2, bw - 34, bh, progress, bar_color)
        pct_txt = f"{progress * 100:.0f}%"
        draw.text((bx + bw - 4, y), pct_txt, fill=(255, 255, 255, 200), font=_FONT_SM)

    # --- GOAL REACHED banner ---
    if goal_flash:
        label = "GOAL REACHED!"
        tw    = len(label) * 14
        bx    = W // 2 - tw // 2 - 14
        by    = H // 2 - 28
        draw.rounded_rectangle([bx, by, bx + tw + 28, by + 40],
                                radius=8, fill=(0, 190, 60, 210))
        draw.text((W // 2 - tw // 2, by + 6), label,
                  fill=(255, 255, 255, 255), font=_FONT_BIG)

    return np.array(Image.alpha_composite(img, over).convert("RGB"), dtype=np.uint8)


def _add_view_label(frame, label):
    """Burn a small view label (e.g. 'External View') into the top-right corner."""
    img  = Image.fromarray(frame).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)
    W    = img.size[0]
    tw   = len(label) * 9
    draw.rounded_rectangle([W - tw - 20, 6, W - 6, 6 + _LINE_H],
                            radius=4, fill=_BOX_COLOR)
    draw.text((W - tw - 12, 9), label, fill=(200, 200, 200, 220), font=_FONT_SM)
    return np.array(Image.alpha_composite(img, over).convert("RGB"), dtype=np.uint8)


def _combine_frames(ext_frame, pov_frame):
    """Stack external POV (left) and robot POV (right) side by side."""
    divider = np.zeros((ext_frame.shape[0], 3, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)
    return np.concatenate([ext_frame, divider, pov_frame], axis=1)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _make_tracking_camera(model):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    cam.distance  = EXT_DISTANCE
    cam.elevation = EXT_ELEVATION
    cam.azimuth   = EXT_AZIMUTH
    return cam


# ---------------------------------------------------------------------------
# Model / normaliser helpers
# ---------------------------------------------------------------------------

def _autodetect_model(experiment):
    ckpt_dir = os.path.join("experiments", experiment, "checkpoints")
    for name in ("best_model.zip", "rover_ppo_final.zip"):
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p[:-4]
    steps = sorted(glob.glob(os.path.join(ckpt_dir, "rover_ppo_*_steps.zip")))
    return steps[-1][:-4] if steps else None


def _load_vecnormalize(experiment, model_path=None):
    """Load the VecNormalize object and return it frozen (training=False).

    Prefers vecnormalize_best.pkl (saved at the same step as best_model.zip)
    over the end-of-run vecnormalize.pkl to avoid obs drift.
    Uses vn.normalize_obs() directly — identical to how training/eval normalised.
    """
    candidates = []
    if model_path and "best_model" in os.path.basename(model_path):
        candidates.append(
            os.path.join("experiments", experiment, "checkpoints", "vecnormalize_best.pkl")
        )
    candidates.append(os.path.join("experiments", experiment, "vecnormalize.pkl"))
    norm_path = next((p for p in candidates if os.path.isfile(p)), None)
    if norm_path is None:
        return None
    try:
        with open(norm_path, "rb") as f:
            vn = pickle.load(f)
        vn.training     = False   # freeze — never update running stats during recording
        vn.norm_reward  = False
        print(f"  VecNormalize    : {norm_path}")
        if hasattr(vn, "obs_rms"):
            m = vn.obs_rms.mean
            v = vn.obs_rms.var
            print(f"  obs_rms mean[:5]: {np.round(m[:5], 4)}")
            print(f"  obs_rms var[:5] : {np.round(v[:5], 4)}")
        return vn
    except Exception as e:
        print(f"  Warning: could not load VecNormalize: {e}")
    return None


def _normalize_obs(obs, vn):
    """Normalise a single (obs_dim,) observation using the loaded VecNormalize."""
    if vn is None:
        return obs.astype(np.float32)
    # normalize_obs expects shape matching the vec-env batch; pass as (1, dim) then squeeze
    normed = vn.normalize_obs(obs[np.newaxis])
    return normed[0].astype(np.float32)


def _next_seed(rng):
    return int(rng.integers(0, MAX_TERRAIN_SEED))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_trajectory_plot(all_episode_data, best_ep_idx, output_dir):
    """Top-down trajectory map.

    all_episode_data: list of dicts with keys:
        traj      — list of (x, y)
        goal      — (gx, gy)
        outcome   — 'GOAL' | 'fall' | 'timeout'
        reward    — float
        obstacles — list of {'pos': (x,y,z), 'size': ...}
    best_ep_idx: index into all_episode_data for the episode to highlight
    """
    if not all_episode_data:
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#13131f")

    OUTCOME_COLOR = {"GOAL": "#33dd66", "fall": "#dd4444", "timeout": "#8888aa"}

    for i, ep in enumerate(all_episode_data):
        if i == best_ep_idx or len(ep["traj"]) < 2:
            continue
        xs, ys = zip(*ep["traj"])
        color  = OUTCOME_COLOR.get(ep["outcome"], "#666688")
        ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.45)

    # Best episode — thick gold line
    best = all_episode_data[best_ep_idx]
    if len(best["traj"]) >= 2:
        xs, ys = zip(*best["traj"])
        ax.plot(xs, ys, color="#ffd700", linewidth=2.8, alpha=0.95,
                label=f"Best (Ep {best_ep_idx + 1}  {best['outcome']}  "
                      f"reward={best['reward']:+,.0f})", zorder=5)
        ax.plot(xs[0], ys[0], "o", color="#ffd700", markersize=9, zorder=6)

    # Obstacles for best episode (grey dots)
    for ob in best.get("obstacles", []):
        ox, oy = float(ob["pos"][0]), float(ob["pos"][1])
        ax.plot(ox, oy, "s", color="#666677", markersize=4, alpha=0.6, zorder=2)

    # Start marker (white circle)
    ax.plot(0, 0, "o", color="white", markersize=9, zorder=7, label="Start")

    # Goal position (green star)
    gx, gy = float(best["goal"][0]), float(best["goal"][1])
    ax.plot(gx, gy, "*", color="#00ff88", markersize=16, zorder=7, label="Goal")
    goal_ring = plt.Circle((gx, gy), 0.55, color="#00ff88", fill=False,
                            linewidth=1.5, alpha=0.7)
    ax.add_patch(goal_ring)

    # Legend for outcomes
    legend_patches = [
        mpatches.Patch(color="#33dd66", label="Goal reached"),
        mpatches.Patch(color="#dd4444", label="Fall"),
        mpatches.Patch(color="#8888aa", label="Timeout"),
    ]
    first_legend = ax.legend(handles=legend_patches, loc="upper left",
                             fontsize=9, framealpha=0.4)
    ax.add_artist(first_legend)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.4)

    # Success rate in title
    n_goals   = sum(1 for ep in all_episode_data if ep["outcome"] == "GOAL")
    n_total   = len(all_episode_data)
    ax.set_title(
        f"Rover Trajectories — {n_goals}/{n_total} goals reached "
        f"({100 * n_goals / n_total:.0f}% success)",
        color="white", fontsize=13,
    )
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_xlim(-3, 48)
    ax.set_ylim(-12, 12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444455")
    ax.grid(color="#333344", linewidth=0.5, alpha=0.6)

    out = os.path.join(output_dir, "trajectories.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved trajectory plot   -> {out}")


def _save_episode_summary(results, output_dir):
    """Bar chart of all episode rewards, colour-coded by outcome."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_facecolor("#2a2a2a")
    fig.patch.set_facecolor("#1a1a1a")

    OUTCOME_COLOR = {"GOAL": "#33dd66", "fall": "#dd4444", "timeout": "#8888aa"}
    best_idx = max(range(len(results)),
                   key=lambda i: (results[i]["outcome"] == "GOAL", results[i]["reward"]))

    for i, r in enumerate(results):
        color = "#ffd700" if i == best_idx else OUTCOME_COLOR.get(r["outcome"], "#888888")
        ax.bar(r["ep"], r["reward"], color=color, alpha=0.9 if i == best_idx else 0.75)

    ax.axhline(0, color="white", linewidth=0.6, alpha=0.35)

    legend_elems = [
        mpatches.Patch(color="#33dd66", label="Goal reached"),
        mpatches.Patch(color="#dd4444", label="Fall"),
        mpatches.Patch(color="#8888aa", label="Timeout"),
        mpatches.Patch(color="#ffd700", label=f"Saved (Ep {results[best_idx]['ep']})"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, framealpha=0.4)

    n_goals = sum(1 for r in results if r["outcome"] == "GOAL")
    ax.set_title(
        f"Episode rewards — {n_goals}/{len(results)} goal reached "
        f"({100 * n_goals / len(results):.0f}%)",
        color="white", fontsize=12,
    )
    ax.set_xlabel("Episode", color="white")
    ax.set_ylabel("Reward", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555555")

    out = os.path.join(output_dir, "episode_summary.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved episode summary   -> {out}")


# ---------------------------------------------------------------------------
# Main recorder
# ---------------------------------------------------------------------------

def record2(model_path=None, n_episodes=DEFAULT_N_EPISODES, fps=DEFAULT_FPS,
            difficulty=0.6, seed=None, enable_obstacles=True,
            experiment=DEFAULT_EXPERIMENT, stochastic=False, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    seed_rng = np.random.default_rng(seed)

    adversary = AdversarialTerrainAgent(start_difficulty=difficulty)
    adversary.update = lambda outcome: None   # freeze difficulty

    env = RoverEnv(render_mode=None, adversary=adversary,
                   enable_obstacles=enable_obstacles,
                   max_episode_steps=50000)

    terrain_seed = _next_seed(seed_rng)
    obs, _ = env.reset(seed=terrain_seed)

    ext_renderer = mujoco.Renderer(env.model, height=RES[0], width=RES[1])
    pov_renderer = mujoco.Renderer(env.model, height=RES[0], width=RES[1])
    track_cam    = _make_tracking_camera(env.model)

    if model_path is None:
        model_path = _autodetect_model(experiment)

    policy, vn = None, None
    if model_path:
        try:
            from sb3_contrib import RecurrentPPO
            policy = RecurrentPPO.load(model_path, device="cpu")
            print(f"Loaded RecurrentPPO from: {model_path}")
        except Exception:
            from stable_baselines3 import PPO
            policy = PPO.load(model_path, device="cpu")
            print(f"Loaded PPO from: {model_path}")
        vn = _load_vecnormalize(experiment, model_path)
    else:
        print("WARNING: no model — using random actions")

    mode_str = "stochastic" if stochastic else "deterministic"
    print(f"\nRunning {n_episodes} {mode_str} episodes  "
          f"(difficulty={difficulty}, obstacles={enable_obstacles})")
    print("Keeping best: goal-reached first, then by reward\n")

    best_comb_frames = []
    best_frame_count = 0
    best_reward      = -np.inf
    best_goal        = False
    best_ep_num      = 0
    best_ep_idx      = 0

    results          = []   # for episode_summary.png
    all_episode_data = []   # for trajectories.png

    for ep in range(1, n_episodes + 1):
        if ep > 1:
            terrain_seed = _next_seed(seed_rng)
            obs, _ = env.reset(seed=terrain_seed)
            del ext_renderer, pov_renderer
            ext_renderer = mujoco.Renderer(env.model, height=RES[0], width=RES[1])
            pov_renderer = mujoco.Renderer(env.model, height=RES[0], width=RES[1])
            track_cam    = _make_tracking_camera(env.model)

        comb_frames  = []
        frame_count  = 0
        ep_reward        = 0.0
        ep_step          = 0
        n_collected      = 0
        goal_reached_step = None
        lstm_states      = None
        episode_start    = np.ones((1,), dtype=bool)
        traj_xy          = []

        # Capture initial dist for progress bar
        goal_pos     = env.goal_pos.copy()
        chassis_pos0 = env._sensor("chassis_pos")
        initial_dist = float(np.linalg.norm(goal_pos[:2] - chassis_pos0[:2]))

        # Obstacle info for trajectory plot
        ep_obstacles = []
        spec = getattr(env, "_terrain_spec", None)
        if spec is not None:
            ep_obstacles = spec.obstacles

        terminated = truncated = False
        while not (terminated or truncated):
            if policy is not None:
                norm_obs = _normalize_obs(obs, vn)
                action, lstm_states = policy.predict(
                    norm_obs[np.newaxis],
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=not stochastic,
                )
                action = action[0]
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_step   += 1

            comp = info.get("reward_components", {})
            if comp.get("collectible", 0) > 0:
                n_collected += int(round(comp["collectible"] / 8000.0))
            if info.get("reached_goal") and goal_reached_step is None:
                goal_reached_step = ep_step

            chassis_pos  = env._sensor("chassis_pos")
            chassis_vel  = env._sensor("chassis_linvel")
            traj_xy.append((float(chassis_pos[0]), float(chassis_pos[1])))
            dist_to_goal = float(np.linalg.norm(goal_pos[:2] - chassis_pos[:2]))
            speed        = float(np.linalg.norm(chassis_vel[:2]))

            if ep_step % FRAME_SKIP == 0 and frame_count < MAX_RECORD_FRAMES:
                goal_flash = (goal_reached_step is not None and
                              ep_step - goal_reached_step < 60)

                ext_renderer.update_scene(env.data, camera=track_cam)
                raw_ext = ext_renderer.render().astype(np.uint8)
                ann_ext = _annotate(raw_ext, ep, ep_step, ep_reward,
                                    n_collected, N_COLLECTIBLES,
                                    dist_to_goal, speed, initial_dist, goal_flash)
                ann_ext = _add_view_label(ann_ext, "External View")

                pov_renderer.update_scene(env.data, camera="robot_pov")
                raw_pov = pov_renderer.render().astype(np.uint8)
                ann_pov = _annotate(raw_pov, ep, ep_step, ep_reward,
                                    n_collected, N_COLLECTIBLES,
                                    dist_to_goal, speed, initial_dist, goal_flash)
                ann_pov = _add_view_label(ann_pov, "Robot POV")

                comb_frames.append(_combine_frames(ann_ext, ann_pov))
                frame_count += 1

        ep_goal = bool(info.get("reached_goal"))
        outcome = ("GOAL" if ep_goal
                   else "fall" if info.get("fell_over")
                   else "timeout")

        results.append({"ep": ep, "reward": ep_reward,
                        "outcome": outcome, "goal": ep_goal, "steps": ep_step})
        all_episode_data.append({
            "traj":      traj_xy,
            "goal":      goal_pos,
            "outcome":   outcome,
            "reward":    ep_reward,
            "obstacles": ep_obstacles,
        })

        is_better = (ep_goal and not best_goal) or \
                    (ep_goal == best_goal and ep_reward > best_reward)
        tag = " ** new best!" if is_better else ""
        print(f"  Ep {ep:2d}/{n_episodes}: {outcome:7s}  "
              f"reward={ep_reward:+,.0f}  steps={ep_step}  "
              f"collected={n_collected}/{N_COLLECTIBLES}{tag}")

        if is_better:
            best_comb_frames = comb_frames
            best_frame_count = frame_count
            best_reward      = ep_reward
            best_goal        = ep_goal
            best_ep_num      = ep
            best_ep_idx      = ep - 1

    n_goals = sum(1 for r in results if r["goal"])
    print(f"\n{'='*60}")
    print(f"  Goals reached : {n_goals}/{n_episodes} ({100*n_goals/n_episodes:.0f}%)")
    print(f"  Best episode  : Ep {best_ep_num}  outcome={results[best_ep_idx]['outcome']}"
          f"  reward={best_reward:+,.0f}")
    print(f"  Frames        : {best_frame_count}")
    print(f"{'='*60}\n")

    # --- Save plots ---
    _save_trajectory_plot(all_episode_data, best_ep_idx, output_dir)
    _save_episode_summary(results, output_dir)

    # --- Save GIFs ---
    comb_path = os.path.join(output_dir, "best_combined.gif")

    print(f"Saving combined (both)  -> {comb_path}")
    imageio.mimsave(comb_path, best_comb_frames, fps=fps, loop=0, quantizer="nq")

    print(f"\n  best_combined.gif      {os.path.getsize(comb_path)  / 1024:.0f} KB")
    print("Done.")

    del ext_renderer, pov_renderer
    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stochastic best-episode recorder for presentations"
    )
    parser.add_argument("--model",      type=str,   default=None)
    parser.add_argument("--experiment", type=str,   default=DEFAULT_EXPERIMENT)
    parser.add_argument("--episodes",   type=int,   default=DEFAULT_N_EPISODES,
                        help="Episodes to try (default 20); more = better odds of goal")
    parser.add_argument("--fps",        type=int,   default=DEFAULT_FPS)
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Terrain seed sequence (reproducible runs)")
    parser.add_argument("--flat",        action="store_true",
                        help="Flat terrain, no obstacles")
    parser.add_argument("--difficulty",  type=float, default=0.6)
    parser.add_argument("--stochastic",  action="store_true",
                        help="Sample actions with noise (default: deterministic mean action)")
    parser.add_argument("--output-dir",  type=str,   default=OUTPUT_DIR,
                        help="Directory to save GIFs and plots (default: output)")
    args = parser.parse_args()

    record2(
        model_path=args.model,
        n_episodes=args.episodes,
        fps=args.fps,
        difficulty=args.difficulty,
        seed=args.seed,
        enable_obstacles=not args.flat,
        experiment=args.experiment,
        stochastic=args.stochastic,
        output_dir=args.output_dir,
    )
