"""One-episode inference diagnostic — prints normalized obs, actions, and trajectory."""
import os, sys, pickle
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from training.env_wrapper import RoverEnv

EXPERIMENT    = "v13_mlp"
DETERMINISTIC = True
MODEL_NAME    = "best_model"   # or "rover_ppo_final"
SEED          = 42

def main():
    # ── Load model ────────────────────────────────────────────────────────────
    model_path = os.path.join("experiments", EXPERIMENT, "checkpoints", MODEL_NAME)
    try:
        from sb3_contrib import RecurrentPPO
        policy = RecurrentPPO.load(model_path, device="cpu")
        print("Loaded: RecurrentPPO")
    except Exception:
        from stable_baselines3 import PPO
        policy = PPO.load(model_path, device="cpu")
        print("Loaded: PPO")

    # ── Load VecNormalize ─────────────────────────────────────────────────────
    vn_suffix = "best" if MODEL_NAME == "best_model" else "final"
    vn_path = os.path.join("experiments", EXPERIMENT, "checkpoints", f"vecnormalize_{vn_suffix}.pkl")
    if not os.path.exists(vn_path):
        vn_path = os.path.join("experiments", EXPERIMENT, "checkpoints", "vecnormalize_best.pkl")
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    vn.training    = False
    vn.norm_reward = False

    print(f"vn.norm_obs          : {vn.norm_obs}")
    print(f"vn.clip_obs          : {vn.clip_obs}")
    print(f"obs_rms.mean[13:16]  : {np.round(vn.obs_rms.mean[13:16], 5)}  ← goal_vec mean")
    print(f"obs_rms.var[13:16]   : {np.round(vn.obs_rms.var[13:16],  5)}  ← goal_vec var")
    print(f"obs_rms.mean[:3]     : {np.round(vn.obs_rms.mean[:3],    5)}  ← pos mean")

    # Sanity-check: vn.normalize_obs vs manual formula
    dummy = np.ones(32, dtype=np.float32)
    auto   = vn.normalize_obs(dummy[np.newaxis])[0]
    manual = np.clip((dummy - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8),
                     -vn.clip_obs, vn.clip_obs)
    print(f"normalize_obs matches manual: {np.allclose(auto, manual, atol=1e-5)}")

    # ── Create env (flat, no obstacles, no adversary) ─────────────────────────
    env = RoverEnv(render_mode=None, adversary=None, enable_obstacles=False)
    obs, _ = env.reset(seed=SEED)

    chassis_pos  = env._sensor("chassis_pos")
    chassis_quat = env._sensor("chassis_quat")
    w, x, y, z  = chassis_quat
    yaw_deg = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))

    print(f"\nGoal pos   : {np.round(env.goal_pos, 3)}")
    print(f"Start pos  : {np.round(chassis_pos, 3)}")
    print(f"Start yaw  : {yaw_deg:.1f} deg  (0=East/+X, 90=North/+Y)")

    goal_vec_raw = obs[13:16]
    print(f"\nRaw  goal_vec (obs[13:16]) : {np.round(goal_vec_raw, 4)}")
    norm_obs = vn.normalize_obs(obs[np.newaxis])[0]
    print(f"Norm goal_vec (obs[13:16]) : {np.round(norm_obs[13:16], 4)}")
    print(f"  (dx_norm, dy_norm, dist/40) — dx>0 means goal is in world +X)")

    # ── First action ──────────────────────────────────────────────────────────
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)
    action, lstm_states = policy.predict(
        norm_obs[np.newaxis], state=lstm_states,
        episode_start=episode_start, deterministic=DETERMINISTIC,
    )
    action        = action[0]
    episode_start = np.zeros((1,), dtype=bool)
    print(f"\nFirst action [left, right]: {np.round(action, 3)}")
    print(f"  (+1,+1)=straight  (+1,-1)=turn-right  (-1,+1)=turn-left")

    # ── Run 500 steps, print every 25 ─────────────────────────────────────────
    goal_pos    = env.goal_pos.copy()
    total_rew   = 0.0

    print(f"\n{'step':>5} {'x':>7} {'y':>7} {'dist':>7} {'L':>6} {'R':>6} {'step_rew':>10}")
    print("-" * 55)

    def _print_row(s, cp, act, rew):
        d = np.linalg.norm(goal_pos[:2] - cp[:2])
        print(f"{s:>5} {cp[0]:>7.2f} {cp[1]:>7.2f} {d:>7.2f} {act[0]:>6.3f} {act[1]:>6.3f} {rew:>10.1f}")

    for step in range(5000):
        obs, reward, terminated, truncated, info = env.step(action)
        total_rew += reward
        cp = env._sensor("chassis_pos")

        if step < 5 or step % 100 == 99:
            _print_row(step + 1, cp, action, reward)

        if terminated or truncated:
            _print_row(step + 1, cp, action, reward)
            outcome = ("GOAL" if info.get("reached_goal") else
                       "FALL" if info.get("fell_over") else "TIMEOUT")
            print(f"\n→ {outcome} at step {step+1}   total_reward={total_rew:.1f}")
            break

        norm_obs = vn.normalize_obs(obs[np.newaxis])[0]
        action, lstm_states = policy.predict(
            norm_obs[np.newaxis], state=lstm_states,
            episode_start=episode_start, deterministic=DETERMINISTIC,
        )
        action = action[0]
    else:
        print(f"\n→ ran 5000 steps  total_reward={total_rew:.1f}  dist={np.linalg.norm(goal_pos[:2]-cp[:2]):.2f}")

    env.close()

if __name__ == "__main__":
    main()
