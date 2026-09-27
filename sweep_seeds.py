import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.env_wrapper import RoverEnv
from stable_baselines3 import PPO

EXP = "v13_mlp"
CKPT_DIR = os.path.join("experiments", EXP, "checkpoints")
N_SEEDS = 30
MAX_STEPS = 10000

MODEL_VN = {
    "best_model":      os.path.join(CKPT_DIR, "vecnormalize_best.pkl"),
    "rover_ppo_final": os.path.join("experiments", EXP, "vecnormalize.pkl"),
}

for model_name in ["best_model", "rover_ppo_final"]:
    policy = PPO.load(os.path.join(CKPT_DIR, model_name), device="cpu")
    vn_path = MODEL_VN[model_name]
    if not os.path.exists(vn_path):
        vn_path = os.path.join(CKPT_DIR, "vecnormalize_best.pkl")
        print(f"  [warn] {MODEL_VN[model_name]} not found, falling back to best")
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    vn.training = False
    vn.norm_reward = False

    print(f"\n=== {model_name}  (vn: {os.path.basename(vn_path)}) ===")
    goals = 0
    for seed in range(N_SEEDS):
        env = RoverEnv(render_mode=None, adversary=None, enable_obstacles=False,
                       max_episode_steps=MAX_STEPS)
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0
        ep_start = np.ones((1,), dtype=bool)
        lstm_states = None
        while not done and step < MAX_STEPS:
            nobs = vn.normalize_obs(obs[np.newaxis])[0]
            action, lstm_states = policy.predict(
                nobs[np.newaxis], state=lstm_states,
                episode_start=ep_start, deterministic=True,
            )
            action = action[0]
            ep_start = np.zeros((1,), dtype=bool)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1
        dist_final = np.linalg.norm(env.goal_pos[:2] - env._sensor("chassis_pos")[:2])
        reached = info.get("reached_goal", False)
        goals += reached
        tag = "GOAL  " if reached else f"dist={dist_final:.1f}m"
        print(f"  seed={seed:2d}  steps={step:4d}  {tag}")
        env.close()
    print(f"  Success rate: {goals}/{N_SEEDS}")
