"""Smoke test for adversarial terrain generation.

Runs N episodes with random or trained policy, lets the adversary adapt,
and prints the difficulty curve plus failure mode mix.

Usage:
    python -m adversary.test --episodes 30
    python -m adversary.test --episodes 30 --model experiments/v6_lstm/checkpoints/rover_ppo_final
"""

import argparse
import numpy as np

from training.env_wrapper import RoverEnv
from adversary.adversary import AdversarialTerrainAgent


def run(model_path=None, episodes=20, max_steps=2000, seed=0):
    adversary = AdversarialTerrainAgent(start_difficulty=0.1)
    env = RoverEnv(adversary=adversary, max_episode_steps=max_steps)

    # Bug7 fix: try RecurrentPPO first (matches v6_lstm training), fall back to PPO
    policy = None
    is_recurrent = False
    if model_path:
        try:
            from sb3_contrib import RecurrentPPO
            policy = RecurrentPPO.load(model_path, device="cpu")
            is_recurrent = True
            print(f"Loaded RecurrentPPO policy: {model_path}")
        except Exception:
            from stable_baselines3 import PPO
            policy = PPO.load(model_path, device="cpu")
            print(f"Loaded PPO policy: {model_path}")
    else:
        print("No model given - using random actions")

    history = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        steps = 0
        info = {}

        # Bug7 fix: maintain LSTM state across steps, reset at episode boundary
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        while not (terminated or truncated):
            if policy is not None:
                if is_recurrent:
                    action, lstm_states = policy.predict(
                        obs[np.newaxis],
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    action = action[0]
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1

        if info.get("reached_goal"):
            outcome = "success"
        elif info.get("fell_over"):
            outcome = "fall"
        else:
            outcome = "timeout"

        history.append(outcome)
        print(
            f"ep {ep:3d}  {outcome:8s}  diff={adversary.difficulty:.2f}  "
            f"steps={steps:4d}  success_rate={adversary.tracker.success_rate:.2f}  "
            f"fall_rate={adversary.tracker.fall_rate:.2f}"
        )

    env.close()

    print("\n=== Summary ===")
    print(f"  episodes : {len(history)}")
    print(f"  success  : {history.count('success')}")
    print(f"  fall     : {history.count('fall')}")
    print(f"  timeout  : {history.count('timeout')}")
    print(f"  final difficulty : {adversary.difficulty:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.model, args.episodes, args.max_steps, args.seed)
