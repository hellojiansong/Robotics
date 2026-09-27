import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback,
)

# Allow running this file directly (python train.py) from training/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from adversary.adversary import AdversarialTerrainAgent
from training.env_wrapper import RoverEnv


EXPERIMENT  = "v13_mlp"
N_ENVS      = 16
TOTAL_STEPS = 30_000_000

EXP_DIR   = os.path.join(PROJECT_ROOT, "experiments", EXPERIMENT)
LOG_BASE  = os.path.join(EXP_DIR, "logs")
CKPT_BASE = os.path.join(EXP_DIR, "checkpoints")
NORM_PATH = os.path.join(EXP_DIR, "vecnormalize.pkl")

# v13 changes vs v12:
#   Action negated (_apply_action): body +x = physical rear, so ctrl was negated so that
#     [+1,+1] = physically forward. v12 policy saturated at [-1,-1] because physical
#     forward required negative ctrl, and the policy mean drifted to -inf with hard clipping.
#   squash_output=True: tanh on policy output prevents saturation at ±1 since gradients
#     flow through tanh everywhere. Hard clipping + no squash caused v12 mean → -4.
#   Spawn constrained: rover now spawns within ±60° of goal direction instead of random
#     full-circle yaw. v12 training ep_rew_mean=63k was from 50% lucky-alignment spawns,
#     not learned steering. Constrained spawn forces the policy to actually navigate.
#   Obs x-components negated: goal_vec[0], linvel_body[0], collectible[0], raycast forward
#     all use physical-forward convention so [+1,+1] = ahead = positive goal_vec[0].
#   TOTAL_STEPS 8M→6M: cleaner obs + constrained spawn should converge faster.
PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=512,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
)


def make_env(rank, seed=0):
    def _init():
        env = RoverEnv(adversary=None, enable_obstacles=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def _build_vec_env(rank_offset, n, norm_reward):
    env = SubprocVecEnv([make_env(rank_offset + i) for i in range(n)])
    env = VecMonitor(env)
    return VecNormalize(
        env,
        norm_obs=True,          # normalise observations using running mean/std
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=PPO_KWARGS["gamma"],
    )


class VecNormSyncCallback(BaseCallback):
    """Keep eval VecNormalize obs_rms in sync with the training env.

    train_env and eval_env are separate VecNormalize instances, so they
    accumulate independent running statistics. Without syncing, the policy
    sees obs normalised with different mean/var during evaluation vs training.
    This callback copies train stats to eval at the end of every rollout,
    just before EvalCallback might trigger.
    """

    def __init__(self, train_env, eval_env, verbose=0):
        super().__init__(verbose)
        self._train = train_env
        self._eval  = eval_env

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        self._eval.obs_rms.mean[:]  = self._train.obs_rms.mean
        self._eval.obs_rms.var[:]   = self._train.obs_rms.var
        self._eval.obs_rms.count    = self._train.obs_rms.count


class BestModelNormCallback(BaseCallback):
    """Save VecNormalize stats alongside best_model.zip whenever EvalCallback saves a new best.

    EvalCallback writes best_model.zip but never saves vecnormalize at that moment.
    The mismatch between best_model weights (e.g. step 12M) and the end-of-run
    vecnormalize.pkl (step 20M) causes obs normalisation drift during recording,
    making the policy misread the goal direction. This callback detects when
    best_model.zip is freshly written and immediately saves matching norm stats.
    """

    def __init__(self, train_env, ckpt_dir, verbose=0):
        super().__init__(verbose)
        self._train_env = train_env
        self._norm_path = os.path.join(ckpt_dir, "vecnormalize_best.pkl")
        self._last_mtime = None

    def _on_step(self) -> bool:
        best_zip = os.path.join(os.path.dirname(self._norm_path), "best_model.zip")
        if os.path.isfile(best_zip):
            mtime = os.path.getmtime(best_zip)
            if self._last_mtime is None or mtime > self._last_mtime:
                self._last_mtime = mtime
                self._train_env.save(self._norm_path)
        return True


class AdversaryCallback(BaseCallback):
    """Single shared adversary that receives outcomes from all parallel envs.

    With SubprocVecEnv the per-env adversary instances cannot share state
    (separate processes). This callback maintains one authoritative adversary
    in the main process, reads episode outcomes from the info dicts that SB3
    passes back on each done step, and pushes the updated difficulty params to
    all envs at the end of each rollout.
    """

    def __init__(self, adversary, verbose=0):
        super().__init__(verbose)
        self.adversary = adversary
        self._had_episode_end = False

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                if info.get("reached_goal"):
                    outcome = "success"
                elif info.get("fell_over"):
                    outcome = "fall"
                else:
                    outcome = "timeout"
                self.adversary.update(outcome)
                self._had_episode_end = True
        return True

    def _on_rollout_end(self):
        if self._had_episode_end:
            params = self.adversary.propose()
            self.training_env.env_method("set_difficulty_params", params)
            self._had_episode_end = False


def train():
    os.makedirs(LOG_BASE, exist_ok=True)
    os.makedirs(CKPT_BASE, exist_ok=True)

    # MlpPolicy inference on a small 256x256 net is CPU-bound; GPU transfer
    # overhead outweighs compute benefit (SB3 warning). Use CPU for MLP runs.
    device = "cpu"

    print(f"  Experiment : {EXPERIMENT}")
    print(f"  Envs       : {N_ENVS}")
    print(f"  Timesteps  : {TOTAL_STEPS:,}")
    print(f"  Logs       : {LOG_BASE}")

    train_env = _build_vec_env(0, N_ENVS, norm_reward=True)
    eval_env  = _build_vec_env(N_ENVS, 1, norm_reward=False)

    # Shared adversary — one instance for all envs
    shared_adversary = AdversarialTerrainAgent(start_difficulty=0.0)

    # Push initial params before any episode runs
    initial_params = AdversarialTerrainAgent.build_params(0.0)
    train_env.env_method("set_difficulty_params", initial_params)

    model = PPO(
        env=train_env,
        verbose=1,
        device=device,
        tensorboard_log=LOG_BASE,
        **PPO_KWARGS,
    )

    save_freq = 100_000 // N_ENVS
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=save_freq,
            save_path=CKPT_BASE,
            name_prefix="rover_ppo",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            eval_freq=save_freq,
            n_eval_episodes=5,
            best_model_save_path=CKPT_BASE,
            log_path=LOG_BASE,
            deterministic=True,
            render=False,
            verbose=1,
        ),
        AdversaryCallback(shared_adversary),
        VecNormSyncCallback(train_env, eval_env),
        BestModelNormCallback(train_env, CKPT_BASE),
    ])

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callbacks,
        tb_log_name="PPO",
        reset_num_timesteps=True,
    )

    final_path = os.path.join(CKPT_BASE, "rover_ppo_final")
    model.save(final_path)
    train_env.save(NORM_PATH)
    print(f"Model saved to {final_path}.zip")
    print(f"VecNormalize stats saved to {NORM_PATH}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
