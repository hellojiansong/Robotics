# Rover RL Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/Simulation-MuJoCo-blue?style=flat)](https://mujoco.org/)
[![RL Algorithm](https://img.shields.io/badge/RL-PPO-brightgreen?style=flat)](https://arxiv.org/abs/1707.06347)
[![Curriculum](https://img.shields.io/badge/Curriculum-Adversarial-red?style=flat)](#)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat&logo=black)](https://github.com/psf/black)

This repository trains a MuJoCo rover policy using PPO and records dual-camera rollouts as GIFs.

## What is included

- Training entrypoint: `training/train.py`
- Environment and reward logic: `training/env_wrapper.py`, `training/reward.py`
- Rollout recorder: `record.py`
- Robot model: `robot/robot.xml`

## Prerequisites

- Python 3.10+
- Bash shell (Git Bash, WSL, or Linux/macOS terminal)
- MuJoCo-compatible graphics runtime for rendering

## Quick start

1. Clone the repo and open a terminal in the project root.
2. Run setup:

```bash
bash scripts/setup.sh
```

3. Train:

```bash
bash scripts/train.sh
```

4. Record rollout GIFs:

```bash
bash scripts/record.sh
```

## Script usage

### Setup

```bash
bash scripts/setup.sh
```

Optional environment variables:

- `PYTHON_BIN` (default: `python3`)
- `VENV_DIR` (default: `.venv` in project root)

### Train

```bash
bash scripts/train.sh
```

Optional environment variables:

- `PYTHON_BIN` (default: `python`)
- `VENV_DIR` (default: `.venv`)

Training outputs are written to `experiments/v1_flat_terrain/`.

### Record

Recording now defaults to randomized rocky terrain and prints a short terrain summary at startup:

```bash
bash scripts/record.sh
```

With trained checkpoint:

```bash
bash scripts/record.sh experiments/v1_flat_terrain/checkpoints/rover_ppo_final.zip 1200 20
```

Flat-ground recording:

```bash
python record.py --flat
```

Arguments:

1. `model_path` (optional)
2. `steps` (optional, default: `1500`)
3. `fps` (optional, default: `20`)

Named flags:

- `--seed` for a reproducible terrain sequence
- `--flat` to force flat ground and disable rocks

GIFs are written to `output/external_pov.gif` and `output/robot_pov.gif`.

## Manual Python commands

If you do not want to use bash scripts:

```bash
python -m training.train
python record.py --model experiments/v1_flat_terrain/checkpoints/rover_ppo_final.zip --steps 600 --fps 20
```

## Notes

- Existing `experiments/` artifacts are large; if you track new checkpoints, consider Git LFS.
- Rendering quality and speed depend on your graphics setup.
- In Line 12 in env_wrapper.py, change "robot" to "robot_front" or "robot_rear" to train the model with varied centers of gravity.
