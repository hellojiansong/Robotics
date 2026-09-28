# Autonomous Rover Navigation with Adversarial Curriculum Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/Simulation-MuJoCo-blue?style=flat)](https://mujoco.org/)
[![RL Algorithm](https://img.shields.io/badge/RL-PPO-brightgreen?style=flat)](https://arxiv.org/abs/1707.06347)
[![Curriculum](https://img.shields.io/badge/Curriculum-Adversarial-red?style=flat)](#)
[![Video Demo](https://img.shields.io/badge/Demo-Google%20Drive-yellow?logo=googledrive)](https://drive.google.com/file/d/1UCdMJtC9LAvuQ25bMk0Q1MgiMB0un2CY/view?usp=sharing)

This repository trains a four-wheeled rover policy in MuJoCo using Proximal Policy Optimization (PPO). The training environment features an asymmetric band-controller adversary that dynamically scales procedural terrain difficulty based on recent success rates and empirical failure modes.

---

## 1. System Overview & Results

- **Control & Observation**: 4-wheel skid-steer velocity control with a 32-dimensional observation vector (kinematics, orientation quaternion, goal-relative vectors, and 9 forward-fan raycasts).
- **Adversarial Curriculum**: A deterministic controller tracks a 50-episode sliding window, scaling scalar difficulty up ($+0.005$ if win rate $> 0.65$) or down ($-0.010$ if $< 0.30$) while biasing features toward vertical obstacles or lateral detours.
- **Procedural Heightfield**: Multi-band sinusoidal elevation combined with craters, mesas, and edge tapering near spawn and goal areas.

### Benchmark Evaluation (30 Seeds)

| Terrain Setting | Difficulty | Goal Success Rate | Behavior |
| :--- | :---: | :---: | :--- |
| **Flat Ground Baseline** | $0.0$ | **30 / 30 (100%)** | Direct, smooth trajectories toward targets. |
| **Midpoint Rocky Terrain** | $0.5$ | **11 / 30 (36.7%)** | Aligns with the adversary's target equilibrium band ($30\% \sim 65\%$). |

---

## 2. File Structure

```text
.
├── adversary/                    # Band controller & sliding-window outcome tracker
├── docs/                         # Documentation and project assets
├── env/                          # Procedural terrain generation & heightfield samplers
├── experiments/                  # Training logs, TensorBoard curves, and model checkpoints
│   └── v13_mlp/                  # Checkpoints from the 30M-step PPO training run
├── output0.0terrain/             # Rendered rollout GIFs and evaluations on flat terrain (diff=0.0)
├── output0.5terrain/             # Rendered rollout GIFs and evaluations on rocky terrain (diff=0.5)
├── robot/
│   └── robot.xml                 # MuJoCo MJCF kinematics, actuators, and sensor definitions
├── scripts/                      # Bash execution launchers
│   ├── setup.sh                  # Virtual environment initialization
│   ├── train.sh                  # Distributed PPO training across 16 CPUs
│   └── record.sh                 # Rollout recording pipeline
├── training/                     # Core RL implementation
│   ├── train.py                  # PPO orchestration using SubprocVecEnv
│   ├── env_wrapper.py            # Observation normalizer, frame rotations, raycasts
│   └── reward.py                 # Multi-objective & staged tilt penalty shaping
├── debug_inference.py            # Single-step inference check and rollout inspection
├── main.py                       # Main project execution interface
├── record.py                     # Headless rollout rendering engine (external POV + robot POV)
├── record2.py                    # Multi-camera / secondary viewpoint recording engine
├── sweep_seeds.py                # Batch evaluation across fixed random seeds (0.0 vs 0.5 difficulty)
└── requirements.txt              # Project dependencies
```

---

## 3. Prerequisites & Environment Setup

### Prerequisites
- Python 3.10+
- Bash shell (WSL, Linux, macOS terminal, or Git Bash)
- OpenGL runtime for MuJoCo offscreen rendering

### Installation
```bash
git clone [https://github.com/hellojiansong/Robotics.git](https://github.com/hellojiansong/Robotics.git)
cd Robotics
bash scripts/setup.sh
```

---

## 4. Execution Guide

### Automated Workflow (Bash Scripts)

```bash
# 1. Launch training (runs across 16 parallel CPUs)
bash scripts/train.sh

# 2. Record rollout GIFs with trained checkpoint
bash scripts/record.sh experiments/v13_mlp/checkpoints/rover_ppo_final.zip 1200 20
```

### Direct Python Commands

```bash
# Train PPO policy
python -m training.train

# Benchmark across 30 evaluation seeds
python sweep_seeds.py

# Render dual-view rollout GIFs
python record.py --model experiments/v13_mlp/checkpoints/rover_ppo_final.zip --steps 1200 --fps 20

# Record on flat ground baseline
python record.py --flat --steps 1200 --fps 20
```

*GIFs are output to `output/external_pov.gif` and `output/robot_pov.gif`.*

---

## 5. Notes & Configurations

- **Center-of-Gravity (CoG) Shift**: In `training/env_wrapper.py` (Line 12), change `"robot"` to `"robot_front"` or `"robot_rear"` to train/evaluate policies under shifted mass distributions.
- **Coordinate Conventions**: The chassis longitudinal axis is inverted in the MJCF model; the wrapper explicitly negates the forward velocity component to align world and body directions.
