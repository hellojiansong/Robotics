# 🤖 4-Wheeled Robot Navigation in Rocky Terrain — Project Plan
### MuJoCo + Gymnasium + Reinforcement Learning | Group of 4

---

## 🎯 Project Overview

Build a 4-wheeled robot in MuJoCo that learns to navigate between two waypoints across a procedurally generated rocky terrain using Reinforcement Learning. The environment features obstacles/no-go zones, colored rocks with reward/penalty semantics, and an adversarial terrain generator (a second AI) that actively adapts difficulty to target the robot's weaknesses.

**Demo Deadline:** May 18, 2025  
**Paper Deadline:** June 1, 2025

---

## 🧱 Technical Stack

| Component | Technology |
|---|---|
| Physics Simulator | MuJoCo |
| RL Framework | Python + Gymnasium |
| RL Algorithms | PPO / SAC (stable-baselines3 or custom) |
| Adversarial Terrain AI | Secondary RL agent or evolutionary optimizer |
| Logging & Visualization | TensorBoard / W&B |
| Paper | LaTeX (IEEE or NeurIPS format) |

---

## 👥 Team Roles & Responsibilities

| Person | Role |
|---|---|
| **Person A** | Robot Mechanics & Simulation Lead |
| **Person B** | RL Training & Reward Engineering Lead |
| **Person C** | Environment & Terrain Systems Lead |
| **Person D** | Adversarial AI & Integration Lead |

> Everyone reads and understands all parts of the codebase. Weekly sync meetings are mandatory. Each person writes their own section of the paper.

---

## 🗓️ Phase Breakdown

---

### 📦 Phase 0 — Shared Onboarding (All 4 People)
**Duration:** ~1 week | **Target End:** ~April 6**

Everyone must complete this before diverging into their individual roles.

- [ ] Install MuJoCo, Gymnasium, stable-baselines3, and Python dependencies
- [ ] Run the [MuJoCo ant or hopper example](https://mujoco.readthedocs.io) to verify setup
- [ ] Read the MuJoCo XML (MJCF) format documentation
- [ ] Skim the [Gymnasium custom environment guide](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [ ] Review the PPO or SAC algorithm at a conceptual level
- [ ] Set up a shared Git repo with a clean structure (`/env`, `/robot`, `/training`, `/paper`)
- [ ] Agree on coordinate conventions, reward scale, and logging format

**Shared deliverable:** Everyone can run a basic Gymnasium environment with MuJoCo rendering locally.

---

### 🔧 Phase 1 — Foundation (April 7 – April 20)

> Build the core robot, terrain, and training loop. Each person owns their domain but outputs must connect.

---

#### 🅐 Person A — Robot Design & Mechanics

**Goal:** A functioning 4-wheeled robot MJCF model that can receive velocity/torque commands.

- [ ] Design the MJCF XML for the 4-wheeled robot body (chassis, 4 wheels, axles)
- [ ] Define actuators: torque or velocity-controlled motors per wheel
- [ ] Add sensors: IMU (orientation/angular velocity), wheel odometry, contact forces
- [ ] Implement basic stability checks: detect if robot has tipped over (roll/pitch thresholds)
- [ ] Expose a clean `step(action)` interface — action = [left_drive, right_drive] or [4x wheel torques]
- [ ] Write unit tests: robot spawns correctly, wheels touch ground, no physics explosions

**Key deliverable:** `robot/robot.xml` + `robot/robot_interface.py`

---

#### 🅑 Person B — RL Agent & Reward Engineering

**Goal:** A training loop that can learn a basic navigation policy.

- [ ] Set up the Gymnasium `Env` wrapper around the MuJoCo simulation
- [ ] Define the **observation space**: robot position, orientation (quaternion or Euler), velocity, goal direction vector, terrain height samples in front of robot, proximity sensors
- [ ] Define the **action space**: continuous wheel torques / velocities
- [ ] Implement the **reward function**:
  - `+large` for reaching the goal waypoint
  - `+small` per step for reducing distance to goal
  - `-medium` per step if robot is tilted beyond threshold (weird angles)
  - `-large` one-time penalty for falling over (episode ends)
  - `-small` per step time penalty (penalize dawdling)
  - `-medium` one-time penalty for running out of time (timeout)
  - `+small` for touching a reward rock (colored green)
  - `-medium` for touching a penalty rock (colored red)
  - `-large` for entering a no-go zone
- [ ] Set up training with PPO (stable-baselines3) on a flat terrain first
- [ ] Add TensorBoard logging: episode reward, episode length, success rate, tilt events

**Key deliverable:** `training/env_wrapper.py`, `training/reward.py`, `training/train.py`

---

#### 🅒 Person C — Terrain & Environment Systems

**Goal:** A procedurally generated rocky terrain with colored rocks, obstacles, and no-go zones.

- [ ] Generate a height map for rocky terrain (use Perlin noise or MuJoCo `hfield`)
- [ ] Implement terrain loading into the MuJoCo scene via MJCF XML generation
- [ ] Place **colored rocks**:
  - Green rocks → reward (`+small` for touching)
  - Red rocks → penalty (`-medium` for touching)
  - Gray/brown rocks → neutral obstacles (collision only)
- [ ] Define **no-go zones**: flat red-tinted patches, trigger penalty on entry
- [ ] Place start waypoint (A) and goal waypoint (B) at fixed or randomized positions
- [ ] Expose an interface: `terrain.generate(seed, difficulty_params) → mjcf_scene`
- [ ] Validate scenes render without physics issues (no rock intersections, valid heights)

**Key deliverable:** `env/terrain_generator.py`, `env/scene_builder.py`

---

#### 🅓 Person D — Adversarial Terrain AI (Architecture Design)

**Goal:** Design and stub out the adversarial terrain generation system.

- [ ] Research adversarial/procedural content generation approaches:
  - PAIRED algorithm (Population Asymmetric Regret)
  - Unsupervised Environment Design (UED)
  - Simple evolutionary hill-climbing on difficulty params
- [ ] Define **terrain parameters** the adversary can tune: rock density, rock height variance, slope steepness, no-go zone placement, colored rock ratios
- [ ] Design the adversary's **objective**: maximize robot failure rate or minimize robot reward
- [ ] Stub out `AdversarialTerrainAgent` class with a `propose_terrain(robot_weakness_stats) → params` interface
- [ ] Write the **stats collector** that profiles robot weaknesses: where it fails (tipping? timeout? wrong path?), used as input to adversary
- [ ] Coordinate with Person C to ensure `terrain_generator.py` accepts the param interface

**Key deliverable:** `adversary/adversary_interface.py`, `adversary/weakness_tracker.py`

---

### 🔗 Phase 2 — Integration & First Training Runs (April 21 – May 4)

> Connect all modules. Get the robot training on real terrain.

- [ ] **A + C:** Integrate robot MJCF into terrain scenes — verify spawning, scale, and wheel-ground contact
- [ ] **B + C:** Connect terrain generator to the Gymnasium env — env resets regenerate terrain
- [ ] **B:** Run first real training experiment on rocky terrain (no adversary yet). Log and share results with group
- [ ] **D + C:** Implement the first version of the adversarial agent (start simple: random terrain param perturbation, then evolve toward harder params based on robot reward)
- [ ] **A:** Add visual debugging: camera follows robot, waypoints are visible, rock colors are correct
- [ ] **All:** Fix integration bugs. Hold 2 integration meetings this phase

**Key deliverable:** A complete training loop that runs end-to-end: spawn → train → log → terrain resets

---

### 🧪 Phase 3 — Experiments & Tuning (May 5 – May 16)

> Run ablations, tune hyperparameters, enable adversarial training.

- [ ] **B:** Hyperparameter sweep: learning rate, discount factor γ, PPO clip ratio
- [ ] **B:** Ablation: train with vs. without colored rocks, with vs. without time penalty
- [ ] **D:** Enable full adversarial training loop: robot trains → adversary observes failure stats → adversary proposes harder terrain → repeat
- [ ] **D:** Compare adversarial training vs. fixed curriculum vs. random terrain
- [ ] **C:** Generate a suite of 5–10 evaluation terrains (fixed seeds, varying difficulty) for final benchmarking
- [ ] **A:** Stress test robot mechanics on extreme terrain: very steep slopes, high rock density
- [ ] **All:** Record evaluation videos for the demo. Capture 3 scenarios: easy, medium, hard terrain

**Key deliverable:** Trained model checkpoints + evaluation metrics table + demo videos

---

### 🎬 Phase 4 — Demo Preparation (May 17 – May 18)

> Polish and present.

- [ ] **All:** Final demo run — robot navigates at least 3 different terrain seeds live
- [ ] **B:** Export trained model and write a `demo.py` script (no training, just inference + rendering)
- [ ] **D:** Live demo of adversarial agent proposing a new terrain patch during demo
- [ ] **A:** Ensure the robot visually looks good in the demo (no clipping, smooth motion)
- [ ] **C:** Prepare a side-by-side: flat terrain vs. rocky terrain navigation

**🏁 DEMO DEADLINE: May 18**

---

### 📝 Phase 5 — Paper Writing (May 19 – June 1)

> Each person writes their own section. Person D leads the paper structure.

| Section | Owner |
|---|---|
| Abstract | D (final polish by all) |
| 1. Introduction & Motivation | D |
| 2. Related Work (RL navigation, adversarial envs) | B |
| 3. Robot Design & Simulation Setup | A |
| 4. Environment: Terrain, Rewards, Observations | C |
| 5. Adversarial Terrain Generation Method | D |
| 6. Experiments & Results | B + D |
| 7. Discussion & Future Work | All |
| 8. Conclusion | D |
| Figures & Tables | All (own section figures) |

- [ ] Use IEEE or NeurIPS LaTeX template
- [ ] Include: training curves, evaluation table, terrain visualizations, robot diagrams
- [ ] Proofread and submit by June 1

**📄 PAPER DEADLINE: June 1**

---

## 📁 Recommended Repository Structure

```
/project-root
│
├── robot/
│   ├── robot.xml              # MJCF robot model (Person A)
│   └── robot_interface.py     # Step/reset API (Person A)
│
├── env/
│   ├── terrain_generator.py   # Procedural terrain (Person C)
│   └── scene_builder.py       # Combines robot + terrain (Person C)
│
├── training/
│   ├── env_wrapper.py         # Gymnasium Env class (Person B)
│   ├── reward.py              # Reward function (Person B)
│   └── train.py               # Training entry point (Person B)
│
├── adversary/
│   ├── adversary_interface.py # Adversarial terrain agent (Person D)
│   └── weakness_tracker.py    # Robot failure profiler (Person D)
│
├── experiments/
│   ├── configs/               # YAML hyperparameter configs
│   ├── checkpoints/           # Saved model weights
│   └── logs/                  # TensorBoard logs
│
├── demo.py                    # Inference-only demo script
├── paper/                     # LaTeX paper
└── README.md
```

---

## ⚠️ Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| MuJoCo physics instability (robot explodes) | Medium | Reduce timestep, add joint damping, test early |
| Adversarial agent too strong → robot can't learn | Medium | Cap adversary difficulty; use regret-based curriculum |
| Slow training convergence | High | Use PPO with parallel envs (SubprocVecEnv); start on flat terrain |
| Integration delays between modules | Medium | Define clear interfaces in Phase 0; weekly syncs |
| Time running out before demo | Low-Medium | Demo can use a partially trained policy; prioritize stability |

---

## ✅ Weekly Milestone Checklist

| Week | Milestone |
|---|---|
| Apr 6 | All 4 members have MuJoCo running; repo initialized |
| Apr 13 | Robot model spawns in terrain; basic env wrapper works |
| Apr 20 | Phase 1 deliverables complete; first flat-terrain training run done |
| Apr 27 | Full integration working; rocky terrain training started |
| May 4 | Adversarial agent (v1) connected; first adversarial training run |
| May 11 | Ablation experiments complete; demo video recorded |
| May 18 | 🏁 **Working Demo** |
| May 25 | All paper sections drafted |
| June 1 | 📄 **Paper submitted** |

---

*Good luck! The adversarial terrain generator is the standout feature — make sure to give it plenty of experiment time in Phase 3.*
