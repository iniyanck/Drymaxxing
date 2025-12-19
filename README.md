# Drymaxxing: AI Paper Folding Simulation

**Drymaxxing** is a reinforcement learning project where an AI agent learns to autonomously manipulate a flexible sheet of paper to minimize its wet surface area when exposed to rain. The agent controls the paper's position, orientation, and curvature to dodge or shield itself from dynamic rain conditions.

## Features

- **Physically-based Paper Simulation**:
  - 6-DoF Rigid Body Dynamics (Position & Rotation).
  - Variable curvature profile simulation allowing the paper to bend and roll.
  - Self-intersection detection and wetness masking.
- **Reinforcement Learning Environment**:
  - Built with `Gymnasium`.
  - Continuous action space controlling velocities and curvature.
  - Reward function incentivizing dryness, stability, and collision avoidance.
- **Advanced Training**:
  - Uses Proximal Policy Optimization (PPO) from `Stable Baselines 3`.
  - Supports parallel training environments for faster convergence.
- **Visualization**:
  - Real-time web-based 3D visualization using FastAPI and WebSockets.
  - Legacy Matplotlib 3D debugging view.
- **Dynamic Weather**:
  - Simulation of rain with variable and drifting directions.
  - Accurate "wetness" calculation using robust Monte Carlo projected area analysis.

## Project Structure

```
Drymaxxing/
├── src/
│   ├── env.py          # Gymnasium Environment (PaperEnv)
│   ├── paper.py        # Core physics and paper simulation logic
│   ├── train_ppo.py    # Training script using PPO
│   ├── server.py       # FastAPI backend for web visualization
│   └── static/         # Frontend assets for visualization
├── models/             # Saved RL models
├── logs/               # Tensorboard logs
└── README.md
```

## Installation

Ensure you have Python 3.8+ installed. Install the dependencies:

```bash
pip install numpy matplotlib gymnasium stable-baselines3 fastapi uvicorn
```

## Usage

### 1. Train the Agent

To start training the PPO agent:

```bash
python src/train_ppo.py
```

Arguments:
- `--steps`: Total training timesteps (default: 1,000,000).
- `--n_envs`: Number of parallel environments (default: auto-detect).
- `--device`: `cpu` or `cuda` (default: auto).
- `--test`: Run a single environment test without training.

### 2. Visualize the Simulation

To watch the agent in action (using trained weights or random actions):

1. Start the visualization server:
   ```bash
   python src/server.py
   ```
2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Implementation Details

- **Observation Space**: The agent observes its own state (position, orientation, curvature) and the rain vector (global and local).
- **Action Space**: The agent outputs velocity commands for movement ($v_x, v_y, v_z$), rotation ($\omega_x, \omega_y, \omega_z$), and curvature rate of change.
- **Wetness Calculation**: The simulation projects the rain direction onto the paper's mesh to calculate the exposed surface area effectively.
