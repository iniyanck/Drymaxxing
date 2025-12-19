import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import os
import sys
import multiprocessing

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import PaperEnv

def make_env(rank, seed=0, **kwargs):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        env = PaperEnv(**kwargs)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000000, help="Number of training steps")
    parser.add_argument("--test", action="store_true", help="Just test the env and model setup")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for training")
    parser.add_argument("--n_envs", type=int, default=0, help="Number of parallel environments (0 = auto detect cpu count)")
    args = parser.parse_args()

    # Determine num envs
    if args.n_envs == 0:
        cpu_count = multiprocessing.cpu_count()
        # Cap at 8 or similar to prevent crashing on high-core consumer PCs
        # Running too many heavy python processes (matplotlib/numpy) can cause instability.
        safe_max = 8
        n_envs = min(cpu_count, safe_max)
        # Ensure at least 1, and assume we want to leave some headroom if we have many cores
        if cpu_count > 4:
            n_envs = min(n_envs, cpu_count - 2)
        else:
             n_envs = max(1, cpu_count - 1)
             
        print(f"Auto-detected {cpu_count} CPUs. Using {n_envs} parallel environments to prevent system overload.")
    else:
        n_envs = args.n_envs

    # Directories
    models_dir = "models/PPO"
    log_dir = "logs"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Environment Setup
    # Turn off rain effects for training to speed up
    env_kwargs = {"render_rain_effects": False}
    
    if args.test:
        print("Testing single environment...")
        env = PaperEnv(**env_kwargs)
        check_env(env)
        print("Environment check passed.")
        vec_env = DummyVecEnv([lambda: env])
        TIMESTEPS = 1000
    else:
        print(f"Creating {n_envs} parallel environments...")
        # Use SubprocVecEnv for true parallelism
        vec_env = make_vec_env(
            PaperEnv, 
            n_envs=n_envs, 
            seed=0, 
            vec_env_cls=SubprocVecEnv,
            env_kwargs=env_kwargs
        )
        TIMESTEPS = args.steps

    # Initialize PPO Agent
    print(f"Using device: {args.device}")
    
    # Tuned Hyperparameters for Speed and Stability
    # larger batch size is better for GPU
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,           # Steps per env per update
        batch_size=2048,        # Large batch size for efficiency
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device=args.device 
    )

    print(f"Training for {TIMESTEPS} timesteps...")
    
    # Callback uses total steps across all envs? No, total_timesteps in learn is global.
    # Checkpoint freq is in steps of the vectorized env.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000 // n_envs, 2048), 
        save_path=models_dir, 
        name_prefix="ppo_paper"
    )
    
    try:
        model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
        print("Training complete.")
        model.save(f"{models_dir}/ppo_paper_final")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        model.save(f"{models_dir}/ppo_paper_interrupted")
        
    vec_env.close()

    if args.test: return

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
