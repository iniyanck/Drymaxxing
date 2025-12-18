import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import PaperEnv

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--test", action="store_true", help="Just test the env and model setup")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device to use for training")
    args = parser.parse_args()

    # Create environment
    env = PaperEnv()
    
    # Check if env is valid
    print("Checking environment...")
    check_env(env)
    print("Environment is valid.")

    # Model directory
    models_dir = "models/PPO"
    log_dir = "logs"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize PPO Agent
    print(f"Using device: {args.device}")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        device=args.device 
    )

    TIMESTEPS = args.steps
    if args.test:
        TIMESTEPS = 1000
    
    print(f"Training for {TIMESTEPS} timesteps...")
    
    checkpoint_callback = CheckpointCallback(save_freq=max(1000, TIMESTEPS//5), save_path=models_dir, name_prefix="ppo_paper")
    
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    print("Training complete.")
    
    # Save final model
    model.save(f"{models_dir}/ppo_paper_final")

    if args.test: return

    # Evaluate
    print("Evaluating...")
    obs, _ = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    main()
