import numpy as np
import torch
from src.paper import PaperSim
from src.torch_sim import TorchPaperSim

def test_compare_cpu_gpu():
    print("Initializing Sims...")
    L, W = 10.0, 5.0
    cpu_sim = PaperSim(L=L, W=W)
    gpu_sim = TorchPaperSim(L=L, W=W, batch_size=1, device="cpu") # Use CPU for torch to easy compare
    
    # State
    pos = np.array([0.0, 0.0, 0.0])
    euler = np.array([0.1, 0.0, 0.0]) # Slight roll
    k_val = 0.1
    # Controls: just constant curvature
    # CPU expects list/array. GPU expects tensor (B, K)
    controls_np = np.array([k_val]*5)
    controls_torch = torch.tensor([k_val]*5).unsqueeze(0).float()
    
    pos_torch = torch.tensor(pos).unsqueeze(0).float()
    euler_torch = torch.tensor(euler).unsqueeze(0).float()
    
    # Update
    print("Updating...")
    cpu_sim.update(pos, euler, controls_np)
    gpu_sim.update(pos_torch, euler_torch, controls_torch)
    
    # Check Vertices
    # Vertices shape: (N_p, N_r, 3)
    cpu_verts = cpu_sim.vertices
    gpu_verts = gpu_sim.vertices.squeeze(0).numpy()
    
    diff = np.abs(cpu_verts - gpu_verts)
    max_diff = np.max(diff)
    print(f"Max Vertex Diff: {max_diff}")
    
    if max_diff < 1e-4:
        print("Geometry matches!")
    else:
        print("Geometry mismatch!")
        
    # Check Rain
    print("Applying Rain...")
    rain_dir = np.array([0.0, 0.0, -1.0])
    rain_dir_torch = torch.tensor(rain_dir).float().unsqueeze(0)
    
    # We can't compare exact drops because of randomness.
    # But we can check wet area calculation function if we fix rain?
    # No, wet area depends on wet mask which depends on rain.
    
    # Let's just run rain and see if it runs without error and produces somewhat plausible result.
    # Or force seed.
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # It's hard to sync random numbers between numpy and torch distributions.
    # Just check if wet area is properly calculated (non-zero).
    
    gpu_sim.apply_rain(rain_dir_torch, n_drops=100)
    gpu_area = gpu_sim.calculate_wet_area(rain_dir_torch).item()
    
    print(f"GPU Wet Area: {gpu_area}")
    
    # Run CPU rain just to see
    cpu_sim.apply_rain(rain_dir, n_drops=100)
    cpu_area = cpu_sim.calculate_wet_area(rain_dir)
    print(f"CPU Wet Area: {cpu_area}")
    
    if gpu_area > 0 and cpu_area > 0:
        print("Both sims registered wetness.")
    
    print("Test Complete.")

if __name__ == "__main__":
    test_compare_cpu_gpu()
