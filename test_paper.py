import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from paper import PaperSim
    print("Import Successful")
except ImportError as e:
    print(f"Import Failed: {e}")
    sys.exit(1)

def test_flat():
    print("Testing Flat Paper...")
    sim = PaperSim(L=10, W=5)
    # New Signature: update(pos, euler, curv)
    sim.update([0,0,0], [0,0,0], [0, 0])
    
    verts = sim.vertices
    print(f"Vertices Shape: {verts.shape}")
    
    # Check bounds
    z_range = verts[:, :, 2].max() - verts[:, :, 2].min()
    print(f"Z Range (flat): {z_range:.4f}")
    if z_range > 1e-5:
        print("ERROR: Flat paper should have 0 Z range.")
    else:
        print("PASS: Flat paper is flat.")

    area = sim.calculate_wet_area()
    print(f"Flat Area: {area}")
    expected = 10 * 5
    if abs(area - expected) < 3.0: # Monte Carlo tolerance
        print("PASS: Area is correct.")
    else:
        print(f"FAIL: Expected {expected}, got {area}")

def test_cylinder():
    print("\nTesting 90-degree bend...")
    sim = PaperSim(L=10, W=5)
    # k = pi / (2 * L) for 90 degree arc
    # actually L = R * theta => 10 = R * (pi/2) => R = 20/pi
    # k = 1/R = pi/20
    k = 3.14159 / 20.0
    sim.update([0,0,0], [0,0,0], [k, k])
    
    verts = sim.vertices
    # Check max Z
    z_range = verts[:, :, 2].max() - verts[:, :, 2].min()
    print(f"Z Range (curved): {z_range:.4f}")
    
    # Check area
    area = sim.calculate_wet_area(resolution=100)
    print(f"Projected Area: {area:.4f}")
    
    # Monte Carlo variance can be high with low samples in default, but let's check basic reduction
    if abs(area - 45.0) < 5.0:
        print("PASS: Area is reduced as expected.")
    else:
        print(f"FAIL: Area seems wrong. Expected ~45.0")

if __name__ == "__main__":
    test_flat()
    test_cylinder()
