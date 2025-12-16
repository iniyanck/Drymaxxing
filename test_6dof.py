import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))
from paper import PaperSim

def test_translation():
    print("\n[Test] Translation")
    sim = PaperSim(L=10, W=5)
    
    # Flat paper at zero
    sim.update([0,0,0], [0,0,0], [0,0])
    v0 = sim.vertices.copy()
    center0 = np.mean(v0, axis=(0,1))
    print(f"Initial Center: {center0}")
    
    # Move to (10, 20, 30)
    target_pos = np.array([10.0, 20.0, 30.0])
    sim.update(target_pos, [0,0,0], [0,0])
    v1 = sim.vertices.copy()
    center1 = np.mean(v1, axis=(0,1))
    print(f"New Center: {center1}")
    
    diff = center1 - center0
    err = np.linalg.norm(diff - target_pos)
    print(f"Translation Error: {err}")
    
    if err < 1e-5:
        print("PASS: Translation correct.")
    else:
        print("FAIL: Translation incorrect.")

def test_rotation():
    print("\n[Test] Rotation (Yaw 90)")
    sim = PaperSim(L=10, W=5)
    
    # Flat paper, width along Y.
    # Vertices should be roughly X: [-5, 5], Y: [-2.5, 2.5], Z: 0
    sim.update([0,0,0], [0,0,0], [0,0])
    v0 = sim.vertices
    
    print(f"Pre-Rot Extents X: {v0[:,:,0].min():.2f} to {v0[:,:,0].max():.2f}")
    print(f"Pre-Rot Extents Y: {v0[:,:,1].min():.2f} to {v0[:,:,1].max():.2f}")
    
    # Rotate 90 deg around Z (Yaw)
    # X should become Y, Y should become -X (or X depending on sign convention)
    sim.update([0,0,0], [0, 0, np.pi/2], [0,0])
    v1 = sim.vertices
    
    print(f"Post-Rot Extents X: {v1[:,:,0].min():.2f} to {v1[:,:,0].max():.2f}")
    print(f"Post-Rot Extents Y: {v1[:,:,1].min():.2f} to {v1[:,:,1].max():.2f}")
    
    # Check if dimensions swapped
    # X extents should match old Y extents
    x_span = v1[:,:,0].max() - v1[:,:,0].min()
    y_span = v1[:,:,1].max() - v1[:,:,1].min()
    
    print(f"New X Span: {x_span:.2f} (Expected 5.0)")
    print(f"New Y Span: {y_span:.2f} (Expected 10.0)")
    
    if abs(x_span - 5.0) < 0.1 and abs(y_span - 10.0) < 0.1:
        print("PASS: Rotation Dimensions match.")
    else:
        print("FAIL: Rotation Dimensions mismatch.")

def test_rain_in_frame():
    print("\n[Test] Rain Contacts Transform")
    sim = PaperSim(L=10, W=10) # 10x10 square
    
    # Rotate 45 degrees Roll. Face is now diagonal.
    # Tilt it so it faces Rain more or less?
    # Rain is hitting from +Z. 
    # If we roll 90 degrees, paper is vertical (YZ plane). 
    # Rain (0,0,-1) should graze it (0 area) or hit the thin edge.
    # Area calculation should be 0.
    
    sim.update([0,0,0], [np.pi/2, 0, 0], [0,0])
    area = sim.calculate_wet_area(resolution=100)
    print(f"Wet Area at 90 deg Roll: {area} (Expected ~0)")
    
    if area < 1.0:
        print("PASS: Vertical paper gets no rain area.")
    else:
        print("FAIL: Vertical paper got significant area.")
        
    # Check contacts
    # Should be empty or very few hits on edge
    contacts = sim.get_rain_contacts(n_drops=100)
    print(f"Contacts count: {len(contacts)}")

if __name__ == "__main__":
    test_translation()
    test_rotation()
    test_rain_in_frame()
