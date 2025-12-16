import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))
from paper import PaperSim

def test_rain_contacts():
    sim = PaperSim(L=10, W=5)
    # Flat paper
    sim.update(angle=0, curvature_controls=[0, 0])
    
    contacts = sim.get_rain_contacts(n_drops=100)
    print(f"Flat Paper Contacts: {len(contacts)}")
    if len(contacts) > 0:
        print(f"Mean Z (should be ~0): {np.mean(contacts[:, 2]):.4f}")
        # Check bounds
        print(f"X range: {contacts[:, 0].min():.2f} to {contacts[:, 0].max():.2f} (should be approx -5 to 5)")
        print(f"Y range: {contacts[:, 1].min():.2f} to {contacts[:, 1].max():.2f} (should be approx -2.5 to 2.5)")

    # Bended paper
    k = np.pi / 10.0 # moderate bend
    sim.update(angle=0, curvature_controls=[k, k])
    contacts = sim.get_rain_contacts(n_drops=100)
    print(f"\nBent Paper Contacts: {len(contacts)}")
    if len(contacts) > 0:
        print(f"Z range: {contacts[:, 2].min():.4f} to {contacts[:, 2].max():.4f}")

    # Rotated paper
    sim.update(angle=np.pi/4, curvature_controls=[0, 0])
    contacts = sim.get_rain_contacts(n_drops=100)
    print(f"\nRotated Flat Paper Contacts: {len(contacts)}")
    # Verify coplanar?
    
if __name__ == "__main__":
    test_rain_contacts()
