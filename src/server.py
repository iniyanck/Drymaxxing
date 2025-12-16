import asyncio
import json
import threading
import time
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np

# Import environment
from env import PaperEnv

app = FastAPI()

# Global State
class SimulationState:
    def __init__(self):
        self.env = PaperEnv()
        self.running = True
        self.latest_data = {}
        self.lock = threading.Lock()
        
        # Manual Control
        self.manual_action = None
        self.last_manual_input = 0.0
        self.manual_timeout = 0.2 # Reset to none if no input for 0.2s

    def run_loop(self):
        obs, _ = self.env.reset()
        
        # Try to load weights
        weights = None
        try:
            weights = np.load("agent_weights.npy")
            print("Loaded agent weights.")
        except:
            print("No weights found. Running random policy.")

        # Agent helper
        def get_action(w, o):
            # Prioritize Manual
            with self.lock:
                if self.manual_action is not None and (time.time() - self.last_manual_input < self.manual_timeout):
                    return self.manual_action.copy()
            
            # Same logic as train.py CEMAgent
            if w is None:
                return self.env.action_space.sample() * 0.1 # Small random moves
            
            # Reconstruct agent dims from env
            obs_dim = self.env.observation_space.shape[0]
            act_dim = self.env.action_space.shape[0]
            
            W = w.reshape(act_dim, obs_dim)
            return np.tanh(W @ o)

        while self.running:
            try:
                action = get_action(weights, obs)
            except Exception as e:
                # If weights are incompatible or any other error, fallback to random
                # print(f"Agent error: {e}")
                action = self.env.action_space.sample() * 0.1

            obs, reward, done, truncated, info = self.env.step(action)
            
            # Prepare data for frontend
            # Vertices: (N_p, N_r, 3)
            verts = self.env.sim.vertices
            
            # Rain contacts (Global) respecting current rain direction
            contacts_uv, contacts_world = self.env.sim.get_rain_contacts(
                n_drops=200, 
                bounds=self.env.workspace_bounds,
                rain_dir=self.env.rain_dir
            )
            
            # Stats
            wet_area = info.get("wet_area", 0)
            
            with self.lock:
                self.latest_data = {
                    "vertices": verts.tolist() if verts is not None else [],
                    "rain_uv": contacts_uv.tolist(), 
                    "rain_hits": contacts_world.tolist(), 
                    "wet_area": float(wet_area),
                    "pos": self.env.pos.tolist(),
                    "euler": self.env.euler.tolist(),
                    "curvatures": self.env.curvatures.tolist(),
                    "rain_dir": self.env.rain_dir.tolist()
                }
            
            
            if done or truncated:
                obs, _ = self.env.reset()
                
            time.sleep(0.05) # ~20 FPS simulation speed

sim_state = SimulationState()

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=sim_state.run_loop, daemon=True)
    t.start()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receiver Task
        async def receive_loop():
            try:
                while True:
                    data = await websocket.receive_text()
                    # We can handle camera events if needed, but for now client handles camera locally.
                    # Just keep alive or log.
                    # msg = json.loads(data)
                    pass
            except Exception as e:
                print(f"WS Rx Error: {e}")

        # Start Receiver
        asyncio.create_task(receive_loop())

        while True:
            # Send latest state
            with sim_state.lock:
                data = sim_state.latest_data
            
            if data:
                await websocket.send_json(data)
            
            await asyncio.sleep(0.033) # 30 FPS send rate
    except Exception as e:
        print(f"WebSocket Error: {e}")

# Mount static files
app.mount("/", StaticFiles(directory="src/static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
