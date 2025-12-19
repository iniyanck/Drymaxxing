import asyncio
import json
import threading
import time
import uvicorn
import os
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
from stable_baselines3 import PPO

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

    def handle_message(self, msg):
        try:
            if msg['type'] == 'reset':
                with self.lock:
                    self.env.reset()
                    # Keep rain mode if it was set to fixed, or maybe reset to dynamic? 
                    # User likely wants to keep their settings, but reset() randomizes rain.
                    # We should probably re-apply desired rain if we are in fixed mode?
                    # For now simpliest is just reset.
            
            elif msg['type'] == 'set_rain':
                d = np.array(msg['dir'])
                norm = np.linalg.norm(d)
                if norm > 0:
                    d = d / norm
                    with self.lock:
                        self.env.rain_dir = d
                        # self.env.rain_mode = 'fixed' # Don't stop drifting, user wants random variation + control

                        
            elif msg['type'] == 'set_rain_mode':
                with self.lock:
                    self.env.rain_mode = msg.get('mode', 'dynamic')

        except Exception as e:
            print(f"Error handling message: {e}")

    def run_loop(self):
        obs, _ = self.env.reset()
        
        # Try to load weights
        weights = None
        ppo_model = None
        
        ppo_path = "models/PPO/ppo_paper_final.zip"
        cem_path = "agent_weights.npy"
        
        if os.path.exists(ppo_path):
            print(f"Loading PPO model from {ppo_path}...")
            try:
                ppo_model = PPO.load(ppo_path)
                print("Loaded PPO model successfully.")
            except Exception as e:
                print(f"Error loading PPO model: {e}")
        
        if ppo_model is None:
            if os.path.exists(cem_path):
                try:
                    weights = np.load(cem_path)
                    print(f"Loaded CEM agent weights from {cem_path}.")
                except Exception as e:
                    print(f"Error loading CEM weights: {e}")
            else:
                print("No weights found. Running random policy.")

        # Agent helper
        def get_action(w, model, o):
            # Prioritize Manual
            with self.lock:
                if self.manual_action is not None and (time.time() - self.last_manual_input < self.manual_timeout):
                    return self.manual_action.copy()
            
            # PPO Model
            if model is not None:
                action, _ = model.predict(o, deterministic=True)
                return action
            
            # CEM Weights
            if w is not None:
                # Reconstruct agent dims from env
                obs_dim = self.env.observation_space.shape[0]
                act_dim = self.env.action_space.shape[0]
                W = w.reshape(act_dim, obs_dim)
                return np.tanh(W @ o)
            
            # Random Fallback
            return self.env.action_space.sample() * 0.1 # Small random moves

        while self.running:
            try:
                action = get_action(weights, ppo_model, obs)
            except Exception as e:
                # If weights are incompatible or any other error, fallback to random
                print(f"Agent error: {e}")
                action = self.env.action_space.sample() * 0.1

            with self.lock:
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
                    try:
                         msg = json.loads(data)
                         sim_state.handle_message(msg)
                    except json.JSONDecodeError:
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
