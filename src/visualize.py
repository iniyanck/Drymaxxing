import os
import sys
import webbrowser
import uvicorn
import threading
import time

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    time.sleep(1.5) # Wait for server
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("Starting Drymaxxing Server...")
    # Start browser in a separate thread so it doesn't block if we were doing other things,
    # but strictly uvicorn blocks main thread.
    threading.Thread(target=open_browser, daemon=True).start()
    
    # We can run via uvicorn direct invocation or via shell. 
    # Importing app from server to run programmatically
    try:
        from server import app
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except ImportError:
        # If running from root without proper python path setup
        os.system("uvicorn src.server:app --reload --port 8000")
