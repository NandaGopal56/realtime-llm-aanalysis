import asyncio
import websockets
import argparse
import os
import sys
from watchfiles import awatch

# Handle incoming WebSocket connections
async def handle_client(websocket):
    try:
        async for message in websocket:
            print(f"Received: {message}")
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")

# Start the WebSocket server
async def start_server():
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    
    # This keeps the server running indefinitely
    await asyncio.Future()  

# Watch for file changes and restart the process
async def watch_for_changes():
    async for _ in awatch("."):  # Watch current directory
        print("Code changed. Restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    if args.dev:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(watch_for_changes())
        loop.run_until_complete(start_server())
    else:
        asyncio.run(start_server())