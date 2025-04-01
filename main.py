import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS for cross-origin WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueueManager:
    _queue = None

    @classmethod
    def get_queue(cls):
        """Returns the global queue, initializing it only once."""
        if cls._queue is None:
            cls._queue = asyncio.Queue()
        return cls._queue

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Ensure WebSocket connection is accepted properly
    queue = QueueManager.get_queue()
    
    print(f"Client connected: {websocket.client}")
    
    try:
        while True:
            message = await websocket.receive_bytes()
            await queue.put(message)
    except WebSocketDisconnect:
        print(f"Client disconnected: {websocket.client}")

async def websocket_audio_generator():
    """Async generator that yields data from the global queue."""
    queue = QueueManager.get_queue()
    print(id(queue))
    while True:
        data = await queue.get()
        print("Len of queue in generator", queue.qsize())
        yield data

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8765)