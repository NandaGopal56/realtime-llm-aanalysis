import asyncio
import uvicorn
import json
import base64
from redis import asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from globals import REDIS_CHANNEL


app = FastAPI()

# Enable CORS for cross-origin WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    """Handles WebSocket connections and directly publishes audio to Redis."""
    await websocket.accept()
    redis = await aioredis.Redis.from_url("redis://localhost")

    print(f"Client connected - User: {user_id}, Session: {session_id}")

    try:
        while True:
            audio_data = await websocket.receive_bytes()

            if audio_data:
                # Base64 encode the audio bytes to make them JSON serializable
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
                # Create the message dictionary
                message = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "audio": audio_base64  # Base64-encoded audio data
                }
            
                # Serialize the message to JSON string before publishing
                message_json = json.dumps(message)
                await redis.publish(REDIS_CHANNEL, message_json)  # Publish JSON string
                print(f"Published audio for User: {user_id}, Session: {session_id}, audio_base64: {audio_data[0]}")

    except WebSocketDisconnect:
        print(f"Client disconnected - User: {user_id}, Session: {session_id}")
    finally:
        await redis.aclose()



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8765)