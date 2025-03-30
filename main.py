import os
import json
import asyncio
import websockets
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

url = "wss://api.openai.com/v1/realtime?intent=transcription"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta": "realtime=v1"
}

message_queue = [
    "What Prince album sold the most copies?",
    "Who was the lead guitarist for Queen?",
    "What year was the first iPhone released?"
]

async def send_message(websocket, text):
    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": text,
                }
            ]
        }
    }
    await websocket.send(json.dumps(event))
    print(f"Sent: {json.dumps(event, indent=2)}")

async def websocket_client():
    async with websockets.connect(url, additional_headers=headers) as websocket:
        print("Connected to server.")
        message_sent = False
        
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print("Received event:", json.dumps(data, indent=2))
            
            if data.get("type") == "transcription_session.created" and not message_sent:
                if message_queue:
                    next_message = message_queue.pop(0)
                    await send_message(websocket, next_message)

                    event = {
                        "type": "response.create",
                        "response": {
                            "modalities": [ "text" ]
                        }
                    }
                    await websocket.send(json.dumps(event))

                    message_sent = True
            
            elif data.get("type") == "response.done":
                if message_queue:
                    next_message = message_queue.pop(0)
                    await send_message(websocket, next_message)

# Run the WebSocket client
asyncio.run(websocket_client())
