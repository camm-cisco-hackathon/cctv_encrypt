from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import cv2
import os
import time
import uuid
import numpy as np
from glob import glob

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create record directory if it doesn't exist
os.makedirs("./record", exist_ok=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    frame_count = 0
    session_id = uuid.uuid4()
    
    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)
            print(f"Received message type: {data['type']}")

            if data["type"] == "video_frame":
                # Decode base64 image
                img_data = base64.b64decode(data["data"])
                
                # Convert to numpy array for OpenCV
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Save image to ./record directory
                timestamp = int(time.time() * 1000)
                filename = f"./record/frame_{session_id}_{timestamp}_{frame_count}.jpg"
                cv2.imwrite(filename, img)
                frame_count += 1
                
                # Send confirmation back to client
                await websocket.send_json({
                    "type": "frame_received",
                    "frame_number": frame_count
                })
            
            elif data["type"] == "stream_request":
                # Get session ID to stream (optional, stream all if not provided)
                target_session = data.get("session_id", None)
                
                if target_session:
                    frames = sorted(glob(f"./record/frame_{target_session}_*.jpg"))
                else:
                    frames = sorted(glob("./record/frame_*.jpg"))
                
                for frame_path in frames:
                    # Read the image
                    img = cv2.imread(frame_path)
                    if img is None:
                        continue
                    
                    # Encode to base64
                    _, buffer = cv2.imencode(".jpg", img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame to client
                    await websocket.send_json({
                        "type": "stream_frame",
                        "data": img_base64,
                        "filename": os.path.basename(frame_path)
                    })
                    
                    # Brief pause between frames
                    time.sleep(0.1)  # Adjust streaming speed as needed
                
                await websocket.send_json({
                    "type": "stream_complete"
                })

        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=52049)