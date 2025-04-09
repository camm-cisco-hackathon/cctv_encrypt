from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import cv2
import os
import time
import numpy as np
import asyncio
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
    streaming = False
    
    while True:
        try:
            # Set up a task to receive messages without blocking stream updates
            receive_task = asyncio.create_task(websocket.receive_text())
            
            # If streaming mode is on, also create a timer task
            if streaming:
                timer_task = asyncio.create_task(asyncio.sleep(1))
                done, pending = await asyncio.wait(
                    [receive_task, timer_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                
                # If the timer completed, send the latest frame
                if timer_task in done:
                    frames = sorted(glob("./record/frame_*.jpg"))
                    if frames:
                        most_recent_frame = frames[-1]
                        img = cv2.imread(most_recent_frame)
                        if img is not None:
                            _, buffer = cv2.imencode(".jpg", img)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            await websocket.send_json({
                                "type": "stream_frame",
                                "data": img_base64,
                                "filename": os.path.basename(most_recent_frame)
                            })
                    continue
            else:
                # If not streaming, just wait for messages
                await receive_task
            
            # Process the received message
            message = receive_task.result()
            data = json.loads(message)
            print(f"Received message type: {data['type']}")

            if data["type"] == "video_frame":
                # Decode base64 image
                img_data = base64.b64decode(data["data"])
                
                # Convert to numpy array for OpenCV
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Save image to ./record directory without session_id
                timestamp = int(time.time() * 1000)
                filename = f"./record/frame_{timestamp}_{frame_count}.jpg"
                cv2.imwrite(filename, img)
                frame_count += 1
                
                # Send confirmation back to client
                await websocket.send_json({
                    "type": "frame_received",
                    "frame_number": frame_count
                })
            
            elif data["type"] == "stream_request":
                # Start streaming mode
                streaming = True
                
                # Send initial frame
                frames = sorted(glob("./record/frame_*.jpg"))
                if frames:
                    most_recent_frame = frames[-1]
                    img = cv2.imread(most_recent_frame)
                    if img is not None:
                        _, buffer = cv2.imencode(".jpg", img)
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        await websocket.send_json({
                            "type": "stream_frame",
                            "data": img_base64,
                            "filename": os.path.basename(most_recent_frame)
                        })
            
            elif data["type"] == "stop_stream":
                # Stop streaming mode
                streaming = False
                await websocket.send_json({
                    "type": "stream_complete"
                })

        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=52049)