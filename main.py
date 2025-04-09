from datetime import datetime
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
import encrypt  # Import our encryption module
import tempfile
import ffmpeg

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create all necessary directories if they don't exist
os.makedirs("./record", exist_ok=True)
os.makedirs("./record_mosaic", exist_ok=True)
os.makedirs("./record_encrypt", exist_ok=True)

INTERVAL = 0.5

# RTSP stream settings
width, height = 1920, 1080
rtsp_url = 'rtsp://192.168.128.11:9000/live'
frame_interval = INTERVAL  # Capture a frame every 0.5 seconds

# Frame processing task
async def process_rtsp_stream():
    # Create directories if they don't exist
    os.makedirs("./record", exist_ok=True)
    os.makedirs("./record_mosaic", exist_ok=True)
    os.makedirs("./record_encrypt", exist_ok=True)
    
    # ffmpeg process to receive RTSP stream
    process = (
        ffmpeg
        .input(rtsp_url, rtsp_transport='tcp')
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
        .run_async(pipe_stdout=True)
    )
    
    frame_count = 0
    last_saved_time = time.time()
    
    try:
        while True:
            # Read frame from RTSP stream
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                print("[ERROR] No bytes received from RTSP stream")
                break
                
            # Convert bytes to numpy array (frame)
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            now = time.time()
            
            # Save frame at specified interval
            if now - last_saved_time >= frame_interval:
                # Save original image
                filename = f"frame_{frame_count:04d}.jpg"
                file_path = f"./record/{filename}"
                cv2.imwrite(file_path, frame)
                last_saved_time = now
                
                try:
                    # Apply face mosaic
                    mosaic_img = encrypt.apply_face_mosaic(frame)
                    mosaic_path = f"./record_mosaic/{filename}"
                    cv2.imwrite(mosaic_path, mosaic_img)
                    
                    # Encrypt original image
                    key = encrypt.generate_key(encrypt.ENCRYPTION_KEY)
                    encrypt_path = f"./record_encrypt/{filename}.enc"
                    encrypt.encrypt_file(file_path, key, encrypt_path)
                    
                    # Delete original image after mosaicking and encryption
                    if os.path.exists(mosaic_path) and os.path.exists(encrypt_path):
                        os.remove(file_path)
                        print(f"[Deleted original] {filename}")

                    print(f"[Processed] {filename}")
                    frame_count += 1
                except Exception as e:
                    print(f"Error processing image: {e}")
            
            # Wait before processing next frame
            await asyncio.sleep(0.01)  # Small delay to prevent CPU hogging
    except Exception as e:
        print(f"RTSP stream error: {e}")
    finally:
        # Clean up
        process.terminate()
        print("[INFO] RTSP stream processing stopped")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    streaming = False
    use_decryption = False
    decryption_key = None
    
    while True:
        try:
            # Set up a task to receive messages without blocking stream updates
            receive_task = asyncio.create_task(websocket.receive_text())
            
            # If streaming mode is on, also create a timer task
            if streaming:
                timer_task = asyncio.create_task(asyncio.sleep(INTERVAL))
                done, pending = await asyncio.wait(
                    [receive_task, timer_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                
                # If the timer completed, send the latest frame
                if timer_task in done:
                    if use_decryption and decryption_key:
                        # Use original encrypted images when decryption key is provided
                        encrypted_frames = sorted(glob("./record_encrypt/frame_*.jpg.enc"))
                        if encrypted_frames:
                            most_recent_frame = encrypted_frames[-1]
                            original_filename = os.path.basename(most_recent_frame)[:-4]  # Remove .enc
                            
                            try:
                                # Create a temporary file for decrypted content
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                    temp_path = temp_file.name
                                
                                # Decrypt the file
                                encrypt.decrypt_file(most_recent_frame, decryption_key, temp_path)
                                
                                # Read the decrypted image
                                img = cv2.imread(temp_path)
                                
                                # Remove temporary file
                                os.unlink(temp_path)
                                
                                if img is not None:
                                    _, buffer = cv2.imencode(".jpg", img)
                                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                                    
                                    await websocket.send_json({
                                        "type": "stream_frame",
                                        "data": img_base64,
                                        "filename": original_filename,
                                        "decrypted": True
                                    })
                            except Exception as e:
                                print(f"Decryption error: {e}")
                                # Fall back to mosaic if decryption fails
                                use_decryption = False
                                await websocket.send_json({
                                    "type": "decryption_error",
                                    "message": "Failed to decrypt images. Falling back to mosaic."
                                })
                    else:
                        # Use mosaic images when no decryption key or decryption is off
                        frames = sorted(glob("./record_mosaic/frame_*.jpg"))
                        if frames:
                            most_recent_frame = frames[-1]
                            img = cv2.imread(most_recent_frame)
                            if img is not None:
                                _, buffer = cv2.imencode(".jpg", img)
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                
                                await websocket.send_json({
                                    "type": "stream_frame",
                                    "data": img_base64,
                                    "filename": os.path.basename(most_recent_frame),
                                    "decrypted": False
                                })
                    continue
            else:
                # If not streaming, just wait for messages
                await receive_task
            
            # Process the received message
            message = receive_task.result()
            data = json.loads(message)
            print(f"Received message type: {data['type']}")
            
            if data["type"] == "stream_request":
                # Start streaming mode
                streaming = True
                
                # Get initial frame based on decryption status
                if use_decryption and decryption_key:
                    encrypted_frames = sorted(glob("./record_encrypt/frame_*.jpg.enc"))
                    if encrypted_frames:
                        most_recent_frame = encrypted_frames[-1]
                        original_filename = os.path.basename(most_recent_frame)[:-4]  # Remove .enc
                        
                        try:
                            # Create a temporary file for decrypted content
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                temp_path = temp_file.name
                            
                            # Decrypt the file
                            encrypt.decrypt_file(most_recent_frame, decryption_key, temp_path)
                            
                            # Read the decrypted image
                            img = cv2.imread(temp_path)
                            
                            # Remove temporary file
                            os.unlink(temp_path)
                            
                            if img is not None:
                                _, buffer = cv2.imencode(".jpg", img)
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                
                                await websocket.send_json({
                                    "type": "stream_frame",
                                    "data": img_base64,
                                    "filename": original_filename,
                                    "decrypted": True
                                })
                        except Exception as e:
                            print(f"Decryption error: {e}")
                            use_decryption = False
                            await websocket.send_json({
                                "type": "decryption_error",
                                "message": "Failed to decrypt images. Falling back to mosaic."
                            })
                else:
                    frames = sorted(glob("./record_mosaic/frame_*.jpg"))
                    if frames:
                        most_recent_frame = frames[-1]
                        img = cv2.imread(most_recent_frame)
                        if img is not None:
                            _, buffer = cv2.imencode(".jpg", img)
                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            await websocket.send_json({
                                "type": "stream_frame",
                                "data": img_base64,
                                "filename": os.path.basename(most_recent_frame),
                                "decrypted": False
                            })
            
            elif data["type"] == "stop_stream":
                # Stop streaming mode
                streaming = False
                await websocket.send_json({
                    "type": "stream_complete"
                })
                
            elif data["type"] == "set_decryption_key":
                # Client is sending a decryption key
                provided_key = data.get("key")
                if provided_key:
                    try:
                        # Generate key from provided password
                        key = encrypt.generate_key(provided_key)
                        
                        # Test decryption on a random encrypted file to validate key
                        encrypted_files = glob("./record_encrypt/*.enc")
                        if encrypted_files:
                            test_file = encrypted_files[0]
                            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                                temp_path = temp_file.name
                            
                            # Try to decrypt a file to verify the key
                            encrypt.decrypt_file(test_file, key, temp_path)
                            os.unlink(temp_path)  # Remove temp file
                            
                            # If no exception was raised, key is valid
                            decryption_key = key
                            use_decryption = True
                            
                            await websocket.send_json({
                                "type": "decryption_key_valid",
                                "message": "Decryption key accepted. Streaming original images."
                            })
                    except Exception as e:
                        print(f"Invalid decryption key: {e}")
                        await websocket.send_json({
                            "type": "decryption_key_invalid",
                            "message": "Invalid decryption key provided."
                        })
                else:
                    # Turn off decryption if empty key is provided
                    use_decryption = False
                    decryption_key = None
                    await websocket.send_json({
                        "type": "decryption_disabled",
                        "message": "Decryption disabled. Streaming mosaic images."
                    })

        except Exception as e:
            print(f"Error: {e}")
            break

def del_files():
    # Delete all files in ./record directory
    for file in os.listdir("./record"):
        os.remove(os.path.join("./record", file))
    for file in os.listdir("./record_mosaic"):
        os.remove(os.path.join("./record_mosaic", file))
    for file in os.listdir("./record_encrypt"):
        os.remove(os.path.join("./record_encrypt", file))

# Start RTSP processing and process existing images when the server starts
@app.on_event("startup")
async def startup_event():
    del_files()

    # Process any existing images
    encrypt.process_files()
    
    # Start RTSP processing task
    asyncio.create_task(process_rtsp_stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=52049)