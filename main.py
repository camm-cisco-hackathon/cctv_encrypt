import asyncio
import json
import logging
import os
import time
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder, MediaRelay

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 저장된 비디오를 위한 디렉토리 생성
RECORDINGS_PATH = Path("recordings")
RECORDINGS_PATH.mkdir(exist_ok=True)

# 프론트엔드 파일을 위한 정적 디렉토리 생성
STATIC_PATH = Path("static")
STATIC_PATH.mkdir(exist_ok=True)

# 활성 피어 연결 추적
active_peer_connections: Dict[str, RTCPeerConnection] = {}
# 여러 피어에게 미디어를 배포하기 위한 릴레이
relay = MediaRelay()
# 배포될 공유 비디오 트랙 추적
shared_video_track = None


class VideoFrameProcessor(MediaStreamTrack):
    """수신된 비디오 프레임을 처리하고 디스크에 저장합니다."""
    
    kind = "video"
    
    def __init__(self, track, client_id):
        super().__init__()
        self.track = track
        self.client_id = client_id
        self.recorder = None
        self.start_recording()
        
    def start_recording(self):
        """비디오 파일 녹화 시작."""
        timestamp = int(time.time())
        filename = f"{RECORDINGS_PATH}/{self.client_id}_{timestamp}.mp4"
        self.recorder = MediaRecorder(filename)
        self.recorder.start()
        logger.info(f"Started recording to {filename}")
    
    async def recv(self):
        """각 비디오 프레임 처리."""
        frame = await self.track.recv()
        
        # 프레임을 레코더에 전송
        if self.recorder:
            self.recorder.addFrame(frame)
        
        # 다른 클라이언트에게 배포하기 위해 프레임 반환
        return frame


async def handle_offer(request):
    """발신자로부터 들어오는 WebRTC 제안 처리."""
    client_id = request.match_info["client_id"]
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    pc = RTCPeerConnection()
    active_peer_connections[client_id] = pc
    
    logger.info(f"Created peer connection for client {client_id}")
    
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state for {client_id}: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
            await cleanup_peer_connection(client_id)
    
    @pc.on("track")
    async def on_track(track):
        global shared_video_track
        
        if track.kind == "video":
            logger.info(f"Received video track from {client_id}")
            
            # 비디오 프레임 처리 및 녹화 설정
            processed_track = VideoFrameProcessor(track, client_id)
            
            # 다른 피어가 소비할 수 있도록 공유 트랙으로 설정
            shared_video_track = relay.subscribe(processed_track)
            
            # 참조를 유지하기 위해 피어 연결에 추가
            pc.addTrack(processed_track)
    
    # 원격 설명 설정
    await pc.setRemoteDescription(offer)
    
    # 응답 생성
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def handle_viewer_offer(request):
    """시청자로부터 들어오는 WebRTC 제안 처리."""
    client_id = request.match_info["client_id"]
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    pc = RTCPeerConnection()
    active_peer_connections[client_id] = pc
    
    logger.info(f"Created peer connection for viewer {client_id}")
    
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state for viewer {client_id}: {pc.iceConnectionState}")
        if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
            await cleanup_peer_connection(client_id)
    
    # 공유 트랙이 있는 경우 이 피어 연결에 추가
    if shared_video_track:
        pc.addTrack(shared_video_track)
    
    # 원격 설명 설정
    await pc.setRemoteDescription(offer)
    
    # 응답 생성
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def cleanup_peer_connection(client_id):
    """피어 연결 리소스 정리."""
    if client_id in active_peer_connections:
        pc = active_peer_connections[client_id]
        logger.info(f"Cleaning up connection for {client_id}")
        
        # 피어 연결 닫기
        await pc.close()
        
        # 활성 연결에서 제거
        del active_peer_connections[client_id]


async def on_shutdown(app):
    """종료 시 모든 리소스 정리."""
    # 모든 피어 연결 닫기
    coros = [pc.close() for pc in active_peer_connections.values()]
    await asyncio.gather(*coros)
    active_peer_connections.clear()


def create_app():
    """웹 애플리케이션 생성 및 구성."""
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    
    # 프론트엔드 통합을 위한 CORS 미들웨어 추가
    @web.middleware
    async def cors_middleware(request, handler):
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    app.middlewares.append(cors_middleware)
    
    # WebRTC API 라우트
    app.router.add_post("/offer/{client_id}", handle_offer)  # 발신자용
    app.router.add_post("/view/{client_id}", handle_viewer_offer)  # 시청자용
    
    # 정적 프론트엔드 파일 서비스
    app.router.add_static("/", STATIC_PATH)
    
    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=52049)
