FROM python:3.12-slim

# 1. 최소 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4. 코드 복사
COPY . .

# 5. 디렉토리 생성
RUN mkdir -p ./record ./record_mosaic ./record_encrypt

# 6. 모델 다운로드 제거 → 컨테이너 실행 시 다운로드하거나, volume으로 마운트 권장
# 필요한 경우 아래 명령어를 ENTRYPOINT로 이동 가능
RUN pip install --no-cache-dir ultralytics && \
    python -c "from ultralytics import YOLO; YOLO('yolov11n-face.pt')"

# 7. 포트 설정
EXPOSE 52049

# 8. 실행
CMD ["python", "main.py"]