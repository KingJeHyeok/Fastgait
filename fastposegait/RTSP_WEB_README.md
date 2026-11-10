# RTSP 실시간 보행 인식 웹 시스템

RTSP 스트림을 실시간으로 받아 보행 패턴을 분석하고 사람을 식별하는 웹 기반 시스템입니다.

## 🎯 주요 기능

### 1. RTSP 스트림 처리
- **실시간 RTSP 수신**: Jetson 또는 IP 카메라의 RTSP 스트림 실시간 처리
- **자동 재연결**: 연결 끊김 시 자동 재연결 시도
- **프레임 버퍼링**: 안정적인 처리를 위한 프레임 큐 관리

### 2. YOLOv8-Pose 기반 사람 검출
- **실시간 디텍션**: YOLOv8-Pose로 사람 검출 + 17개 키포인트 추출
- **스마트 선택**: 화면 중앙, 크기, confidence를 고려한 최적 인물 선택
- **필터링**: 최소 크기 30x30, confidence > 0.6, 스켈레톤 필수

### 3. 상태 머신 기반 추적
```
IDLE → DETECTING → COLLECTING → LOST → (IDLE or COLLECTING)
```
- **IDLE**: 대기 상태
- **DETECTING**: 사람 감지됨
- **COLLECTING**: 스켈레톤 수집 중 (버퍼 채우는 중)
- **LOST**: 사람 놓침 (1초 대기 후 재인식 또는 종료)

### 4. 버퍼 시스템
- **초기 버퍼**: 300 프레임 (10초) - 첫 추론용
- **누적 간격**: 90 프레임 (3초) - 주기적 추론
- **최대 버퍼**: 400 프레임 - 메모리 관리

### 5. GPGait++ 모델 추론
- **512차원 임베딩**: L2 정규화된 보행 특징 벡터
- **GPU 가속**: CUDA 지원 (자동 CPU fallback)
- **배치 처리**: 버퍼 크기 자동 조절

### 6. ImprovedPersonRegistry (멀티 임베딩 매칭)
- **엄격한 매칭**: Cosine ≥ 0.85 AND Euclidean ≤ 0.50
- **멀티 임베딩**: 최대 10개 임베딩 저장 + EMA 업데이트
- **세션 잠금**: 높은 confidence 시 ID 고정 (오인식 방지)
- **영속성**: Person DB 자동 저장/로드 (pickle)

### 7. Flask 웹 UI
- **실시간 비디오 스트리밍**: MJPEG 스트림
- **상태 시각화**: 현재 상태 (IDLE/DETECTING/COLLECTING/LOST)
- **통계 대시보드**: FPS, 검출률, 추론 횟수
- **Person 정보**: 등록된 인물 목록 + 매칭 메트릭
- **이벤트 로그**: 최근 20개 이벤트 실시간 표시
- **Registry 리셋**: 버튼 클릭으로 DB 초기화

### 8. 상세 로깅
- **이벤트 로그**: 상태 전환, 추론, ID 변경 등 모든 이벤트
- **매칭 로그**: 추론 결과 상세 (cosine, euclidean, buffer size)
- **JSON 저장**: 세션 종료 시 자동 저장

## 📋 요구사항

```bash
# Python 패키지
pip install flask opencv-python numpy torch ultralytics

# 필수 모델 파일
- yolov8n-pose.pt (YOLOv8-Pose 모델)
- GPGait++ checkpoint (예: GPGait_PlusPlus-40000.pt)
- GPGait++ config (예: gpgait++.yaml)
```

## 🚀 사용 방법

### 1. 기본 실행 (모델 없이 - YOLO만)
```bash
python rtsp_web_gait_tracker.py --rtsp rtsp://172.30.85.166:8554/stream
```

### 2. 모델 포함 실행 (전체 기능)
```bash
python rtsp_web_gait_tracker.py \
    --rtsp rtsp://172.30.85.166:8554/stream \
    --enable-model \
    --config configs/gpgait++/gpgait++.yaml \
    --checkpoint output/CASIA-B/GPGait_PlusPlus/GPGait_PlusPlus/checkpoints/GPGait_PlusPlus-40000.pt \
    --yolo-model yolov8n-pose.pt \
    --output-dir rtsp_output \
    --port 5000 \
    --host 0.0.0.0
```

### 3. 웹 UI 접속
- 로컬: http://localhost:5000
- 네트워크: http://<서버IP>:5000

### 4. 종료
- `Ctrl+C`로 종료
- 자동으로 Person DB 저장 및 로그 저장

## 📊 웹 UI 설명

### 상단: 비디오 + 상태
- **Live RTSP Stream**: 실시간 비디오 (YOLO bbox + 스켈레톤 오버레이)
- **상태 카드**:
  - 상태: IDLE/DETECTING/COLLECTING/LOST (색상 코드)
  - 버퍼 크기: 현재 스켈레톤 버퍼
  - 세션 상태: LOCKED/UNLOCKED
  - FPS: 실시간 프레임레이트

### 우측: 통계 + 인물 정보
- **통계 카드**:
  - 수신 프레임: RTSP로 받은 총 프레임
  - 처리 프레임: 처리 완료 프레임
  - YOLO 검출: 사람 검출 횟수
  - 스켈레톤 추출: 성공한 스켈레톤 추출
  - 모델 추론: GPGait++ 추론 횟수
  - 검출률: (detected / processed) * 100

- **현재 인식 중**:
  - Person ID (NEW/MATCHED 표시)
  - Match Score (전체 점수)
  - Cosine Similarity (진행 바 + 값)
  - Euclidean Distance (진행 바 + 값)

- **등록된 인물**:
  - Person #ID
  - 검출 횟수
  - 저장된 임베딩 개수

### 하단: 이벤트 로그
- 최근 20개 이벤트 실시간 표시
- 타임스탬프 + 이벤트 타입 + 메시지
- 색상 코드:
  - DETECTING: 청록색
  - COLLECTING: 초록색
  - LOST: 주황색
  - INFERENCE: 보라색
  - SESSION_LOCK: 초록색
  - ID_CHANGE: 노란색

## 🔧 주요 파라미터

### RTSP 연결
- `--rtsp`: RTSP URL (필수)
- `--host`: Flask 서버 호스트 (기본: 0.0.0.0)
- `--port`: Flask 서버 포트 (기본: 5000)

### 모델 설정
- `--enable-model`: GPGait++ 모델 활성화
- `--config`: GPGait++ config 파일 경로
- `--checkpoint`: GPGait++ checkpoint 경로
- `--yolo-model`: YOLOv8-Pose 모델 (기본: yolov8n-pose.pt)

### 출력
- `--output-dir`: 출력 디렉토리 (기본: rtsp_output)
  - person_database.pkl: Person Registry
  - rtsp_log_YYYYMMDD_HHMMSS.json: 이벤트/매칭 로그

## 📁 출력 파일

### 1. person_database.pkl
- Person Registry 영속성 저장
- registry: {person_id: {...}}
- next_id: 다음 사용할 ID

### 2. rtsp_log_YYYYMMDD_HHMMSS.json
```json
{
  "event_log": [
    {
      "frame": 123,
      "timestamp": 1700123456.789,
      "time_str": "12:34:56",
      "type": "INFERENCE_PERIODIC",
      "message": "Person #1 (score: 0.923)"
    }
  ],
  "matching_log": [
    {
      "frame": 123,
      "person_id": 1,
      "is_new": false,
      "match_score": 0.923,
      "cosine_similarity": 0.95,
      "euclidean_distance": 0.25,
      "buffer_size": 300,
      "session_locked": true
    }
  ],
  "person_registry": {
    "1": {
      "count": 5,
      "embeddings": 10
    }
  }
}
```

## 🎨 시각화 예시

### 비디오 오버레이
- 녹색 바운딩 박스: YOLO 검출 (conf 표시)
- 녹색 선 + 빨간 점: 스켈레톤 (17 keypoints)
- 상단 정보:
  - State: COLLECTING | Buffer: 245
  - Person #1 (LOCKED)
  - Score: 0.923
  - Cosine: 0.9500
  - Euclid: 0.2500
- 우측: Person DB 목록

### 웹 UI
- 그라데이션 배경 (보라색 계열)
- 반투명 카드 (glass morphism)
- 실시간 업데이트 (1초 간격)
- 애니메이션 효과 (이벤트 슬라이드인)

## 🔍 디버깅

### 로그 확인
```bash
# 터미널 출력 (실시간)
[12:34:56] DETECTING              | Person detected (conf: 0.89)
[12:34:57] COLLECTING             | Started collecting (target: 300)
[12:35:07] INFERENCE_PERIODIC     | Person #1 (NEW, score: 0.923, LOCKED)
```

### 웹 UI 상태 확인
1. 상태 카드 색상 확인
2. 버퍼 크기 증가 확인
3. 이벤트 로그에서 흐름 추적
4. Person List에서 등록 확인

### 일반적인 문제

#### RTSP 연결 실패
```bash
❌ [Receiver] Failed to connect to RTSP stream
```
- RTSP URL 확인 (rtsp://IP:PORT/PATH)
- Jetson RTSP 서버 실행 확인
- 네트워크 연결 확인
- 방화벽 설정 확인

#### YOLO 검출 안됨
- 조명 확인 (너무 어두우면 검출 어려움)
- 사람이 화면에 충분히 크게 나오는지 확인 (최소 30x30)
- yolov8n-pose.pt 파일 존재 확인

#### 모델 추론 안됨
- --enable-model 옵션 사용 확인
- config/checkpoint 파일 경로 확인
- GPU 메모리 확인 (CUDA out of memory)

#### 버퍼가 안 찬다
- 스켈레톤 추출 실패 (YOLO keypoints 확인)
- 사람이 계속 화면에 있는지 확인
- LOST 상태로 자주 전환되는지 로그 확인

## 💡 성능 최적화

### GPU 사용
- CUDA 사용 시 자동으로 GPU에서 모델 추론
- YOLO도 GPU 가속 (자동)

### 프레임 스킵
- 큐가 꽉 차면 자동으로 프레임 드롭
- maxsize=5 (0.16초 버퍼)

### 메모리 관리
- 버퍼 최대 400 프레임 제한
- 오래된 이벤트 자동 삭제 (최근 20개)

## 🆚 scenario_tester.py와 차이점

### scenario_tester.py (녹화 영상)
- 파일 입력 (mp4, avi 등)
- 전체 프레임 처리
- 상세 로그 (detection, inference, matching)
- 결과 비디오 저장
- 오프라인 분석

### rtsp_web_gait_tracker.py (실시간 RTSP)
- RTSP 스트림 입력
- 실시간 처리 (프레임 드롭 허용)
- Flask 웹 UI
- MJPEG 스트리밍
- 실시간 모니터링

## 🔄 업데이트 히스토리

### v1.0 (2025-01-16)
- 초기 버전
- RTSP 스트림 지원
- YOLOv8-Pose 통합
- 상태 머신 구현
- ImprovedPersonRegistry 사용
- Flask 웹 UI
- 실시간 이벤트 로그

## 📞 문의

문제 발생 시 다음 정보와 함께 문의하세요:
1. 명령어 (전체 파라미터 포함)
2. 터미널 출력 (특히 오류 메시지)
3. 로그 파일 (rtsp_log_*.json)
4. 환경 정보 (OS, Python 버전, GPU 유무)

## 📄 라이센스

이 프로젝트는 FastPoseGait 프로젝트의 일부입니다.
