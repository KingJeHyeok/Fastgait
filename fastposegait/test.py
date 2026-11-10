# 30초 1레벨 경고 60초 2레벨 경고

import os
import sys
import subprocess
import importlib

# ================== 자동 의존성 설치 ==================
def check_and_install_package(package_name, import_name=None, pip_name=None):
    """패키지 설치 여부 확인 및 자동 설치"""
    if import_name is None:
        import_name = package_name
    if pip_name is None:
        pip_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"⚠️ {package_name} 모듈이 없습니다. 자동 설치를 진행합니다...")
        try:
            # --quiet 옵션 대신 --no-warn-script-location 사용 (더 나은 출력)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "--no-warn-script-location"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ {package_name} 설치 완료")
            # 설치 후 다시 import 시도
            try:
                importlib.import_module(import_name)
                return True
            except ImportError:
                print(f"⚠️ {package_name} 설치 후에도 import 실패. 프로그램을 재시작해주세요.")
                return False
        except subprocess.CalledProcessError as e:
            print(f"❌ {package_name} 설치 실패")
            print(f"   오류: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            print(f"   수동 설치: pip install {pip_name}")
            return False
        except Exception as e:
            print(f"❌ {package_name} 설치 중 예기치 않은 오류: {e}")
            print(f"   수동 설치: pip install {pip_name}")
            return False

# 필수 패키지 확인 및 설치
REQUIRED_PACKAGES = [
    ("opencv-contrib-python", "cv2", "opencv-contrib-python"),
    ("numpy", "numpy", "numpy"),
    ("torch", "torch", "torch"),  # PyTorch는 CUDA 버전이 필요할 수 있음
    ("requests", "requests", "requests"),
    ("ultralytics", "ultralytics", "ultralytics"),
    ("firebase_admin", "firebase_admin", "firebase-admin"),
    ("flask", "flask", "flask"),
    ("aiortc", "aiortc", "aiortc"),
    ("av", "av", "av"),  # PyAV
]

print("🔍 필요한 모듈 확인 중...")
for package_name, import_name, pip_name in REQUIRED_PACKAGES:
    if not check_and_install_package(package_name, import_name, pip_name):
        print(f"❌ 필수 모듈 {package_name} 설치에 실패했습니다. 수동으로 설치해주세요:")
        print(f"   pip install {pip_name}")
        sys.exit(1)

print("✅ 모든 필수 모듈 확인 완료\n")

# 모듈 import
import time, shutil, threading, asyncio, fractions
import cv2
import numpy as np
import torch
import requests
from pathlib import Path
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, storage, firestore
from flask import Flask, request, jsonify
import traceback
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame
import torch.nn.functional as F

# GPGait++ 및 Person Registry 모듈 (조건부, 선택적 자동 설치)
OPTIONAL_PACKAGES = [
    ("realtime_skeleton_gait", "realtime_skeleton_gait", "realtime-skeleton-gait"),
    ("improved_person_registry", "improved_person_registry", "improved-person-registry"),
]

print("🔍 선택적 모듈 확인 중...")
for package_name, import_name, pip_name in OPTIONAL_PACKAGES:
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} 이미 설치됨")
    except ImportError:
        print(f"⚠️ {package_name} 모듈이 없습니다. 자동 설치를 시도합니다...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "--no-warn-script-location"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ {package_name} 설치 완료")
            # 설치 후 다시 import 시도
            try:
                importlib.import_module(import_name)
            except ImportError:
                print(f"⚠️ {package_name} 설치 후에도 import 실패. 기능이 비활성화됩니다.")
        except subprocess.CalledProcessError:
            # pip 패키지가 아닐 수 있음 (커스텀 모듈 등)
            print(f"⚠️ {package_name} pip 패키지로 설치 불가. 수동 설치 필요:")
            print(f"   (커스텀 모듈이거나 GitHub에서 설치해야 할 수 있습니다)")
        except Exception as e:
            print(f"⚠️ {package_name} 설치 중 오류: {e}")

print()  # 빈 줄

# 모듈 import 시도
try:
    from realtime_skeleton_gait import GaitModelLoader, SkeletonExtractor
    GAIT_MODEL_AVAILABLE = True
except ImportError:
    GAIT_MODEL_AVAILABLE = False

try:
    from improved_person_registry import ImprovedPersonRegistry
    PERSON_REGISTRY_AVAILABLE = True
except ImportError:
    PERSON_REGISTRY_AVAILABLE = False

# ================== 설정 (Level1=30s / Level2=60s) ==================
JETSON_IP   = "172.30.111.124"
RTSP_URL    = f"rtsp://{JETSON_IP}:8554/test"
RTSP_RETRY_COUNT = 5  # RTSP 연결 재시도 횟수
RTSP_RETRY_DELAY = 3  # 재시도 간격 (초)

# YOLOv8n: 사람만 탐지 + 저지연
YOLO_MODEL      = "yolov8n.pt"
YOLO_CONF       = 0.5
YOLO_IMGSZ      = 416          # 352~448 권장
TRACK_EVERY_N   = 2            # YOLO 추적 호출 간격 (2~3 추천)

# YOLOv8-Pose: 스켈레톤 추출용
YOLO_POSE_MODEL = "yolov8n-pose.pt"
USE_POSE_MODEL  = True         # True면 Pose 모델 사용, False면 일반 YOLO만 사용
POSE_CONF       = 0.6          # Pose 모델 confidence threshold

# 얼굴 인식: YuNet + SFace
ID_DIR          = r"C:\Users\SJH\Desktop\id_photos"   # (로컬) 등록 폴더 - Firebase 동기화 후 사용
YUNET_PATH      = "face_detection_yunet_2023mar.onnx"
SFACE_PATH      = "face_recognition_sface_2021dec.onnx"
URL_YUNET       = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
URL_SFACE       = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YUNET_SCORE     = 0.6
YUNET_NMS       = 0.3
SIM_THR         = 0.38         # Known/Unknown 기준
FACE_START_CONF = 0.9          # 얼굴 인식 시작 최소 confidence (핵심!)
FACE_LOCK_SEC   = 3.0          # 3초간 연속 인식되면 Lock on
FACE_EVERY_SEC  = 1.0          # Lock-on 전 얼굴 인식 주기 (2.0 → 1.0)
FACE_LOCK_UNKNOWN_TOLERANCE = 2  # Lock-on 후 Unknown 허용 횟수
LABEL_TTL       = 3.0          # 라벨 캐시 유지(초)
MIN_FACE_SIDE   = 40
UPSCALE_FACE_TO = 360          # 얼굴 ROI 최대 변 기준 업샘플

# Firebase Storage 설정
FIREBASE_BUCKET = os.environ.get("FIREBASE_STORAGE_BUCKET")  # 환경 변수가 없으면 서비스 계정 키에서 자동 감지
FACES_PREFIX    = os.environ.get("FIREBASE_FACES_PREFIX", "faces/")
FACES_CACHE_DIR = Path(os.environ.get("FACE_CACHE_DIR", Path(__file__).parent / "faces_cache"))
FIREBASE_CREDENTIAL_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")  # 환경변수 설정 권장
TARGET_USER_ID = os.environ.get("TARGET_USER_ID")  # 특정 사용자만 로드하고 싶을 때 사용
DANGER_USER_ID = os.environ.get("DANGER_USER_ID") or TARGET_USER_ID

# ================== Firebase 키 파일 찾기 ==================
def find_firebase_key_files():
    """서비스 계정 키 파일 위치 찾기"""
    found_files = []
    
    # 1. 환경 변수에서 경로 확인
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).exists():
        found_files.append(("환경 변수", str(Path(env_path).absolute())))
    
    # 2. 현재 디렉토리와 상위 디렉토리에서 검색
    script_dir = Path(__file__).parent
    search_dirs = [script_dir, script_dir.parent]
    
    key_patterns = [
        "serviceAccountKey.json",
        "firebase-adminsdk*.json",
        "*service-account*.json",
        "*firebase*.json",
        "*credentials*.json"
    ]
    
    for search_dir in search_dirs:
        for pattern in key_patterns:
            for key_file in search_dir.glob(pattern):
                if key_file.is_file() and key_file.suffix == '.json':
                    # JSON 파일인지 확인 (Firebase 키 파일 형식 체크)
                    try:
                        import json
                        with open(key_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Firebase 서비스 계정 키 파일인지 확인
                            if isinstance(data, dict) and 'type' in data and 'private_key' in data:
                                rel_path = key_file.relative_to(script_dir) if key_file.is_relative_to(script_dir) else key_file
                                found_files.append(("자동 탐지", str(key_file.absolute())))
                    except Exception:
                        pass
    
    # 3. 일반적인 다운로드 폴더 위치 확인
    common_locations = []
    if os.name == 'nt':  # Windows
        downloads_path = Path.home() / "Downloads"
        desktop_path = Path.home() / "Desktop"
        common_locations = [downloads_path, desktop_path]
    else:  # Linux/Mac
        downloads_path = Path.home() / "Downloads"
        desktop_path = Path.home() / "Desktop"
        common_locations = [downloads_path, desktop_path]
    
    for common_dir in common_locations:
        if common_dir.exists():
            for pattern in key_patterns:
                for key_file in common_dir.glob(pattern):
                    if key_file.is_file() and key_file.suffix == '.json':
                        try:
                            import json
                            with open(key_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, dict) and 'type' in data and 'private_key' in data:
                                    if str(key_file.absolute()) not in [f[1] for f in found_files]:
                                        found_files.append(("일반 위치", str(key_file.absolute())))
                        except Exception:
                            pass
    
    return found_files

# 체류 경고 및 위험도 적산
DANGER_BASE_SEC = 10.0         # 10초 이후부터 위험도 적산
ACTIVE_USER_REFRESH_SEC = 1.0  # 활성 사용자 재조회 주기 (딜레이 감소)
ABSENCE_GRACE   = 2.0          # 2초 이상 안 보이면 트랙 소멸로 판단 (1.5 → 2.0)

# 스켈레톤 시스템 설정
ENABLE_GAIT_MODEL = os.environ.get("ENABLE_GAIT_MODEL", "false").lower() == "true"  # 환경변수로 활성화
GAIT_CONFIG_PATH = os.environ.get("GAIT_CONFIG_PATH", "configs/gpgait++/gpgait++.yaml")
GAIT_CHECKPOINT_PATH = os.environ.get("GAIT_CHECKPOINT_PATH", "")
SKELETON_INITIAL_BUFFER = 300  # 초기 버퍼 크기 (10초 @ 30fps)
SKELETON_ACCUM_INTERVAL = 90   # 주기적 추론 간격 (3초 @ 30fps)
SKELETON_MAX_BUFFER = 400      # 최대 버퍼 크기
SKELETON_MIN_BUFFER = 30       # 최소 버퍼 크기 (추론 가능)
SKELETON_STATE_ENABLED = True  # 상태 머신 활성화

# Person DB 설정
PERSON_DB_DIR = Path(os.environ.get("PERSON_DB_DIR", Path(__file__).parent / "person_db"))
PERSON_DB_PATH = PERSON_DB_DIR / "person_database.pkl"
PERSON_DB_AUTOSAVE_INTERVAL = 10  # 10회 추론마다 자동 저장

WEBRTC_HOST = os.environ.get("WEBRTC_HOST", "0.0.0.0")
WEBRTC_PORT = int(os.environ.get("WEBRTC_PORT", "5001"))

# 표시/입력
FRAME_W, FRAME_H = 800, 600
FONT = cv2.FONT_HERSHEY_SIMPLEX

# FFmpeg 저지연(백업)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|fflags;nobuffer|max_delay;0|probesize;32000|analyzeduration;0|stimeout;2000000"
)

# ================== 유틸 ==================
def ensure_file(url, path):
    if os.path.exists(path):
        return
    print(f"[DL] {path}")
    r = requests.get(url, stream=True); r.raise_for_status()
    with open(path, "wb") as f:
        for ch in r.iter_content(1024): f.write(ch)
    print(f"✅ {path} 다운로드 완료")

def l2norm(v):
    v = v.astype(np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def extract_feature_from_image(img, yunet, recognizer):
    if img is None or img.size == 0:
        return None

    working = img.copy()
    h, w = working.shape[:2]
    if max(h, w) < UPSCALE_FACE_TO:
        scale = UPSCALE_FACE_TO / max(1, max(h, w))
        working = cv2.resize(working, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    h2, w2 = working.shape[:2]
    yunet.setInputSize((w2, h2))
    ok, faces = yunet.detect(working)

    if ok and faces is not None and len(faces) > 0:
        faces = faces[np.argsort(faces[:, 2] * faces[:, 3])[::-1]]
        try:
            aligned = recognizer.alignCrop(working, faces[0])
            return l2norm(recognizer.feature(aligned))
        except Exception:
            pass

    # Fallback: 중앙 크롭
    h, w = working.shape[:2]
    size = min(h, w)
    if size <= 0:
        return None
    x0 = max((w - size) // 2, 0)
    y0 = max((h - size) // 2, 0)
    crop = working[y0:y0 + size, x0:x0 + size]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
    try:
        return l2norm(recognizer.feature(crop))
    except Exception:
        return None

def cosine_sim(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return -1.0
    return float(np.dot(a, b) / (na * nb))


# ================== Person DB 관리 ==================
def load_person_db(person_registry, db_path):
    """Person DB 로드"""
    if not os.path.exists(db_path):
        print(f"ℹ️ Person DB 없음 (새로 시작): {db_path}")
        return False
    
    try:
        import pickle
        with open(db_path, 'rb') as f:
            data = pickle.load(f)
            person_registry.registry = data['registry']
            person_registry.next_id = data['next_id']
            print(f"✅ Person DB 로드 완료: {len(person_registry.registry)} persons")
            for pid, info in person_registry.registry.items():
                emb_count = len(info.get('embeddings_list', []))
                print(f"   Person #{pid}: {info['count']} detections, {emb_count} embeddings")
        return True
    except Exception as e:
        print(f"⚠️ Person DB 로드 실패: {e}")
        return False


def save_person_db(person_registry, db_path):
    """Person DB 저장"""
    try:
        import pickle
        # 디렉토리 생성
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'registry': person_registry.registry,
            'next_id': person_registry.next_id
        }
        with open(db_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Person DB 저장 완료: {len(person_registry.registry)} persons")
        return True
    except Exception as e:
        print(f"⚠️ Person DB 저장 실패: {e}")
        return False


# ================== 스켈레톤 시스템 함수 ==================
def detect_persons_yolo_pose(pose_model, frame, conf_threshold=POSE_CONF):
    """YOLOv8-Pose로 사람 디텍션 + 스켈레톤 추출"""
    if pose_model is None:
        return []
    
    try:
        results = pose_model(frame, verbose=False)
        persons = []
        
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    
                    # 스켈레톤 추출
                    skeleton = None
                    if keypoints is not None and len(keypoints) > idx:
                        kpts = keypoints[idx].xy.cpu().numpy()[0]  # (17, 2)
                        if len(kpts) == 17:
                            skeleton = kpts.astype(np.float32)
                    
                    # 필터링
                    if (conf > conf_threshold and bbox_w > 30 and bbox_h > 30 and skeleton is not None):
                        persons.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'skeleton': skeleton
                        })
        
        return persons
    except Exception as e:
        print(f"⚠️ YOLO-Pose error: {e}")
        return []


def select_best_person(persons, frame_shape):
    """가장 좋은 사람 선택 (중앙 + 크기 + confidence)"""
    if not persons:
        return None
    
    h, w = frame_shape[:2]
    center_x, center_y = w // 2, h // 2
    
    best_person = None
    best_score = -1
    
    for person in persons:
        bbox = person['bbox']
        x1, y1, x2, y2 = bbox
        
        # 중심점
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # 크기
        area = (x2 - x1) * (y2 - y1)
        
        # 거리
        dist = np.sqrt((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)
        
        # 점수 계산
        size_score = min(area / (w * h), 0.5) * 2  # 0~1
        dist_score = 1 - min(dist / (w/2), 1)  # 0~1
        conf_score = person['confidence']  # 0~1
        
        score = size_score * 0.4 + dist_score * 0.2 + conf_score * 0.4
        
        if score > best_score:
            best_score = score
            best_person = person
    
    return best_person


def draw_skeleton(frame, skeleton):
    """스켈레톤 그리기"""
    if skeleton is None:
        return frame
    
    display_frame = frame.copy()
    
    # COCO 스켈레톤 연결 정의 (17개 키포인트)
    coco_skeleton = [
        [0, 1], [0, 2], [1, 3], [2, 4],          # 머리 → 어깨 → 팔꿈치 → 손목
        [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], # 어깨 → 팔꿈치 → 손목 (양쪽)
        [5, 11], [6, 12], [11, 12],              # 어깨 → 엉덩이 연결
        [11, 13], [12, 14], [13, 15], [14, 16]   # 엉덩이 → 무릎 → 발목
    ]
    
    # 뼈대 그리기
    for joint1, joint2 in coco_skeleton:
        pt1 = skeleton[joint1]
        pt2 = skeleton[joint2]
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(display_frame, 
                    (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), 
                    (0, 255, 0), 2)
    
    # 관절 그리기
    for i, (x, y) in enumerate(skeleton):
        if x > 0 and y > 0:
            cv2.circle(display_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.circle(display_frame, (int(x), int(y)), 6, (255, 255, 255), 1)
    
    return display_frame


def process_skeleton_inference(tid, frame_idx, gait_model, person_registry, device, is_final=False):
    """스켈레톤 추론 처리 (tid별)"""
    global skeleton_buffers, skeleton_person_ids, skeleton_session_locked, skeleton_tracking_confidence
    global skeleton_has_initial_inference, skeleton_states, skeleton_frames_since_inference
    
    if gait_model is None or person_registry is None:
        return None
    
    buffer = skeleton_buffers.get(tid, [])
    if len(buffer) < SKELETON_MIN_BUFFER:
        return None
    
    try:
        # 스켈레톤 시퀀스 준비
        skeleton_sequence = np.array(buffer)
        buffer_size = len(skeleton_sequence)
        
        if buffer_size > SKELETON_MAX_BUFFER:
            indices = np.linspace(0, buffer_size-1, SKELETON_MAX_BUFFER, dtype=int)
            skeleton_sequence = skeleton_sequence[indices]
        
        # GPGait++ 추론
        data = GaitModelLoader.preprocess_skeleton_sequence(skeleton_sequence)
        data = data.to(device, non_blocking=True).contiguous()
        
        with torch.no_grad():
            T = data.shape[2]
            seqL = torch.LongTensor([T]).to(device, non_blocking=True)
            inputs = [[data], None, None, None, [seqL]]
            outputs = gait_model.forward(inputs)
            
            feat = outputs['inference_feat']['embeddings']
            if feat.dim() == 3:
                feat = feat.mean(dim=2)
            feat = F.normalize(feat, p=2, dim=1)
        
        # Person Registry 매칭
        current_person_id = skeleton_person_ids.get(tid, None)
        session_locked = skeleton_session_locked.get(tid, False)
        
        if session_locked and current_person_id is not None:
            person_id, is_new, match_score, cosine_sim, euclid_dist = person_registry.force_update_person(
                current_person_id, feat
            )
        else:
            person_id, is_new, match_score, cosine_sim, euclid_dist = person_registry.match_or_register(
                feat, frame_count=len(skeleton_sequence)
            )
        
        # 세션 잠금 처리
        has_init = skeleton_has_initial_inference.get(tid, False)
        if not has_init:
            # 첫 추론
            skeleton_person_ids[tid] = person_id
            skeleton_has_initial_inference[tid] = True
            if tid not in skeleton_tracking_confidence:
                skeleton_tracking_confidence[tid] = 0.0
            skeleton_tracking_confidence[tid] = match_score
            
            if match_score > 0.85:
                skeleton_session_locked[tid] = True
                print(f"🔒 Skeleton Lock: tid={tid} -> Person #{person_id} (score: {match_score:.3f})")
        else:
            # 세션 잠금 처리
            if session_locked:
                if match_score < 0.70:
                    skeleton_session_locked[tid] = False
                else:
                    if person_id != current_person_id:
                        person_id = current_person_id
                        is_new = False
            else:
                # ID 변경 처리
                if person_id != current_person_id:
                    if match_score > skeleton_tracking_confidence.get(tid, 0.0) + 0.10:
                        skeleton_person_ids[tid] = person_id
                        skeleton_tracking_confidence[tid] = match_score
                        if match_score > 0.85:
                            skeleton_session_locked[tid] = True
                            print(f"🔒 Skeleton Lock: tid={tid} -> Person #{person_id} (score: {match_score:.3f})")
                    else:
                        person_id = current_person_id
                        is_new = False
                else:
                    skeleton_tracking_confidence[tid] = max(skeleton_tracking_confidence.get(tid, 0.0), match_score)
        
        inference_type = "Final" if is_final else "Periodic"
        status = "NEW" if is_new else "MATCHED"
        lock_status = "LOCKED" if skeleton_session_locked.get(tid, False) else "UNLOCKED"
        print(f"🦴 Skeleton {inference_type}: tid={tid} -> Person #{person_id} ({status}, score: {match_score:.3f}, {lock_status})")
        
        if is_final:
            # 세션 리셋
            skeleton_buffers[tid] = []
            skeleton_states[tid] = 'IDLE'
            skeleton_frames_since_inference[tid] = 0
            skeleton_has_initial_inference[tid] = False
            skeleton_person_ids[tid] = None
            skeleton_session_locked[tid] = False
        else:
            skeleton_frames_since_inference[tid] = 0
        
        return {
            'person_id': person_id,
            'is_new': is_new,
            'match_score': match_score,
            'cosine_similarity': float(cosine_sim),
            'euclidean_distance': float(euclid_dist)
        }
    except Exception as e:
        print(f"⚠️ Skeleton inference error (tid={tid}): {e}")
        return None


class FrameGrabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread = None
        self.last_update = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.last_update = time.time()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None, 0.0
            frame = self.frame.copy()
            ts = self.last_update
        return True, frame, ts

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class ProcessedVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, queue: "asyncio.Queue[np.ndarray]"):
        super().__init__()
        self._queue = queue
        self._pts = 0
        self._time_base = fractions.Fraction(1, 30)

    async def recv(self):
        frame = await self._queue.get()
        # 디버깅: 첫 몇 프레임 로그
        if self._pts < 5:
            print(f"[WebRTC] 프레임 수신 {self._pts+1}")
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        self._pts += 1
        video_frame.pts = self._pts
        video_frame.time_base = self._time_base
        return video_frame

    async def enqueue(self, frame: np.ndarray):
        try:
            while self._queue.full():
                self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        await self._queue.put(frame)


class WebRTCSessionException(Exception):
    pass


webrtc_app = Flask(__name__)
_webrtc_loop = asyncio.new_event_loop()
_webrtc_loop_thread = None
_webrtc_video_track: ProcessedVideoTrack | None = None
_webrtc_queue: asyncio.Queue[np.ndarray] | None = None
_peer_connections: set[RTCPeerConnection] = set()


def _start_webrtc_loop():
    asyncio.set_event_loop(_webrtc_loop)
    _webrtc_loop.run_forever()


async def _create_video_track():
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=3)
    track = ProcessedVideoTrack(queue)
    return queue, track


async def _cleanup_peer(pc: RTCPeerConnection):
    try:
        await pc.close()
    finally:
        _peer_connections.discard(pc)


async def _wait_ice_complete(pc: RTCPeerConnection, timeout: float = 5.0):
    if pc.iceGatheringState == "complete":
        return

    future: asyncio.Future[None] = _webrtc_loop.create_future()

    @pc.on("icegatheringstatechange")
    async def _on_ice_gathering_state_change():
        if pc.iceGatheringState == "complete" and not future.done():
            future.set_result(None)

    try:
        await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        pass


async def _handle_offer(params: dict) -> dict:
    if _webrtc_video_track is None:
        raise WebRTCSessionException("WebRTC 영상 트랙이 초기화되지 않았습니다.")

    pc = RTCPeerConnection()
    _peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def _on_state_change():
        state = pc.connectionState
        if state in ("failed", "closed", "disconnected"):
            await _cleanup_peer(pc)

    offer = RTCSessionDescription(sdp=params["sdp"], type=params.get("type", "offer"))
    await pc.setRemoteDescription(offer)

    video_sender = None
    for transceiver in pc.getTransceivers():
        if transceiver.kind == 'video':
            transceiver.direction = 'sendonly'
            video_sender = transceiver.sender
            break

    if video_sender is None:
        raise WebRTCSessionException("클라이언트 Offer에 video 트랜시버가 없습니다.")

    video_sender.replaceTrack(_webrtc_video_track)
    print("[WebRTC] 비디오 트랙 연결 완료")

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    await _wait_ice_complete(pc)

    local = pc.localDescription
    if local is None:
        raise WebRTCSessionException("Answer SDP 생성에 실패했습니다.")

    response = {
        "sdp": local.sdp,
        "type": local.type,
        "iceCandidates": [],
    }
    return response


@webrtc_app.route("/offer", methods=["POST"])
def webrtc_offer():
    payload = request.get_json(force=True, silent=False)
    if not isinstance(payload, dict) or "sdp" not in payload:
        return jsonify({"error": "invalid_offer"}), 400

    future = asyncio.run_coroutine_threadsafe(_handle_offer(payload), _webrtc_loop)
    try:
        result = future.result(timeout=10)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


def start_webrtc_services(host: str = "0.0.0.0", port: int = 5001):
    global _webrtc_loop_thread, _webrtc_video_track, _webrtc_queue

    if _webrtc_loop_thread is None:
        _webrtc_loop_thread = threading.Thread(target=_start_webrtc_loop, daemon=True)
        _webrtc_loop_thread.start()

    if _webrtc_video_track is None:
        queue, track = asyncio.run_coroutine_threadsafe(_create_video_track(), _webrtc_loop).result()
        _webrtc_queue = queue
        _webrtc_video_track = track

    threading.Thread(
        target=lambda: webrtc_app.run(host=host, port=port, use_reloader=False, threaded=True),
        daemon=True,
    ).start()


def push_frame_to_webrtc(frame: np.ndarray):
    if _webrtc_queue is None or _webrtc_video_track is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(_webrtc_video_track.enqueue(frame.copy()), _webrtc_loop)
    except RuntimeError:
        pass


def find_active_user_id(db_client):
    try:
        docs = db_client.collection('users').stream()
        for doc in docs:
            data = doc.to_dict() or {}
            current_mode = data.get('currentMode')
            if current_mode is not None:
                return doc.id
    except Exception as e:
        print(f"⚠️ 활성 사용자 조회 실패: {e}")
    return None

def load_yunet(path, score=YUNET_SCORE, nms=YUNET_NMS):
    # OpenCV 버전별 생성 호환
    if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
        det = cv2.FaceDetectorYN.create(path, "", (0,0))
    elif hasattr(cv2, "FaceDetectorYN_create"):
        det = cv2.FaceDetectorYN_create(path, "", (0,0))
    else:
        raise RuntimeError("FaceDetectorYN 생성 실패 (opencv-contrib-python 필요)")
    try:
        det.setScoreThreshold(score); det.setNMSThreshold(nms)
    except Exception:
        pass
    return det

def load_sface(path):
    if hasattr(cv2, "FaceRecognizerSF") and hasattr(cv2.FaceRecognizerSF, "create"):
        return cv2.FaceRecognizerSF.create(path, "")
    elif hasattr(cv2, "FaceRecognizerSF_create"):
        return cv2.FaceRecognizerSF_create(path, "")
    raise RuntimeError("FaceRecognizerSF 생성 실패 (opencv-contrib-python 필요)")

def build_face_db(yunet, sface, root_dir: str, target_user: str | None = None):
    db = {}
    label_names = {}
    root = Path(root_dir)
    if not root.exists():
        print(f"⚠️ 얼굴 DB 폴더 없음: {root_dir}")
        return db, label_names

    def iter_label_dirs(base: Path):
        # 사용자 폴더(target_user 지정 시 해당 폴더만)
        if target_user:
            candidate = base / target_user
            if candidate.exists() and candidate.is_dir():
                for label_dir in candidate.iterdir():
                    if label_dir.is_dir():
                        yield candidate.name, label_dir
            else:
                print(f"⚠️ 지정 사용자({target_user}) 폴더 없음")
        else:
            for user_dir in base.iterdir():
                if not user_dir.is_dir():
                    continue
                for label_dir in user_dir.iterdir():
                    if label_dir.is_dir():
                        yield user_dir.name, label_dir

    known_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 사용자/라벨 폴더 구조 처리
    for user_id, label_dir in iter_label_dirs(root):
        feats = []
        for imgp in label_dir.glob("*.*"):
            if imgp.suffix.lower() not in known_extensions:
                continue
            img = cv2.imread(str(imgp))
            feat = extract_feature_from_image(img, yunet, sface)
            if feat is not None:
                feats.append(feat)
        if feats:
            label_key = label_dir.name if target_user else f"{user_id}/{label_dir.name}"
            db[label_key] = l2norm(np.mean(np.stack(feats, 0), 0))
            label_names[label_key] = label_dir.name
            print(f"✅ DB(폴더): {label_key} ({len(feats)}장)")

    # 루트 바로 아래 개별 파일(이전 호환)
    for imgp in root.glob("*.*"):
        if not imgp.is_file() or imgp.suffix.lower() not in known_extensions:
            continue
        img = cv2.imread(str(imgp))
        feat = extract_feature_from_image(img, yunet, sface)
        if feat is None:
            continue
        label = imgp.stem
        db[label] = l2norm((db[label] + feat) / 2) if label in db else feat
        label_names[label] = label
        print(f"✅ DB(파일): {label}")

    if not db:
        print("⚠️ DB 비어있음 → 전원 Unknown")
    else:
        print(f"📦 FaceDB 라벨 수: {len(db)}")
    return db, label_names

def extract_project_id_from_key_file(key_file_path):
    """서비스 계정 키 파일에서 project_id 추출"""
    try:
        import json
        with open(key_file_path, 'r', encoding='utf-8') as f:
            key_data = json.load(f)
            project_id = key_data.get('project_id') or key_data.get('projectId')
            return project_id
    except Exception as e:
        return None

def get_bucket_names_from_project_id(project_id):
    """project_id로부터 Firebase Storage 버킷 이름 후보 리스트 반환"""
    if not project_id:
        return []
    
    # Firebase Storage 버킷 이름 형식들 (우선순위 순)
    bucket_candidates = [
        f"{project_id}.firebasestorage.app",  # 새로운 형식 (우선 시도)
        f"{project_id}.appspot.com",          # 기본 형식
    ]
    
    return bucket_candidates

def try_bucket_name(bucket_name):
    """버킷 이름이 유효한지 테스트"""
    try:
        init_firebase()
        bucket = storage.bucket(bucket_name)
        # 간단한 접근 테스트
        list(bucket.list_blobs(max_results=1))
        return True
    except Exception:
        return False

def init_firebase():
    """Firebase 초기화 (환경 변수 또는 서비스 계정 키 파일 사용)"""
    global FIREBASE_BUCKET
    
    if firebase_admin._apps:
        return
    
    used_key_file = None
    project_id = None
    
    # 환경 변수에서 서비스 계정 키 경로 확인
    if FIREBASE_CREDENTIAL_PATH and Path(FIREBASE_CREDENTIAL_PATH).exists():
        used_key_file = FIREBASE_CREDENTIAL_PATH
        project_id = extract_project_id_from_key_file(FIREBASE_CREDENTIAL_PATH)
        try:
            cred = credentials.Certificate(FIREBASE_CREDENTIAL_PATH)
            # 버킷 이름이 없으면 project_id에서 자동 생성 (여러 형식 시도)
            if not FIREBASE_BUCKET and project_id:
                bucket_candidates = get_bucket_names_from_project_id(project_id)
                # 첫 번째 후보를 기본값으로 사용
                FIREBASE_BUCKET = bucket_candidates[0] if bucket_candidates else None
                if FIREBASE_BUCKET:
                    print(f"📦 Storage 버킷 자동 감지: {FIREBASE_BUCKET} (project_id: {project_id})")
                    if len(bucket_candidates) > 1:
                        print(f"   (실패 시 다음 후보 시도: {', '.join(bucket_candidates[1:])})")
            
            if not FIREBASE_BUCKET:
                raise ValueError("Storage 버킷 이름을 찾을 수 없습니다.")
            
            firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
            print(f"✅ Firebase 초기화 완료 (서비스 계정 키: {FIREBASE_CREDENTIAL_PATH})")
            return
        except Exception as e:
            print(f"⚠️ 서비스 계정 키 파일 로드 실패: {e}")
    
    # 환경 변수가 없거나 파일이 없는 경우, 자동으로 찾기 시도
    # 현재 디렉토리와 상위 디렉토리에서 JSON 키 파일 찾기
    potential_key_files = []
    script_dir = Path(__file__).parent
    
    # 일반적인 서비스 계정 키 파일 이름들
    key_patterns = ["serviceAccountKey.json", "firebase-adminsdk*.json", "*service-account*.json", "*firebase*.json"]
    
    for pattern in key_patterns:
        potential_key_files.extend(script_dir.glob(pattern))
        potential_key_files.extend((script_dir.parent).glob(pattern))
    
    # 찾은 키 파일 중 유효한 것 사용
    for key_file in potential_key_files:
        try:
            if key_file.exists() and key_file.suffix == '.json':
                # project_id 추출
                project_id = extract_project_id_from_key_file(key_file)
                used_key_file = str(key_file)
                
                # 버킷 이름이 없으면 project_id에서 자동 생성 (여러 형식 시도)
                if not FIREBASE_BUCKET and project_id:
                    bucket_candidates = get_bucket_names_from_project_id(project_id)
                    # 첫 번째 후보를 기본값으로 사용
                    FIREBASE_BUCKET = bucket_candidates[0] if bucket_candidates else None
                    if FIREBASE_BUCKET:
                        print(f"📦 Storage 버킷 자동 감지: {FIREBASE_BUCKET} (project_id: {project_id})")
                        if len(bucket_candidates) > 1:
                            print(f"   (실패 시 다음 후보 시도: {', '.join(bucket_candidates[1:])})")
                
                if not FIREBASE_BUCKET:
                    continue
                
                cred = credentials.Certificate(str(key_file))
                firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
                print(f"✅ Firebase 초기화 완료 (자동 탐지: {key_file})")
                return
        except Exception:
            continue
    
    # Application Default Credentials 시도 (gcloud auth application-default login 필요)
    try:
        # 버킷 이름 확인
        if not FIREBASE_BUCKET:
            print(f"⚠️ Storage 버킷 이름이 설정되지 않았습니다.")
            print(f"   환경 변수 FIREBASE_STORAGE_BUCKET를 설정하거나")
            print(f"   서비스 계정 키 파일을 사용해주세요.")
            FIREBASE_BUCKET = "your-project-id.appspot.com"  # 기본값 (실패할 것임)
        
        firebase_admin.initialize_app(options={"storageBucket": FIREBASE_BUCKET})
        print(f"✅ Firebase 초기화 완료 (Application Default Credentials)")
        return
    except Exception as e:
        error_msg = str(e)
        print(f"\n⚠️ Firebase 초기화 실패: {error_msg}")
        print(f"\n📝 Firebase 인증 설정이 필요합니다:")
        print(f"   방법 1: 환경 변수 설정 (권장)")
        print(f"   PowerShell:")
        print(f"   $env:GOOGLE_APPLICATION_CREDENTIALS=\"C:\\path\\to\\serviceAccountKey.json\"")
        print(f"   ")
        print(f"   CMD:")
        print(f"   set GOOGLE_APPLICATION_CREDENTIALS=C:\\path\\to\\serviceAccountKey.json")
        print(f"   ")
        print(f"   Linux/Mac:")
        print(f"   export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/serviceAccountKey.json\"")
        print(f"   ")
        print(f"   방법 2: 서비스 계정 키 파일을 다음 위치에 배치:")
        print(f"   - {script_dir / 'serviceAccountKey.json'}")
        print(f"   - 또는 test.py와 같은 디렉토리")
        print(f"   ")
        print(f"   방법 3: gcloud Application Default Credentials 설정")
        print(f"   gcloud auth application-default login")
        print(f"   ")
        print(f"   현재 확인한 경로: {FIREBASE_CREDENTIAL_PATH if FIREBASE_CREDENTIAL_PATH else '(환경 변수 없음)'}")
        raise

def sync_faces_from_storage(local_root: Path):
    """Firebase Storage에서 얼굴 이미지 동기화"""
    global FIREBASE_BUCKET
    
    try:
        init_firebase()
    except Exception as e:
        raise Exception(f"Firebase 초기화 실패: {e}")
    
    # project_id 추출을 위한 키 파일 찾기
    project_id = None
    if FIREBASE_CREDENTIAL_PATH and Path(FIREBASE_CREDENTIAL_PATH).exists():
        project_id = extract_project_id_from_key_file(FIREBASE_CREDENTIAL_PATH)
    else:
        # 자동으로 찾은 키 파일 사용
        script_dir = Path(__file__).parent
        key_patterns = ["serviceAccountKey.json", "firebase-adminsdk*.json", "*service-account*.json", "*firebase*.json"]
        for pattern in key_patterns:
            for key_file in list(script_dir.glob(pattern)) + list(script_dir.parent.glob(pattern)):
                if key_file.exists() and key_file.suffix == '.json':
                    project_id = extract_project_id_from_key_file(key_file)
                    break
            if project_id:
                break
    
    # 버킷 이름 후보들 (여러 형식 시도)
    bucket_candidates = []
    if FIREBASE_BUCKET:
        bucket_candidates.append(FIREBASE_BUCKET)
    
    if project_id:
        bucket_candidates.extend(get_bucket_names_from_project_id(project_id))
        # 중복 제거 (순서 유지)
        seen = set()
        unique_candidates = []
        for candidate in bucket_candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        bucket_candidates = unique_candidates
    
    # 각 버킷 이름 후보를 시도
    last_error = None
    for bucket_name in bucket_candidates:
        try:
            # Firebase 앱 재초기화 (다른 버킷 이름으로)
            if firebase_admin._apps:
                firebase_admin.delete_app(firebase_admin.get_app())
            
            if FIREBASE_CREDENTIAL_PATH and Path(FIREBASE_CREDENTIAL_PATH).exists():
                cred = credentials.Certificate(FIREBASE_CREDENTIAL_PATH)
                firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
            else:
                # 키 파일 자동 찾기
                script_dir = Path(__file__).parent
                key_patterns = ["serviceAccountKey.json", "firebase-adminsdk*.json", "*service-account*.json", "*firebase*.json"]
                key_file = None
                for pattern in key_patterns:
                    for kf in list(script_dir.glob(pattern)) + list(script_dir.parent.glob(pattern)):
                        if kf.exists() and kf.suffix == '.json':
                            key_file = kf
                            break
                    if key_file:
                        break
                
                if key_file:
                    cred = credentials.Certificate(str(key_file))
                    firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
                else:
                    firebase_admin.initialize_app(options={"storageBucket": bucket_name})
            
            bucket = storage.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=FACES_PREFIX)
            
            # 성공하면 FIREBASE_BUCKET 업데이트
            FIREBASE_BUCKET = bucket_name
            if bucket_name != bucket_candidates[0]:
                print(f"📦 버킷 이름 자동 변경: {bucket_candidates[0]} → {bucket_name}")
            
            break  # 성공했으므로 루프 종료
        except Exception as e:
            last_error = e
            if bucket_name == bucket_candidates[-1]:  # 마지막 후보도 실패
                raise Exception(f"Firebase Storage 접근 실패 (모든 후보 시도 실패): {e}")
            continue  # 다음 후보 시도

    shutil.rmtree(local_root, ignore_errors=True)
    for blob in blobs:
        name = blob.name
        if not name or name.endswith('/'):
            continue
        relative = name[len(FACES_PREFIX):] if name.startswith(FACES_PREFIX) else name
        if not relative:
            continue
        local_path = local_root / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

def has_gst():
    try: return "GStreamer" in cv2.getBuildInformation()
    except: return False

def gst_pipe_windows():
    # d3d11 하드웨어 디코딩 + 늦은 패킷 드롭
    return (
        f"rtspsrc location={RTSP_URL} latency=0 protocols=udp ! "
        f"rtpjitterbuffer latency=0 drop-on-late=true ! "
        f"rtph264depay ! h264parse ! d3d11h264dec ! d3d11convert ! "
        f"video/x-raw,format=BGR ! videoscale ! "
        f"video/x-raw,width={FRAME_W},height={FRAME_H},framerate=30/1 ! "
        f"appsink drop=1 sync=false max-buffers=1"
    )

# ================== 메인 ==================
def main():
    # 모델 파일 준비
    ensure_file(URL_YUNET, YUNET_PATH)
    ensure_file(URL_SFACE, SFACE_PATH)

    # 디바이스
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    # YOLO 모델들
    if USE_POSE_MODEL:
        # YOLOv8-Pose 모델 (스켈레톤 추출용)
        pose_model = YOLO(YOLO_POSE_MODEL).to(device)
        if device != "cpu":
            try: pose_model.model.half()
            except: pass
        print(f"✅ YOLOv8-Pose 로드: {YOLO_POSE_MODEL}")
        # 일반 YOLO는 Pose 모델로 대체 (track 기능 사용)
        model = pose_model
    else:
        # 일반 YOLO 모델 (얼굴 인식용만)
        model = YOLO(YOLO_MODEL).to(device)
        if device != "cpu":
            try: model.model.half()
            except: pass
        pose_model = None
        print(f"✅ YOLOv8 로드: {YOLO_MODEL}")

    # 얼굴 모델
    recognizer = load_sface(SFACE_PATH)
    yunet_det  = load_yunet(YUNET_PATH)
    
    # GPGait++ 모델 (조건부)
    gait_model = None
    if ENABLE_GAIT_MODEL and GAIT_MODEL_AVAILABLE and GAIT_CHECKPOINT_PATH:
        try:
            if os.path.exists(GAIT_CHECKPOINT_PATH):
                gait_model = GaitModelLoader.load_model(GAIT_CONFIG_PATH, GAIT_CHECKPOINT_PATH, str(device))
                print(f"✅ GPGait++ 모델 로드 완료")
            else:
                print(f"⚠️ GPGait++ 체크포인트 없음: {GAIT_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"⚠️ GPGait++ 모델 로드 실패: {e}")
    
    # Person Registry (조건부)
    person_registry = None
    inference_count = 0  # 추론 횟수 카운터 (자동 저장용)
    if PERSON_REGISTRY_AVAILABLE:
        try:
            person_registry = ImprovedPersonRegistry(
                match_threshold_cos_strict=0.92,
                match_threshold_euc_strict=0.36,
                match_threshold_cos_relaxed=0.90,
                match_threshold_euc_relaxed=0.40,
                max_embeddings=10,
                ema_alpha=0.3
            )
            print(f"✅ Person Registry 초기화 완료")
            
            # Person DB 로드
            load_person_db(person_registry, PERSON_DB_PATH)
        except Exception as e:
            print(f"⚠️ Person Registry 초기화 실패: {e}")

    # Firebase 키 파일 위치 확인 (프로그램 시작 시)
    print("\n🔍 Firebase 서비스 계정 키 파일 검색 중...")
    found_keys = find_firebase_key_files()
    if found_keys:
        print(f"✅ 발견된 키 파일 ({len(found_keys)}개):")
        for i, (source, file_path) in enumerate(found_keys, 1):
            print(f"   {i}. [{source}] {file_path}")
    else:
        print(f"⚠️ Firebase 서비스 계정 키 파일을 찾을 수 없습니다.")
        print(f"   키 파일이 필요합니다. 위의 안내를 따라 환경 변수를 설정하거나")
        print(f"   키 파일을 test.py와 같은 폴더에 배치하세요.")
    print()
    
    # 얼굴 DB 동기화 (Firebase Storage → 로컬 캐시)
    face_db_root = ID_DIR
    try:
        sync_faces_from_storage(FACES_CACHE_DIR)
        face_db_root = str(FACES_CACHE_DIR)
        print(f"✅ Firebase faces 동기화 완료 → {face_db_root}")
    except Exception as e:
        print(f"⚠️ Firebase faces 동기화 실패: {e}\n   → 로컬 폴더({ID_DIR}) 사용")

    face_db, face_labels = build_face_db(yunet_det, recognizer, face_db_root, TARGET_USER_ID)

    danger_ref = None
    active_user_id = None
    last_active_refresh = 0.0
    db_client = None
    
    # Firestore 클라이언트 초기화 (Firebase 초기화 필요)
    try:
        # Firebase가 초기화되지 않았다면 초기화 시도
        if not firebase_admin._apps:
            init_firebase()
        db_client = firestore.client()
        print("✅ Firestore 클라이언트 초기화 완료")
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Firestore 초기화 실패: {error_msg}")
        if "credentials" in error_msg.lower() or "authentication" in error_msg.lower():
            print(f"   💡 Firebase 인증이 필요합니다. 위의 Firebase 인증 설정 방법을 확인하세요.")

    if DANGER_USER_ID and db_client is not None:
        active_user_id = DANGER_USER_ID
        danger_ref = db_client.collection('users').document(active_user_id)
        print(f"✅ 위험도 갱신 대상 고정: users/{active_user_id}")
    elif db_client is not None:
        print("ℹ️ 활성 사용자 자동 탐색 모드")
    else:
        print("⚠️ Firestore 클라이언트 없음 → 위험도 갱신 생략")

    start_webrtc_services(host=WEBRTC_HOST, port=WEBRTC_PORT)
    print(f"✅ WebRTC 신호 서버 실행: http://{WEBRTC_HOST}:{WEBRTC_PORT}/offer")

    # 캡처(지연↓) - 재시도 로직 포함
    print(f"\n🔗 RTSP 연결 시도: {RTSP_URL}")
    cap = None
    
    # 연결 방법 목록 (우선순위 순)
    connection_methods = []
    
    # 1. GStreamer (가장 빠름)
    if has_gst():
        connection_methods.append(("GStreamer", lambda: cv2.VideoCapture(gst_pipe_windows(), cv2.CAP_GSTREAMER)))
    
    # 2. FFmpeg UDP
    connection_methods.append(("FFmpeg UDP", lambda: cv2.VideoCapture(RTSP_URL+"?rtsp_transport=udp", cv2.CAP_FFMPEG)))
    
    # 3. FFmpeg TCP (UDP 실패 시)
    connection_methods.append(("FFmpeg TCP", lambda: cv2.VideoCapture(RTSP_URL+"?rtsp_transport=tcp", cv2.CAP_FFMPEG)))
    
    # 4. FFmpeg 기본
    connection_methods.append(("FFmpeg 기본", lambda: cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)))
    
    # 재시도 루프
    for attempt in range(RTSP_RETRY_COUNT):
        for method_name, create_cap in connection_methods:
            try:
                print(f"   [{attempt+1}/{RTSP_RETRY_COUNT}] {method_name} 시도 중...", end=" ")
                cap = create_cap()
                
                if cap is not None and cap.isOpened():
                    # 버퍼 크기 설정
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # 연결 테스트: 프레임 읽기 시도
                    test_ret, test_frame = cap.read()
                    if test_ret and test_frame is not None:
                        print(f"✅ 성공!")
                        print(f"✅ RTSP 연결 완료: {method_name}")
                        break
                    else:
                        print("❌ 프레임 읽기 실패")
                        cap.release()
                        cap = None
                else:
                    print("❌ 열기 실패")
                    if cap is not None:
                        cap.release()
                        cap = None
            except Exception as e:
                print(f"❌ 오류: {e}")
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                    cap = None
        
        if cap is not None and cap.isOpened():
            break
        
        if attempt < RTSP_RETRY_COUNT - 1:
            print(f"   {RTSP_RETRY_DELAY}초 후 재시도...")
            time.sleep(RTSP_RETRY_DELAY)
    
    if cap is None or not cap.isOpened():
        print(f"\n❌ RTSP 연결 실패: {RTSP_URL}")
        print(f"   가능한 원인:")
        print(f"   1. RTSP 서버가 실행 중이 아닙니다")
        print(f"   2. 네트워크 연결 문제 (IP: {JETSON_IP})")
        print(f"   3. 방화벽이 포트 8554를 차단하고 있습니다")
        print(f"   4. RTSP URL이 올바르지 않습니다")
        print(f"\n   해결 방법:")
        print(f"   - RTSP 서버 상태 확인")
        print(f"   - 네트워크 연결 확인")
        print(f"   - 방화벽 설정 확인")
        print(f"   - RTSP_URL 설정 확인: {RTSP_URL}")
        raise RuntimeError(f"❌ RTSP 입력을 열 수 없습니다: {RTSP_URL}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 상태
    dwell = {}         # tid -> {'start','last'}
    last_seen = {}     # tid -> ts
    last_identity = {}    # tid -> {'key':..., 'text':..., 'known':bool}
    last_face_ts = {}  # tid -> 마지막 얼굴 인식 시각
    face_lock_on = {}  # tid -> {'locked':bool, 'known_start':float, 'lock_time':float}
    FACE_LOCK_SEC = 2.0  # 3초간 연속 인식되면 Lock on
    
    # 스켈레톤 시스템 상태 (tid별)
    skeleton_buffers = {}      # tid -> [skeleton, ...]
    skeleton_states = {}       # tid -> 'IDLE' | 'DETECTING' | 'COLLECTING' | 'LOST'
    skeleton_frames_since_inference = {}  # tid -> int
    skeleton_frames_since_lost = {}       # tid -> int
    skeleton_has_initial_inference = {}   # tid -> bool
    skeleton_person_ids = {}              # tid -> person_id (GPGait++ 매칭 결과)
    skeleton_session_locked = {}          # tid -> bool
    skeleton_tracking_confidence = {}     # tid -> float

    frame_idx = 0
    last_results = None

    grabber = FrameGrabber(cap)
    grabber.start()

    print("\n✅ 실행 중… (Q 종료)\n")
    try:
        while True:
            ok, frame, frame_ts = grabber.read()
            if not ok:
                time.sleep(0.005)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            loop_wall = time.time()
            if (
                db_client is not None
                and DANGER_USER_ID is None
                and loop_wall - last_active_refresh >= ACTIVE_USER_REFRESH_SEC
            ):
                last_active_refresh = loop_wall
                new_active = find_active_user_id(db_client)
                if new_active != active_user_id:
                    # None에서 활성화될 때 또는 활성 사용자가 변경될 때 dwell 초기화
                    was_none = (active_user_id is None)
                    was_active = (active_user_id is not None)
                    active_user_id = new_active
                    if active_user_id:
                        danger_ref = db_client.collection('users').document(active_user_id)
                        print(f"✅ 위험도 갱신 대상: users/{active_user_id}")
                        # None에서 활성화되었으면 dwell 딕셔너리 초기화 (0부터 다시 시작)
                        if was_none:
                            dwell.clear()
                            print("🔄 시간 누적 리셋 (새 모니터링 시작)")
                    else:
                        # 활성에서 None으로 변경되면 즉시 dwell 초기화 (딜레이 제거)
                        if was_active:
                            dwell.clear()
                            print("🛑 시간 누적 즉시 중지 (모니터링 종료)")
                        danger_ref = None
                        print("ℹ️ 활성 사용자 없음 → 위험도 갱신 대기")

            frame_idx += 1
            now = frame_ts if frame_ts > 0 else time.time()
            current_ids = []
            danger_to_add = 0

            # --- YOLO 추적: TRACK_EVERY_N 프레임마다만 실행 ---
            do_track = (frame_idx % TRACK_EVERY_N == 0) or (last_results is None)
            if do_track:
                if USE_POSE_MODEL and pose_model is not None:
                    # Pose 모델로 track + 스켈레톤 추출
                    results = pose_model.track(
                        frame, persist=True, conf=POSE_CONF, imgsz=YOLO_IMGSZ,
                        classes=[0], device=device, verbose=False
                    )
                else:
                    # 일반 YOLO track
                    results = model.track(
                        frame, persist=True, conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                        classes=[0], device=device, verbose=False
                    )
                last_results = results
            else:
                results = last_results  # 최근 결과 재사용
            
            # YOLOv8-Pose로 추가 디텍션 (스켈레톤 추출용, track이 없을 때 사용)
            pose_persons = []
            if USE_POSE_MODEL and pose_model is not None and (do_track or last_results is None):
                pose_persons = detect_persons_yolo_pose(pose_model, frame, POSE_CONF)

            # --- 결과 처리 ---
            # 스켈레톤 매핑: pose_persons에서 tid별 스켈레톤 추출
            tid_to_skeleton = {}  # tid -> skeleton
            if USE_POSE_MODEL and pose_model is not None and results:
                res = results[0]
                boxes = res.boxes
                keypoints = res.keypoints if hasattr(res, 'keypoints') else None
                
                if boxes is not None and keypoints is not None and len(boxes) > 0:
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(boxes))
                    for idx, tid in enumerate(ids):
                        if idx < len(keypoints):
                            kpts = keypoints[idx].xy.cpu().numpy()[0] if len(keypoints[idx].xy) > 0 else None
                            if kpts is not None and len(kpts) == 17:
                                tid_to_skeleton[tid] = kpts.astype(np.float32)
            
            if results:
                res = results[0]
                boxes = res.boxes
                if boxes is not None and len(boxes) > 0:
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(boxes))
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    for tid, (x1,y1,x2,y2), conf in zip(ids, xyxy, confs):
                        if conf < YOLO_CONF:
                            continue
                        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                        current_ids.append(tid)
                        
                        # 스켈레톤 추출 및 버퍼 관리
                        skeleton = tid_to_skeleton.get(tid, None)
                        
                        # 스켈레톤 시스템 초기화 (tid별)
                        if tid not in skeleton_buffers:
                            skeleton_buffers[tid] = []
                            skeleton_states[tid] = 'IDLE'
                            skeleton_frames_since_inference[tid] = 0
                            skeleton_frames_since_lost[tid] = 0
                            skeleton_has_initial_inference[tid] = False
                            skeleton_person_ids[tid] = None
                            skeleton_session_locked[tid] = False
                            skeleton_tracking_confidence[tid] = 0.0
                        
                        # 상태 머신 업데이트 (스켈레톤 기반)
                        if SKELETON_STATE_ENABLED and skeleton is not None:
                            current_state = skeleton_states[tid]
                            conf = conf if 'conf' in locals() else POSE_CONF
                            
                            if current_state == 'IDLE':
                                skeleton_states[tid] = 'DETECTING'
                                skeleton_buffers[tid].clear()
                                skeleton_has_initial_inference[tid] = False
                                skeleton_frames_since_inference[tid] = 0
                            elif current_state == 'DETECTING':
                                skeleton_states[tid] = 'COLLECTING'
                            elif current_state == 'LOST':
                                if skeleton_frames_since_lost[tid] < 30:  # 1초 @ 30fps
                                    skeleton_states[tid] = 'COLLECTING'
                                    skeleton_frames_since_lost[tid] = 0
                                else:
                                    skeleton_states[tid] = 'DETECTING'
                                    skeleton_buffers[tid].clear()
                                    skeleton_has_initial_inference[tid] = False
                                    skeleton_frames_since_inference[tid] = 0
                        
                        # 스켈레톤 버퍼 관리
                        if skeleton is not None and SKELETON_STATE_ENABLED:
                            if skeleton_states[tid] == 'COLLECTING':
                                skeleton_buffers[tid].append(skeleton)
                                skeleton_frames_since_inference[tid] += 1
                                
                                buffer_size = len(skeleton_buffers[tid])
                                
                                # 첫 추론 (초기 버퍼 채워지면)
                                if not skeleton_has_initial_inference[tid] and buffer_size >= SKELETON_INITIAL_BUFFER:
                                    if gait_model is not None and person_registry is not None:
                                        result = process_skeleton_inference(
                                            tid, frame_idx, gait_model, person_registry, device,
                                            is_final=False
                                        )
                                        if result:
                                            skeleton_person_ids[tid] = result['person_id']
                                            skeleton_has_initial_inference[tid] = True
                                            skeleton_tracking_confidence[tid] = result['match_score']
                                            
                                            # DB 자동 저장 (NEW 등록 시 또는 주기적)
                                            inference_count += 1
                                            if result['is_new'] or inference_count % PERSON_DB_AUTOSAVE_INTERVAL == 0:
                                                save_person_db(person_registry, PERSON_DB_PATH)
                                        if result:
                                            skeleton_person_ids[tid] = result['person_id']
                                            skeleton_has_initial_inference[tid] = True
                                            skeleton_tracking_confidence[tid] = result['match_score']
                                
                                # 주기적 추론
                                elif (skeleton_has_initial_inference[tid] and 
                                      skeleton_frames_since_inference[tid] >= SKELETON_ACCUM_INTERVAL):
                                    if gait_model is not None and person_registry is not None:
                                        result = process_skeleton_inference(
                                            tid, frame_idx, gait_model, person_registry, device,
                                            is_final=False
                                        )
                                        if result:
                                            skeleton_person_ids[tid] = result['person_id']
                                            skeleton_tracking_confidence[tid] = result['match_score']
                                            
                                            # DB 자동 저장 (NEW 등록 시 또는 주기적)
                                            inference_count += 1
                                            if result['is_new'] or inference_count % PERSON_DB_AUTOSAVE_INTERVAL == 0:
                                                save_person_db(person_registry, PERSON_DB_PATH)
                                        result = process_skeleton_inference(
                                            tid, frame_idx, gait_model, person_registry, device,
                                            is_final=False
                                        )
                                        if result:
                                            skeleton_person_ids[tid] = result['person_id']
                                            skeleton_tracking_confidence[tid] = result['match_score']
                        elif skeleton is None and SKELETON_STATE_ENABLED:
                            # 스켈레톤 없으면 LOST 처리
                            if skeleton_states[tid] == 'COLLECTING':
                                skeleton_states[tid] = 'LOST'
                                skeleton_frames_since_lost[tid] = 0
                            elif skeleton_states[tid] == 'LOST':
                                skeleton_frames_since_lost[tid] += 1
                                
                                # LOST 상태에서 최종 추론
                                if skeleton_frames_since_lost[tid] >= 30 and len(skeleton_buffers[tid]) > 0:
                                    if gait_model is not None and person_registry is not None:
                                        result = process_skeleton_inference(
                                            tid, frame_idx, gait_model, person_registry, device,
                                            is_final=True
                                        )
                                    # 세션 리셋
                                    skeleton_buffers[tid].clear()
                                    skeleton_states[tid] = 'IDLE'
                                    skeleton_frames_since_lost[tid] = 0
                                    skeleton_has_initial_inference[tid] = False
                                    skeleton_person_ids[tid] = None
                                    skeleton_session_locked[tid] = False

                        # dwell - danger_ref가 있을 때만 시간 누적
                        if danger_ref is not None:
                            if tid not in dwell:
                                dwell[tid] = {'start': now, 'last': now, 'danger_awarded': 0}
                            else:
                                dwell[tid]['last'] = now
                            dur = dwell[tid]['last'] - dwell[tid]['start']
                        else:
                            # danger_ref가 없으면 시간 누적하지 않음 (항상 0으로 표시)
                            dur = 0

                        # 얼굴 인식 (트랙당 FACE_EVERY_SEC 간격)
                        identity = last_identity.get(tid)
                        dwell_start_time = dwell[tid]['start'] if tid in dwell else now
                        
                        # Lock on 상태 확인
                        lock_info = face_lock_on.get(tid, {
                            'locked': False, 
                            'known_start': 0.0, 
                            'lock_time': 0.0,
                            'unknown_count': 0
                        })
                        
                        # 조건 1: Lock-on 되었으면 얼굴 인식 완전히 건너뛰기
                        if lock_info['locked']:
                            identity = last_identity.get(tid, {'text': 'Unknown', 'known': False})
                            # 스켈레톤만으로 추적 중 - 추가 연산 없음
                        
                        # 조건 2: confidence가 0.9 이상이고 Lock-on 안 되었을 때만 얼굴 인식
                        elif conf >= FACE_START_CONF:
                            # 얼굴 인식 주기 체크
                            should_refresh = (
                                identity is None  # 처음 등장
                                or (now - last_face_ts.get(tid, 0.0) >= FACE_EVERY_SEC)  # 주기 도래
                            )
                            
                            if should_refresh:
                                # YuNet + SFace 얼굴 인식 수행
                                detected_key = None
                                detected_text = "Unknown"
                                detected_known = False
                                roi = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                                if roi.size > 0 and face_db:
                                    h, w = roi.shape[:2]
                                    if max(h,w) < UPSCALE_FACE_TO:
                                        scale = UPSCALE_FACE_TO / max(1, max(h,w))
                                        roi_in = cv2.resize(roi, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
                                    else:
                                        roi_in = roi

                                    try:
                                        yunet_det.setInputSize((roi_in.shape[1], roi_in.shape[0]))
                                    except Exception:
                                        pass

                                    okf, faces = yunet_det.detect(roi_in)
                                    if okf and faces is not None and len(faces) > 0:
                                        # 가장 큰 얼굴 1개
                                        inds = np.argsort(faces[:,2]*faces[:,3])[::-1]
                                        f = faces[inds[0]]
                                        w0, h0 = int(f[2]), int(f[3])
                                        if min(w0, h0) >= MIN_FACE_SIDE:
                                            try:
                                                aligned = recognizer.alignCrop(roi_in, f)
                                                feat = l2norm(recognizer.feature(aligned))
                                                best = -1.0
                                                best_key = None
                                                for key, emb in face_db.items():
                                                    s = cosine_sim(emb, feat)
                                                    if s > best:
                                                        best = s
                                                        best_key = key
                                                if best_key is not None and best >= SIM_THR:
                                                    detected_key = best_key
                                                    detected_known = True
                                                    detected_text = face_labels.get(best_key, best_key.split('/')[-1])
                                            except Exception:
                                                pass

                                identity = {
                                    'key': detected_key,
                                    'text': detected_text,
                                    'known': detected_known,
                                }
                                last_identity[tid] = identity
                                last_face_ts[tid] = now
                                
                                # Lock on 로직
                                if detected_known:
                                    if lock_info['known_start'] == 0.0:
                                        # 처음 known 인식 시작
                                        face_lock_on[tid] = {
                                            'locked': False,
                                            'known_start': now,
                                            'lock_time': 0.0,
                                            'unknown_count': 0
                                        }
                                        print(f"✅ 얼굴 인식 시작: tid={tid} -> {detected_text}")
                                    else:
                                        # 이미 known 인식 중
                                        known_duration = now - lock_info['known_start']
                                        
                                        if known_duration >= FACE_LOCK_SEC and not lock_info['locked']:
                                            # 3초간 연속 인식 -> Lock on
                                            face_lock_on[tid] = {
                                                'locked': True,
                                                'known_start': lock_info['known_start'],
                                                'lock_time': now,
                                                'unknown_count': 0,
                                                'face_name': detected_text,
                                                'face_key': detected_key
                                            }
                                            print(f"🔒 얼굴 Lock-on 완료: tid={tid} -> {detected_text}")
                                            print(f"   → 이제부터 스켈레톤 보행 패턴으로만 추적")
                                            
                                            # 스켈레톤 Person ID 연동
                                            skeleton_person_id = skeleton_person_ids.get(tid, None)
                                            if skeleton_person_id is not None:
                                                print(f"   ✅ 스켈레톤 연동: Person #{skeleton_person_id} = {detected_text}")
                                        else:
                                            # Unknown 카운터 리셋
                                            face_lock_on[tid]['unknown_count'] = 0
                                else:
                                    # Unknown 감지
                                    if lock_info['locked']:
                                        # Lock-on 상태에서 Unknown
                                        unknown_count = lock_info.get('unknown_count', 0) + 1
                                        
                                        if unknown_count >= FACE_LOCK_UNKNOWN_TOLERANCE:
                                            # Unknown 허용 횟수 초과 -> Lock 해제
                                            face_lock_on[tid] = {
                                                'locked': False,
                                                'known_start': 0.0,
                                                'lock_time': 0.0,
                                                'unknown_count': 0
                                            }
                                            print(f"🔓 Lock-on 해제: tid={tid} (Unknown {unknown_count}회)")
                                        else:
                                            # 허용 범위 내 - Lock 유지
                                            face_lock_on[tid]['unknown_count'] = unknown_count
                                            print(f"⚠️ Unknown 감지: tid={tid} ({unknown_count}/{FACE_LOCK_UNKNOWN_TOLERANCE})")
                                    else:
                                        # Lock-on 전 Unknown -> 리셋
                                        if lock_info['known_start'] > 0.0:
                                            face_lock_on[tid] = {
                                                'locked': False,
                                                'known_start': 0.0,
                                                'lock_time': 0.0,
                                                'unknown_count': 0
                                            }
                            else:
                                # 주기 안 됨 - 기존 identity 유지
                                identity = last_identity.get(tid, {'text': 'Unknown', 'known': False})
                        
                        # 조건 3: confidence < 0.9 이면 얼굴 인식 안 함
                        else:
                            identity = last_identity.get(tid, {'text': 'Unknown', 'known': False})
                            # confidence가 낮아서 얼굴 인식 건너뜀

                        # 박스/라벨 그리기 - 색상 및 추적 모드 표시
                        label_text = identity.get('text', 'Unknown') if identity else 'Unknown'
                        is_known = identity.get('known', False) if identity else False
                        skeleton_person_id = skeleton_person_ids.get(tid, None)
                        
                        # 색상 및 추적 모드
                        if lock_info['locked']:
                            color = (255, 128, 0)  # 청록색 - 스켈레톤 추적 중
                            tracking_mode = "[SKELETON]"
                        elif conf >= FACE_START_CONF:
                            color = (0, 255, 0) if is_known else (0, 0, 255)  # 초록/빨강 - 얼굴 인식 중
                            tracking_mode = "[FACE]"
                        else:
                            color = (128, 128, 128)  # 회색 - 대기 중 (conf < 0.9)
                            tracking_mode = "[WAIT]"
                        
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        
                        # 라벨
                        label_display = f"{tracking_mode} {label_text}"
                        if skeleton_person_id is not None and SKELETON_STATE_ENABLED:
                            label_display += f" (P#{skeleton_person_id})"
                        
                        # confidence 표시 추가
                        label_display += f" {int(dur)}s | C:{conf:.2f}"
                        
                        cv2.putText(frame, label_display,
                                    (x1, max(15, y1-6)), FONT, 0.7, color, 2)
                        
                        # 스켈레톤 그리기
                        if skeleton is not None and USE_POSE_MODEL:
                            frame = draw_skeleton(frame, skeleton)
                        
                        # 스켈레톤 상태 정보 표시 (우측 상단)
                        if SKELETON_STATE_ENABLED:
                            state = skeleton_states.get(tid, 'IDLE')
                            buffer_size = len(skeleton_buffers.get(tid, []))
                            if buffer_size > 0 or state != 'IDLE':
                                state_color = {
                                    'IDLE': (128, 128, 128),
                                    'DETECTING': (0, 255, 255),
                                    'COLLECTING': (0, 255, 0),
                                    'LOST': (0, 165, 255)
                                }.get(state, (255, 255, 255))
                                
                                state_text = f"S:{state} B:{buffer_size}"
                                if skeleton_person_id is not None:
                                    state_text += f" ID:#{skeleton_person_id}"
                                cv2.putText(frame, state_text, (x2-150, y1-6), 
                                          FONT, 0.5, state_color, 1)

                        # 위험도 적산 (10초 이후 초당 1점) - danger_ref가 있을 때만 계산
                        if danger_ref is not None and dur >= DANGER_BASE_SEC:
                            elapsed = dur - DANGER_BASE_SEC
                            awarded = dwell[tid].get('danger_awarded', 0)
                            grant = int(elapsed) - awarded
                            if grant > 0:
                                danger_to_add += grant
                                dwell[tid]['danger_awarded'] = awarded + grant

            # 사라진 트랙 정리
            for tid in list(dwell.keys()):
                if tid not in current_ids:
                    if tid not in last_seen:
                        last_seen[tid] = now
                    
                    # ABSENCE_GRACE 이상 안 보이면 완전히 정리
                    if now - last_seen[tid] > ABSENCE_GRACE:
                        print(f"👋 트랙 소멸: tid={tid} (재등장 시 처음부터 시작)")
                        
                        # 모든 상태 초기화
                        dwell.pop(tid, None)
                        last_identity.pop(tid, None)
                        last_face_ts.pop(tid, None)
                        last_seen.pop(tid, None)
                        face_lock_on.pop(tid, None)  # Lock-on 해제
                        
                        # 스켈레톤 시스템 정리
                        if SKELETON_STATE_ENABLED:
                            skeleton_buffers.pop(tid, None)
                            skeleton_states.pop(tid, None)
                            skeleton_frames_since_inference.pop(tid, None)
                            skeleton_frames_since_lost.pop(tid, None)
                            skeleton_has_initial_inference.pop(tid, None)
                            skeleton_person_ids.pop(tid, None)
                            skeleton_session_locked.pop(tid, None)
                            skeleton_tracking_confidence.pop(tid, None)
                else:
                    last_seen[tid] = now

            if danger_ref is not None and danger_to_add > 0:
                try:
                    danger_ref.update({'dangerLevel': firestore.Increment(danger_to_add)})
                    print(f"⚠️ 위험도 +{danger_to_add} (누적) → {active_user_id}")
                except Exception as e:
                    print(f"⚠️ 위험도 갱신 실패: {e}")

            push_frame_to_webrtc(frame)

            cv2.imshow("YOLOv8n Person + YuNet/SFace (30s/60s Dwell Alert)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        grabber.stop()
        
        # 프로그램 종료 시 Person DB 최종 저장
        if person_registry is not None:
            print("\n💾 프로그램 종료 - Person DB 최종 저장 중...")
            save_person_db(person_registry, PERSON_DB_PATH)

    cap.release()
    cv2.destroyAllWindows()

    for pc in list(_peer_connections):
        asyncio.run_coroutine_threadsafe(_cleanup_peer(pc), _webrtc_loop)



if __name__ == "__main__":
    main()
