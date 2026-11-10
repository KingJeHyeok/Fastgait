# rtsp_web_gait_tracker.py - RTSP 실시간 보행 인식 웹 시스템
"""
RTSP 스트림에서 실시간 보행 인식 + Flask 웹 시각화
- YOLOv8-Pose 기반 사람 디텍션 + 스켈레톤 추출
- 상태 머신 기반 추적 (IDLE → DETECTING → COLLECTING → LOST)
- GPGait++ 모델 추론
- ImprovedPersonRegistry (멀티 임베딩 비교)
- Flask 웹 UI (실시간 시각화)
"""
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from datetime import datetime
import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not installed. Install with: pip install ultralytics")

# Project path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import from existing modules
from realtime_skeleton_gait import GaitModelLoader, SkeletonExtractor
from improved_person_registry import ImprovedPersonRegistry

app = Flask(__name__)


class RTSPWebGaitTracker:
    """RTSP 기반 실시간 보행 인식 웹 시스템"""
    
    def __init__(self, rtsp_url, config_path=None, checkpoint_path=None,
                 enable_model=True, yolo_model='yolov8n-pose.pt', output_dir='rtsp_output'):
        """초기화"""
        print("="*80)
        print("🚀 RTSP Web Gait Tracker Initialization")
        print("="*80)
        
        self.rtsp_url = rtsp_url
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"📡 RTSP URL: {rtsp_url}")
        print(f"💾 Output Dir: {output_dir}\n")
        
        # === YOLO 초기화 ===
        print("🔧 Initializing YOLO...")
        self.yolo_model = None
        self.yolo_enabled = False
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(yolo_model)
                self.yolo_enabled = True
                print(f"✅ YOLO loaded: {yolo_model}\n")
            except Exception as e:
                print(f"❌ YOLO load failed: {e}\n")
        
        # === 상태 머신 ===
        print("🔧 Setting up state machine...")
        self.tracking_state = "IDLE"  # IDLE, DETECTING, COLLECTING, LOST
        self.detection_confidence_threshold = 0.75
        self.lost_delay_seconds = 1.0  # LOST 후 1초 대기
        self.frames_since_lost = 0
        self.lost_delay_frames = 30  # 30 frames = 1 sec @ 30fps
        print("✅ State machine ready\n")
        
        # === 버퍼 시스템 ===
        print("🔧 Configuring buffer system...")
        self.initial_buffer_size = 300  # 10초 (첫 추론)
        self.accumulation_interval = 90  # 3초 (주기적 추론)
        self.max_buffer_size = 400  # 최대 버퍼
        self.skeleton_buffer = []
        self.frames_since_last_inference = 0
        self.has_initial_inference = False
        print(f"   - Initial buffer: {self.initial_buffer_size} frames (10s)")
        print(f"   - Accumulation: {self.accumulation_interval} frames (3s)")
        print(f"   - Max buffer: {self.max_buffer_size} frames\n")
        
        # === Person ID 관리 ===
        print("🔧 Setting up person tracking...")
        self.current_person_id = None
        self.session_person_locked = False
        self.tracking_confidence = 0.0
        print("✅ Person tracking ready\n")
        
        # === GPGait++ 모델 ===
        print("🔧 Loading GPGait++ model...")
        self.enable_model = enable_model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   - Device: {self.device}")
        
        if enable_model and config_path and checkpoint_path:
            try:
                self.model = GaitModelLoader.load_model(config_path, checkpoint_path, str(self.device))
                print(f"✅ Model loaded successfully\n")
            except Exception as e:
                print(f"❌ Model load failed: {e}\n")
                raise
        
        # === Person Registry ===
        print("🔧 Creating Person Registry...")
        self.person_registry = ImprovedPersonRegistry(
            match_threshold_cos_strict=0.92,   # Cosine ≥ 0.85
            match_threshold_euc_strict=0.36,   # Euclidean ≤ 0.50
            match_threshold_cos_relaxed=0.90,
            match_threshold_euc_relaxed=0.40,
            max_embeddings=10,
            ema_alpha=0.3
        )
        print("   - Thresholds: Cos≥0.85 AND Euc≤0.50")
        
        # Person DB 로드
        self.person_db_path = os.path.join(output_dir, 'person_database.pkl')
        if os.path.exists(self.person_db_path):
            self.load_person_db()
        else:
            print("   - No existing DB, starting fresh\n")
        
        # === 스레드 통신 ===
        print("🔧 Setting up threading...")
        self.frame_queue = Queue(maxsize=5)
        self.display_frame = None
        self.display_lock = threading.Lock()
        self.running = False
        print("✅ Threading ready\n")
        
        # === 웹 데이터 ===
        print("🔧 Initializing web data structures...")
        self.web_data = {
            'current_person': None,
            'person_list': [],
            'stats': {
                'received_frames': 0,
                'processed_frames': 0,
                'detected_frames': 0,
                'inference_count': 0,
                'yolo_detections': 0,
                'skeleton_extractions': 0,
                'fps': 0
            },
            'state_info': {
                'current_state': 'IDLE',
                'buffer_size': 0,
                'frames_since_lost': 0
            },
            'event_log': []  # 최근 20개 이벤트
        }
        self.web_data_lock = threading.Lock()
        print("✅ Web data ready\n")
        
        # === 로깅 ===
        print("🔧 Setting up logging...")
        self.event_log = []
        self.matching_log = []
        self.log_file = os.path.join(output_dir, f'rtsp_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        print(f"   - Log file: {os.path.basename(self.log_file)}\n")
        
        # FPS 계산
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # RTSP 캡처
        self.cap = None
        
        print("="*80)
        print("✅ Initialization Complete!")
        print("="*80 + "\n")
    
    def load_person_db(self):
        """Person DB 로드"""
        import pickle
        try:
            with open(self.person_db_path, 'rb') as f:
                data = pickle.load(f)
                self.person_registry.registry = data['registry']
                self.person_registry.next_id = data['next_id']
                print(f"   - Loaded {len(self.person_registry.registry)} persons from DB")
                for pid, info in self.person_registry.registry.items():
                    emb_count = len(info.get('embeddings_list', []))
                    print(f"     Person #{pid}: {emb_count} embeddings")
                print()
        except Exception as e:
            print(f"   ⚠️ Failed to load DB: {e}\n")
    
    def save_person_db(self):
        """Person DB 저장"""
        import pickle
        try:
            data = {
                'registry': self.person_registry.registry,
                'next_id': self.person_registry.next_id
            }
            with open(self.person_db_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"\n💾 Person DB saved: {len(self.person_registry.registry)} persons")
        except Exception as e:
            print(f"⚠️ Save DB failed: {e}")
    
    def log_event(self, frame_idx, event_type, message):
        """이벤트 로깅"""
        timestamp = time.time()
        event = {
            'frame': frame_idx,
            'timestamp': timestamp,
            'time_str': datetime.fromtimestamp(timestamp).strftime("%H:%M:%S"),
            'type': event_type,
            'message': message
        }
        self.event_log.append(event)
        
        # 웹 UI용 이벤트 로그 (최근 20개)
        with self.web_data_lock:
            self.web_data['event_log'].append(event)
            if len(self.web_data['event_log']) > 20:
                self.web_data['event_log'].pop(0)
        
        print(f"[{event['time_str']}] {event_type:20s} | {message}")
    
    def detect_persons_yolo_pose(self, frame):
        """YOLOv8-Pose로 사람 디텍션 + 스켈레톤 추출"""
        if not self.yolo_enabled or self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, verbose=False)
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
                        if (conf > 0.6 and bbox_w > 30 and bbox_h > 30 and skeleton is not None):
                            persons.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'skeleton': skeleton
                            })
            
            return persons
        except Exception as e:
            print(f"⚠️ YOLO error: {e}")
            return []
    
    def select_best_person(self, persons, frame_shape):
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
    
    def update_tracking_state(self, best_person, frame_idx):
        """상태 머신 업데이트"""
        current_state = self.tracking_state
        
        if best_person and best_person['confidence'] >= self.detection_confidence_threshold:
            if current_state == "IDLE":
                self.log_event(frame_idx, "DETECTING", f"Person detected (conf: {best_person['confidence']:.2f})")
                self.tracking_state = "DETECTING"
                self.skeleton_buffer.clear()
                self.has_initial_inference = False
                self.frames_since_last_inference = 0
                self.current_person_id = None
                self.session_person_locked = False
            elif current_state == "DETECTING":
                self.tracking_state = "COLLECTING"
                self.log_event(frame_idx, "COLLECTING", f"Started collecting (target: {self.initial_buffer_size})")
            elif current_state == "LOST":
                if self.frames_since_lost < self.lost_delay_frames:
                    self.log_event(frame_idx, "COLLECTING", f"Re-detected ({self.frames_since_lost} frames lost)")
                    self.tracking_state = "COLLECTING"
                    self.frames_since_lost = 0
                else:
                    self.log_event(frame_idx, "DETECTING", "New person after delay")
                    self.tracking_state = "DETECTING"
                    self.skeleton_buffer.clear()
                    self.has_initial_inference = False
                    self.frames_since_last_inference = 0
                    self.current_person_id = None
                    self.session_person_locked = False
        else:
            if current_state == "COLLECTING":
                self.log_event(frame_idx, "LOST", f"Person lost (delay: {self.lost_delay_seconds}s)")
                self.tracking_state = "LOST"
                self.frames_since_lost = 0
            elif current_state == "LOST":
                self.frames_since_lost += 1
                
                if self.frames_since_lost >= self.lost_delay_frames:
                    if len(self.skeleton_buffer) > 0:
                        self.log_event(frame_idx, "FINALIZING", f"Delay complete ({len(self.skeleton_buffer)} frames)")
                    else:
                        self.log_event(frame_idx, "IDLE", "Empty buffer, returning to IDLE")
                        self.tracking_state = "IDLE"
    
    def process_inference(self, frame_idx, is_final=False):
        """추론 처리"""
        if not self.enable_model or len(self.skeleton_buffer) == 0:
            if is_final:
                self.reset_tracking_session(frame_idx)
            return None
        
        buffer_size = len(self.skeleton_buffer)
        
        if buffer_size < 30:
            if is_final:
                self.reset_tracking_session(frame_idx)
            return None
        
        # 스켈레톤 시퀀스 준비
        skeleton_sequence = np.array(self.skeleton_buffer)
        
        if buffer_size > self.max_buffer_size:
            indices = np.linspace(0, buffer_size-1, self.max_buffer_size, dtype=int)
            skeleton_sequence = skeleton_sequence[indices]
        
        result = self.run_gait_model(skeleton_sequence, frame_idx)
        
        if result:
            inference_type = "Final" if is_final else "Periodic"
            
            # 첫 추론
            if not self.has_initial_inference:
                self.current_person_id = result['person_id']
                self.has_initial_inference = True
                self.tracking_confidence = result['match_score']
                
                # 세션 잠금 결정: is_new가 아니고(기존 매칭) score가 높으면 즉시 잠금
                if not result['is_new'] and result['match_score'] > 0.85:
                    self.session_person_locked = True
                    lock_status = "LOCKED"
                elif result['match_score'] > 0.85:
                    self.session_person_locked = True
                    lock_status = "LOCKED"
                else:
                    self.session_person_locked = False
                    lock_status = "UNLOCKED"
                
                status = "NEW" if result['is_new'] else "MATCHED"
                self.log_event(frame_idx, f"INFERENCE_{inference_type.upper()}", 
                             f"Person #{result['person_id']} ({status}, score: {result['match_score']:.3f}, cos: {result['cosine_similarity']:.3f}, euc: {result['euclidean_distance']:.3f}, {lock_status})")
                
                # NEW 사람인 경우 알림
                if result['is_new']:
                    print(f"   ✨ NEW Person #{result['person_id']} registered")
            else:
                # 세션 잠금 처리
                if self.session_person_locked:
                    if result['match_score'] < 0.70:
                        self.session_person_locked = False
                        self.log_event(frame_idx, "SESSION_UNLOCK", f"Low confidence ({result['match_score']:.3f})")
                    else:
                        if result['person_id'] != self.current_person_id:
                            result['person_id'] = self.current_person_id
                            result['is_new'] = False
                
                # ID 변경 처리
                if not self.session_person_locked:
                    if result['person_id'] != self.current_person_id:
                        if result['match_score'] > self.tracking_confidence + 0.10:
                            old_id = self.current_person_id
                            self.current_person_id = result['person_id']
                            self.tracking_confidence = result['match_score']
                            self.log_event(frame_idx, "ID_CHANGE", f"#{old_id} → #{result['person_id']} (score: {result['match_score']:.3f})")
                            
                            if result['match_score'] > 0.85:
                                self.session_person_locked = True
                                self.log_event(frame_idx, "SESSION_LOCK", f"Locked to Person #{result['person_id']}")
                        else:
                            result['person_id'] = self.current_person_id
                            result['is_new'] = False
                    else:
                        self.tracking_confidence = max(self.tracking_confidence, result['match_score'])
                
                self.log_event(frame_idx, f"INFERENCE_{inference_type.upper()}", 
                             f"Person #{result['person_id']} (score: {result['match_score']:.3f}, cos: {result['cosine_similarity']:.3f}, euc: {result['euclidean_distance']:.3f})")
            
            with self.web_data_lock:
                self.web_data['stats']['inference_count'] += 1
        
        if is_final:
            self.reset_tracking_session(frame_idx)
        else:
            self.frames_since_last_inference = 0
        
        return result
    
    def run_gait_model(self, skeleton_sequence, frame_idx):
        """GPGait++ 추론"""
        if self.model is None:
            return None
        
        try:
            data = GaitModelLoader.preprocess_skeleton_sequence(skeleton_sequence)
            # GPU로 명시적 이동 + contiguous
            data = data.to(self.device, non_blocking=True).contiguous()
            
            with torch.no_grad():
                T = data.shape[2]
                seqL = torch.LongTensor([T]).to(self.device, non_blocking=True)

                inputs = [[data], None, None, None, [seqL]]
                outputs = self.model.forward(inputs)
                
                feat = outputs['inference_feat']['embeddings']
                if feat.dim() == 3:
                    feat = feat.mean(dim=2)
                feat = F.normalize(feat, p=2, dim=1)
            
            # 매칭
            if self.session_person_locked and self.current_person_id is not None:
                person_id, is_new, match_score, cosine_sim, euclid_dist = self.person_registry.force_update_person(
                    self.current_person_id, feat
                )
            else:
                person_id, is_new, match_score, cosine_sim, euclid_dist = self.person_registry.match_or_register(
                    feat, frame_count=len(skeleton_sequence)
                )
            
            # 로깅
            self.matching_log.append({
                'frame': frame_idx,
                'person_id': person_id,
                'is_new': is_new,
                'match_score': match_score,
                'cosine_similarity': float(cosine_sim),
                'euclidean_distance': float(euclid_dist),
                'buffer_size': len(skeleton_sequence),
                'session_locked': self.session_person_locked
            })
            
            # 자동 저장 (NEW 등록 시 또는 10회 추론마다)
            if is_new or len(self.matching_log) % 10 == 0:
                self.save_person_db()
            
            return {
                'person_id': person_id,
                'is_new': is_new,
                'match_score': match_score,
                'cosine_similarity': float(cosine_sim),
                'euclidean_distance': float(euclid_dist)
            }
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            return None
    
    def reset_tracking_session(self, frame_idx):
        """세션 리셋"""
        self.log_event(frame_idx, "SESSION_END", f"Person #{self.current_person_id} session complete")
        self.skeleton_buffer.clear()
        self.tracking_state = "IDLE"
        self.frames_since_lost = 0
        self.has_initial_inference = False
        self.current_person_id = None
        self.session_person_locked = False
    
    def draw_visualization(self, frame, skeleton, best_person, current_result):
        """시각화"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 바운딩 박스
        if best_person:
            bbox = best_person['bbox']
            x1, y1, x2, y2 = bbox
            conf = best_person['confidence']
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Person {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 스켈레톤
        if skeleton is not None:
            coco_skeleton = [
                [0, 1], [0, 2], [1, 3], [2, 4],
                [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [12, 14], [13, 15], [14, 16]
            ]
            
            for joint1, joint2 in coco_skeleton:
                pt1 = skeleton[joint1]
                pt2 = skeleton[joint2]
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(display_frame, 
                            (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), 
                            (0, 255, 0), 2)
            
            for i, (x, y) in enumerate(skeleton):
                if x > 0 and y > 0:
                    cv2.circle(display_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.circle(display_frame, (int(x), int(y)), 6, (255, 255, 255), 1)
        
        # 상태 정보 (상단)
        state_color = {
            "IDLE": (128, 128, 128),
            "DETECTING": (0, 255, 255),
            "COLLECTING": (0, 255, 0),
            "LOST": (0, 165, 255)
        }.get(self.tracking_state, (255, 255, 255))
        
        # 배경
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (5, 5), (650, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # 상태
        info_text = f"State: {self.tracking_state} | Buffer: {len(self.skeleton_buffer)}"
        if self.current_person_id:
            lock_status = "LOCKED" if self.session_person_locked else "UNLOCKED"
            info_text += f" | Person #{self.current_person_id} ({lock_status})"
        
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # 현재 결과
        if current_result:
            status = "NEW" if current_result['is_new'] else "MATCHED"
            status_color = (0, 255, 255) if current_result['is_new'] else (0, 255, 0)
            
            result_text = f"Person #{current_result['person_id']} ({status})"
            cv2.putText(display_frame, result_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            score_text = f"Score: {current_result['match_score']:.3f}"
            cv2.putText(display_frame, score_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cosine_text = f"Cosine: {current_result['cosine_similarity']:.4f}"
            cv2.putText(display_frame, cosine_text, (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
            
            euclid_text = f"Euclid: {current_result['euclidean_distance']:.4f}"
            cv2.putText(display_frame, euclid_text, (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        
        # Person Registry (우측 상단)
        if len(self.person_registry.registry) > 0:
            overlay2 = display_frame.copy()
            box_height = 35 + len(self.person_registry.registry) * 25
            cv2.rectangle(overlay2, (w-200, 5), (w-5, box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.6, display_frame, 0.4, 0, display_frame)
            
            cv2.putText(display_frame, "Person DB:", (w-190, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset = 50
            for pid in sorted(self.person_registry.registry.keys()):
                info = self.person_registry.registry[pid]
                text_color = (0, 255, 0) if pid == self.current_person_id else (200, 200, 200)
                person_text = f"#{pid}: {info['count']} det"
                cv2.putText(display_frame, person_text, (w-190, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
                y_offset += 25
        
        return display_frame
    
    def update_fps(self):
        """FPS 업데이트"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            with self.web_data_lock:
                self.web_data['stats']['fps'] = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def update_web_ui(self, current_result):
        """웹 UI 데이터 업데이트"""
        with self.web_data_lock:
            # 현재 인식 중인 사람
            if current_result:
                self.web_data['current_person'] = {
                    'id': current_result['person_id'],
                    'is_new': current_result['is_new'],
                    'match_score': current_result['match_score'],
                    'cosine_similarity': current_result['cosine_similarity'],
                    'euclidean_distance': current_result['euclidean_distance']
                }
            
            # Person List
            person_list = []
            for pid, info in sorted(self.person_registry.registry.items()):
                person_list.append({
                    'id': pid,
                    'count': info['count'],
                    'embeddings': len(info.get('embeddings_list', []))
                })
            self.web_data['person_list'] = person_list
            
            # 상태 정보
            self.web_data['state_info'] = {
                'current_state': self.tracking_state,
                'buffer_size': len(self.skeleton_buffer),
                'frames_since_lost': self.frames_since_lost,
                'session_locked': self.session_person_locked
            }
    
    def rtsp_receiver_thread(self):
        """RTSP 수신 + 처리 스레드"""
        print("📡 [Receiver] Thread started")
        print(f"📡 [Receiver] Connecting to {self.rtsp_url}")
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        
        if not self.cap.isOpened():
            print("❌ [Receiver] Failed to connect to RTSP stream")
            return
        
        print("✅ [Receiver] Connected successfully\n")
        
        frame_idx = 0
        current_result = None
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️ [Receiver] No frame received")
                time.sleep(0.1)
                continue
            
            with self.web_data_lock:
                self.web_data['stats']['received_frames'] += 1
                self.web_data['stats']['processed_frames'] += 1
            
            self.update_fps()
            
            # YOLOv8-Pose 디텍션
            persons = self.detect_persons_yolo_pose(frame)
            best_person = self.select_best_person(persons, frame.shape)
            
            if persons:
                with self.web_data_lock:
                    self.web_data['stats']['yolo_detections'] += len(persons)
            
            # 상태 업데이트
            self.update_tracking_state(best_person, frame_idx)
            
            # 스켈레톤 처리
            skeleton = None
            if best_person and 'skeleton' in best_person:
                skeleton = best_person['skeleton']
                
                if skeleton is not None:
                    with self.web_data_lock:
                        self.web_data['stats']['detected_frames'] += 1
                        self.web_data['stats']['skeleton_extractions'] += 1
            
            # 버퍼 관리
            if self.tracking_state == "COLLECTING" and skeleton is not None:
                self.skeleton_buffer.append(skeleton)
                self.frames_since_last_inference += 1
                
                buffer_size = len(self.skeleton_buffer)
                
                # 첫 추론
                if not self.has_initial_inference and buffer_size >= self.initial_buffer_size:
                    current_result = self.process_inference(frame_idx, is_final=False)
                    if current_result:
                        self.update_web_ui(current_result)
                
                # 주기적 추론
                elif (self.has_initial_inference and 
                      self.frames_since_last_inference >= self.accumulation_interval):
                    current_result = self.process_inference(frame_idx, is_final=False)
                    if current_result:
                        self.update_web_ui(current_result)
            
            # LOST 최종 처리
            if (self.tracking_state == "LOST" and 
                self.frames_since_lost >= self.lost_delay_frames and
                len(self.skeleton_buffer) > 0):
                current_result = self.process_inference(frame_idx, is_final=True)
                if current_result:
                    self.update_web_ui(current_result)
            
            # 시각화
            display_frame = self.draw_visualization(frame, skeleton, best_person, current_result)
            
            # JPEG 인코딩 (웹 스트리밍용)
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # 프레임 큐에 추가
            try:
                self.frame_queue.put(buffer.tobytes(), block=False)
            except:
                pass  # 큐가 꽉 차면 스킵
            
            # display_frame도 락으로 보호
            with self.display_lock:
                self.display_frame = display_frame
            
            frame_idx += 1
        
        # 종료
        if self.cap:
            self.cap.release()
        
        print("\n📡 [Receiver] Thread stopped")
    
    def start(self):
        """시스템 시작"""
        print("\n🚀 Starting RTSP Web Gait Tracker...")
        self.running = True
        
        # RTSP 수신 스레드
        receiver = threading.Thread(target=self.rtsp_receiver_thread, daemon=True)
        receiver.start()
        
        print("✅ System started\n")
    
    def stop(self):
        """시스템 종료"""
        print("\n🛑 Stopping system...")
        self.running = False
        time.sleep(1)
        
        # Person DB 저장
        self.save_person_db()
        
        # 로그 저장
        self.save_logs()
        
        print("✅ System stopped\n")
    
    def save_logs(self):
        """로그 저장"""
        try:
            log_data = {
                'event_log': self.event_log,
                'matching_log': self.matching_log,
                'person_registry': {}
            }
            
            for pid, info in self.person_registry.registry.items():
                log_data['person_registry'][str(pid)] = {
                    'count': info['count'],
                    'embeddings': len(info.get('embeddings_list', []))
                }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Logs saved: {os.path.basename(self.log_file)}")
        except Exception as e:
            print(f"⚠️ Save logs failed: {e}")
    
    def generate_frames(self):
        """프레임 생성기 (Flask 스트리밍용)"""
        while True:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Empty:
                # 타임아웃 시 마지막 프레임 반복
                with self.display_lock:
                    if self.display_frame is not None:
                        _, buffer = cv2.imencode('.jpg', self.display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# === Flask 앱 ===
tracker = None

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('rtsp_gait_tracker.html')

@app.route('/video_feed')
def video_feed():
    """비디오 스트림"""
    if tracker:
        return Response(tracker.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response("", mimetype='text/plain')

@app.route('/api/stats')
def api_stats():
    """통계 API"""
    if tracker:
        with tracker.web_data_lock:
            return jsonify(tracker.web_data)
    return jsonify({})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Registry 리셋"""
    if tracker:
        # 백업 생성
        import pickle
        import shutil
        from datetime import datetime
        
        backup_path = tracker.person_db_path.replace('.pkl', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        
        if os.path.exists(tracker.person_db_path):
            try:
                shutil.copy2(tracker.person_db_path, backup_path)
                print(f"💾 Backup created: {os.path.basename(backup_path)}")
            except Exception as e:
                print(f"⚠️ Backup failed: {e}")
        
        person_count = len(tracker.person_registry.registry)
        print(f"⚠️ Registry reset requested - {person_count} persons will be deleted")
        
        tracker.person_registry.registry.clear()
        tracker.person_registry.next_id = 1
        tracker.save_person_db()
        
        print("🔄 Registry reset complete")
        return jsonify({'status': 'ok', 'backup': backup_path})
    return jsonify({'status': 'error'})


def main():
    global tracker
    
    parser = argparse.ArgumentParser(description='RTSP Web Gait Tracker')
    parser.add_argument('--rtsp', type=str, required=True,
                       help='RTSP URL (e.g., rtsp://172.30.85.166:8554/stream)')
    parser.add_argument('--enable-model', action='store_true',
                       help='Enable GPGait++ model')
    parser.add_argument('--config', type=str,
                       default='configs/gpgait++/gpgait++.yaml',
                       help='Model config')
    parser.add_argument('--checkpoint', type=str,
                       default=r'C:\FastGait\FastPoseGait\fastposegait\output\CASIA-B\GPGait_PlusPlus\GPGait_PlusPlus\checkpoints\GPGait_PlusPlus-40000.pt',
                       help='Checkpoint path')
    parser.add_argument('--yolo-model', type=str, default='yolov8n-pose.pt',
                       help='YOLOv8-Pose model')
    parser.add_argument('--output-dir', type=str, default='rtsp_output',
                       help='Output directory')
    parser.add_argument('--port', type=int, default=5000,
                       help='Flask port (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Flask host (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    # Config path
    if args.enable_model and not os.path.isabs(args.config):
        args.config = os.path.join(PROJECT_ROOT, args.config)
    
    # Tracker 생성
    tracker = RTSPWebGaitTracker(
        rtsp_url=args.rtsp,
        config_path=args.config if args.enable_model else None,
        checkpoint_path=args.checkpoint if args.enable_model else None,
        enable_model=args.enable_model,
        yolo_model=args.yolo_model,
        output_dir=args.output_dir
    )
    
    # 시작
    tracker.start()
    
    print("="*80)
    print("🌐 Flask Web Server Starting...")
    print(f"   URL: http://{args.host}:{args.port}")
    print("="*80 + "\n")
    
    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
