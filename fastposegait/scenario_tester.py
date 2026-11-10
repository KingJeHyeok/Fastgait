# scenario_tester.py - 시나리오 영상 테스트 및 시각화
"""
녹화된 영상으로 보행 인식 시스템 테스트
- YOLOv8-Pose 기반 사람 디텍션 + 스켈레톤 추출
- GPGait++ 모델 추론
- 실시간 시각화 및 로그 저장
"""
import cv2
import numpy as np
import argparse
import os
import sys
import time
import json
from datetime import datetime
import torch
import torch.nn.functional as F

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available")

# Project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from realtime_skeleton_gait import GaitModelLoader
from improved_person_registry import ImprovedPersonRegistry


class ScenarioTester:
    """시나리오 테스트 시스템"""
    
    def __init__(self, video_path, config_path=None, checkpoint_path=None,
                 enable_model=True, yolo_model='yolov8n.pt', output_dir='scenario_output',
                 show_display=True, rotate=0, display_scale=0.5, fast_mode=False):
        self.video_path = video_path
        self.output_dir = output_dir
        self.show_display = show_display
        self.rotate = rotate  # 0, 90, 180, 270
        self.display_scale = display_scale  # 화면 표시 배율 (0.1 ~ 1.0)
        self.fast_mode = fast_mode  # 빠른 모드 (UI 업데이트 간격 증가)
        os.makedirs(output_dir, exist_ok=True)
        
        # 영상 정보
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"❌ Cannot open video: {video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 회전 시 width/height 교체
        if self.rotate in [90, 270]:
            self.width, self.height = self.height, self.width
        
        print(f"📹 Video: {self.video_name}")
        print(f"   Resolution: {self.width}x{self.height}")
        if self.rotate > 0:
            print(f"   Rotation: {self.rotate}°")
        print(f"   FPS: {self.fps}")
        print(f"   Total Frames: {self.total_frames}")
        print(f"   Duration: {self.total_frames/self.fps:.1f}s\n")
        
        # YOLOv8-Pose 초기화
        self.yolo_model = None
        self.yolo_enabled = False
        if YOLO_AVAILABLE:
            try:
                print(f"🔄 Loading YOLOv8-Pose: {yolo_model}")
                self.yolo_model = YOLO(yolo_model)
                self.yolo_enabled = True
                print("✅ YOLOv8-Pose loaded\n")
            except Exception as e:
                print(f"❌ YOLOv8-Pose failed: {e}\n")
        
        # 추적 상태
        self.tracking_state = "IDLE"
        self.detection_confidence_threshold = 0.75
        self.lost_delay_seconds = 1.0
        self.frames_since_lost = 0
        self.lost_delay_frames = int(self.lost_delay_seconds * self.fps)
        
        # 버퍼 시스템
        self.initial_buffer_size = 300  # 120 → 300 (충분한 gait signature 확보)
        self.accumulation_interval = 90  # 30 → 90 (주기적 추론 간격 증가)
        self.max_buffer_size = 400  # 300 → 400 (초기 버퍼 증가에 맞춤)
        self.skeleton_buffer = []
        self.frames_since_last_inference = 0
        self.has_initial_inference = False
        
        # Person ID 관리
        self.current_person_id = None
        self.session_person_locked = False
        self.tracking_confidence = 0.0
        
        # GPGait++ 모델
        self.enable_model = enable_model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if enable_model and config_path and checkpoint_path:
            print(f"🔄 Loading GPGait++ model...")
            self.model = GaitModelLoader.load_model(config_path, checkpoint_path, str(self.device))
            print(f"✅ Model loaded on {self.device}\n")
        
        # Person Registry (DB 저장 경로) - ImprovedPersonRegistry 사용
        self.person_db_path = os.path.join(output_dir, 'person_database.pkl')
        self.person_registry = ImprovedPersonRegistry(
            match_threshold_cos_strict=0.92,   # 엄격 (STRICT 모드)
            match_threshold_euc_strict=0.35,
            match_threshold_cos_relaxed=0.88,  # 완화 (RELAXED 모드, 사용 안함)
            match_threshold_euc_relaxed=0.45,
            max_embeddings=10,
            ema_alpha=0.3
        )
        
        # 기존 DB 로드
        if os.path.exists(self.person_db_path):
            self.load_person_db()
            print(f"📂 Loaded existing Person DB: {len(self.person_registry.registry)} persons\n")
        else:
            print(f"📂 No existing Person DB found - Starting fresh\n")
        
        # 로깅
        self.event_log = []
        self.frame_results = []
        self.detection_log = []  # YOLO 디텍션 상세 로그
        self.inference_log = []  # 추론 상세 로그
        self.matching_log = []   # 매칭 상세 로그
        
        # 통계
        self.stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'inference_count': 0,
            'persons_detected': 0,
            'yolo_detections': 0,
            'skeleton_extractions': 0,
            'state_transitions': {}
        }
        
        # 출력 비디오
        output_video_path = os.path.join(output_dir, f"{self.video_name}_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))
        
        print(f"💾 Output: {output_video_path}\n")
    
    def load_person_db(self):
        """Person DB 로드"""
        import pickle
        try:
            with open(self.person_db_path, 'rb') as f:
                data = pickle.load(f)
                self.person_registry.registry = data['registry']
                self.person_registry.next_id = data['next_id']
                print(f"✅ Loaded {len(self.person_registry.registry)} persons from DB")
                for pid, info in self.person_registry.registry.items():
                    print(f"   Person #{pid}: {len(info.get('embeddings', []))} embeddings")
        except Exception as e:
            print(f"⚠️ Failed to load Person DB: {e}")
    
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
            print(f"⚠️ Failed to save Person DB: {e}")
    
    def rotate_frame(self, frame):
        """프레임 회전"""
        if self.rotate == 0:
            return frame
        elif self.rotate == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def detect_persons_yolo_pose(self, frame, frame_idx):
        """YOLOv8-Pose 사람 디텍션 + 스켈레톤 추출 (로깅 포함)"""
        if not self.yolo_enabled or self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, verbose=False)
            
            h, w = frame.shape[:2]
            persons = []
            all_detections = []  # 필터링 전 모든 디텍션
            
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints  # YOLO-Pose keypoints
                
                if boxes is not None and len(boxes) > 0:
                    for idx, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 0
                        
                        # Keypoints 추출 (COCO 17 keypoints)
                        skeleton = None
                        if keypoints is not None and len(keypoints) > idx:
                            kpts = keypoints[idx].xy.cpu().numpy()[0]  # (17, 2)
                            if len(kpts) == 17:
                                skeleton = kpts.astype(np.float32)
                        
                        detection_info = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'width': float(bbox_w),
                            'height': float(bbox_h),
                            'aspect_ratio': float(aspect_ratio),
                            'has_skeleton': skeleton is not None
                        }
                        all_detections.append(detection_info)
                        
                        # 필터링
                        passed_conf = conf > 0.6
                        passed_size = bbox_w > 30 and bbox_h > 30
                        passed_skeleton = skeleton is not None
                        
                        detection_info['passed_filters'] = {
                            'confidence': passed_conf,
                            'size': passed_size,
                            'skeleton': passed_skeleton,
                            'overall': passed_conf and passed_size and passed_skeleton
                        }
                        
                        if passed_conf and passed_size and passed_skeleton:
                            persons.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'skeleton': skeleton
                            })
                            self.stats['yolo_detections'] += 1
            
            # 로깅
            self.detection_log.append({
                'frame': frame_idx,
                'total_detections': len(all_detections),
                'filtered_detections': len(persons),
                'detections': all_detections
            })
            
            return persons
        except Exception as e:
            print(f"❌ YOLOv8-Pose error: {e}")
            return []
    
    def select_best_person(self, persons):
        """가장 좋은 사람 선택"""
        if not persons:
            return None
        
        center_x, center_y = self.width // 2, self.height // 2
        best_person = None
        best_score = -1
        
        for person in persons:
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            area = bbox_w * bbox_h
            
            dist = np.sqrt((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)
            
            size_score = min(area / (self.width * self.height), 0.5) * 2
            dist_score = 1 - min(dist / (self.width/2), 1)
            conf_score = person['confidence']
            
            score = size_score * 0.4 + dist_score * 0.2 + conf_score * 0.4
            
            if score > best_score:
                best_score = score
                best_person = person
        
        return best_person
    
    def update_tracking_state(self, best_person, frame_idx):
        """상태 머신 업데이트 (로깅 포함)"""
        current_state = self.tracking_state
        new_state = current_state
        
        if best_person and best_person['confidence'] >= self.detection_confidence_threshold:
            if current_state == "IDLE":
                self.log_event(frame_idx, "DETECTING", f"Person detected (conf: {best_person['confidence']:.2f})")
                new_state = "DETECTING"
                self.tracking_state = "DETECTING"
                self.skeleton_buffer.clear()
                self.has_initial_inference = False
                self.frames_since_last_inference = 0
                self.current_person_id = None
                self.session_person_locked = False
            elif current_state == "DETECTING":
                new_state = "COLLECTING"
                self.tracking_state = "COLLECTING"
                self.log_event(frame_idx, "COLLECTING", f"Started collecting (target: {self.initial_buffer_size} frames)")
            elif current_state == "LOST":
                if self.frames_since_lost < self.lost_delay_frames:
                    self.log_event(frame_idx, "COLLECTING", f"Re-detected ({self.frames_since_lost} frames lost)")
                    new_state = "COLLECTING"
                    self.tracking_state = "COLLECTING"
                    self.frames_since_lost = 0
                else:
                    self.log_event(frame_idx, "DETECTING", "New person after delay")
                    new_state = "DETECTING"
                    self.tracking_state = "DETECTING"
                    self.skeleton_buffer.clear()
                    self.has_initial_inference = False
                    self.frames_since_last_inference = 0
                    self.current_person_id = None
                    self.session_person_locked = False
        else:
            if current_state == "COLLECTING":
                self.log_event(frame_idx, "LOST", f"Person lost (delay: {self.lost_delay_seconds}s)")
                new_state = "LOST"
                self.tracking_state = "LOST"
                self.frames_since_lost = 0
            elif current_state == "LOST":
                self.frames_since_lost += 1
                
                if self.frames_since_lost >= self.lost_delay_frames:
                    if len(self.skeleton_buffer) > 0:
                        self.log_event(frame_idx, "FINALIZING", f"Delay complete ({len(self.skeleton_buffer)} frames)")
                    else:
                        self.log_event(frame_idx, "IDLE", "Empty buffer, returning to IDLE")
                        new_state = "IDLE"
                        self.tracking_state = "IDLE"
        
        # 상태 전환 통계
        if new_state != current_state:
            transition = f"{current_state} → {new_state}"
            self.stats['state_transitions'][transition] = self.stats['state_transitions'].get(transition, 0) + 1
    
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
        
        skeleton_sequence = np.array(self.skeleton_buffer)
        
        if buffer_size > self.max_buffer_size:
            indices = np.linspace(0, buffer_size-1, self.max_buffer_size, dtype=int)
            skeleton_sequence = skeleton_sequence[indices]
        
        result = self.run_gait_model(skeleton_sequence, frame_idx)
        
        if result:
            inference_type = "Final" if is_final else "Periodic"
            
            if not self.has_initial_inference:
                self.current_person_id = result['person_id']
                self.has_initial_inference = True
                self.tracking_confidence = result['match_score']
                
                if result['match_score'] > 0.85:
                    self.session_person_locked = True
                    lock_status = "LOCKED"
                else:
                    self.session_person_locked = False
                    lock_status = "UNLOCKED"
                
                status = "NEW" if result['is_new'] else "MATCHED"
                self.log_event(frame_idx, f"INFERENCE_{inference_type.upper()}", 
                             f"Person #{result['person_id']} ({status}, score: {result['match_score']:.3f}, {lock_status})")
                
                if result['is_new']:
                    self.stats['persons_detected'] += 1
            else:
                if self.session_person_locked:
                    if result['match_score'] < 0.70:
                        self.session_person_locked = False
                        self.log_event(frame_idx, "SESSION_UNLOCK", f"Low confidence ({result['match_score']:.3f})")
                    else:
                        if result['person_id'] != self.current_person_id:
                            result['person_id'] = self.current_person_id
                            result['is_new'] = False
                
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
                             f"Person #{result['person_id']} (score: {result['match_score']:.3f})")
            
            self.stats['inference_count'] += 1
        
        if is_final:
            self.reset_tracking_session(frame_idx)
        else:
            self.frames_since_last_inference = 0
        
        return result
    
    def run_gait_model(self, skeleton_sequence, frame_idx):
        """GPGait++ 추론 (로깅 포함)"""
        if self.model is None:
            return None
        
        try:
            data = GaitModelLoader.preprocess_skeleton_sequence(skeleton_sequence)
            # GPU로 명시적 이동 + contiguous
            data = data.to(self.device, non_blocking=True).contiguous()
            
            with torch.no_grad():
                T = data.shape[2]
                # seqL도 GPU로 명시적 이동
                seqL_tensor = torch.LongTensor([T]).to(self.device, non_blocking=True)
                seqL = [seqL_tensor]
                inputs = [[data], None, None, None, seqL]
                outputs = self.model.forward(inputs)
                
                feat = outputs['inference_feat']['embeddings']
                if feat.dim() == 3:
                    feat = feat.mean(dim=2)
                feat = F.normalize(feat, p=2, dim=1)
            
            # 매칭 전 상태 로깅
            matching_info = {
                'frame': frame_idx,
                'buffer_size': len(skeleton_sequence),
                'session_locked': self.session_person_locked,
                'current_person_id': self.current_person_id,
                'mode': 'force_update' if (self.session_person_locked and self.current_person_id) else 'match_or_register'
            }
            
            if self.session_person_locked and self.current_person_id is not None:
                person_id, is_new, match_score, cosine_sim, euclid_dist = self.person_registry.force_update_person(
                    self.current_person_id, feat
                )
                matching_info['method'] = 'force_update'
            else:
                # 매칭 전 DB 상태 기록
                matching_info['registry_size'] = len(self.person_registry.registry)
                matching_info['existing_persons'] = list(self.person_registry.registry.keys())
                
                # ImprovedPersonRegistry: frame_count 전달
                person_id, is_new, match_score, cosine_sim, euclid_dist = self.person_registry.match_or_register(
                    feat,
                    frame_count=len(skeleton_sequence)
                )
                matching_info['method'] = 'match_or_register'
            
            # 매칭 결과 로깅 (cosine/euclidean 추가)
            matching_info.update({
                'result_person_id': person_id,
                'is_new': is_new,
                'match_score': match_score,
                'cosine_similarity': float(cosine_sim),
                'euclidean_distance': float(euclid_dist)
            })
            self.matching_log.append(matching_info)
            
            # ImprovedPersonRegistry에는 check_stalking이 없으므로 기본값
            is_stalking, stalk_count, stalk_duration = False, 0, 0.0
            
            # 추론 결과 로깅 (cosine/euclidean 추가)
            self.inference_log.append({
                'frame': frame_idx,
                'person_id': person_id,
                'is_new': is_new,
                'match_score': match_score,
                'cosine_similarity': float(cosine_sim),
                'euclidean_distance': float(euclid_dist),
                'buffer_size': len(skeleton_sequence),
                'session_locked': self.session_person_locked,
                'is_stalking': is_stalking
            })
            
            return {
                'person_id': person_id,
                'is_new': is_new,
                'match_score': match_score,
                'cosine_similarity': float(cosine_sim),
                'euclidean_distance': float(euclid_dist),
                'is_stalking': is_stalking,
                'stalk_count': stalk_count,
                'stalk_duration': stalk_duration
            }
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            import traceback
            traceback.print_exc()
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
    
    def log_event(self, frame_idx, event_type, message):
        """이벤트 로깅"""
        timestamp = frame_idx / self.fps
        event = {
            'frame': frame_idx,
            'time': f"{timestamp:.2f}s",
            'type': event_type,
            'message': message
        }
        self.event_log.append(event)
        print(f"[Frame {frame_idx:4d} | {timestamp:6.2f}s] {event_type:20s} | {message}")
    
    def draw_visualization(self, frame, skeleton, best_person, current_result):
        """시각화"""
        display_frame = frame.copy()
        
        # YOLO 바운딩 박스
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
                    cv2.circle(display_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.circle(display_frame, (int(x), int(y)), 7, (255, 255, 255), 2)
        
        # 상태 정보
        state_color = {
            "IDLE": (128, 128, 128),
            "DETECTING": (0, 255, 255),
            "COLLECTING": (0, 255, 0),
            "LOST": (0, 165, 255)
        }.get(self.tracking_state, (255, 255, 255))
        
        # 반투명 배경 (상단 정보 - 더 크게)
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (5, 5), (650, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # 상단 정보
        info_text = f"State: {self.tracking_state} | Buffer: {len(self.skeleton_buffer)}"
        if self.current_person_id:
            lock_status = "LOCKED" if self.session_person_locked else "UNLOCKED"
            info_text += f" | Person #{self.current_person_id} ({lock_status})"
        
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # 현재 결과 (크게 표시)
        if current_result:
            status = "NEW" if current_result['is_new'] else "MATCHED"
            status_color = (0, 255, 255) if current_result['is_new'] else (0, 255, 0)
            
            result_text = f"Person #{current_result['person_id']} ({status})"
            cv2.putText(display_frame, result_text, (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            
            # Combined Score
            score_text = f"Score: {current_result['match_score']:.3f}"
            cv2.putText(display_frame, score_text, (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Cosine Similarity
            cosine_text = f"Cosine: {current_result['cosine_similarity']:.4f}"
            cv2.putText(display_frame, cosine_text, (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            # Euclidean Distance
            euclid_text = f"Euclid: {current_result['euclidean_distance']:.4f}"
            cv2.putText(display_frame, euclid_text, (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        
        # Person Registry 정보 (우측 상단)
        if len(self.person_registry.registry) > 0:
            overlay2 = display_frame.copy()
            box_height = 35 + len(self.person_registry.registry) * 30
            cv2.rectangle(overlay2, (display_frame.shape[1]-200, 5), 
                         (display_frame.shape[1]-5, box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.6, display_frame, 0.4, 0, display_frame)
            
            cv2.putText(display_frame, "Person DB:", 
                       (display_frame.shape[1]-190, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset = 50
            for pid in sorted(self.person_registry.registry.keys()):
                info = self.person_registry.registry[pid]
                text_color = (0, 255, 0) if pid == self.current_person_id else (200, 200, 200)
                person_text = f"#{pid}: {info['count']} detections"
                cv2.putText(display_frame, person_text, 
                           (display_frame.shape[1]-190, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                y_offset += 30
        
        return display_frame
    
    def run(self):
        """테스트 실행"""
        print("="*80)
        print(f"🚀 Starting Scenario Test: {self.video_name}")
        print("="*80 + "\n")
        
        frame_idx = 0
        current_result = None
        debug_counter = 0  # 디버깅용 카운터
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 프레임 회전
            frame = self.rotate_frame(frame)
            
            self.stats['total_frames'] += 1
            
            # YOLOv8-Pose 디텍션 + 스켈레톤 추출
            persons = self.detect_persons_yolo_pose(frame, frame_idx)
            best_person = self.select_best_person(persons)
            
            # 디버깅: 처음 30프레임 상세 로그
            debug_counter += 1
            if debug_counter <= 30 or frame_idx % 30 == 0:
                print(f"[Frame {frame_idx:4d}] YOLO: {len(persons)} persons, Best: {'Yes' if best_person else 'No'}, State: {self.tracking_state}, Buffer: {len(self.skeleton_buffer)}")
            
            # 상태 업데이트
            self.update_tracking_state(best_person, frame_idx)
            
            # 스켈레톤 가져오기 (이미 YOLO-Pose에서 추출됨)
            skeleton = None
            if best_person and 'skeleton' in best_person:
                skeleton = best_person['skeleton']
                
                if skeleton is not None:
                    self.stats['detected_frames'] += 1
                    self.stats['skeleton_extractions'] += 1
                    
                    # 디버깅: 스켈레톤 추출 성공 로그
                    if debug_counter <= 30:
                        bbox = best_person['bbox']
                        x1, y1, x2, y2 = bbox
                        print(f"  ✅ Skeleton extracted from bbox [{x1}, {y1}, {x2}, {y2}]")
                else:
                    if debug_counter <= 30:
                        print(f"  ❌ Skeleton extraction failed")
            
            # 버퍼 관리
            if self.tracking_state == "COLLECTING" and skeleton is not None:
                self.skeleton_buffer.append(skeleton)
                self.frames_since_last_inference += 1
                
                buffer_size = len(self.skeleton_buffer)
                
                # 첫 추론
                if not self.has_initial_inference and buffer_size >= self.initial_buffer_size:
                    current_result = self.process_inference(frame_idx, is_final=False)
                
                # 주기적 추론
                elif (self.has_initial_inference and 
                      self.frames_since_last_inference >= self.accumulation_interval):
                    current_result = self.process_inference(frame_idx, is_final=False)
            
            # LOST 상태 최종 처리
            if (self.tracking_state == "LOST" and 
                self.frames_since_lost >= self.lost_delay_frames and
                len(self.skeleton_buffer) > 0):
                current_result = self.process_inference(frame_idx, is_final=True)
            
            # 시각화
            display_frame = self.draw_visualization(frame, skeleton, best_person, current_result)
            
            # 비디오 저장
            self.video_writer.write(display_frame)
            
            # 실시간 미리보기 (옵션)
            if self.show_display:
                # Fast 모드: 5프레임마다만 UI 업데이트
                update_interval = 5 if self.fast_mode else 1
                
                if frame_idx % update_interval == 0:
                    # 화면 크기 조절
                    if self.display_scale != 1.0:
                        display_w = int(display_frame.shape[1] * self.display_scale)
                        display_h = int(display_frame.shape[0] * self.display_scale)
                        display_resized = cv2.resize(display_frame, (display_w, display_h))
                    else:
                        display_resized = display_frame
                    
                    cv2.imshow(f'Scenario Test: {self.video_name}', display_resized)
                    
                    # ESC 키로 종료
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print("\n⚠️ User interrupted")
                        break
                    elif key == 32:  # SPACE
                        print("\n⏸️  Paused (Press any key to continue)")
                        cv2.waitKey(0)
            
            # 프레임 결과 저장
            self.frame_results.append({
                'frame': frame_idx,
                'state': self.tracking_state,
                'person_detected': best_person is not None,
                'skeleton_detected': skeleton is not None,
                'buffer_size': len(self.skeleton_buffer),
                'current_person_id': self.current_person_id,
                'session_locked': self.session_person_locked
            })
            
            frame_idx += 1
            
            # 진행률 표시 (매 60프레임, 디버그 로그와 겹치지 않게)
            if frame_idx > 30 and frame_idx % 60 == 0:
                progress = (frame_idx / self.total_frames) * 100
                print(f"\n📊 Progress: {progress:.1f}% ({frame_idx}/{self.total_frames}) | State: {self.tracking_state} | Buffer: {len(self.skeleton_buffer)}\n")
        
        # 종료
        self.cap.release()
        self.video_writer.release()
        
        if self.show_display:
            cv2.destroyAllWindows()
        
        # Person DB 저장
        self.save_person_db()
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """결과 저장 (상세 로그 포함)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 이벤트 로그
        log_path = os.path.join(self.output_dir, f"{self.video_name}_events.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.event_log, f, indent=2, ensure_ascii=False)
        
        # 프레임 결과
        frame_path = os.path.join(self.output_dir, f"{self.video_name}_frames.json")
        with open(frame_path, 'w', encoding='utf-8') as f:
            json.dump(self.frame_results, f, indent=2)
        
        # YOLO 디텍션 로그
        detection_path = os.path.join(self.output_dir, f"{self.video_name}_detections.json")
        with open(detection_path, 'w', encoding='utf-8') as f:
            json.dump(self.detection_log, f, indent=2)
        
        # 추론 로그
        inference_path = os.path.join(self.output_dir, f"{self.video_name}_inferences.json")
        with open(inference_path, 'w', encoding='utf-8') as f:
            json.dump(self.inference_log, f, indent=2)
        
        # 매칭 로그
        matching_path = os.path.join(self.output_dir, f"{self.video_name}_matching.json")
        with open(matching_path, 'w', encoding='utf-8') as f:
            json.dump(self.matching_log, f, indent=2)
        
        # Person Registry
        registry_path = os.path.join(self.output_dir, f"{self.video_name}_registry.json")
        registry_data = {}
        for pid, info in self.person_registry.registry.items():
            registry_data[str(pid)] = {
                'count': info['count'],
                'first_seen': info['first_seen'],
                'last_seen': info['last_seen'],
                'num_embeddings': len(info.get('embeddings', []))
            }
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=2)
        
        # 통합 분석 리포트
        report_path = os.path.join(self.output_dir, f"{self.video_name}_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"SCENARIO TEST REPORT: {self.video_name}\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            f.write("## BASIC STATISTICS\n")
            f.write(f"Total Frames: {self.stats['total_frames']}\n")
            f.write(f"YOLO Detections: {self.stats['yolo_detections']}\n")
            f.write(f"Skeleton Extractions: {self.stats['skeleton_extractions']}\n")
            f.write(f"Inference Count: {self.stats['inference_count']}\n")
            f.write(f"Persons Detected: {self.stats['persons_detected']}\n\n")
            
            f.write("## STATE TRANSITIONS\n")
            for transition, count in sorted(self.stats['state_transitions'].items()):
                f.write(f"{transition}: {count}\n")
            f.write("\n")
            
            f.write("## PERSON REGISTRY\n")
            for pid, info in sorted(self.person_registry.registry.items()):
                f.write(f"Person #{pid}:\n")
                f.write(f"  Detections: {info['count']}\n")
                f.write(f"  Embeddings: {len(info.get('embeddings', []))}\n")
                f.write(f"  First Seen: {info['first_seen']:.2f}s\n")
                f.write(f"  Last Seen: {info['last_seen']:.2f}s\n")
                f.write(f"  Duration: {info['last_seen'] - info['first_seen']:.2f}s\n\n")
            
            f.write("## KEY EVENTS TIMELINE\n")
            for event in self.event_log:
                f.write(f"[{event['time']:>8s}] {event['type']:20s} | {event['message']}\n")
            f.write("\n")
            
            f.write("## INFERENCE DETAILS\n")
            for inf in self.inference_log:
                status = "NEW" if inf['is_new'] else "MATCHED"
                locked = "LOCKED" if inf['session_locked'] else "UNLOCKED"
                f.write(f"Frame {inf['frame']:4d} | Person #{inf['person_id']} ({status}) | Score: {inf['match_score']:.3f} | {locked} | Buffer: {inf['buffer_size']}\n")
            f.write("\n")
            
            f.write("## POTENTIAL ISSUES\n")
            # 문제점 분석
            issues = []
            
            # 1. 동일 인물이 여러 ID로 등록되었는지 체크
            if len(self.person_registry.registry) > self.stats['persons_detected']:
                issues.append(f"⚠️ Registry has {len(self.person_registry.registry)} persons but {self.stats['persons_detected']} were marked as NEW")
            
            # 2. 낮은 스켈레톤 추출률
            extraction_rate = self.stats['skeleton_extractions'] / self.stats['yolo_detections'] * 100 if self.stats['yolo_detections'] > 0 else 0
            if extraction_rate < 50:
                issues.append(f"⚠️ Low skeleton extraction rate: {extraction_rate:.1f}%")
            
            # 3. 잦은 ID 변경
            id_changes = [e for e in self.event_log if e['type'] == 'ID_CHANGE']
            if len(id_changes) > 5:
                issues.append(f"⚠️ Frequent ID changes: {len(id_changes)} times")
            
            # 4. 낮은 매칭 스코어
            low_score_inferences = [inf for inf in self.inference_log if inf['match_score'] < 0.7]
            if len(low_score_inferences) > len(self.inference_log) * 0.3:
                issues.append(f"⚠️ Many low-confidence matches: {len(low_score_inferences)}/{len(self.inference_log)}")
            
            if issues:
                for issue in issues:
                    f.write(issue + "\n")
            else:
                f.write("✅ No major issues detected\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\n💾 Results saved to: {self.output_dir}/")
        print(f"   - Events: {os.path.basename(log_path)}")
        print(f"   - Frames: {os.path.basename(frame_path)}")
        print(f"   - Detections: {os.path.basename(detection_path)}")
        print(f"   - Inferences: {os.path.basename(inference_path)}")
        print(f"   - Matching: {os.path.basename(matching_path)}")
        print(f"   - Registry: {os.path.basename(registry_path)}")
        print(f"   - Report: {os.path.basename(report_path)}")
    
    def print_summary(self):
        """요약 출력 (상세)"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"Video: {self.video_name}")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"YOLO Detections: {self.stats['yolo_detections']}")
        print(f"Skeleton Extractions: {self.stats['skeleton_extractions']}")
        
        if self.stats['yolo_detections'] > 0:
            extraction_rate = self.stats['skeleton_extractions'] / self.stats['yolo_detections'] * 100
            print(f"Extraction Rate: {extraction_rate:.1f}%")
        
        print(f"Inference Count: {self.stats['inference_count']}")
        print(f"Persons Detected (NEW): {self.stats['persons_detected']}")
        
        print(f"\n📈 State Transitions:")
        for transition, count in sorted(self.stats['state_transitions'].items()):
            print(f"  {transition}: {count}")
        
        print(f"\n👥 Person Registry ({len(self.person_registry.registry)} persons):")
        for pid, info in sorted(self.person_registry.registry.items()):
            duration = info['last_seen'] - info['first_seen']
            print(f"  Person #{pid}: {info['count']} detections, {len(info.get('embeddings', []))} embeddings, {duration:.1f}s duration")
        
        # 문제점 요약
        print(f"\n⚠️  Potential Issues:")
        issues = []
        
        if len(self.person_registry.registry) > self.stats['persons_detected']:
            issues.append(f"  - Registry mismatch: {len(self.person_registry.registry)} IDs vs {self.stats['persons_detected']} NEW detections")
        
        extraction_rate = self.stats['skeleton_extractions'] / self.stats['yolo_detections'] * 100 if self.stats['yolo_detections'] > 0 else 0
        if extraction_rate < 50:
            issues.append(f"  - Low skeleton extraction rate: {extraction_rate:.1f}%")
        
        id_changes = [e for e in self.event_log if e['type'] == 'ID_CHANGE']
        if len(id_changes) > 5:
            issues.append(f"  - Frequent ID changes: {len(id_changes)} times")
        
        low_score_inferences = [inf for inf in self.inference_log if inf['match_score'] < 0.7]
        if len(self.inference_log) > 0 and len(low_score_inferences) > len(self.inference_log) * 0.3:
            issues.append(f"  - Low-confidence matches: {len(low_score_inferences)}/{len(self.inference_log)} ({len(low_score_inferences)/len(self.inference_log)*100:.1f}%)")
        
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("  ✅ No major issues detected")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Scenario Tester with Real-time Visualization')
    parser.add_argument('--video', type=str, required=True,
                       help='Video file path')
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
    parser.add_argument('--output-dir', type=str, default='scenario_output',
                       help='Output directory')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display (faster processing)')
    parser.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270],
                       help='Rotate video (0, 90, 180, 270 degrees)')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Display scale (0.1-1.0, default: 0.5)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: Skip UI updates for speed (update every 5 frames)')
    
    args = parser.parse_args()
    
    # Config path
    if args.enable_model and not os.path.isabs(args.config):
        args.config = os.path.join(PROJECT_ROOT, args.config)
    
    # Tester 생성 및 실행
    tester = ScenarioTester(
        video_path=args.video,
        config_path=args.config if args.enable_model else None,
        checkpoint_path=args.checkpoint if args.enable_model else None,
        enable_model=args.enable_model,
        yolo_model=args.yolo_model,
        output_dir=args.output_dir,
        show_display=not args.no_display,
        rotate=args.rotate,
        display_scale=args.scale,
        fast_mode=args.fast
    )
    
    print(f"\n{'🖥️  Real-time display: ENABLED' if not args.no_display else '⚡ Fast mode: Display DISABLED'}")
    print("💡 Controls: ESC=Exit, SPACE=Pause\n")
    
    tester.run()


if __name__ == '__main__':
    main()
