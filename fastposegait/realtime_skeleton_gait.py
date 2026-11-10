# realtime_skeleton_gait.py - 실시간 스켈레톤 추출 및 보행 인식
"""
RTSP 스트림에서 실시간으로 스켈레톤을 추출하고 GPGait++ 모델로 보행 인식
3-Thread Pipeline Architecture:
  [Frame Receiver] → [Skeleton Extractor] → [Model Inference]
"""
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from queue import Queue, Empty, Full
from collections import deque
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import sys

# Project path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from improved_person_registry import ImprovedPersonRegistry


class GaitModelLoader:
    """GPGait++ 모델 로더"""
    
    @staticmethod
    def load_model(config_path, checkpoint_path, device='cuda'):
        """모델 로드"""
        print(f"📦 Loading GPGait++ model...")
        
        # Config 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            cfgs = yaml.safe_load(f)
        
        # 모델 초기화
        from fastposegait.modeling import models
        model_cfg = cfgs['model_cfg']
        model_name = model_cfg['model']
        ModelClass = getattr(models, model_name)
        model = nn.Module.__new__(ModelClass)
        nn.Module.__init__(model)
        model.build_network(model_cfg)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model']
        
        # 'module.' prefix 제거
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
        # Device로 이동 및 eval 모드 설정
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            # 모든 파라미터와 버퍼를 명시적으로 GPU로 이동
            for param in model.parameters():
                param.data = param.data.cuda()
            for buffer in model.buffers():
                buffer.data = buffer.data.cuda()
            print("✅ Model loaded on GPU")
        else:
            print("⚠️ Model loaded on CPU (slow)")
        
        model.eval()
        # Gradient 계산 비활성화
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    @staticmethod
    def preprocess_skeleton_sequence(skeleton_seq):
        """
        스켈레톤 시퀀스를 GPGait++ 입력 형식으로 변환
        
        Args:
            skeleton_seq: numpy array [T, 17, 2]
        
        Returns:
            torch.Tensor [1, 5, T, 17, 1]
        """
        skeleton_seq = np.array(skeleton_seq)
        T, V, C = skeleton_seq.shape  # [T, 17, 2]
        
        # 얼굴 키포인트(0~4) 마스킹
        skeleton_seq[:, 0:5, :] = 0
        
        # 중심화
        center = skeleton_seq.mean(axis=1, keepdims=True)
        skeleton_norm = skeleton_seq - center
        
        # Bone 계산
        bone = np.zeros_like(skeleton_norm)
        bone_pairs = [
            [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[6,8],[7,9],[8,10],
            [11,12],[5,11],[6,12],[11,13],[12,14],[13,15],[14,16]
        ]
        for v1, v2 in bone_pairs:
            if v1 < V and v2 < V:
                bone[:, v1, :] = skeleton_norm[:, v1, :] - skeleton_norm[:, v2, :]
        
        bone = bone - bone.mean(axis=1, keepdims=True)
        
        # Angle (간단 버전)
        angle = np.zeros((T, 17, 1))
        
        # [C, T, V, M] 형식으로 변환
        pose = skeleton_norm.transpose(2, 0, 1)[..., np.newaxis]  # [2, T, V, 1]
        bone = bone.transpose(2, 0, 1)[..., np.newaxis]  # [2, T, V, 1]
        angle = angle.transpose(2, 0, 1)[..., np.newaxis]  # [1, T, V, 1]
        
        # Concatenate: [5, T, V, M]
        data = np.concatenate([pose, bone, angle], axis=0)
        
        # Batch dimension 추가: [1, 5, T, V, M]
        data = torch.FloatTensor(data).unsqueeze(0)
        
        return data


class PersonRegistry:
    """실시간 Person Tracking 및 매칭 시스템 (Embedding 업데이트 지원)"""
    
    def __init__(self, match_threshold_cos=0.70, match_threshold_euc=0.50,
                 match_threshold_cos_strict=0.85, match_threshold_euc_strict=0.40):
        self.registry = {}  # {person_id: {'embedding': tensor, 'embeddings': list, 'last_seen': time, 'count': int}}
        self.next_id = 1
        
        # 적응형 임계값
        self.match_threshold_cos = match_threshold_cos  # 초기/기본 임계값
        self.match_threshold_euc = match_threshold_euc
        self.match_threshold_cos_strict = match_threshold_cos_strict  # 누적/엄격 임계값
        self.match_threshold_euc_strict = match_threshold_euc_strict
        
        self.stalking_threshold = 3  # 3번 이상 보이면 스토킹 의심
        
        # Embedding 업데이트 설정
        self.max_embeddings_per_person = 10  # 최대 저장 개수
        self.embedding_update_weight = 0.3  # 새 embedding 가중치 (EMA)
        self.use_multi_embedding = True  # 다중 embedding 사용 여부
        
    def match_or_register(self, embedding, use_strict=False):
        """
        새로운 embedding을 기존 registry와 매칭하거나 새로 등록
        매칭 성공 시 기존 embedding 업데이트
        
        Args:
            embedding: torch.Tensor [1, D] (normalized)
            use_strict: bool - True면 엄격한 임계값 사용
        
        Returns:
            person_id: int (매칭된 ID 또는 새 ID)
            is_new: bool (새로운 사람인지)
            match_score: float (매칭 점수, cosine similarity)
        """
        if not self.registry:
            # 첫 번째 사람 등록
            person_id = self.next_id
            self.next_id += 1
            self.registry[person_id] = {
                'embedding': embedding.clone(),
                'embeddings': [embedding.clone()],  # 다중 embedding 리스트
                'last_seen': time.time(),
                'count': 1,
                'first_seen': time.time()
            }
            print(f"   ✨ [Registry] First person registered: #{person_id}")
            return person_id, True, 1.0, 1.0, 0.0  # 첫 등록은 완벽한 매칭으로 간주
        
        # 기존 registry와 비교 (다중 embedding 지원)
        best_match_id = None
        best_cosine = -1.0
        best_euclid = float('inf')
        best_match_emb_idx = 0
        
        for pid, info in self.registry.items():
            if self.use_multi_embedding and len(info.get('embeddings', [])) > 0:
                # 다중 embedding과 비교 (최고 점수 선택)
                for idx, stored_emb in enumerate(info['embeddings']):
                    cosine_sim = torch.mm(embedding, stored_emb.t()).item()
                    euclid_dist = torch.cdist(embedding, stored_emb, p=2).item()
                    
                    if cosine_sim > best_cosine:
                        best_cosine = cosine_sim
                        best_euclid = euclid_dist
                        best_match_id = pid
                        best_match_emb_idx = idx
            else:
                # 단일 embedding 비교
                stored_emb = info['embedding']
                cosine_sim = torch.mm(embedding, stored_emb.t()).item()
                euclid_dist = torch.cdist(embedding, stored_emb, p=2).item()
                
                if cosine_sim > best_cosine:
                    best_cosine = cosine_sim
                    best_euclid = euclid_dist
                    best_match_id = pid
        
        # 임계값 선택
        cos_threshold = self.match_threshold_cos_strict if use_strict else self.match_threshold_cos
        euc_threshold = self.match_threshold_euc_strict if use_strict else self.match_threshold_euc
        
        # 로그 출력
        threshold_type = "Strict" if use_strict else "Relaxed"
        print(f"\n[Match] 비교 결과 ({threshold_type}):")
        print(f"  최고 매칭 Person #{best_match_id}")
        print(f"  코사인 유사도: {best_cosine:.4f} (>= {cos_threshold})")
        print(f"  유클리디안 거리: {best_euclid:.4f} (<= {euc_threshold})")
        
        # 매칭 조건: AND 로직 (둘 다 만족해야 매칭)
        cosine_match = best_cosine >= cos_threshold
        euclid_match = best_euclid <= euc_threshold
        
        if cosine_match and euclid_match:
            # 기존 사람 - Embedding 업데이트
            print(f"  ✅ 매칭 성공! Person #{best_match_id} (Cosine:{best_cosine:.3f}, Euclid:{best_euclid:.3f})")
            self.update_person_embedding(best_match_id, embedding)
            self.registry[best_match_id]['last_seen'] = time.time()
            self.registry[best_match_id]['count'] += 1
            
            combined_score = (best_cosine + (1 - best_euclid)) / 2
            return best_match_id, False, combined_score, best_cosine, best_euclid
        else:
            # 새로운 사람
            fail_reasons = []
            if not cosine_match:
                fail_reasons.append(f"Cosine:{best_cosine:.3f}<{cos_threshold}")
            if not euclid_match:
                fail_reasons.append(f"Euclid:{best_euclid:.3f}>{euc_threshold}")
            print(f"  ❌ 매칭 실패! ({', '.join(fail_reasons)}) 새로운 Person #{self.next_id} 등록")
            person_id = self.next_id
            self.next_id += 1
            self.registry[person_id] = {
                'embedding': embedding.clone(),
                'embeddings': [embedding.clone()],
                'last_seen': time.time(),
                'count': 1,
                'first_seen': time.time()
            }
            combined_score = (best_cosine + (1 - best_euclid)) / 2
            return person_id, True, combined_score, best_cosine, best_euclid
    
    def force_update_person(self, person_id, new_embedding):
        """
        세션 잠금 모드: 강제로 특정 Person ID의 embedding 업데이트
        
        Args:
            person_id: int - 업데이트할 Person ID
            new_embedding: torch.Tensor [1, D] (normalized)
        
        Returns:
            person_id: int
            is_new: bool (항상 False)
            match_score: float (1.0 - 강제 매칭)
        """
        if person_id not in self.registry:
            # Person이 없으면 등록
            print(f"   ⚠️ [Force] Person #{person_id} not found, registering...")
            self.registry[person_id] = {
                'embedding': new_embedding.clone(),
                'embeddings': [new_embedding.clone()],
                'last_seen': time.time(),
                'count': 1,
                'first_seen': time.time()
            }
            return person_id, False, 1.0, 0.0, 1.0  # 새 Person 등록 시 기본값
        
        # 기존 Person 업데이트
        self.update_person_embedding(person_id, new_embedding)
        self.registry[person_id]['last_seen'] = time.time()
        self.registry[person_id]['count'] += 1
        
        print(f"   ✅ [Force] Updated Person #{person_id} (count: {self.registry[person_id]['count']})")
        
        return person_id, False, 1.0, 1.0, 0.0  # force update는 완벽한 매칭으로 간주
    
    def update_person_embedding(self, person_id, new_embedding):
        """
        기존 사람의 embedding 업데이트
        
        전략:
        1. Exponential Moving Average (EMA)로 주 embedding 업데이트
        2. 다중 embedding 리스트에 추가 (최대 개수 제한)
        3. 오래된 embedding 제거
        
        Args:
            person_id: int
            new_embedding: torch.Tensor [1, D] (normalized)
        """
        if person_id not in self.registry:
            return
        
        info = self.registry[person_id]
        
        # 1. 주 embedding 업데이트 (EMA)
        old_emb = info['embedding']
        alpha = self.embedding_update_weight  # 새 embedding 가중치
        updated_emb = alpha * new_embedding + (1 - alpha) * old_emb
        # L2 정규화
        updated_emb = updated_emb / updated_emb.norm(p=2, dim=1, keepdim=True)
        info['embedding'] = updated_emb
        
        # 2. 다중 embedding 리스트 업데이트
        if 'embeddings' not in info:
            info['embeddings'] = []
        
        info['embeddings'].append(new_embedding.clone())
        
        # 3. 최대 개수 초과 시 오래된 것 제거 (FIFO)
        if len(info['embeddings']) > self.max_embeddings_per_person:
            info['embeddings'].pop(0)
        
        print(f"   🔄 [Registry] Updated Person #{person_id} embedding")
        print(f"      Main embedding: EMA(α={alpha:.2f})")
        print(f"      Stored embeddings: {len(info['embeddings'])}/{self.max_embeddings_per_person}")
    
    def check_stalking(self, person_id):
        """스토킹 의심 여부 체크"""
        if person_id in self.registry:
            count = self.registry[person_id]['count']
            duration = time.time() - self.registry[person_id]['first_seen']
            
            # 짧은 시간(60초)에 여러 번 감지되면 스토킹 의심
            if count >= self.stalking_threshold and duration < 60:
                return True, count, duration
        return False, 0, 0
    
    def get_aggregated_embedding(self, person_id):
        """
        Person의 통합 embedding 반환 (다중 embedding 평균)
        
        Args:
            person_id: int
        
        Returns:
            torch.Tensor [1, D]: 통합된 embedding
        """
        if person_id not in self.registry:
            return None
        
        info = self.registry[person_id]
        
        if self.use_multi_embedding and len(info.get('embeddings', [])) > 0:
            # 모든 embedding의 평균
            embeddings = torch.stack(info['embeddings'], dim=0)  # [N, 1, D]
            embeddings = embeddings.squeeze(1)  # [N, D]
            avg_emb = embeddings.mean(dim=0, keepdim=True)  # [1, D]
            # L2 정규화
            avg_emb = avg_emb / avg_emb.norm(p=2, dim=1, keepdim=True)
            return avg_emb
        else:
            return info['embedding']
    
    def get_person_info(self, person_id):
        """Person 정보 조회 (embedding 통계 포함)"""
        if person_id in self.registry:
            info = self.registry[person_id]
            return {
                'id': person_id,
                'count': info['count'],
                'duration': time.time() - info['first_seen'],
                'last_seen': info['last_seen'],
                'embedding_count': len(info.get('embeddings', [])),
                'has_multi_embedding': len(info.get('embeddings', [])) > 1
            }
        return None
    
    def cleanup_old_entries(self, timeout=300):
        """오래된 항목 정리 (5분 이상 안 보인 사람)"""
        current_time = time.time()
        to_remove = []
        for pid, info in self.registry.items():
            if current_time - info['last_seen'] > timeout:
                to_remove.append(pid)
        
        for pid in to_remove:
            del self.registry[pid]
        
        return len(to_remove)


class SkeletonExtractor:
    """MediaPipe 기반 실시간 스켈레톤 추출기"""
    
    def __init__(self):
        # MediaPipe 설정
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # COCO 17 keypoint 매핑
        self.mp_to_coco_mapping = {
            0: 0,   # nose
            2: 1,   # left_eye  
            5: 2,   # right_eye
            7: 3,   # left_ear
            8: 4,   # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16  # right_ankle
        }
        
        # COCO skeleton 연결
        self.coco_skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # 얼굴
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],  # 팔
            [5, 11], [6, 12], [11, 12],  # 몸통
            [11, 13], [12, 14], [13, 15], [14, 16]  # 다리
        ]
    
    def extract_coco_skeleton(self, landmarks, width, height):
        """MediaPipe landmarks를 COCO 17 format으로 변환"""
        skeleton = np.zeros((17, 2))
        
        for mp_idx, coco_idx in self.mp_to_coco_mapping.items():
            if mp_idx < len(landmarks):
                landmark = landmarks[mp_idx]
                skeleton[coco_idx, 0] = landmark.x * width
                skeleton[coco_idx, 1] = landmark.y * height
        
        return skeleton
    
    def process_frame(self, frame):
        """프레임에서 스켈레톤 추출"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            skeleton = self.extract_coco_skeleton(results.pose_landmarks.landmark, w, h)
            return skeleton, results.pose_landmarks
        
        return None, None
    
    def draw_skeleton(self, frame, skeleton, pose_landmarks=None):
        """프레임에 스켈레톤 그리기"""
        h, w = frame.shape[:2]
        
        # MediaPipe 스켈레톤 그리기
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # COCO 관절점 그리기
        if skeleton is not None:
            for i, (x, y) in enumerate(skeleton):
                if x > 0 and y > 0 and x < w and y < h:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 2)
        
        return frame


class RealtimeGaitRecognizer:
    """실시간 보행 인식 시스템"""
    
    def __init__(self, jetson_ip="172.30.101.189", port=8554, sequence_length=300,
                 config_path=None, checkpoint_path=None, enable_model=False):
        self.jetson_ip = jetson_ip
        self.port = port
        
        # 버퍼 시스템 (scenario_tester와 동일)
        self.initial_buffer_size = 300  # 충분한 gait signature 확보
        self.accumulation_interval = 90  # 주기적 추론 간격
        self.max_buffer_size = 400  # 최대 버퍼 크기
        self.sequence_length = sequence_length  # 호환성 유지
        self.frames_since_last_inference = 0
        self.has_initial_inference = False
        
        # 스레드 간 통신 큐
        self.frame_queue = Queue(maxsize=10)  # Receiver → Extractor
        self.skeleton_queue = Queue(maxsize=30)  # Extractor → Inference
        self.display_queue = Queue(maxsize=5)  # Extractor → Display
        self.result_queue = Queue(maxsize=10)  # Inference → Display (결과)
        
        # 스레드 제어
        self.running = False
        self.threads = []
        
        # 스켈레톤 추출기
        self.skeleton_extractor = SkeletonExtractor()
        
        # 스켈레톤 시퀀스 버퍼 (maxlen 제거 - 수동 관리)
        self.skeleton_buffer = []
        
        # GPGait++ 모델 (옵션)
        self.enable_model = enable_model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if enable_model:
            if config_path is None or checkpoint_path is None:
                raise ValueError("config_path and checkpoint_path required when enable_model=True")
            self.model = GaitModelLoader.load_model(config_path, checkpoint_path, str(self.device))
        
        # Person DB 저장 경로 (scenario_tester와 동일)
        self.person_db_path = 'realtime_person_database.pkl'
        
        # Person Registry (ImprovedPersonRegistry 사용 - scenario_tester와 동일)
        self.person_registry = ImprovedPersonRegistry(
            match_threshold_cos_strict=0.92,   # 엄격 (STRICT 모드)
            match_threshold_euc_strict=0.35,
            match_threshold_cos_relaxed=0.88,  # 완화 (RELAXED 모드)
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
        
        # Person ID 관리 (세션 추적)
        self.current_person_id = None
        self.session_person_locked = False
        self.tracking_confidence = 0.0
        
        # 최근 인식 결과 (display용)
        self.latest_recognition = {
            'person_id': None,
            'is_new': False,
            'match_score': 0.0,
            'timestamp': 0
        }
        
        # 통계
        self.stats = {
            'received_frames': 0,
            'processed_frames': 0,
            'detected_frames': 0,
            'inference_count': 0,
            'dropped_frames': 0,
            'receiver_fps': 0,
            'extractor_fps': 0,
            'display_fps': 0
        }
        
        # FPS 계산용
        self.fps_counters = {
            'receiver': {'count': 0, 'last_time': time.time()},
            'extractor': {'count': 0, 'last_time': time.time()},
            'display': {'count': 0, 'last_time': time.time()}
        }
    
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
            for pid, info in self.person_registry.registry.items():
                print(f"   Person #{pid}: {info['count']} detections, {len(info.get('embeddings', []))} embeddings")
        except Exception as e:
            print(f"⚠️ Failed to save Person DB: {e}")
    
    def update_fps(self, counter_name):
        """FPS 업데이트"""
        counter = self.fps_counters[counter_name]
        counter['count'] += 1
        
        current_time = time.time()
        if current_time - counter['last_time'] >= 1.0:
            self.stats[f'{counter_name}_fps'] = counter['count']
            counter['count'] = 0
            counter['last_time'] = current_time
    
    def frame_receiver_thread(self):
        """Thread 1: RTSP 스트림에서 프레임 수신"""
        rtsp_url = f"rtsp://{self.jetson_ip}:{self.port}/stream"
        print(f"📡 [Receiver] Connecting to {rtsp_url}")
        
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("❌ [Receiver] Failed to connect to RTSP stream")
            self.running = False
            return
        
        print("✅ [Receiver] Connected successfully")
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ [Receiver] Failed to receive frame")
                time.sleep(0.1)
                continue
            
            self.stats['received_frames'] += 1
            self.update_fps('receiver')
            
            try:
                # Queue가 가득 차면 오래된 프레임 버리고 최신 프레임 넣기
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # 오래된 프레임 버림
                        self.stats['dropped_frames'] += 1
                    except Empty:
                        pass
                
                self.frame_queue.put(frame, block=False)
            except Full:
                self.stats['dropped_frames'] += 1
        
        cap.release()
        print("🔚 [Receiver] Thread stopped")
    
    def skeleton_extractor_thread(self):
        """Thread 2: 스켈레톤 추출 및 시각화"""
        print("🚀 [Extractor] Starting skeleton extraction")
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                continue
            
            self.stats['processed_frames'] += 1
            
            # 스켈레톤 추출
            skeleton, pose_landmarks = self.skeleton_extractor.process_frame(frame)
            
            if skeleton is not None:
                self.stats['detected_frames'] += 1
                
                # 스켈레톤을 inference queue에 추가
                try:
                    self.skeleton_queue.put_nowait({
                        'skeleton': skeleton,
                        'timestamp': time.time(),
                        'frame_id': self.stats['processed_frames']
                    })
                except Full:
                    # Inference가 느리면 스켈레톤 버림
                    pass
                
                # 시각화를 위한 프레임 준비
                display_frame = frame.copy()
                display_frame = self.skeleton_extractor.draw_skeleton(
                    display_frame, skeleton, pose_landmarks
                )
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, "NO POSE", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display queue에 추가
            try:
                if self.display_queue.full():
                    self.display_queue.get_nowait()  # 오래된 프레임 버림
                self.display_queue.put_nowait(display_frame)
            except (Empty, Full):
                pass
            
            self.update_fps('extractor')
        
        print("🔚 [Extractor] Thread stopped")
    
    def model_inference_thread(self):
        """Thread 3: 모델 추론 - scenario_tester와 동일한 로직"""
        print("🚀 [Inference] Starting model inference")
        print(f"   Initial buffer: {self.initial_buffer_size} frames (약 {self.initial_buffer_size/30:.1f}초)")
        print(f"   Accumulation interval: {self.accumulation_interval} frames (약 {self.accumulation_interval/30:.1f}초)")
        print(f"   Mode: 초기 {self.initial_buffer_size}프레임 수집 후 {self.accumulation_interval}프레임마다 추론")
        
        while self.running:
            try:
                skeleton_data = self.skeleton_queue.get(timeout=1.0)
            except Empty:
                continue
            
            # 스켈레톤 버퍼에 추가
            self.skeleton_buffer.append(skeleton_data['skeleton'])
            self.frames_since_last_inference += 1
            
            buffer_size = len(self.skeleton_buffer)
            
            # 초기 추론: initial_buffer_size 도달 시
            if not self.has_initial_inference and buffer_size >= self.initial_buffer_size:
                print(f"\n[Inference] 초기 추론 시작 ({buffer_size} frames)")
                skeleton_sequence = np.array(self.skeleton_buffer[-self.initial_buffer_size:])
                
                # GPGait++ 모델 추론
                if self.enable_model:
                    result = self.run_gait_model(skeleton_sequence)
                    self.has_initial_inference = True
                    self.frames_since_last_inference = 0
                    
                    if result:
                        self.stats['inference_count'] += 1
                        print(f"\n🤖 [Inference] 초기 인식 완료")
                        print(f"   Person ID: #{result['person_id']}")
                        print(f"   Status: {'🆕 NEW' if result['is_new'] else '✅ MATCHED'}")
                        print(f"   Match Score: {result['match_score']:.4f}")
                        
            # 주기적 추론: accumulation_interval 간격으로
            elif self.has_initial_inference and self.frames_since_last_inference >= self.accumulation_interval:
                # 버퍼 크기 제한
                if buffer_size > self.max_buffer_size:
                    # 균등 샘플링
                    indices = np.linspace(0, buffer_size-1, self.max_buffer_size, dtype=int)
                    inference_buffer = np.array(self.skeleton_buffer)[indices]
                else:
                    inference_buffer = np.array(self.skeleton_buffer)
                
                print(f"\n[Inference] 주기적 추론 ({len(inference_buffer)} frames)")
                
                if self.enable_model:
                    result = self.run_gait_model(inference_buffer)
                    self.frames_since_last_inference = 0
                    
                    if result:
                        self.stats['inference_count'] += 1
                        print(f"\n🤖 [Inference] 주기 인식 완료")
                        print(f"   Person ID: #{result['person_id']}")
                        print(f"   Status: {'🆕 NEW' if result['is_new'] else '✅ MATCHED'}")
                        print(f"   Match Score: {result['match_score']:.4f}")
                        
                        # 스토킹 경고
                        if result['is_stalking']:
                            print(f"   ⚠️ STALKING ALERT!")
                            print(f"   Detected {result['stalk_count']} times in {result['stalk_duration']:.1f}s")
                        
                        # Person Registry 상태
                        print(f"   Registered Persons: {len(self.person_registry.registry)}")
        
        print("🔚 [Inference] Thread stopped")
    
    def run_gait_model(self, skeleton_sequence):
        """
        GPGait++ 모델 추론
        
        Args:
            skeleton_sequence: (T, 17, 2) numpy array
        
        Returns:
            dict: {'person_id': int, 'is_new': bool, 'match_score': float, 'embedding': tensor}
        """
        if self.model is None:
            return None
        
        try:
            # 1. 전처리: (T, 17, 2) -> [1, 5, T, 17, 1]
            data = GaitModelLoader.preprocess_skeleton_sequence(skeleton_sequence)
            data = data.to(self.device)
            
            # 2. 모델 추론
            with torch.no_grad():
                T = data.shape[2]
                seqL = [torch.LongTensor([T]).to(self.device)]
                
                # contiguous 보장
                data = data.contiguous()
                
                # GPGait++ input format
                inputs = [[data], None, None, None, seqL]
                
                outputs = self.model.forward(inputs)
                
                # Feature 추출
                feat = outputs['inference_feat']['embeddings']  # [N, D, S] or [N, D]
                
                # Temporal pooling
                if feat.dim() == 3:
                    feat = feat.mean(dim=2)  # [N, D]
                
                # L2 normalize
                feat = F.normalize(feat, p=2, dim=1)  # [1, D]
            
            # 3. Person Registry에 매칭 또는 등록 (scenario_tester 로직)
            if self.session_person_locked and self.current_person_id is not None:
                # 세션 잠금 모드: 강제 업데이트
                person_id, is_new, match_score = self.person_registry.force_update_person(
                    self.current_person_id, feat
                )
                cosine_sim = 1.0
                euclid_dist = 0.0
            else:
                # 매칭 또는 신규 등록
                person_id, is_new, match_score = self.person_registry.match_or_register(
                    feat, frame_count=len(skeleton_sequence)
                )
                cosine_sim = match_score
                euclid_dist = 1.0 - match_score
                
                # 초기 추론 시 세션 관리
                if not self.has_initial_inference:
                    self.current_person_id = person_id
                    self.tracking_confidence = match_score
                    
                    if match_score > 0.85:
                        self.session_person_locked = True
                        print(f"   🔒 [Session] Person #{person_id} locked (score: {match_score:.3f})")
                    else:
                        self.session_person_locked = False
                        print(f"   🔓 [Session] Person #{person_id} unlocked (score: {match_score:.3f})")
                else:
                    # 세션 잠금 해제 조건
                    if self.session_person_locked and match_score < 0.70:
                        self.session_person_locked = False
                        print(f"   🔓 [Session] Unlocked (low confidence: {match_score:.3f})")
                    
                    # ID 변경 로직
                    if not self.session_person_locked:
                        if person_id != self.current_person_id:
                            if match_score > self.tracking_confidence + 0.10:
                                old_id = self.current_person_id
                                self.current_person_id = person_id
                                self.tracking_confidence = match_score
                                print(f"   🔄 [ID Change] #{old_id} → #{person_id} (score: {match_score:.3f})")
                                
                                if match_score > 0.85:
                                    self.session_person_locked = True
                                    print(f"   🔒 [Session] Locked to Person #{person_id}")
                            else:
                                person_id = self.current_person_id
                                is_new = False
                        else:
                            self.tracking_confidence = max(self.tracking_confidence, match_score)
            
            # 4. 스토킹 체크 (ImprovedPersonRegistry에는 없으므로 기본값)
            is_stalking, stalk_count, stalk_duration = False, 0, 0
            
            result = {
                'person_id': person_id,
                'is_new': is_new,
                'match_score': match_score,
                'cosine_similarity': float(cosine_sim),
                'euclidean_distance': float(euclid_dist),
                'embedding': feat,
                'is_stalking': is_stalking,
                'stalk_count': stalk_count,
                'stalk_duration': stalk_duration,
                'timestamp': time.time()
            }
            
            # 최근 결과 업데이트
            self.latest_recognition = result
            
            # Result queue에 전송 (display용)
            try:
                self.result_queue.put_nowait(result)
            except Full:
                pass
            
            return result
            
        except Exception as e:
            print(f"⚠️ [Inference] Error in model inference: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def display_thread(self):
        """Display Thread: 실시간 시각화"""
        print("🚀 [Display] Starting display")
        
        cv2.namedWindow("Realtime Gait Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Realtime Gait Recognition", 1280, 720)
        
        while self.running:
            try:
                frame = self.display_queue.get(timeout=1.0)
            except Empty:
                continue
            
            # 통계 정보 오버레이
            self.draw_stats(frame)
            
            cv2.imshow("Realtime Gait Recognition", frame)
            self.update_fps('display')
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 [Display] Quit requested")
                self.running = False
                break
            elif key == ord('s'):
                filename = f'gait_snapshot_{int(time.time())}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 [Display] Saved: {filename}")
        
        cv2.destroyAllWindows()
        print("🔚 [Display] Thread stopped")
    
    def draw_stats(self, frame):
        """통계 정보 표시"""
        h, w = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 제목
        cv2.putText(frame, "Realtime Gait Recognition", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 통계 정보
        y_offset = 65
        line_height = 25
        
        stats_text = [
            f"Source: {self.jetson_ip}:{self.port}",
            f"Receiver FPS: {self.stats['receiver_fps']}",
            f"Extractor FPS: {self.stats['extractor_fps']}",
            f"Display FPS: {self.stats['display_fps']}",
            f"Received: {self.stats['received_frames']}",
            f"Detected: {self.stats['detected_frames']}/{self.stats['processed_frames']}",
            f"Dropped: {self.stats['dropped_frames']}",
            f"Inferences: {self.stats['inference_count']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (20, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 버퍼 상태
        target_size = self.initial_buffer_size if not self.has_initial_inference else self.max_buffer_size
        buffer_fill = len(self.skeleton_buffer) / target_size
        cv2.rectangle(frame, (20, 230), (430, 245), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 230), (20 + int(410 * buffer_fill), 245), 
                     (0, 255, 0) if buffer_fill >= 1.0 else (0, 165, 255), -1)
        
        buffer_text = f"Buffer: {len(self.skeleton_buffer)}/{target_size}"
        if self.has_initial_inference:
            buffer_text += f" (Next: {self.accumulation_interval - self.frames_since_last_inference})"
        cv2.putText(frame, buffer_text, 
                   (25, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 인식 결과 표시 (오른쪽 상단)
        if self.enable_model and self.latest_recognition['person_id'] is not None:
            # 배경
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (w-320, 10), (w-10, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
            
            # Person ID (크게)
            person_id = self.latest_recognition['person_id']
            cv2.putText(frame, f"Person #{person_id}", (w-310, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Status
            status_text = "NEW" if self.latest_recognition['is_new'] else "MATCHED"
            status_color = (0, 255, 0) if not self.latest_recognition['is_new'] else (255, 255, 0)
            cv2.putText(frame, status_text, (w-310, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Match Score
            score = self.latest_recognition['match_score']
            cv2.putText(frame, f"Score: {score:.3f}", (w-310, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Cosine Similarity
            if 'cosine_similarity' in self.latest_recognition:
                cos_text = f"Cos: {self.latest_recognition['cosine_similarity']:.4f}"
                cv2.putText(frame, cos_text, (w-310, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
            
            # Euclidean Distance
            if 'euclidean_distance' in self.latest_recognition:
                euc_text = f"Euc: {self.latest_recognition['euclidean_distance']:.4f}"
                cv2.putText(frame, euc_text, (w-310, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
            
            # Person Info
            person_info = self.person_registry.get_person_info(person_id)
            if person_info:
                cv2.putText(frame, f"Count: {person_info['count']}", (w-310, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Embeddings: {person_info['embedding_count']}", (w-310, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Stalking Alert (ImprovedPersonRegistry에는 없으므로 주석 처리)
            # is_stalking, stalk_count, _ = self.person_registry.check_stalking(person_id)
            if False:
                cv2.putText(frame, "⚠️ STALKING ALERT", (w-310, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 컨트롤 정보
        cv2.putText(frame, "Press 'q' to quit, 's' for screenshot", (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 상태 표시
        status_color = (0, 255, 0) if self.stats['receiver_fps'] > 20 else (0, 165, 255)
        cv2.circle(frame, (w-30, 30), 15, status_color, -1)
        cv2.putText(frame, "LIVE", (w-100, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    def start(self):
        """시스템 시작"""
        print("🚀 Starting Realtime Gait Recognition System")
        print(f"📡 Target: {self.jetson_ip}:{self.port}")
        print(f"📊 Sequence length: {self.sequence_length} frames")
        print(f"🧵 Starting 4 threads...")
        
        self.running = True
        
        # 스레드 생성 및 시작
        threads_config = [
            ("Receiver", self.frame_receiver_thread),
            ("Extractor", self.skeleton_extractor_thread),
            ("Inference", self.model_inference_thread),
            ("Display", self.display_thread)
        ]
        
        for name, target in threads_config:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)
            print(f"✅ {name} thread started")
            time.sleep(0.2)  # 스레드 초기화 대기
        
        print("\n✅ All threads started successfully")
        print("📺 Press 'q' in the display window to quit\n")
        
        # 메인 스레드는 display 스레드 대기
        try:
            self.threads[3].join()  # Display thread
        except KeyboardInterrupt:
            print("\n⚠️ Ctrl+C pressed - shutting down...")
            self.running = False
        
        # 모든 스레드 종료 대기
        print("\n🛑 Stopping all threads...")
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Person DB 저장 (scenario_tester와 동일)
        if self.enable_model:
            self.save_person_db()
        
        # 최종 통계
        self.print_final_stats()
    
    def print_final_stats(self):
        """최종 통계 출력"""
        print("\n" + "="*60)
        print("📊 Final Statistics")
        print("="*60)
        print(f"Received frames:     {self.stats['received_frames']}")
        print(f"Processed frames:    {self.stats['processed_frames']}")
        print(f"Detected frames:     {self.stats['detected_frames']}")
        print(f"Dropped frames:      {self.stats['dropped_frames']}")
        print(f"Model inferences:    {self.stats['inference_count']}")
        
        if self.stats['processed_frames'] > 0:
            detection_rate = (self.stats['detected_frames'] / self.stats['processed_frames']) * 100
            print(f"Detection rate:      {detection_rate:.1f}%")
        
        if self.stats['received_frames'] > 0:
            drop_rate = (self.stats['dropped_frames'] / self.stats['received_frames']) * 100
            print(f"Drop rate:           {drop_rate:.1f}%")
        
        print("="*60)
        print("👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(description='Realtime Gait Recognition System')
    parser.add_argument('--ip', type=str, default='172.30.117.33', 
                       help='Jetson IP address')
    parser.add_argument('--port', type=int, default=8554, 
                       help='RTSP port (default: 8554)')
    parser.add_argument('--seq-length', type=int, default=300,
                       help='Initial buffer size for model input (default: 300)')
    parser.add_argument('--inference-interval', type=int, default=90,
                       help='Inference interval in frames after initial (default: 90)')
    parser.add_argument('--enable-model', action='store_true',
                       help='Enable GPGait++ model inference')
    parser.add_argument('--config', type=str, 
                       default='configs/gpgait++/gpgait++.yaml',
                       help='Model config path')
    parser.add_argument('--checkpoint', type=str,
                       default=r'C:\FastGait\FastPoseGait\fastposegait\output\CASIA-B\GPGait_PlusPlus\GPGait_PlusPlus\checkpoints\GPGait_PlusPlus-40000.pt',
                       help='Model checkpoint path')
    
    args = parser.parse_args()
    
    # Config path를 절대 경로로 변환
    if args.enable_model:
        if not os.path.isabs(args.config):
            args.config = os.path.join(PROJECT_ROOT, args.config)
    
    recognizer = RealtimeGaitRecognizer(
        jetson_ip=args.ip,
        port=args.port,
        sequence_length=args.seq_length,
        config_path=args.config if args.enable_model else None,
        checkpoint_path=args.checkpoint if args.enable_model else None,
        enable_model=args.enable_model
    )
    
    recognizer.start()


if __name__ == '__main__':
    main()
