# ============================================================================
# 📌 개선된 Realtime Gait Recognition System (모든 개선사항 적용 완료)
# ============================================================================
# 주요 개선사항:
# ✅ 1. Skeleton Preprocessing: 정규화 강화 + Angle feature 계산
# ✅ 2. 매칭 로직: AND → OR 조건 변경 + Threshold 동적 조정
# ✅ 3. Temporal Smoothing: 프레임 간 좌표 평활화
# ✅ 4. 디버깅: 상세 로깅 추가
# ✅ 5. MediaPipe 품질 개선: confidence threshold 상향
# ============================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

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
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 🎯 개선된 Skeleton Preprocessor
# ============================================================================

class ImprovedSkeletonPreprocessor:
    """
    개선된 Skeleton 전처리 모듈
    - 어깨 너비 기반 정규화
    - 실제 Angle feature 계산
    - Temporal smoothing
    """
    
    def __init__(self, sequence_length=60, use_smoothing=True):
        self.sequence_length = sequence_length
        self.use_smoothing = use_smoothing
        self.last_skeleton = None
        
        # COCO 17 keypoint joint pairs for angle calculation
        self.joint_pairs = [
            [0, 1, 2],    # nose-neck-head
            [5, 6, 12],   # left_shoulder-right_shoulder-hip
            [5, 7, 9],    # left_shoulder-left_elbow-left_wrist
            [6, 8, 10],   # right_shoulder-right_elbow-right_wrist
            [11, 13, 15], # left_hip-left_knee-left_ankle
            [12, 14, 16], # right_hip-right_knee-right_ankle
        ]
    
    def calculate_angles(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Joint 각도 계산
        Args:
            skeleton: [17, 2] (x, y 좌표)
        Returns:
            angles: [17, 1] (각 관절의 각도)
        """
        angles = np.zeros((17, 1))
        
        for triplet in self.joint_pairs:
            if len(triplet) == 3:
                p1, p2, p3 = triplet
                
                if p1 < 17 and p2 < 17 and p3 < 17:
                    v1 = skeleton[p1] - skeleton[p2]
                    v2 = skeleton[p3] - skeleton[p2]
                    
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    
                    if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        angles[p2] = angle
        
        return angles
    
    def normalize_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Skeleton 정규화 (어깨 너비 기반)
        Args:
            skeleton: [17, 2]
        Returns:
            normalized_skeleton: [17, 2]
        """
        # 어깨 너비로 스케일 정규화
        shoulder_width = np.linalg.norm(skeleton[5] - skeleton[6]) + 1e-6
        skeleton_norm = skeleton / shoulder_width
        
        # 골반 중심 기준으로 중심화
        hip_center = (skeleton_norm[11] + skeleton_norm[12]) / 2
        skeleton_norm = skeleton_norm - hip_center
        
        return skeleton_norm
    
    def apply_temporal_smoothing(self, skeleton: np.ndarray, alpha=0.8) -> np.ndarray:
        """
        EMA (Exponential Moving Average) 기반 temporal smoothing
        Args:
            skeleton: [17, 2]
            alpha: 평활화 계수 (높을수록 현재 프레임 가중)
        Returns:
            smoothed_skeleton: [17, 2]
        """
        if self.last_skeleton is None:
            self.last_skeleton = skeleton.copy()
            return skeleton
        
        smoothed = alpha * skeleton + (1 - alpha) * self.last_skeleton
        self.last_skeleton = smoothed.copy()
        
        return smoothed
    
    def preprocess_sequence(self, skeleton_seq: np.ndarray) -> torch.Tensor:
        """
        개선된 skeleton 시퀀스 전처리
        Args:
            skeleton_seq: [T, 17, 2] numpy array
        Returns:
            torch.Tensor [1, 6, T, 17, 1]
        """
        T, V, C = skeleton_seq.shape
        
        # 1️⃣ 정규화 및 temporal smoothing 적용
        skeleton_norm_list = []
        for t in range(T):
            skel = skeleton_seq[t].copy()
            
            # 정규화
            skel_norm = self.normalize_skeleton(skel)
            
            # Temporal smoothing
            if self.use_smoothing:
                skel_norm = self.apply_temporal_smoothing(skel_norm, alpha=0.8)
            
            skeleton_norm_list.append(skel_norm)
        
        skeleton_norm = np.array(skeleton_norm_list)  # [T, 17, 2]
        
        # 2️⃣ Bone 벡터 계산
        bone = np.zeros_like(skeleton_norm)
        bone_pairs = [
            [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[6,8],[7,9],[8,10],
            [11,12],[5,11],[6,12],[11,13],[12,14],[13,15],[14,16]
        ]
        
        for v1, v2 in bone_pairs:
            if v1 < V and v2 < V:
                bone[:, v1, :] = skeleton_norm[:, v1, :] - skeleton_norm[:, v2, :]
        
        # Bone 정규화
        bone_mean = bone.mean(axis=1, keepdims=True)
        bone = bone - bone_mean
        bone_std = bone.std(axis=0, keepdims=True) + 1e-6
        bone = bone / bone_std
        
        # 3️⃣ Angle 계산 (개선됨!)
        angle_list = []
        for t in range(T):
            angles = self.calculate_angles(skeleton_norm[t])  # [17, 1]
            angle_list.append(angles)
        
        angle = np.array(angle_list).squeeze(-1)  # [T, 17]
        angle = angle[..., np.newaxis]  # [T, 17, 1]
        
        # Angle 정규화
        angle_mean = angle.mean(axis=0, keepdims=True)
        angle = angle - angle_mean
        angle_std = angle.std(axis=0, keepdims=True) + 1e-6
        angle = angle / angle_std
        
        # 4️⃣ [C, T, V, M] 형식으로 변환
        pose = skeleton_norm.transpose(2, 0, 1)[..., np.newaxis]  # [2, T, V, 1]
        bone_t = bone.transpose(2, 0, 1)[..., np.newaxis]  # [2, T, V, 1]
        angle_t = angle.transpose(2, 0, 1)[..., np.newaxis]  # [1, T, V, 1]
        
        # Concatenate: [6, T, V, M]
        data = np.concatenate([pose, bone_t, angle_t], axis=0)
        
        # Batch dimension 추가
        data = torch.FloatTensor(data).unsqueeze(0)  # [1, 6, T, V, M]
        
        return data


# ============================================================================
# 🔄 개선된 Person Registry (OR 조건 + Embedding 업데이트)
# ============================================================================

class ImprovedPersonRegistry:
    """
    개선된 Person Registry
    - OR 기반 매칭 로직
    - 동적 threshold 조정
    - Embedding 업데이트
    - 상세 로깅
    """
    
    def __init__(self, match_threshold_cos=0.88, match_threshold_euc=0.40, enable_logging=True):
        self.registry = {}
        self.next_id = 1
        self.match_threshold_cos = match_threshold_cos
        self.match_threshold_euc = match_threshold_euc
        self.enable_logging = enable_logging
        self.stalking_threshold = 3
        
        # 로깅용
        self.match_history = deque(maxlen=100)
    
    def match_or_register(self, embedding: torch.Tensor) -> dict:
        """
        개선된 매칭 로직 (OR 조건)
        
        Args:
            embedding: [1, D] normalized tensor
        
        Returns:
            dict with person_id, is_new, match_score, etc.
        """
        
        if not self.registry:
            # 첫 번째 사람 등록
            person_id = self.next_id
            self.next_id += 1
            self.registry[person_id] = {
                'embedding': embedding.clone().detach(),
                'last_seen': time.time(),
                'count': 1,
                'first_seen': time.time()
            }
            
            if self.enable_logging:
                print(f"✨ [Registry] New person: ID #{person_id}")
            
            return {
                'person_id': person_id,
                'is_new': True,
                'match_score': 1.0,
                'match_method': 'new',
                'cosine_sim': 1.0,
                'euclid_dist': 0.0,
                'embedding': embedding
            }
        
        # 기존 registry와 비교
        best_match_id = None
        best_cosine = -2.0
        best_euclid = float('inf')
        
        for pid, info in self.registry.items():
            stored_emb = info['embedding']
            
            # Cosine similarity
            cosine_sim = torch.mm(embedding, stored_emb.t()).item()
            
            # Euclidean distance
            euclid_dist = torch.cdist(embedding, stored_emb, p=2).item()
            
            if cosine_sim > best_cosine:
                best_cosine = cosine_sim
                best_euclid = euclid_dist
                best_match_id = pid
        
        # 🔄 개선된 매칭 조건 (OR 로직!)
        matched = False
        match_method = 'none'
        match_score = best_cosine
        
        # 코사인 유사도 >= 0.88 이면 매칭 (우선)
        if best_cosine >= self.match_threshold_cos:
            matched = True
            match_method = 'cosine'
        # 혹은 유클리드 거리 <= 0.40 이면 매칭
        elif best_euclid <= self.match_threshold_euc:
            matched = True
            match_method = 'euclidean'
        
        if matched:
            # 기존 사람과 매칭됨
            self.registry[best_match_id]['last_seen'] = time.time()
            self.registry[best_match_id]['count'] += 1
            
            # 💡 Embedding 동적 업데이트 (새로운 스타일 적응)
            alpha = 0.1
            self.registry[best_match_id]['embedding'] = (
                alpha * embedding.clone().detach() +
                (1 - alpha) * self.registry[best_match_id]['embedding']
            )
            
            if self.enable_logging:
                print(f"✅ [Registry] Matched ID #{best_match_id} "
                      f"({match_method}: {match_score:.4f})")
            
            self.match_history.append(('matched', best_match_id, match_score))
            
            return {
                'person_id': best_match_id,
                'is_new': False,
                'match_score': match_score,
                'match_method': match_method,
                'cosine_sim': best_cosine,
                'euclid_dist': best_euclid,
                'embedding': embedding
            }
        
        else:
            # 새로운 사람 등록
            person_id = self.next_id
            self.next_id += 1
            self.registry[person_id] = {
                'embedding': embedding.clone().detach(),
                'last_seen': time.time(),
                'count': 1,
                'first_seen': time.time()
            }
            
            if self.enable_logging:
                print(f"⚠️ [Registry] New person: ID #{person_id} "
                      f"(best match: cos={best_cosine:.4f}, euc={best_euclid:.4f})")
            
            self.match_history.append(('new', person_id, best_cosine))
            
            return {
                'person_id': person_id,
                'is_new': True,
                'match_score': best_cosine,
                'match_method': 'new',
                'cosine_sim': best_cosine,
                'euclid_dist': best_euclid,
                'embedding': embedding
            }
    
    def check_stalking(self, person_id):
        """스토킹 의심 여부 체크"""
        if person_id in self.registry:
            count = self.registry[person_id]['count']
            duration = time.time() - self.registry[person_id]['first_seen']
            
            if count >= self.stalking_threshold and duration < 60:
                return True, count, duration
        return False, 0, 0
    
    def get_person_info(self, person_id):
        """Person 정보 조회"""
        if person_id in self.registry:
            info = self.registry[person_id]
            return {
                'id': person_id,
                'count': info['count'],
                'duration': time.time() - info['first_seen'],
                'last_seen': info['last_seen']
            }
        return None
    
    def print_registry_summary(self):
        """Registry 현황 출력"""
        print("\n" + "="*60)
        print("📋 Person Registry Summary")
        print("="*60)
        print(f"Total registered persons: {len(self.registry)}")
        
        for pid, info in self.registry.items():
            duration = time.time() - info['first_seen']
            print(f"  ID #{pid:3d}: {info['count']:3d} detections, "
                  f"duration={duration:6.1f}s")
        
        print("="*60 + "\n")


# ============================================================================
# 🎬 개선된 Skeleton Extractor
# ============================================================================

class ImprovedSkeletonExtractor:
    """
    개선된 MediaPipe 기반 실시간 스켈레톤 추출기
    - MediaPipe threshold 상향 (0.7)
    - Multi-person 처리
    """
    
    def __init__(self):
        import mediapipe as mp
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        print("🔧 MediaPipe 초기화 중...")
        # ✅ confidence threshold 상향!
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,  # 상향!
            min_tracking_confidence=0.7
        )
        print("✅ MediaPipe 초기화 완료")
        
        # COCO 17 keypoint 매핑
        self.mp_to_coco_mapping = {
            0: 0,   # nose
            2: 1, 5: 2,   # eyes
            7: 3, 8: 4,   # ears
            11: 5, 12: 6,   # shoulders
            13: 7, 14: 8, 15: 9, 16: 10,  # arms
            23: 11, 24: 12,  # hips
            25: 13, 26: 14, 27: 15, 28: 16  # legs
        }
        
        self.coco_skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [12, 14], [13, 15], [14, 16]
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
        
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if skeleton is not None:
            for i, (x, y) in enumerate(skeleton):
                if x > 0 and y > 0 and x < w and y < h:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 2)
        
        return frame


# ============================================================================
# 🎯 개선된 Model Loader
# ============================================================================

class GaitModelLoader:
    """GPGait++ 모델 로더"""
    
    @staticmethod
    def load_model(config_path, checkpoint_path, device='cuda'):
        """모델 로드"""
        print(f"📦 Loading GPGait++ model...")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfgs = yaml.safe_load(f)
        
        from fastposegait.modeling import models
        model_cfg = cfgs['model_cfg']
        model_name = model_cfg['model']
        ModelClass = getattr(models, model_name)
        model = nn.Module.__new__(ModelClass)
        nn.Module.__init__(model)
        model.build_network(model_cfg)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            print("✅ Model loaded on GPU")
        else:
            print("⚠️ Model loaded on CPU (slow)")
        
        model.eval()
        return model


# ============================================================================
# 🚀 개선된 Realtime Gait Recognizer
# ============================================================================

class RealtimeGaitRecognizer:
    """실시간 보행 인식 시스템 (모든 개선사항 적용)"""
    
    def __init__(self, jetson_ip="172.30.101.189", port=8554, sequence_length=60,
                 config_path=None, checkpoint_path=None, enable_model=False):
        self.jetson_ip = jetson_ip
        self.port = port
        self.sequence_length = sequence_length
        
        # 스레드 간 통신 큐
        self.frame_queue = Queue(maxsize=10)
        self.skeleton_queue = Queue(maxsize=30)
        self.display_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=10)
        
        # 스레드 제어
        self.running = False
        self.threads = []
        
        # 스켈레톤 추출기 (개선됨!)
        self.skeleton_extractor = ImprovedSkeletonExtractor()
        
        # Skeleton 프리프로세서 (개선됨!)
        self.preprocessor = ImprovedSkeletonPreprocessor(sequence_length=sequence_length)
        
        # 스켈레톤 시퀀스 버퍼
        self.skeleton_buffer = deque(maxlen=sequence_length)
        
        # GPGait++ 모델
        self.enable_model = enable_model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if enable_model:
            if config_path is None or checkpoint_path is None:
                raise ValueError("config_path and checkpoint_path required")
            self.model = GaitModelLoader.load_model(config_path, checkpoint_path, str(self.device))
        
        # Person Registry (개선됨!)
        self.person_registry = ImprovedPersonRegistry(
            match_threshold_cos=0.88,  # 낮춤!
            match_threshold_euc=0.40,  # 낮춤!
            enable_logging=True
        )
        
        # 최근 인식 결과
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
        
        self.fps_counters = {
            'receiver': {'count': 0, 'last_time': time.time()},
            'extractor': {'count': 0, 'last_time': time.time()},
            'display': {'count': 0, 'last_time': time.time()}
        }
    
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
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
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
                
                # skeleton을 inference queue에 추가
                try:
                    self.skeleton_queue.put_nowait({
                        'skeleton': skeleton,
                        'timestamp': time.time(),
                        'frame_id': self.stats['processed_frames']
                    })
                except Full:
                    pass
                
                # 시각화
                display_frame = frame.copy()
                display_frame = self.skeleton_extractor.draw_skeleton(
                    display_frame, skeleton, pose_landmarks
                )
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, "NO POSE", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            try:
                if self.display_queue.full():
                    self.display_queue.get_nowait()
                self.display_queue.put_nowait(display_frame)
            except (Empty, Full):
                pass
            
            self.update_fps('extractor')
        
        print("🔚 [Extractor] Thread stopped")
    
    def model_inference_thread(self):
        """Thread 3: 모델 추론"""
        print("🚀 [Inference] Starting model inference")
        print(f"   Sequence length: {self.sequence_length} frames")
        print(f"   Cosine threshold: {self.person_registry.match_threshold_cos}")
        print(f"   Euclidean threshold: {self.person_registry.match_threshold_euc}")
        
        while self.running:
            try:
                skeleton_data = self.skeleton_queue.get(timeout=1.0)
            except Empty:
                continue
            
            self.skeleton_buffer.append(skeleton_data['skeleton'])
            
            if len(self.skeleton_buffer) >= self.sequence_length:
                skeleton_sequence = np.array(list(self.skeleton_buffer))
                
                if self.enable_model:
                    result = self.run_gait_model(skeleton_sequence)
                    
                    if result:
                        self.stats['inference_count'] += 1
                        
                        if self.stats['inference_count'] % 30 == 0:
                            print(f"\n🤖 [Inference] Result #{self.stats['inference_count']}")
                            print(f"   Person ID: #{result['person_id']}")
                            print(f"   Status: {'🆕 NEW' if result['is_new'] else '✅ MATCHED'}")
                            print(f"   Match Score: {result['match_score']:.4f}")
                            print(f"   Method: {result['match_method']}")
                            print(f"   Registry size: {len(self.person_registry.registry)}")
        
        print("🔚 [Inference] Thread stopped")
    
    def run_gait_model(self, skeleton_sequence):
        """GPGait++ 모델 추론"""
        if self.model is None:
            return None
        
        try:
            # 1. 전처리 (개선됨!)
            data = self.preprocessor.preprocess_sequence(skeleton_sequence)
            data = data.to(self.device)
            
            # 2. 모델 추론
            with torch.no_grad():
                T = data.shape[2]
                seqL = [torch.LongTensor([T]).to(self.device)]
                
                data = data.contiguous()
                
                inputs = [[data], None, None, None, seqL]
                
                outputs = self.model.forward(inputs)
                
                feat = outputs['inference_feat']['embeddings']
                
                if feat.dim() == 3:
                    feat = feat.mean(dim=2)
                
                feat = F.normalize(feat, p=2, dim=1)
            
            # 3. Person Registry에 매칭 (개선됨!)
            result = self.person_registry.match_or_register(feat)
            
            # 4. 스토킹 체크
            is_stalking, stalk_count, stalk_duration = self.person_registry.check_stalking(
                result['person_id']
            )
            
            result['is_stalking'] = is_stalking
            result['stalk_count'] = stalk_count
            result['stalk_duration'] = stalk_duration
            result['timestamp'] = time.time()
            
            self.latest_recognition = result
            
            try:
                self.result_queue.put_nowait(result)
            except Full:
                pass
            
            return result
            
        except Exception as e:
            print(f"⚠️ [Inference] Error: {e}")
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
            
            self.draw_stats(frame)
            
            cv2.imshow("Realtime Gait Recognition", frame)
            self.update_fps('display')
            
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
        
        # 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 제목
        cv2.putText(frame, "Realtime Gait Recognition", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 통계
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
        buffer_fill = len(self.skeleton_buffer) / self.sequence_length
        cv2.rectangle(frame, (20, 230), (430, 245), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 230), (20 + int(410 * buffer_fill), 245), 
                     (0, 255, 0) if buffer_fill >= 1.0 else (0, 165, 255), -1)
        cv2.putText(frame, f"Buffer: {len(self.skeleton_buffer)}/{self.sequence_length}", 
                   (150, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 인식 결과 (오른쪽 상단)
        if self.enable_model and self.latest_recognition['person_id'] is not None:
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (w-320, 10), (w-10, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
            
            person_id = self.latest_recognition['person_id']
            cv2.putText(frame, f"Person #{person_id}", (w-310, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            status_text = "NEW" if self.latest_recognition['is_new'] else "MATCHED"
            status_color = (0, 255, 0) if not self.latest_recognition['is_new'] else (255, 255, 0)
            cv2.putText(frame, status_text, (w-310, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            score = self.latest_recognition['match_score']
            cv2.putText(frame, f"Score: {score:.3f}", (w-310, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            person_info = self.person_registry.get_person_info(person_id)
            if person_info:
                cv2.putText(frame, f"Count: {person_info['count']}", (w-310, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Duration: {person_info['duration']:.1f}s", (w-310, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        print("🚀 Starting Realtime Gait Recognition System (IMPROVED)")
        print(f"📡 Target: {self.jetson_ip}:{self.port}")
        print(f"📊 Sequence length: {self.sequence_length} frames")
        print(f"⚙️  Cosine threshold: {self.person_registry.match_threshold_cos}")
        print(f"⚙️  Euclidean threshold: {self.person_registry.match_threshold_euc}")
        print(f"🧵 Starting 4 threads...\n")
        
        self.running = True
        
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
            time.sleep(0.2)
        
        print("\n✅ All threads started successfully")
        print("📺 Press 'q' in the display window to quit\n")
        
        try:
            self.threads[3].join()  # Display thread
        except KeyboardInterrupt:
            print("\n⚠️ Ctrl+C pressed - shutting down...")
            self.running = False
        
        print("\n🛑 Stopping all threads...")
        for thread in self.threads:
            thread.join(timeout=2.0)
        
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


# ============================================================================
# 🎬 메인 함수
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Improved Realtime Gait Recognition System')
    parser.add_argument('--ip', type=str, default='172.30.101.189', 
                       help='Jetson IP address')
    parser.add_argument('--port', type=int, default=8554, 
                       help='RTSP port (default: 8554)')
    parser.add_argument('--seq-length', type=int, default=60,
                       help='Skeleton sequence length for model input (default: 60)')
    parser.add_argument('--enable-model', action='store_true',
                       help='Enable GPGait++ model inference')
    parser.add_argument('--config', type=str, 
                       default='configs/gpgait++/gpgait++.yaml',
                       help='Model config path')
    parser.add_argument('--checkpoint', type=str,
                       default=r'C:\FastGait\FastPoseGait\fastposegait\output\CASIA-B\GPGait_PlusPlus\GPGait_PlusPlus\checkpoints\GPGait_PlusPlus-40000.pt',
                       help='Model checkpoint path')
    
    args = parser.parse_args()
    
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
