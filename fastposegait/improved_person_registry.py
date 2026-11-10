# improved_person_registry.py - 개선된 인물 등록 시스템
"""
다중 임베딩 비교 + 엄격한 검증으로 오인식 방지
"""

import torch
import torch.nn.functional as F
import time
from collections import deque

class ImprovedPersonRegistry:
    """개선된 Person Registry - 다른 사람 구분 강화"""
    
    def __init__(self,
                 match_threshold_cos_strict=0.92,   # 엄격 (AND 조건)
                 match_threshold_euc_strict=0.35,
                 match_threshold_cos_relaxed=0.88,  # 완화 (OR 조건)
                 match_threshold_euc_relaxed=0.45,
                 max_embeddings=10,
                 ema_alpha=0.3):
        
        print("[ImprovedPersonRegistry] __init__ started")
        # ====== 엄격한 임계값 (AND 조건 - 둘 다 만족) ======
        self.threshold_cos_strict = match_threshold_cos_strict
        self.threshold_euc_strict = match_threshold_euc_strict
        print("[ImprovedPersonRegistry] Strict thresholds set")
        
        # ====== 완화된 임계값 (OR 조건 - 하나만 만족) ======
        self.threshold_cos_relaxed = match_threshold_cos_relaxed
        self.threshold_euc_relaxed = match_threshold_euc_relaxed
        print("[ImprovedPersonRegistry] Relaxed thresholds set")
        
        self.max_embeddings = max_embeddings
        self.ema_alpha = ema_alpha
        print("[ImprovedPersonRegistry] Max embeddings and EMA alpha set")
        
        # Person 데이터베이스
        self.registry = {}
        self.next_id = 1
        print("[ImprovedPersonRegistry] __init__ completed")
    
    def match_or_register(self, feat, frame_count=120):
        """
        다중 임베딩 비교 + 2단계 검증
        
        Args:
            feat: Feature embedding
            frame_count: 수집된 프레임 수 (검증 강도 결정)
        
        Returns:
            person_id, is_new, match_score
        """
        
        if len(self.registry) == 0:
            # 첫 사람
            pid = self._register_new_person(feat)
            return pid, True, 1.0, 1.0, 0.0
        
        # ====== 모든 등록된 사람과 비교 ======
        candidates = []
        
        # feat을 CPU로 이동 (DB는 CPU에 저장됨)
        feat_cpu = feat.detach().cpu()
        
        for pid, data in self.registry.items():
            # 다중 임베딩 비교
            max_cos = -1.0
            min_euc = float('inf')
            avg_cos = 0.0
            avg_euc = 0.0
            
            embeddings_list = data.get('embeddings_list', [data['feature']])
            
            for stored_emb in embeddings_list:
                # CPU 기준으로 비교
                stored_cpu = stored_emb.cpu() if stored_emb.is_cuda else stored_emb
                
                cos_sim = F.cosine_similarity(feat_cpu, stored_cpu, dim=1).item()
                euc_dist = torch.norm(feat_cpu - stored_cpu, p=2).item()
                
                max_cos = max(max_cos, cos_sim)
                min_euc = min(min_euc, euc_dist)
                avg_cos += cos_sim
                avg_euc += euc_dist
            
            avg_cos /= len(embeddings_list)
            avg_euc /= len(embeddings_list)
            
            # 복합 점수 (최대값 + 평균값)
            composite_score = (max_cos * 0.7 + avg_cos * 0.3) - (min_euc * 0.5 + avg_euc * 0.5) * 0.3
            
            candidates.append({
                'pid': pid,
                'max_cos': max_cos,
                'min_euc': min_euc,
                'avg_cos': avg_cos,
                'avg_euc': avg_euc,
                'composite_score': composite_score,
                'embedding_count': len(embeddings_list)
            })
        
        # 후보 정렬 (composite_score 기준)
        candidates.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # 상위 3개 후보 출력
        print(f"\n   [Match] 상위 후보들:")
        for i, cand in enumerate(candidates[:3]):
            print(f"      #{i+1}. Person #{cand['pid']}: Cos={cand['max_cos']:.4f}, Euc={cand['min_euc']:.4f}, Score={cand['composite_score']:.4f}")
        
        best = candidates[0]
        
        # ====== 2단계 검증 ======
        
        # 항상 엄격한 검증 사용 (다른 사람 구분 강화)
        use_strict = True
        threshold_cos = self.threshold_cos_strict
        threshold_euc = self.threshold_euc_strict
        print(f"      Mode: STRICT (frames: {frame_count})")
        print(f"      Thresholds: Cos≥{threshold_cos:.2f} AND Euc≤{threshold_euc:.2f}")
        
        # AND 조건 (둘 다 만족해야 매칭)
        cos_pass = best['max_cos'] >= threshold_cos
        euc_pass = best['min_euc'] <= threshold_euc
        
        print(f"      Best: Person #{best['pid']}")
        print(f"      ✓ Cos: {best['max_cos']:.4f} >= {threshold_cos:.2f} ? {cos_pass}")
        print(f"      ✓ Euc: {best['min_euc']:.4f} <= {threshold_euc:.2f} ? {euc_pass}")
        
        if cos_pass and euc_pass:
            # ✅ 매칭 성공
            print(f"      ✅ MATCHED with Person #{best['pid']}")
            self._update_person(best['pid'], feat)
            return best['pid'], False, best['composite_score'], best['max_cos'], best['min_euc']
        else:
            # ❌ 매칭 실패 - 새 사람
            print(f"      ❌ NEW PERSON (threshold not met)")
            print(f"         → Person #{best['pid']}가 가장 가깝지만 임계값 미달")
            pid = self._register_new_person(feat)
            # NEW 등록 시에도 best 후보의 유사도 반환 (디버깅용)
            return pid, True, best['composite_score'], best['max_cos'], best['min_euc']
    
    def _register_new_person(self, feat):
        """새 사람 등록"""
        pid = self.next_id
        self.next_id += 1
        
        # CPU로 이동 후 저장 (GPU 메모리 누수 방지)
        feat_cpu = feat.detach().cpu()
        
        self.registry[pid] = {
            'feature': feat_cpu.clone(),
            'embeddings_list': [feat_cpu.clone()],
            'count': 1,
            'first_seen': time.time(),
            'last_seen': time.time(),
            'total_frames': 0
        }
        
        print(f"   ✨ [Registry] New person registered: #{pid}")
        print(f"      Embedding shape: {feat_cpu.shape}, Device: {feat_cpu.device}")
        return pid
    
    def _update_person(self, pid, feat):
        """기존 사람 업데이트"""
        data = self.registry[pid]
        
        # CPU로 이동
        feat_cpu = feat.detach().cpu()
        feature_cpu = data['feature'].cpu() if data['feature'].is_cuda else data['feature']
        
        # EMA 업데이트
        data['feature'] = (1 - self.ema_alpha) * feature_cpu + self.ema_alpha * feat_cpu
        
        # 임베딩 리스트 추가
        if len(data['embeddings_list']) < self.max_embeddings:
            data['embeddings_list'].append(feat_cpu.clone())
        else:
            # 가장 오래된 것 교체
            data['embeddings_list'].pop(0)
            data['embeddings_list'].append(feat_cpu.clone())
        
        data['count'] += 1
        data['last_seen'] = time.time()
        
        print(f"   🔄 [Registry] Updated Person #{pid}")
        print(f"      Embeddings: {len(data['embeddings_list'])}/{self.max_embeddings}")
    
    def force_update_person(self, pid, feat):
        """강제 업데이트 (세션 잠금 시) - 실제 유사도 계산"""
        if pid not in self.registry:
            print(f"   ⚠️ Person #{pid} not in registry - registering as new")
            new_pid = self._register_new_person(feat)
            return new_pid, True, 1.0, 1.0, 0.0
        
        # ====== 실제 유사도 계산 (검증용) ======
        data = self.registry[pid]
        embeddings_list = data.get('embeddings_list', [data['feature']])
        
        max_cos = -1.0
        min_euc = float('inf')
        
        for stored_emb in embeddings_list:
            cos_sim = F.cosine_similarity(feat, stored_emb, dim=1).item()
            euc_dist = torch.norm(feat - stored_emb, p=2).item()
            
            max_cos = max(max_cos, cos_sim)
            min_euc = min(min_euc, euc_dist)
        
        # 복합 점수
        composite_score = max_cos * 0.8 - min_euc * 0.2
        
        # ⚠️ 경고: 유사도가 너무 낮으면 잘못된 매칭일 수 있음
        if max_cos < 0.85:
            print(f"   ⚠️ [Force Update] Low similarity detected!")
            print(f"      Person #{pid} | Cos: {max_cos:.4f} | Euc: {min_euc:.4f}")
            print(f"      This might be a DIFFERENT person!")
        
        # 임베딩 업데이트
        self._update_person(pid, feat)
        
        return pid, False, composite_score, max_cos, min_euc
    
    def get_person_info(self, pid):
        """인물 정보 조회"""
        if pid not in self.registry:
            return None
        
        data = self.registry[pid]
        return {
            'count': data['count'],
            'duration': time.time() - data['first_seen'],
            'embeddings': len(data['embeddings_list'])
        }


# ============================================
# scenario_tester.py에 통합하는 방법
# ============================================

"""
scenario_tester.py 수정:

1. PersonRegistry import 변경:
   from improved_person_registry import ImprovedPersonRegistry as PersonRegistry

2. 초기화 시 임계값 설정:
   self.person_registry = PersonRegistry(
       match_threshold_cos_strict=0.92,
       match_threshold_euc_strict=0.35,
       match_threshold_cos_relaxed=0.88,
       match_threshold_euc_relaxed=0.45,
       max_embeddings=10
   )

3. 추론 시 프레임 수 전달:
   person_id, is_new, match_score = self.person_registry.match_or_register(
       feat, 
       frame_count=len(skeleton_sequence)
   )
"""


# ============================================
# 테스트 시나리오 예상 결과
# ============================================

"""
[개선 전]
Scenario 1 (Person A):
  Person #1 (Frame 0-651)   ✅
  Person #2 (Frame 675-795) ❌ 같은 사람인데 분리

Scenario 2 (Person B):
  Person #1 (모든 세션)     ❌ 다른 사람인데 같은 ID

[개선 후]
Scenario 1 (Person A):
  Person #1 (모든 세션)     ✅ 올바르게 통합

Scenario 2 (Person B):
  Person #2 (모든 세션)     ✅ 다른 사람으로 구분

예상 임계값 효과:
- 0.92/0.35 (Strict): 같은 사람 재등장 시 통과
- 다른 사람 (0.9578): 0.92 미달로 거부
"""
