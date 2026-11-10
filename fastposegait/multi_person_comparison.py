# multi_person_comparison.py - 다중 인물 비교 시스템
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import argparse
import sys
import yaml
import torch.nn.functional as F
from itertools import combinations
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

def init_distributed():
    if not dist.is_initialized():
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

def load_model_simple(config_path, checkpoint_iter=40000):
    print("Loading model...")
    with open(os.path.join(PROJECT_ROOT, config_path), 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)
    init_distributed()
    from fastposegait.modeling import models
    
    model_cfg = cfgs['model_cfg']
    model_name = model_cfg['model']
    ModelClass = getattr(models, model_name)
    model = nn.Module.__new__(ModelClass)
    nn.Module.__init__(model)
    model.build_network(model_cfg)
    
    checkpoint_path = r"C:\FastGait\FastPoseGait\fastposegait\output\CASIA-B\GPGait_PlusPlus\GPGait_PlusPlus\checkpoints\GPGait_PlusPlus-40000.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("Using CPU")
    model.eval()
    return model

def preprocess_skeleton(skeleton_file):
    if not os.path.isabs(skeleton_file):
        skeleton_file = os.path.join(SCRIPT_DIR, skeleton_file)
    
    skeleton = np.load(skeleton_file)
    T, V, C = skeleton.shape
    if V != 17:
        raise ValueError(f"Expected 17 joints (COCO), got {V}")
    
    skeleton = skeleton[:, :, :, np.newaxis]
    bone = np.zeros_like(skeleton)
    bone_pairs = [
        [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[6,8],[7,9],[8,10],
        [11,12],[5,11],[6,12],[11,13],[12,14],[13,15],[14,16]
    ]
    
    for v1, v2 in bone_pairs:
        if v1 < V and v2 < V:
            bone[:, v1, :, :] = skeleton[:, v1, :, :] - skeleton[:, v2, :, :]
    
    center = skeleton.mean(axis=1, keepdims=True)
    skeleton = skeleton - center
    bone = bone - bone.mean(axis=1, keepdims=True)
    
    pose = skeleton.transpose(2, 0, 1, 3)
    bone = bone.transpose(2, 0, 1, 3)
    angle = np.zeros((1, T, 17, 1))
    data = np.concatenate([pose, bone, angle], axis=0)
    data = torch.FloatTensor(data).unsqueeze(0)
    
    return data

def extract_features(model, data):
    device = next(model.parameters()).device
    with torch.no_grad():
        T = data.shape[2]
        seqL = [torch.LongTensor([T]).to(device)]
        data = data.to(device).contiguous()
        inputs = [[data], None, None, None, seqL]
        outputs = model.forward(inputs)
        feat = outputs['inference_feat']['embeddings']
        if feat.dim() == 3:
            feat = feat.mean(dim=2)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
    return feat

def statistical_analysis(skeleton):
    """개인 특성 분석"""
    def get_measurements(skel):
        shoulder_width = np.mean(np.linalg.norm(skel[:, 5, :] - skel[:, 6, :], axis=1))
        left_leg = np.mean(np.linalg.norm(skel[:, 11, :] - skel[:, 15, :], axis=1))
        right_leg = np.mean(np.linalg.norm(skel[:, 12, :] - skel[:, 16, :], axis=1))
        torso_length = np.mean(np.linalg.norm(
            (skel[:, 5, :] + skel[:, 6, :]) / 2 - 
            (skel[:, 11, :] + skel[:, 12, :]) / 2, axis=1))
        
        # 보행 패턴
        ankle_distances = np.linalg.norm(skel[:, 15, :] - skel[:, 16, :], axis=1)
        stride_variance = np.var(ankle_distances)
        stride_mean = np.mean(ankle_distances)
        
        # 움직임 패턴
        joint_movements = np.var(skel, axis=0)
        total_movement = np.mean(joint_movements)
        
        return {
            'shoulder_width': shoulder_width,
            'left_leg': left_leg,
            'right_leg': right_leg,
            'leg_ratio': left_leg / (right_leg + 1e-8),
            'torso_length': torso_length,
            'stride_variance': stride_variance,
            'stride_mean': stride_mean,
            'total_movement': total_movement
        }
    
    return get_measurements(skeleton)

def compare_statistical_features(stats1, stats2):
    """통계적 특성 비교"""
    # 체형 차이
    shoulder_diff = abs(stats1['shoulder_width'] - stats2['shoulder_width'])
    leg_diff = abs(stats1['left_leg'] - stats2['left_leg'])
    torso_diff = abs(stats1['torso_length'] - stats2['torso_length'])
    
    # 보행 패턴 차이
    stride_var_diff = abs(stats1['stride_variance'] - stats2['stride_variance'])
    stride_mean_diff = abs(stats1['stride_mean'] - stats2['stride_mean'])
    movement_diff = abs(stats1['total_movement'] - stats2['total_movement'])
    
    # 종합 점수 (낮을수록 유사)
    body_score = (shoulder_diff + leg_diff + torso_diff) / 3
    movement_score = (stride_var_diff + movement_diff) / 2
    
    return {
        'body_score': body_score,
        'movement_score': movement_score,
        'total_score': body_score * 0.6 + movement_score * 0.4,
        'details': {
            'shoulder_diff': shoulder_diff,
            'leg_diff': leg_diff,
            'torso_diff': torso_diff,
            'stride_var_diff': stride_var_diff,
            'movement_diff': movement_diff
        }
    }

def compare_deep_learning_features(feat1, feat2):
    """딥러닝 특성 비교"""
    device = feat1.device
    feat1 = feat1.to(device)
    feat2 = feat2.to(device)
    
    euclidean_dist = torch.cdist(feat1, feat2, p=2).mean().item()
    cosine_sim = torch.mm(feat1, feat2.t()).mean().item()
    
    return {
        'euclidean_distance': euclidean_dist,
        'cosine_similarity': cosine_sim
    }

def analyze_person(person_name, skeleton_file, model):
    """개인 분석"""
    print(f"\n--- Analyzing {person_name} ({skeleton_file}) ---")
    
    # 스켈레톤 로드
    skeleton_raw = np.load(skeleton_file)
    print(f"Shape: {skeleton_raw.shape}")
    
    # 통계적 특성
    stats = statistical_analysis(skeleton_raw)
    print(f"Body: Shoulder={stats['shoulder_width']:.1f}, Leg={stats['left_leg']:.1f}, Torso={stats['torso_length']:.1f}")
    print(f"Gait: Stride_var={stats['stride_variance']:.1f}, Movement={stats['total_movement']:.1f}")
    
    # 딥러닝 특성
    data = preprocess_skeleton(skeleton_file)
    features = extract_features(model, data)
    
    return {
        'name': person_name,
        'file': skeleton_file,
        'stats': stats,
        'features': features,
        'raw_skeleton': skeleton_raw
    }

def find_similar_people(people_data, stat_threshold=50.0, dl_cos_threshold=0.85):
    """유사한 사람들 찾기"""
    
    print("\n" + "="*80)
    print("PAIRWISE COMPARISON MATRIX")
    print("="*80)
    
    n = len(people_data)
    comparison_matrix = []
    
    # 헤더 출력
    print(f"{'':12}", end="")
    for person in people_data:
        print(f"{person['name']:12}", end="")
    print()
    
    for i in range(n):
        row_data = []
        print(f"{people_data[i]['name']:12}", end="")
        
        for j in range(n):
            if i == j:
                print(f"{'SELF':12}", end="")
                row_data.append({'type': 'self'})
                continue
                
            # 통계적 비교
            stat_comp = compare_statistical_features(
                people_data[i]['stats'], 
                people_data[j]['stats']
            )
            
            # 딥러닝 비교
            dl_comp = compare_deep_learning_features(
                people_data[i]['features'], 
                people_data[j]['features']
            )
            
            # 결과 저장
            comparison = {
                'person1': people_data[i]['name'],
                'person2': people_data[j]['name'],
                'statistical': stat_comp,
                'deep_learning': dl_comp
            }
            row_data.append(comparison)
            
            # 판단
            stat_similar = stat_comp['total_score'] < stat_threshold
            dl_similar = dl_comp['cosine_similarity'] > dl_cos_threshold
            
            if stat_similar and dl_similar:
                result = "SAME✓"
            elif stat_similar:
                result = "STAT✓"
            elif dl_similar:
                result = "DL✓"
            else:
                result = "DIFF✗"
            
            print(f"{result:12}", end="")
        
        comparison_matrix.append(row_data)
        print()
    
    return comparison_matrix

def generate_detailed_report(people_data, comparison_matrix):
    """상세 리포트 생성"""
    
    print("\n" + "="*80)
    print("DETAILED SIMILARITY ANALYSIS")
    print("="*80)
    
    similar_pairs = []
    
    for i in range(len(people_data)):
        for j in range(i+1, len(people_data)):
            comp = comparison_matrix[i][j]
            if 'type' in comp and comp['type'] == 'self':
                continue
                
            stat_comp = comp['statistical']
            dl_comp = comp['deep_learning']
            
            print(f"\n{comp['person1']} vs {comp['person2']}:")
            print(f"  Statistical Score: {stat_comp['total_score']:.2f}")
            print(f"    - Body differences: {stat_comp['body_score']:.2f}")
            print(f"    - Movement differences: {stat_comp['movement_score']:.2f}")
            print(f"  Deep Learning:")
            print(f"    - Cosine similarity: {dl_comp['cosine_similarity']:.4f}")
            print(f"    - Euclidean distance: {dl_comp['euclidean_distance']:.4f}")
            
            # 종합 판단
            stat_similar = stat_comp['total_score'] < 50.0
            dl_similar = dl_comp['cosine_similarity'] > 0.85
            
            if stat_similar and dl_similar:
                print(f"  🟢 CONCLUSION: LIKELY SAME PERSON")
                similar_pairs.append((comp['person1'], comp['person2'], 'HIGH'))
            elif stat_similar:
                print(f"  🟡 CONCLUSION: SIMILAR BODY (different gait pattern)")
                similar_pairs.append((comp['person1'], comp['person2'], 'MEDIUM'))
            elif dl_similar:
                print(f"  🟡 CONCLUSION: SIMILAR GAIT (different body)")
            else:
                print(f"  🔴 CONCLUSION: DIFFERENT PERSON")
            
            print("-" * 60)
    
    return similar_pairs

def main():
    parser = argparse.ArgumentParser(description='Multi-Person Gait Comparison')
    parser.add_argument('--skeletons', type=str, nargs='+', required=True,
                       help='Skeleton files to compare')
    parser.add_argument('--names', type=str, nargs='+', required=True,
                       help='Names for each skeleton file')
    parser.add_argument('--config', type=str, default='configs/gpgait++/gpgait++.yaml',
                       help='Model config file')
    parser.add_argument('--stat-threshold', type=float, default=50.0,
                       help='Statistical similarity threshold')
    parser.add_argument('--cos-threshold', type=float, default=0.85,
                       help='Cosine similarity threshold')
    parser.add_argument('--save-report', type=str, default=None,
                       help='Save detailed report to JSON file')
    
    args = parser.parse_args()
    
    if len(args.skeletons) != len(args.names):
        raise ValueError("Number of skeleton files must match number of names")
    
    print("="*80)
    print("MULTI-PERSON GAIT RECOGNITION ANALYSIS")
    print("="*80)
    print(f"Analyzing {len(args.skeletons)} people:")
    for i, (name, file) in enumerate(zip(args.names, args.skeletons)):
        print(f"  {i+1}. {name}: {file}")
    
    # 모델 로드
    model = load_model_simple(args.config)
    print("✓ Model loaded successfully")
    
    # 각 인물 분석
    people_data = []
    for name, skeleton_file in zip(args.names, args.skeletons):
        person_data = analyze_person(name, skeleton_file, model)
        people_data.append(person_data)
    
    # 비교 매트릭스 생성
    comparison_matrix = find_similar_people(
        people_data, 
        stat_threshold=args.stat_threshold,
        dl_cos_threshold=args.cos_threshold
    )
    
    # 상세 리포트
    similar_pairs = generate_detailed_report(people_data, comparison_matrix)
    
    # 최종 결과
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if similar_pairs:
        print("🔍 POTENTIALLY SAME PEOPLE:")
        for person1, person2, confidence in similar_pairs:
            print(f"  • {person1} ⟷ {person2} (Confidence: {confidence})")
    else:
        print("✅ ALL PEOPLE ARE DIFFERENT")
    
    # JSON 저장
    if args.save_report:
        report_data = {
            'people': [{'name': p['name'], 'file': p['file'], 'stats': p['stats']} for p in people_data],
            'similar_pairs': similar_pairs,
            'thresholds': {
                'statistical': args.stat_threshold,
                'cosine': args.cos_threshold
            }
        }
        
        with open(args.save_report, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Detailed report saved to: {args.save_report}")
    
    print("="*80)

if __name__ == '__main__':
    main()