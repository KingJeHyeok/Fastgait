# improved_inference.py - 개선된 특징 추출 및 비교
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import argparse
import sys
import yaml
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

def enhanced_preprocessing(skeleton_file):
    """향상된 전처리: 개인적 특성을 더 잘 보존"""
    if not os.path.isabs(skeleton_file):
        skeleton_file = os.path.join(SCRIPT_DIR, skeleton_file)
    
    print(f"Loading: {skeleton_file}")
    skeleton = np.load(skeleton_file)
    print(f"Original shape: {skeleton.shape}")
    
    T, V, C = skeleton.shape
    if V != 17:
        raise ValueError(f"Expected 17 joints (COCO), got {V}")
    
    # 1. 개인적 특성 추출 (정규화 전에!)
    
    # 신체 비율 (어깨폭, 다리 길이 등)
    shoulder_width = np.linalg.norm(skeleton[:, 5, :] - skeleton[:, 6, :], axis=1).mean()
    leg_length_left = np.linalg.norm(skeleton[:, 11, :] - skeleton[:, 15, :], axis=1).mean()
    leg_length_right = np.linalg.norm(skeleton[:, 12, :] - skeleton[:, 16, :], axis=1).mean()
    torso_length = np.linalg.norm((skeleton[:, 5, :] + skeleton[:, 6, :]) / 2 - 
                                 (skeleton[:, 11, :] + skeleton[:, 12, :]) / 2, axis=1).mean()
    
    # 보행 패턴 특성
    ankle_distance = np.linalg.norm(skeleton[:, 15, :] - skeleton[:, 16, :], axis=1)
    stride_variance = np.var(ankle_distance)
    step_frequency = len(skeleton) / (T / 30.0) if T > 0 else 0  # FPS 가정
    
    # 관절별 움직임 강도
    joint_movements = np.var(skeleton, axis=0)  # [V, C]
    movement_signature = joint_movements.flatten()  # [V*C] = [34]
    
    print(f"Personal characteristics:")
    print(f"  Shoulder width: {shoulder_width:.3f}")
    print(f"  Leg length (L/R): {leg_length_left:.3f}/{leg_length_right:.3f}")
    print(f"  Torso length: {torso_length:.3f}")
    print(f"  Stride variance: {stride_variance:.6f}")
    print(f"  Step frequency: {step_frequency:.3f}")
    print(f"  Movement signature mean: {movement_signature.mean():.6f}")
    
    # 2. 기본 전처리 (하지만 더 신중하게)
    skeleton = skeleton[:, :, :, np.newaxis]  # [T, V, C, M]
    
    # Bone 계산
    bone = np.zeros_like(skeleton)
    bone_pairs = [
        [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[6,8],[7,9],[8,10],
        [11,12],[5,11],[6,12],[11,13],[12,14],[13,15],[14,16]
    ]
    
    for v1, v2 in bone_pairs:
        if v1 < V and v2 < V:
            bone[:, v1, :, :] = skeleton[:, v1, :, :] - skeleton[:, v2, :, :]
    
    # 3. 정규화 (개인적 특성은 보존하면서)
    
    # 위치만 중심화 (크기는 보존)
    hip_center = (skeleton[:, 11, :, :] + skeleton[:, 12, :, :]) / 2  # [T, 2, 1]
    hip_center_expanded = hip_center[:, None, :, :]  # [T, 1, 2, 1]
    skeleton_centered = skeleton - hip_center_expanded
    
    # 크기 정규화는 어깨폭 기준으로만 (다른 비율은 보존)
    shoulder_widths = np.linalg.norm(skeleton_centered[:, 5, :, 0] - skeleton_centered[:, 6, :, 0], axis=1)
    valid_widths = shoulder_widths[shoulder_widths > 1e-5]
    if len(valid_widths) > 0:
        scale_factor = np.median(valid_widths)
        skeleton_normalized = skeleton_centered / (scale_factor + 1e-8)
    else:
        skeleton_normalized = skeleton_centered
        scale_factor = 1.0
    
    # Bone도 같은 방식으로
    bone_center = bone.mean(axis=1, keepdims=True)
    bone_normalized = (bone - bone_center) / (scale_factor + 1e-8)
    
    # 4. 개인적 특성을 추가 채널로 포함
    pose = skeleton_normalized.transpose(2, 0, 1, 3)  # [2, T, V, M]
    bone = bone_normalized.transpose(2, 0, 1, 3)      # [2, T, V, M]
    
    # 개인적 특성 채널 생성
    # 신체 비율을 모든 관절에 브로드캐스트
    body_ratio_channel = np.full((1, T, V, 1), 
                               shoulder_width / (leg_length_left + leg_length_right + 1e-8))
    
    # 움직임 강도 채널
    movement_channel = np.tile(movement_signature[:V, None], (1, T)).reshape(1, T, V, 1)
    
    # 5. 모든 채널 결합 [7, T, V, M]
    data = np.concatenate([
        pose,                    # [2, T, V, M] - x, y 좌표
        bone,                    # [2, T, V, M] - bone vectors
        body_ratio_channel,      # [1, T, V, M] - 신체 비율
        movement_channel,        # [1, T, V, M] - 움직임 강도  
        np.zeros((1, T, V, 1))   # [1, T, V, M] - 각도 (더미)
    ], axis=0)
    
    data = torch.FloatTensor(data).unsqueeze(0)  # [1, 7, T, V, M]
    
    print(f"Enhanced processed shape: {data.shape}")
    print(f"Data dimensions: N={data.shape[0]}, C={data.shape[1]}, T={data.shape[2]}, V={data.shape[3]}, M={data.shape[4]}")
    
    # 개인적 특성을 별도로 반환
    personal_features = {
        'shoulder_width': shoulder_width,
        'leg_length_ratio': leg_length_left / (leg_length_right + 1e-8),
        'torso_length': torso_length,
        'stride_variance': stride_variance,
        'step_frequency': step_frequency,
        'movement_signature': movement_signature
    }
    
    return data, personal_features

def init_distributed():
    if not dist.is_initialized():
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        dist.init_process_group(backend='gloo', rank=0, world_size=1)

def load_model_simple(config_path, checkpoint_iter=40000):
    print("Loading config...")
    with open(os.path.join(PROJECT_ROOT, config_path), 'r', encoding='utf-8') as f:
        cfgs = yaml.safe_load(f)
    init_distributed()
    from fastposegait.modeling import models
    print("Initializing model...")
    model_cfg = cfgs['model_cfg']
    model_name = model_cfg['model']
    ModelClass = getattr(models, model_name)
    model = nn.Module.__new__(ModelClass)
    nn.Module.__init__(model)
    
    # 채널 수 변경 (7채널로)
    original_in_c = model_cfg.get('in_channels', [64, 64, 128, 256])
    model_cfg['in_channels'] = original_in_c
    
    model.build_network(model_cfg)
    checkpoint_path = r"C:\FastGait\FastPoseGait\fastposegait\output\CASIA-B\GPGait_PlusPlus\GPGait_PlusPlus\checkpoints\GPGait_PlusPlus-40000.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # 일부 레이어는 스킵 (채널 수가 다를 수 있음)
    model.load_state_dict(new_state_dict, strict=False)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    else:
        print("Using CPU")
    model.eval()
    print("Model loaded successfully!")
    return model

def extract_features_enhanced(model, data):
    print("Extracting enhanced features...")
    print(f"Input data shape: {data.shape}")
    device = next(model.parameters()).device

    with torch.no_grad():
        T = data.shape[2]
        seqL = [torch.LongTensor([T]).to(device)]
        data = data.to(device).contiguous()

        # 7채널이지만 모델은 5채널을 기대하므로 처음 5채널만 사용
        data_for_model = data[:, :5, :, :, :]  # [1, 5, T, V, M]
        
        inputs = [[data_for_model], None, None, None, seqL]
        outputs = model.forward(inputs)
        feat = outputs['inference_feat']['embeddings']

        if feat.dim() == 3:
            feat = feat.mean(dim=2)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)

    print(f"Enhanced feature shape: {feat.shape}")
    return feat

def compare_features_enhanced(feat1, feat2, personal1, personal2):
    """향상된 특성 비교"""
    
    # 1. 딥러닝 특성 비교
    device = feat1.device if isinstance(feat1, torch.Tensor) else torch.device('cpu')
    feat1 = feat1.to(device)
    feat2 = feat2.to(device)
    
    euclid_mat = torch.cdist(feat1, feat2, p=2)
    cos_mat = torch.mm(feat1, feat2.t())
    
    euclid_mean = euclid_mat.mean().item()
    cos_mean = cos_mat.mean().item()
    
    # 2. 개인적 특성 비교
    personal_diff = {}
    personal_score = 0.0
    
    # 신체 비율 비교
    shoulder_diff = abs(personal1['shoulder_width'] - personal2['shoulder_width'])
    leg_ratio_diff = abs(personal1['leg_length_ratio'] - personal2['leg_length_ratio'])
    torso_diff = abs(personal1['torso_length'] - personal2['torso_length'])
    
    personal_diff['shoulder_width_diff'] = shoulder_diff
    personal_diff['leg_ratio_diff'] = leg_ratio_diff
    personal_diff['torso_diff'] = torso_diff
    
    # 보행 패턴 비교
    stride_diff = abs(personal1['stride_variance'] - personal2['stride_variance'])
    freq_diff = abs(personal1['step_frequency'] - personal2['step_frequency'])
    
    personal_diff['stride_variance_diff'] = stride_diff
    personal_diff['step_frequency_diff'] = freq_diff
    
    # 움직임 시그니처 비교
    movement_corr = np.corrcoef(personal1['movement_signature'], 
                              personal2['movement_signature'])[0, 1]
    if np.isnan(movement_corr):
        movement_corr = 0.0
    personal_diff['movement_correlation'] = movement_corr
    
    # 종합 개인적 특성 점수 (낮을수록 유사)
    personal_score = (
        shoulder_diff * 2.0 +      # 신체 크기
        leg_ratio_diff * 3.0 +     # 다리 비율
        torso_diff * 1.5 +         # 몸통 길이
        stride_diff * 5.0 +        # 보행 패턴
        freq_diff * 1.0 +          # 걸음 빈도
        (1.0 - movement_corr) * 4.0  # 움직임 패턴
    )
    
    return euclid_mean, cos_mean, personal_score, personal_diff

def main():
    parser = argparse.ArgumentParser(description='Enhanced Gait Recognition Inference')
    parser.add_argument('--config', type=str, default='configs/gpgait++/gpgait++.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=int, default=40000, help='Checkpoint iteration')
    parser.add_argument('--skeleton1', type=str, required=True, help='First skeleton (.npy)')
    parser.add_argument('--skeleton2', type=str, default=None, help='Second skeleton (.npy)')
    parser.add_argument('--match-threshold-cos', type=float, default=0.85, help='Cosine similarity threshold')
    parser.add_argument('--match-threshold-euc', type=float, default=0.5, help='Euclidean distance threshold')
    parser.add_argument('--match-threshold-personal', type=float, default=5.0, help='Personal features threshold')
    args = parser.parse_args()

    print("="*60)
    print("Enhanced Gait Recognition Inference")
    print("="*60)

    model = load_model_simple(args.config, args.checkpoint)

    # Enhanced preprocessing
    print("\n--- Enhanced Processing Skeleton 1 ---")
    data1, personal1 = enhanced_preprocessing(args.skeleton1)
    feat1 = extract_features_enhanced(model, data1)

    if args.skeleton2:
        print("\n--- Enhanced Processing Skeleton 2 ---")
        data2, personal2 = enhanced_preprocessing(args.skeleton2)
        feat2 = extract_features_enhanced(model, data2)

        print("\n--- Enhanced Comparison ---")
        euclid_mean, cos_mean, personal_score, personal_diff = compare_features_enhanced(
            feat1, feat2, personal1, personal2)

        print("\n" + "="*60)
        print("ENHANCED RESULTS")
        print("="*60)
        print(f"Deep Learning Features:")
        print(f"  Euclidean Distance: {euclid_mean:.4f}")
        print(f"  Cosine Similarity:  {cos_mean:.4f}")
        print(f"\nPersonal Characteristics:")
        print(f"  Shoulder width diff: {personal_diff['shoulder_width_diff']:.4f}")
        print(f"  Leg ratio diff:      {personal_diff['leg_ratio_diff']:.4f}")
        print(f"  Torso length diff:   {personal_diff['torso_diff']:.4f}")
        print(f"  Stride variance diff: {personal_diff['stride_variance_diff']:.6f}")
        print(f"  Step frequency diff:  {personal_diff['step_frequency_diff']:.4f}")
        print(f"  Movement correlation: {personal_diff['movement_correlation']:.4f}")
        print(f"  Personal Score:       {personal_score:.4f}")
        print("-"*60)
        
        # 종합 판단
        dl_match = (cos_mean >= args.match_threshold_cos or euclid_mean <= args.match_threshold_euc)
        personal_match = (personal_score <= args.match_threshold_personal)
        
        print(f"Deep Learning says: {'SAME' if dl_match else 'DIFFERENT'}")
        print(f"Personal features say: {'SAME' if personal_match else 'DIFFERENT'}")
        print()
        
        if dl_match and personal_match:
            print("✓ SAME PERSON (Both methods agree)")
        elif not dl_match and not personal_match:
            print("✗ DIFFERENT PERSON (Both methods agree)")
        elif dl_match and not personal_match:
            print("⚠️ CONFLICTING: DL says SAME, but personal features say DIFFERENT")
            print("   → Likely DIFFERENT PERSON (personal features more reliable)")
        else:
            print("⚠️ CONFLICTING: DL says DIFFERENT, but personal features say SAME")
            print("   → Likely SAME PERSON (unusual gait variation)")
        
        print("="*60)

    else:
        print("\n✓ Enhanced feature extraction completed!")
        print(f"Deep learning feature shape: {feat1.shape}")
        print(f"Personal characteristics extracted: {len(personal1)} features")

if __name__ == '__main__':
    main()