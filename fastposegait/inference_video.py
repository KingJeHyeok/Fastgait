# inference_video.py (updated for stable embeddings + simple registry)
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
def skeleton_to_gpgait_input(skel):
    # skel: [T, 17, 2]

    T, V, _ = skel.shape

    # 1) velocity (joint movement)
    velocity = np.diff(skel, axis=0, prepend=skel[0:1])  # [T,17,2]

    # 2) bone vectors & lengths
    bone_pairs = [
        (11, 13), (13, 15),   # left hip-knee / knee-ankle
        (12, 14), (14, 16)    # right hip-knee / knee-ankle
    ]
    bone_vecs = []
    for a, b in bone_pairs:
        bone_vecs.append(skel[:, b] - skel[:, a])
    bone_vecs = np.stack(bone_vecs, axis=1)  # [T,4,2]
    bone_lengths = np.linalg.norm(bone_vecs, axis=2)  # [T,4]

    # 3) bone angles (rotation signature)
    bone_angles = np.arctan2(bone_vecs[..., 1], bone_vecs[..., 0])  # [T,4]

    # 4) expand bone info to joints (broadcast)
    bone_lengths_exp = np.repeat(bone_lengths.mean(axis=1, keepdims=True), V, axis=1)  # [T,17]
    bone_angles_exp = np.repeat(bone_angles.mean(axis=1, keepdims=True), V, axis=1)    # [T,17]

    x = skel[:, :, 0]
    y = skel[:, :, 1]
    vx = velocity[:, :, 0]
    vy = velocity[:, :, 1]

    # stack into C=5 channel format
    data = np.stack([x, y, vx, vy, bone_angles_exp], axis=0)  # [5,T,V]

    # add person dimension (M=1)
    data = data[..., None]  # [5,T,V,1]

    return data  # final: [C=5, T, V, M]


def init_distributed():
    """분산 환경 초기화 (에러 처리 강화)"""
    if not dist.is_initialized():
        try:
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')  # localhost 대신 127.0.0.1
            os.environ.setdefault('MASTER_PORT', '12355')
            
            # Windows에서 gloo 초기화
            dist.init_process_group(
                backend='gloo',
                init_method='tcp://127.0.0.1:12355',
                rank=0,
                world_size=1
            )
            print("Distributed environment initialized")
        except Exception as e:
            print(f"Warning: Could not initialize distributed backend: {e}")
            print("Continuing without distributed initialization...")
            # 분산 환경 없이도 작동하도록 더미 설정
            pass

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
    model.load_state_dict(new_state_dict, strict=False)
    
    # GPU 강제 사용
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    else:
        # CPU 사용 시 경고
        print("⚠ WARNING: No GPU detected. This model requires GPU.")
        print("Please install CUDA or use a machine with GPU.")
        raise RuntimeError("GPU required for this model")
    
    model.eval()
    print("Model loaded successfully!")
    return model

def preprocess_skeleton(skeleton_file):
    if not os.path.isabs(skeleton_file):
        skeleton_file = os.path.join(SCRIPT_DIR, skeleton_file)
    print(f"Loading: {skeleton_file}")
    skeleton = np.load(skeleton_file)
    print(f"Original shape: {skeleton.shape}")
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
    pose = skeleton.transpose(2, 0, 1, 3)  # [2, T, V, M]
    bone = bone.transpose(2, 0, 1, 3)
    angle = np.zeros((1, T, 17, 1))
    data = np.concatenate([pose, bone, angle], axis=0)  # [5, T, V, M]
    data = torch.FloatTensor(data).unsqueeze(0)  # [1,5,T,V,M]
    print(f"Processed shape: {data.shape}")
    print(f"Data dimensions: N={data.shape[0]}, C={data.shape[1]}, T={data.shape[2]}, V={data.shape[3]}, M={data.shape[4]}")
    return data

def extract_features(model, data):
    print("Extracting features...")
    print(f"Input data shape: {data.shape}")
    device = next(model.parameters()).device

    with torch.no_grad():
        T = data.shape[2]

        # ✅ seqL must be a list
        seqL = [torch.LongTensor([T]).to(device)]

        # ✅ keep batch dimension & contiguous
        data = data.to(device).contiguous()   # [1, 5, T, V, M]

        # ✅ correct GPGait++ input structure (LIST inside LIST)
        inputs = [[data], None, None, None, seqL]
        #             ↑     pose input list

        outputs = model.forward(inputs)

        feat = outputs['inference_feat']['embeddings']  # [N, D, S] or [N, D]

        # ✅ temporal pooling
        if feat.dim() == 3:
            feat = feat.mean(dim=2)  # [N, D]

        # ✅ L2 normalize for cosine stability
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)

    print(f"Feature shape (post-agg & normalized): {feat.shape}")
    return feat


def statistical_precheck(skeleton1, skeleton2):
    """통계적 사전 검증"""
    def get_body_measurements(skeleton):
        shoulder_width = np.mean(np.linalg.norm(skeleton[:, 5, :] - skeleton[:, 6, :], axis=1))
        left_leg = np.mean(np.linalg.norm(skeleton[:, 11, :] - skeleton[:, 15, :], axis=1))
        torso_length = np.mean(np.linalg.norm(
            (skeleton[:, 5, :] + skeleton[:, 6, :]) / 2 - 
            (skeleton[:, 11, :] + skeleton[:, 12, :]) / 2, axis=1))
        stride_variance = np.var(np.linalg.norm(skeleton[:, 15, :] - skeleton[:, 16, :], axis=1))
        return shoulder_width, left_leg, torso_length, stride_variance
    
    s1_shoulder, s1_leg, s1_torso, s1_stride = get_body_measurements(skeleton1)
    s2_shoulder, s2_leg, s2_torso, s2_stride = get_body_measurements(skeleton2)
    
    # 차이 계산
    shoulder_diff = abs(s1_shoulder - s2_shoulder)
    leg_diff = abs(s1_leg - s2_leg)
    torso_diff = abs(s1_torso - s2_torso)
    stride_diff = abs(s1_stride - s2_stride)
    
    # 임계값 기반 판단
    body_different = (shoulder_diff > 10.0 or leg_diff > 10.0 or torso_diff > 10.0)
    movement_different = stride_diff > 100.0
    
    statistical_confidence = "HIGH" if (body_different and movement_different) else \
                           "MEDIUM" if (body_different or movement_different) else "LOW"
    
    return {
        'body_different': body_different,
        'movement_different': movement_different,
        'confidence': statistical_confidence,
        'differences': {
            'shoulder': shoulder_diff,
            'leg': leg_diff,
            'torso': torso_diff,
            'stride': stride_diff
        }
    }

def compare_features(feat1, feat2):
    """
    feat1, feat2: tensors of shape [N1, D], [N2, D]
    Returns: (euclidean_mean, cosine_mean, detailed_matrix)
    """
    # ensure float tensors on same device
    device = feat1.device if isinstance(feat1, torch.Tensor) else torch.device('cpu')
    feat1 = feat1.to(device)
    feat2 = feat2.to(device)
    # Euclidean distance matrix
    # Using pairwise cdist
    euclid_mat = torch.cdist(feat1, feat2, p=2)  # [N1, N2]
    # Cosine similarity matrix: since normalized, dot product == cosine
    cos_mat = torch.mm(feat1, feat2.t())  # [N1, N2]
    # aggregate statistics
    euclid_mean = euclid_mat.mean().item()
    cos_mean = cos_mat.mean().item()
    return euclid_mean, cos_mean, {'euclid_mat': euclid_mat, 'cos_mat': cos_mat}

def save_embedding(name, feat, registry_dir="registry"):
    os.makedirs(registry_dir, exist_ok=True)
    path = os.path.join(registry_dir, f"{name}.npy")
    # feat is tensor [D] or [1,D]
    if isinstance(feat, torch.Tensor):
        arr = feat.cpu().numpy()
    else:
        arr = np.array(feat)
    # ensure 1D
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    np.save(path, arr)
    print(f"Saved embedding to {path}")

def load_registry(registry_dir="registry"):
    registry = {}
    if not os.path.exists(registry_dir):
        return registry
    for fn in os.listdir(registry_dir):
        if fn.endswith(".npy"):
            name = os.path.splitext(fn)[0]
            arr = np.load(os.path.join(registry_dir, fn))
            # normalize in case
            tensor = torch.from_numpy(arr).float()
            tensor = F.normalize(tensor.unsqueeze(0), p=2, dim=1)  # [1, D]
            registry[name] = tensor
    return registry

def main():
    parser = argparse.ArgumentParser(description='Gait Recognition Inference')
    parser.add_argument('--config', type=str, default='configs/gpgait++/gpgait++.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=int, default=40000, help='Checkpoint iteration')
    parser.add_argument('--skeleton1', type=str, required=True, help='First skeleton (.npy)')
    parser.add_argument('--skeleton2', type=str, default=None, help='Second skeleton (.npy)')
    parser.add_argument('--register', type=str, default=None, help='Register this skeleton under NAME (save embedding)')
    parser.add_argument('--registry-dir', type=str, default='registry', help='Directory to store/load registered embeddings')
    parser.add_argument('--match-threshold-cos', type=float, default=0.94, help='Cosine similarity threshold to consider "same" (default 0.94)')
    parser.add_argument('--match-threshold-euc', type=float, default=0.36, help='Euclidean distance threshold (default 0.36)')
    parser.add_argument('--enable-enhanced', action='store_true', help='Enable enhanced personal feature analysis')
    args = parser.parse_args()

    print("="*60)
    print("Gait Recognition Inference")
    print("="*60)

    model = load_model_simple(args.config, args.checkpoint)

    # process first skeleton
    print("\n--- Processing Skeleton 1 ---")
    data1 = preprocess_skeleton(args.skeleton1)
    feat1 = extract_features(model, data1)  # [N, D]

    # optional register
    if args.register:
        # if batch>1, take first element
        tensor = feat1 if feat1.dim() == 2 else feat1.mean(dim=0, keepdim=True)
        save_embedding(args.register, tensor, registry_dir=args.registry_dir)

    # if skeleton2 provided, extract and compare directly
    if args.skeleton2:
        print("\n--- Processing Skeleton 2 ---")
        data2 = preprocess_skeleton(args.skeleton2)
        feat2 = extract_features(model, data2)
        
        print("\n--- Deep Learning Gait Analysis ---")
        euclid_mean, cos_mean, mats = compare_features(feat1, feat2)

        print("\n" + "="*60)
        print("GAIT RECOGNITION RESULTS")
        print("="*60)
        print(f"Euclidean Distance: {euclid_mean:.4f}")
        print(f"Cosine Similarity:  {cos_mean:.4f}")
        print("-"*60)
        
        # Decision based on deep learning only
        dl_match = (cos_mean >= args.match_threshold_cos or euclid_mean <= args.match_threshold_euc)
        
        if dl_match:
            print("✅ CONCLUSION: SAME PERSON")
            print(f"   → High similarity detected (distance: {euclid_mean:.4f}, similarity: {cos_mean:.4f})")
        else:
            print("❌ CONCLUSION: DIFFERENT PERSON") 
            print(f"   → Low similarity detected (distance: {euclid_mean:.4f}, similarity: {cos_mean:.4f})")
        
        print("="*60)

    else:
        print("\n✓ Feature extraction completed!")
        print(f"Feature shape: {feat1.shape}")

    # If registry exists, try matching against it for stalking detection
    registry = load_registry(args.registry_dir)
    if registry:
        print("\n--- Matching against registry ---")
        # build registry tensor
        names = list(registry.keys())
        regs = torch.cat([registry[n] for n in names], dim=0).to(feat1.device)  # [K, D]
        # compare feat1 (could be batch) to regs
        euclid_mean, cos_mean, mats = compare_features(feat1, regs)
        # show per-name scores (use first sample if batch)
        print("Per-registered scores (first sample vs each):")
        cos_row = mats['cos_mat'][0]  # [K]
        euc_row = mats['euclid_mat'][0]
        for i, n in enumerate(names):
            print(f" - {n}: cosine={cos_row[i].item():.4f}, euclid={euc_row[i].item():.4f}")
        # decide any above threshold
        hits = [names[i] for i in range(len(names)) if (cos_row[i].item() >= args.match_threshold_cos or euc_row[i].item() <= args.match_threshold_euc)]
        if hits:
            print("!! MATCH FOUND in registry: ", hits)
        else:
            print("No close match in registry.")

if __name__ == '__main__':
    main()
