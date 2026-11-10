# GPGait++ 학습 결과 (CASIA-B)

- 모델: GPGait++ (`configs/gpgait++/gpgait++.yaml`)
- 데이터셋: CASIA-B (HRNet pose), 경로 `datasets/CASIA-B/CASIA-B_HRNet` (저장소에는 미포함)
- 장비: NVIDIA GeForce RTX 3090 × 1
- 배치: TripletSampler `[4, 8]` (원본 설정은 `[4, 32]`), frames 30 고정
- 스케줄러: OneCycleLR, max_lr 0.001, total_steps 40000
- 원본 tensorboard 로그: `output/CASIA-B/GPGait_PlusPlus/GPGait_PlusPlus/summary/`

## 학습 시간

| 구간 | 기간 | 소요 시간 |
|---|---|---|
| Run 1: 0 → 40,000 iter | 2025-11-06 23:29 → 11-07 02:12 | 약 2시간 43분 (평가 포함) |
| Run 2: 40,000 → 60,000 iter (40k 체크포인트에서 재개) | 2025-11-07 11:14 → 12:41 | 약 1시간 27분 (평가 포함) |

## 테스트 정확도 (Rank-1, %)

| Iter | NM | BG | CL |
|---|---|---|---|
| 10,000 | 85.63 | 70.07 | 60.44 |
| 20,000 | 89.35 | 74.81 | 68.78 |
| 30,000 | 92.06 | 77.14 | 73.89 |
| **40,000** | **93.75** | **80.55** | **76.25** |
| 50,000 | 92.94 | 79.07 | 73.85 |
| 60,000 | 91.90 | 78.16 | 72.95 |

**40,000 iter 체크포인트가 가장 좋음.** 40k 이후 이어서 학습했을 때 learning rate가 다시 올라가면서(50k에서 0.0003, 60k에서 0.0008) 정확도가 조금 떨어졌어요.

## Loss / 학습 정확도

| Iter | softmax loss | triplet hard loss | softmax acc | LR |
|---|---|---|---|---|
| 100 | 4.7384 | 2.4439 | 0.018 | ~0 |
| 10,000 | 1.7406 | 0.6043 | – | 0.0005 |
| 20,000 | 0.6908 | 0.3535 | – | 0.0010 |
| 30,000 | 0.2829 | 0.2397 | – | 0.0004 |
| 40,000 | 0.3330 | 0.2291 | 0.944 | ~0 |
| 50,000 | 0.4282 | 0.2451 | – | 0.0003 |
| 60,000 | 0.3474 | 0.2644 | 0.926 | 0.0008 |

## 원본 대비 코드 변경 사항

- `configs/gpgait++/gpgait++.yaml`: 로컬 데이터셋 경로, `num_workers: 4`, 배치 `[4, 8]`, 평가 batch_size 1, 40k 체크포인트에서 restore
- `fastposegait/modeling/**/__init__.py`: `iter_modules([str(package_dir)])` — `pkgutil.iter_modules`에 `Path`가 아니라 `str`을 넘기도록 수정
