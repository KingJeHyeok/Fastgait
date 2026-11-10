# ============================================
# 완벽한 얼굴 인식 + 스켈레톤 통합 로직
# ============================================

# ============================================
# 1단계: 설정 수정 (라인 133-170 부근)
# ============================================

# YOLO 설정
YOLO_CONF = 0.5                        # YOLO 기본 detection confidence
FACE_START_CONF = 0.9                  # 얼굴 인식 시작 최소 confidence (핵심!)
POSE_CONF = 0.6                        # Pose 모델 confidence

# 얼굴 인식 설정
FACE_LOCK_SEC = 3.0                    # 3초간 연속 인식되면 Lock on
FACE_EVERY_SEC = 1.0                   # Lock-on 전 얼굴 인식 주기
FACE_LOCK_UNKNOWN_TOLERANCE = 2        # Lock-on 후 Unknown 허용 횟수

# 트랙 소멸 설정
ABSENCE_GRACE = 2.0                    # 2초 이상 안 보이면 트랙 소멸로 판단


# ============================================
# 2단계: 메인 루프 수정 (라인 1424-1548)
# ============================================

for tid, (x1,y1,x2,y2), conf in zip(ids, xyxy, confs):
    # YOLO confidence 체크
    if conf < YOLO_CONF:
        continue
    
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    current_ids.append(tid)
    
    # ============================================
    # A. 스켈레톤 시스템 초기화
    # ============================================
    if tid not in skeleton_buffers:
        skeleton_buffers[tid] = []
        skeleton_states[tid] = 'IDLE'
        skeleton_frames_since_inference[tid] = 0
        skeleton_frames_since_lost[tid] = 0
        skeleton_has_initial_inference[tid] = False
        skeleton_person_ids[tid] = None
        skeleton_session_locked[tid] = False
        skeleton_tracking_confidence[tid] = 0.0
        print(f"🆕 새 트랙 등장: tid={tid} (conf={conf:.2f})")
    
    # ============================================
    # B. 스켈레톤 추출 및 버퍼링 (항상 수행)
    # ============================================
    skeleton = tid_to_skeleton.get(tid, None)
    
    if SKELETON_STATE_ENABLED and skeleton is not None:
        # 스켈레톤 상태 머신 업데이트
        current_state = skeleton_states[tid]
        
        if current_state == 'IDLE':
            skeleton_states[tid] = 'DETECTING'
            skeleton_frames_since_lost[tid] = 0
            print(f"🎯 스켈레톤 검출 시작: tid={tid}")
        
        elif current_state == 'DETECTING':
            skeleton_buffers[tid].append(skeleton)
            if len(skeleton_buffers[tid]) >= SKELETON_DETECT_FRAMES:
                skeleton_states[tid] = 'COLLECTING'
                print(f"📊 스켈레톤 수집 시작: tid={tid}")
        
        elif current_state == 'COLLECTING':
            skeleton_buffers[tid].append(skeleton)
            if len(skeleton_buffers[tid]) > SKELETON_MAX_BUFFER:
                skeleton_buffers[tid].pop(0)
            
            skeleton_frames_since_inference[tid] += 1
            buffer_size = len(skeleton_buffers[tid])
            
            # 첫 추론 (초기 버퍼 채워지면)
            if not skeleton_has_initial_inference[tid] and buffer_size >= SKELETON_INITIAL_BUFFER:
                if gait_model is not None and person_registry is not None:
                    result = process_skeleton_inference(
                        tid, frame_idx, gait_model, person_registry, device, is_final=False
                    )
                    if result:
                        skeleton_person_ids[tid] = result['person_id']
                        skeleton_has_initial_inference[tid] = True
                        skeleton_tracking_confidence[tid] = result['match_score']
                        print(f"🚶 스켈레톤 추론 완료: tid={tid} -> Person #{result['person_id']}")
            
            # 주기적 추론 (Lock-on 후에도 계속 수행!)
            elif (skeleton_has_initial_inference[tid] and 
                  skeleton_frames_since_inference[tid] >= SKELETON_ACCUM_INTERVAL):
                if gait_model is not None and person_registry is not None:
                    result = process_skeleton_inference(
                        tid, frame_idx, gait_model, person_registry, device, is_final=False
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
            if skeleton_frames_since_lost[tid] >= 30:
                # 30프레임 동안 안 보이면 리셋
                skeleton_buffers[tid].clear()
                skeleton_states[tid] = 'IDLE'
                skeleton_frames_since_lost[tid] = 0
                skeleton_has_initial_inference[tid] = False
    
    # ============================================
    # C. 얼굴 인식 (조건부 수행)
    # ============================================
    identity = last_identity.get(tid)
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
                    roi_in = cv2.resize(roi, (int(w*scale), int(h*scale)), 
                                       interpolation=cv2.INTER_LINEAR)
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
                                detected_text = face_labels.get(best_key, 
                                                               best_key.split('/')[-1])
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
    
    # ============================================
    # D. 화면 표시
    # ============================================
    label_text = identity.get('text', 'Unknown') if identity else 'Unknown'
    is_known = identity.get('known', False) if identity else False
    skeleton_person_id = skeleton_person_ids.get(tid, None)
    
    # 색상 및 추적 모드 표시
    if lock_info['locked']:
        color = (255, 128, 0)  # 청록색 - 스켈레톤 추적 중
        tracking_mode = "[SKELETON]"
    elif conf >= FACE_START_CONF:
        color = (0, 255, 0) if is_known else (0, 0, 255)  # 초록/빨강 - 얼굴 인식 중
        tracking_mode = "[FACE]"
    else:
        color = (128, 128, 128)  # 회색 - 대기 중 (conf < 0.9)
        tracking_mode = "[WAIT]"
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 라벨
    label_display = f"{tracking_mode} {label_text}"
    if skeleton_person_id is not None and SKELETON_STATE_ENABLED:
        label_display += f" (P#{skeleton_person_id})"
    
    # confidence 표시 추가
    label_display += f" {int(dur)}s | C:{conf:.2f}"
    
    cv2.putText(frame, label_display, (x1, max(15, y1-6)), 
                FONT, 0.7, color, 2)
    
    # 스켈레톤 그리기
    if skeleton is not None and USE_POSE_MODEL:
        frame = draw_skeleton(frame, skeleton)


# ============================================
# 3단계: 트랙 소멸 처리 (라인 1676-1698)
# ============================================

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
