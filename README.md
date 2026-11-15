# Bullet Bills — Final 50M Model & Parallel Agent (`fin`)

슈퍼 마리오 브라더스(NES) 환경에서 양방향으로 발사되는 킬러(대포)를 피하기 위한 **최종 50M 스텝 병렬 PPO 에이전트 및 모델** 전용 브랜치입니다.

---

## 🌟 최종 모델 안내

- **최종 모델 파일**: `final/parallel50m_final.pt`
- **학습 알고리즘**: 16개 독립 CPU 환경 병렬 수집 + CUDA PPO / DAgger
- **총 학습량**: **50,000,000 스텝 (50M)**
- **입력 및 행동 공간**:
  - 상태 공간: **178차원** (물리 예측, 상대 TTC, 레인 및 고도 제어)
  - 행동 공간: **15-Action** (수평 이동 연속 제어 + 5단계 점프)
- **200판 시뮬레이션 평가 결과 (`final/log/eval_200_log.txt`)**:
  - **평균 생존 시간**: **130.29초** (중앙값: **130.46초**)
  - **180초 클리어**: **43 / 200판 (21.5%)**

---

## 📁 디렉토리 구조 및 구성

```text
final/
├── agent_ppo_v15_50m_parallel.py         # [메인] 단일 파일 PPO 에이전트 및 16-Worker 병렬 학습기
├── parallel50m_final.pt                  # ★ 최종 50M 스텝 학습 완료 모델 가중치
├── requirements.txt                      # 실행 및 학습 의존성 목록 (PyTorch, Numba 등)
├── log/
│   └── eval_200_log.txt                  # 200판 평가 상세 로그
└── 보조도구/
    ├── agent_ppo_v15_50m_parallel_eval_log.py  # 상세 로깅 및 평가 검증 도구
    └── parallel50m_resume.pt             # 학습 재개 및 상태 복원용 체크포인트
```

---

## 🚀 실행 가이드

### 0. 의존성 설치
```bash
pip install -r final/requirements.txt
```

### 1. 데모 시연 (화면 렌더링)
```bash
python final/agent_ppo_v15_50m_parallel.py --demo --model final/parallel50m_final.pt --seed 42
```

### 2. 200판 고속 평가
```bash
python final/agent_ppo_v15_50m_parallel.py --eval 200 --model final/parallel50m_final.pt
```

### 3. 상세 로그 저장 평가 (보조도구)
```bash
python final/보조도구/agent_ppo_v15_50m_parallel_eval_log.py --eval 200 --model final/parallel50m_final.pt
```

### 4. 이어서 학습 (Resume)
```bash
python final/agent_ppo_v15_50m_parallel.py --resume final/보조도구/parallel50m_resume.pt --steps 50000000 --envs 16 --device cuda
```
