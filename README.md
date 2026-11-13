# Bullet Bills — Real Package & Trained Models (`MystaLEE_real_x1`)

슈퍼 마리오 브라더스(NES) 환경에서 양방향으로 발사되는 킬러(대포)를 피하기 위한 강화학습 모델 구현 테스트입니다.

---

## 🌟 최신 추천 모델 안내

- **최신 모델 파일**: `real/v/model_v15_30m.pt`
- **실제 누적 학습량**: **36,675,584스텝 + 10,000,000스텝 (기존 v14 모델) = 총 46,675,584스텝**
- **시뮬레이션 200판 결과**:
  - **평균 생존 시간**: **135.12초**
  - **180초 완주율**: **32.5% (65 / 200판)**
  - **120초 이상 생존율**: **59.5% (119 / 200판)**
  - **150초 이상 생존율**: **43.5% (87 / 200판)**

---

## 📁 디렉토리 구조 및 구성

```text
real/
├── submission/                      # [공식 제출 패키지] 지침서 규격 충족 최종 제출 폴더
│   ├── agent.py                     # [필수 1] YourAgent(kym.Agent) 구현 (Numba 물리 예측 + PPO)
│   ├── best_model.pt                # [필수 2] 공식 평가용 최우수 가중치 (obs_dim=148, hidden=256)
│   ├── requirements.txt             # [필수 3] 실행 의존성 목록
│   ├── evaluate.py                  # 공식 시연용 kym.evaluate() 스크립트 (화면/BGM 활성화)
│   ├── simulate_eval.py             # 화면 없는 고속 다중 시드 평가 검증 스크립트
│   └── README_SUBMISSION.md         # 공식 제출 및 발표 요약 가이드
│
├── v/                               # [체크포인트 및 평가 결과 저장소]
│   ├── model_v15_30m.pt             # ★ 최신 모델 (누적 46.6M 스텝)
│   ├── checkpoints/
│   │   └── v15_100m/
│   │       ├── model_v15_step36675584.pt   # 최종 체크포인트 (36.6M)
│   │       ├── model_v15_step16056320.pt
│   │       └── ...
│   ├── eval_results/
│   │   ├── v15_36m_200ep/           # 최신 200판 시뮬레이션 평가 결과 (평균 135.12초)
│   │   │   ├── episodes.csv
│   │   │   └── summary.json
│   │   └── v15_16m_headless/
│   ├── model_v14_10m.pt             # V14 10M 베이스라인 가중치
│   └── model_v15_100m_best.pt       # V15 100M 학습 베스트 가중치
│
└── model_v4/                        # [네트워크 아키텍처 및 학습/진단 스크립트]
    ├── agent_ppo_v14.py / agent_ppo_v15.py  # 버전별 PPO 에이전트 구현
    ├── train_v15_fast.py            # V15 고속 병렬 추가 학습기
    ├── model_v4_net.py / model_v41_net.py
    ├── diagnose_actions_v4.py
    └── README_V15_FAST_KO.md        # V15 학습 가이드
```

---

## 🚀 빠른 실행 방법

### 1. 공식 제출 모델 평가 (화면 및 사운드)
```bash
cd real/submission
python evaluate.py
```

### 2. 최신 모델(`model_v15_30m.pt`) 평가
`real/submission/agent.py`의 기본 로드 경로를 지정하거나 아래와 같이 실행할 수 있습니다:
```bash
python -c "
import kymnasium as kym
from real.submission.agent import YourAgent

agent = YourAgent.load('real/v/model_v15_30m.pt')
kym.evaluate(
    env_id='kymnasium/BulletBill-Continuous-Hard-Stage-2',
    agent=agent,
    render_mode='human',
    bgm=True,
    seed=42
)
"
```
