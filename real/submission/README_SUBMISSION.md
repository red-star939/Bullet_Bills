# 2026-02 KNU Advanced RL Competition Round 2 — Bullet Bills 제출 보고서 및 가이드

본 패키지는 **KNU Advanced RL Competition Round 2: Bullet Bills!** 공식 지침서(`Instruction.pdf`)의 모든 제출 규격 및 평가 기준을 충족하도록 구성된 최종 제출물입니다.

---

## 1. 제출 파일 목록 및 구조

```text
submission/
├── agent.py               # [필수 1] YourAgent(kym.Agent) 및 train() 구현 (Numba 물리 예측 + PPO)
├── best_model.pt          # [필수 2] 최우수 성능 체크포인트 (obs_dim=148, hidden=256)
├── requirements.txt       # [필수 3] 실행 의존성 목록
├── evaluate.py            # [시연용] 지침서 5페이지 공식 kym.evaluate() 실행 스크립트
├── simulate_eval.py       # [검증용] 화면 없는 다중 시드 평가 스크립트
└── README_SUBMISSION.md   # 본 설명서 (발표 및 제출 문서)
```

---

## 2. 평가 및 실행 방법

### ① 공식 평가 시연 (지침서 규격)
게임 화면과 BGM이 활성화된 상태에서 공식 평가를 실행합니다:
```bash
python evaluate.py
```
내부적으로 다음 지침서 코드가 수행됩니다:
```python
import kymnasium as kym
from agent import YourAgent

agent = YourAgent.load('best_model.pt')
kym.evaluate(
    env_id='kymnasium/BulletBill-Continuous-Hard-Stage-2',
    agent=agent,
    render_mode='human',
    bgm=True,
    seed=42
)
```

### ② 화면 없는 고속 다중 시드 평가
```bash
python simulate_eval.py
```

---

## 3. 핵심 기술 및 발표(Presentation) 요약 자료

### A. 환경 및 문제 정의 (2%)
* **과제 목표**: 고밀도로 날아오는 Bullet Bill(총알)들을 피해 3분(180초) 동안 생존(`cleared`).
* **상태 공간(State)**:
  - 마리오 위치/속도/상태(지면, 공중, 무적) 14차원
  - 가장 위험한 킬러 6개의 상대 위치 및 TTC(충돌 예상 시간) 42차원
  - 별(무적 아이템) 위치 및 도달 가능성 12차원
  - 킬러 생성 타이밍 3차원: 생성 간격 $1.5 \times (0.1 / 1.5)^{t/105}$초에 기반한 잔여 시간
  - **2단계 생존 물리 예측 30차원**: 1.6초(145개 계획) $\rightarrow$ 3.2초(13개 확장 계획) 가상 시뮬레이션 기반 위험도
  - $\rightarrow$ **총 148차원 통합 관측 벡터**
* **행동 공간(Action)**:
  - 수평 이동 연속 제어: $a_{\text{move}} \in [-1, 1]$
  - 점프 강도 5단계 이산 선택: $a_{\text{jump}} \in \{0.0, 0.5, 0.6, 0.75, 1.0\}$

### B. 강화학습의 이해와 선택 근거 (3%)
* **하이브리드 PPO (Proximal Policy Optimization)**:
  - 연속 이동($\text{Beta/Gaussian}$)과 이산 점프($\text{Categorical}$)를 단일 신경망에서 동시 출력하는 하이브리드 액터-크리틱 아키텍처.
* **Potential-based Reward Shaping (Ng et al.)**:
  - 단순 생존(+0.1)과 사망(-10)만 주면 에이전트가 "가만히 서 있는 국소 최적해(Freeze)"에 빠지는 치명적 문제 해결.
  - 최선의 미래 회피 경로 위험도를 기반으로 한 포텐셜 함수 $\Phi(s)$를 설계하여, 최적 정책의 수학적 불변성을 유지하면서 안전한 회피 경로를 선제적으로 선택하도록 유도.
* **스폰 순간 고도 제어**:
  - 킬러 생성 순간 마리오가 48px 이상 공중에 떠 있으면 생성된 킬러가 착지한 마리오 위를 스쳐 지나간다는 환경 물리 규칙을 활용.

### C. 타인에게 주는 실질적 교훈 (3%)
1. **무반응/정지(Freeze) 현상의 방지**:
   - 연속 행동 PPO에서 초기 보상이 희소할 때 평가 시 결정적($\mu$) 출력이 0으로 얼어붙는 현상은 매우 흔함.
   - 이를 극복하려면 회피 인지 특징(TTC, 상대 속도)과 포텐셜 기반 위험도 보상을 결합해야 함.
2. **장기 미래 시뮬레이션의 중요성**:
   - 76초 이후 킬러가 화면에 30개 가득 차는 포화 구간에서는 1초 앞만 보고 피하면 2초 뒤가 막힌 막다른 골목에 갇힘. 3.2초 다단계 계획을 관측 특징으로 넣어주어야 포화 구간을 돌파 가능.

### D. 한계 및 개선점 (1%)
* **한계**: 180초 이후의 킬러 생성 주기는 공식 환경 코드 외삽이 불확실하여 보수적으로 대응함.
* **개선 방향**: 모델 기반 폐루프 플래너(Closed-loop Trajectory Search)를 신경망 추론과 병렬화하여 실시간성 극대화.

