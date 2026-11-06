# Bullet Bills! Continuous Hard Stage 2 — 생존 에이전트 개발

> **프로젝트:** 강화학습-v / Mario BulletBill 연구 프로젝트  
> **환경 ID:** `kymnasium/BulletBill-Continuous-Hard-Stage-2`  
> **문서 갱신:** 2026-11-06  
> **현재 단계:** 정밀 물리·충돌 예측 및 규칙 기반 행동 평가기 개발 완료, 플랫폼 안전성 분석과 **신규 시드 일반화 검증 진행 중**  
> **학습된 `.pt` 체크포인트:** 아직 없음. 아래 실험은 강화학습 성능이 아니라 **규칙 기반 정책** 성능임.

## 1. 목표와 개발 전략

마리오가 무작위로 출현하는 Bullet Bill(포탄)을 피하고, 발판에서 추락하지 않으면서 최대한 오래 생존하는 에이전트를 개발한다. 최종적으로 PyTorch 신경망 정책을 학습·평가하고 `.pt` 파일로 저장하는 것이 목표다.

개발 전략은 다음과 같다.

1. 게임 엔진의 물리·충돌 규칙을 검증한다.
2. 현재 관측값을 기반으로 미래 상태를 시뮬레이션한다.
3. 학습 없이 작동하는 **재현 가능한 규칙 기반 정책**을 성능 기준선으로 확보한다.
4. 사망 원인과 일반화 성능을 검증하고 정책을 개선한다.
5. 검증된 정책으로 학습 데이터를 생성하여 Behavior Cloning(BC)과 이후 강화학습(PPO 등)으로 연결한다.

**중요:** 현재의 `ForwardSimulator`는 *관측된 포탄을 기준으로 한 조건부 미래 예측*이다. 아직 등장하지 않은 무작위 포탄을 미리 알지 못하므로, 미래 전체의 생존을 보장하지 않는다.

## 2. 환경·물리 규칙 요약

| 항목 | 확인된 설정 |
|---|---|
| 월드 크기 | 1296 × 864 px |
| 내부 물리 갱신 | 60 FPS |
| 기본 `frame_skip` | 4 (행동 1스텝 ≈ 0.0667초) |
| 수평 레인 | 48px 높이 기준 18개 |
| 수평 입력 | `action[0]` ∈ [-1, 1] |
| 점프 입력 | `action[1]` ∈ [0, 1] |
| 행동 자료형 | `np.float32` 배열 사용 |
| 중력 | 2000 px/s² |
| 점프 힘 | 1000 |
| 포탄 밟기 반동 | 수직 속도 -500 px/s |
| 지상/공중 최고 수평속도 | 750 / 500 px/s |
| 지상/공중 최대 이동 힘 | 100 / 75 |
| 수평 마찰 | 매 내부 프레임 속도 × 0.9 |

관측의 주요 키는 `mario`, `bullet_bills`다. 마리오는 `[left, top, right, bottom, vx, vy, invincible, jumping]`, 포탄은 `[left, top, right, bottom, vx]` 형식을 사용한다.

실제 엔진은 마리오의 위치 이동에서 `int(velocity / 60)` 형태의 **정수 픽셀 이동**을 사용한다. 따라서 실수 속도의 극히 작은 오차도 경계에서 1px 차이와 충돌 시간 1프레임 차이로 이어질 수 있다.

### 충돌 판정에서 검증된 사실

- 활성 포탄과 마리오의 AABB가 실제로 겹쳐야 충돌이 발생한다. 경계만 맞닿는 것은 충돌이 아니다.
- 위에서 내려오며 포탄을 밟는 조건을 만족하면 포탄이 `fallen`으로 전환되고 마리오는 반동한다.
- 무적 상태에서 포탄과 겹치면 포탄이 `fallen`으로 전환된다.
- 그 외 활성 포탄과의 치명적 겹침에서는 사망한다.
- 포탄과 별의 내부 상태, 생성 이벤트 등 관측에 없는 정보는 장기 예측의 불확실성 요인이다.

## 3. 프로젝트 구조

다음은 **지금까지 구현·검증에 사용한 주요 파일**이다. 실제 Windows 프로젝트에서는 관련 파일을 동일한 상대 경로에 배치해야 한다.

```text
mario_pj/
├── agent_env.py                         # 사용자 로컬 환경 생성 헬퍼
├── map_data/
│   └── object_tiles.csv                 # 정적 발판/물체 지도
├── models/
│   ├── lane_manager.py                  # 레인 분류 (초기 단계)
│   ├── bullet_tracker.py                # 포탄 추적 (초기 단계)
│   ├── risk_estimator.py                # TTC/위험도 (초기 단계, 함수 API)
│   ├── mario_motion_predictor.py        # 마리오 물리 예측
│   ├── collision_predictor.py           # 엔진 충돌 규칙 재현
│   ├── interaction_simulator.py         # 마리오-포탄 상호작용
│   ├── collision_event_audit.py         # 실제 충돌 사건 판정
│   ├── forward_simulator.py             # Nominal/Robust 미래 예측
│   ├── action_evaluator.py              # v1 규칙 기반 행동 평가
│   ├── action_evaluator_v2.py           # v2 장기 추락 예측 보강
│   ├── failure_analyzer.py              # 실패 분석 v1
│   ├── failure_analyzer_v2.py           # 실제 치명적 포탄 연결
│   ├── failure_analyzer_v21.py          # 사망 경로/내부 포탄 ID 진단
│   ├── mpc_controller.py                # 실험적 MPC v1 (성능 저하, 보류)
│   └── platform_safety_evaluator.py     # 착지/지지/발판 경계 진단
├── collect_failure_cases.py
├── collect_failure_cases_v2.py
├── collect_failure_cases_v21.py
├── benchmark_action_evaluator_v2.py
├── benchmark_mpc.py
├── benchmark_generalization.py
├── test_forward_simulator.py
├── test_forward_simulator_robust.py
├── test_action_evaluator.py
├── test_action_evaluator_v2.py
├── test_failure_analyzer_v21.py
├── test_mpc_controller.py
├── test_platform_safety_evaluator.py
├── test_benchmark_generalization.py
└── logs/                                # 실험 실행 시 생성
```

> `agent_env.py` 및 `map_data/object_tiles.csv` 등은 로컬 프로젝트 의존 파일이다. 파일 구성은 개발 시점과 실제 프로젝트 복사 상태에 따라 달라질 수 있다.

## 4. 핵심 모듈과 구현 경과

| 단계 | 모듈 | 구현·검증 요약 |
|---|---|---|
| 환경 분석 | 레인/포탄 추적/TTC | 포탄 최초 관측 위치·초기 사거리·속도 및 위험 추정 구현 |
| 물리 예측 | `MarioMotionPredictor` | 점프, 가속·마찰·중력, 발판 충돌 및 픽셀 단위 계산 재현 |
| 상호작용 | `CollisionPredictor`, `InteractionSimulator` | 포탄 밟기·무적 충돌·사망 이벤트를 엔진 규칙으로 시뮬레이션 |
| 충돌 실환경 검증 | 다중 시드 이벤트 감사 v3 | **10,000 에피소드**, 평가된 충돌 이벤트 **7,880/7,880 일치**; 생성/무적 변경 등 일부 프레임 제외 |
| 미래 예측 | `ForwardSimulator` | 현재 관측 포탄을 기반으로 행동 시퀀스의 60FPS 미래 궤적 계산 |
| 정밀도 보강 | Robust ForwardSimulator | `float32` 속도 양자화 경계의 9개 후보 예측으로 1프레임 충돌 시점 민감성 표시 |
| 정책 기준선 | `ActionEvaluator` v1 | 8개 행동 후보의 단기 미래 위험 비교 |
| 사망 분석 | FailureAnalyzer v1 → v2 → v2.1 | 실제 포탄 충돌과 추락을 구분, 내부 사망 증거 및 ID 연결 |
| 실험적 다단계 제어 | MPC v1 | Beam Search 방식 도입, 실환경 성능 악화로 **사용 보류** |
| 규칙 기반 정책 개선 | `ActionEvaluatorV2` | v1 선택을 유지하되 장기 추락이 예측되면 안전한 대안으로 교체 |
| 플랫폼 진단 | `PlatformSafetyEvaluator` | 발판 지지, 착지, 겹침 폭, 가장자리 여유, 추락 프레임 분석 |
| 일반화 검증 | `benchmark_generalization.py` | Reference/Holdout 분리 실험 **진행 중** |

### 4.1 ForwardSimulator: 경계 민감성 사례

`seed=726388407`, 53번째 행동 스텝의 4번째 내부 프레임에서 Nominal은 사망을 예측했지만 실제 엔진은 다음 프레임에서 사망했다. 원인은 관측 `float32` 속도와 실제 내부 속도의 아주 작은 차이가 `int(vy/60)` 이동을 `-9px` 대 `-10px`로 갈라놓은 데 있다.

Robust 버전은 X/Y 속도의 양자화 구간에서 9개 후보를 검사하며, 해당 사례에서 사망 시점을 **4 또는 5프레임**으로 표시했다. 이는 **샘플 기반 민감도 검사**이지 모든 가능한 물리 상태의 완전 탐색이나 사망 확률 추정이 아니다.

### 4.2 ActionEvaluator v1/v2의 차이

- **v1:** 정지·좌/우·점프·약한 수평 이동 등 고정된 8개 행동 후보를 비교한다. 기본적으로 행동 하나를 약 8스텝(0.53초) 유지하는 가정으로 예측한다.
- **v2:** v1의 행동을 기본값으로 유지한다. 단기 평가에서 안전한 상위 후보에 대해 추가로 최대 20스텝(약 1.33초)까지 예측하여, 기존 선택에 **명시적 추락**이 예상되고 대안의 예측이 완주할 때만 선택을 바꾼다.
- **MPC v1:** 행동을 구간별로 바꾸는 Beam Search 실험이었지만, 아래 성능 비교에서 크게 악화되어 현재 정책 기준선으로 사용하지 않는다.
- **PlatformSafetyEvaluator:** 현재는 *읽기 전용 진단 도구*이며 v2의 점수/행동 변경에 직접 통합하지 않았다.

## 5. 주요 실환경 벤치마크 결과

수치는 **사용자가 Windows 실제 게임에서 실행하여 공유한 로그**에 기반한다. 합성 회귀 테스트 통과와 실게임 성능을 구분해야 한다.

### 5.1 초기 규칙 기반 정책: 무작위 행동 vs ActionEvaluator v1

동일한 100개 시드, 행동 상한 300스텝(20초)에서:

| 항목 | 무작위 정책 | v1 (Nominal) |
|---|---:|---:|
| 평균 **관측** 생존 시간 | 약 3.18초 | 약 14.43초 |
| 20초까지 생존 | 0/100 | 30/100 |
| 시드별 승패 | — | 99승 1패 |
| 정책 계산/결정 | — | 약 9.12ms |

이는 상한 검열(censoring)이 있는 관측 평균이다. 20초에서 살아 있던 30개의 실제 사망 시간은 이 실험만으로 알 수 없다.

### 5.2 Robust ablation (100개 시드)

| 모드 | 평균 관측 생존 | 평균 계산 시간/결정 | 해석 |
|---|---:|---:|---|
| `none` | 14.434초 | 9.12ms | 가장 저렴한 기준선 |
| `top_k` | 14.434초 | 40.99ms | 계산 증가, 생존 개선 관측 안 됨 |
| `all` | 14.434초 | 91.32ms | 계산 증가, 생존 개선 관측 안 됨 |

모든 모드에서 시드별 생존 시간이 같았다. 따라서 Robust를 정책 전체에 상시 적용할 근거는 부족하며, 경계 진단용으로 제한적으로 유지한다.

### 5.3 FailureAnalyzer v2.1: 사망 원인 분류

동일한 100개 시드의 60초 제한 실험에서 모든 에피소드가 제한시간 전에 사망했고, 평균 생존 시간은 **17.7085초**였다.

| 실제 물리 사망 경로 | 건수 |
|---|---:|
| 포탄 충돌 | 45 |
| 화면 아래 추락 | 55 |
| 합계 | 100 |

이전 v2에서 `fatal_projectile_unverified`로 남았던 55건이 추락으로 확인됐다. 포탄 연결 감사에서는 44건의 단일 치명적 겹침을 식별했고 1건은 복수 포탄으로 모호했다. 또한 동일 엔진 포탄의 관측 추적 ID가 여러 개로 갈라지는 사례가 22건 있어 추적 안정화 여지가 남아 있다.

**주의:** 물리적 사망 원인과 정책의 의사결정 실패 원인은 다르다. 사망 직전 예측상 안전한 대안이 있었다는 사실은 그 대안이 실제 환경에서도 생존했을 것이라는 반사실적 증명이 아니다.

### 5.4 MPC v1: 보류한 실험

| 지표 | ActionEvaluator v1 | MPC v1 |
|---|---:|---:|
| 100개 시드 평균 생존 | 17.708초 | **2.601초** |
| 평균 계산/결정 | 9.26ms | 44.40ms |
| MPC가 이긴 시드 | — | **0/100** |

MPC의 조기 가지치기/행동 다양성 및 플랫폼 위험 평가 문제가 의심되지만, 원인 확인 없이 특정 버그로 단정하지 않는다. **v1 MPC는 현재 실전 정책으로 사용하지 않는다.**

### 5.5 ActionEvaluator v2: 현재 최고 규칙 기반 결과

동일한 100개 시드, 900스텝(최대 60초), 기본 단기 8스텝·장기 20스텝·상위 4개 감사 기준:

| 지표 | v1 | **v2** |
|---|---:|---:|
| 평균 **관측** 생존 | 17.708초 | **34.134초** |
| 20초 생존율 | 30% | **73%** |
| 40초 생존율 | 6% | **39%** |
| 60초 생존율 | 0% | **19%** |
| 계산/결정 평균 | 9.06ms | **19.18ms** |
| 동일 시드 승리/동률/패배 | — | **75 / 2 / 23** |
| v2 추락 회피 Override | — | **8,173회** |

v2의 관측 평균은 v1 대비 약 **1.93배**이며, 평균 개선 폭은 **+16.426초**다. 단, 60초까지 생존한 19개 에피소드가 있어 34.134초는 **제한된 관측 평균**이다. Override 횟수는 정책 결정 변경 횟수이지 실제 추락을 예방한 횟수가 아니다.

**퇴행 시드 예시**: `3082272262`(v1 51.33초 → v2 12.18초), `1158896200`(40.03초 → 4.20초). 평균 성능만큼 **23개 퇴행 사례**도 중요하게 다룬다.

## 6. 현재 진행 중인 일반화 벤치마크

`benchmark_generalization.py`는 기존과 동일한 `reference` 100개와 **중복되지 않는 신규 `holdout` 100개 이상**을 분리해 v1/v2를 비교한다. 각 시드를 같은 정책 간 비교에 사용하지만, 두 정책의 행동 차이로 게임 내부 RNG 소비가 달라질 수 있으므로 **동일 미래 포탄 시나리오의 완벽한 반사실 실험은 아니다**.

**2026-11-06 현재 최근 공유된 터미널 진행 상황:**

```text
PROGRESS 25 / 200 split reference mean_delta_s 13.453
PROGRESS 50 / 200 split reference mean_delta_s 16.09
```

- 현재 **50/200 에피소드 쌍(25%)**까지 진행된 것으로 공유됐다.
- **Reference** 구간의 진행 중 평균 개선 폭은 +16.09초다.
- **Holdout** 구간은 아직 결과가 공유되지 않았으므로 **일반화 성능 검증 완료라고 말할 수 없다**.
- **현 구현은 결과 CSV/JSON을 루프 종료 후 저장한다.** 긴 평가 전에 중간 저장과 `--resume` 기능 추가가 권장된다.

### 권장 실행 순서 (Windows PowerShell)

프로젝트 루트에서 실행:

```powershell
# 1) 플랫폼 진단 및 일반화 벤치마크 회귀 테스트
.\.venv\Scripts\python.exe test_platform_safety_evaluator.py
.\.venv\Scripts\python.exe test_benchmark_generalization.py

# 2) Reference 100 + Holdout 100 스모크 테스트
.\.venv\Scripts\python.exe benchmark_generalization.py `
  --reference-episodes 100 `
  --holdout-episodes 100 `
  --steps 900 `
  --report-every 25 `
  --out logs/generalization_smoke

# 3) 스모크 통과 후 Holdout 1,000개 확장
.\.venv\Scripts\python.exe benchmark_generalization.py `
  --reference-episodes 100 `
  --holdout-episodes 1000 `
  --steps 900 `
  --report-every 50 `
  --out logs/generalization_1000
```

**참고:** `--steps 900`은 환경 행동 900스텝 기준 최대 60초다. Reference 100 + Holdout 100은 각 정책을 모두 평가하므로 최대 **400개 에피소드 실행**을 의미한다. 기존 프로세스가 실행 중이면 같은 출력 디렉터리로 두 번째 작업을 동시에 실행하지 않는다.

실행 완료 후 기본 출력:

```text
logs/generalization_smoke/
├── episode_pairs.csv     # split/reference/holdout, 시드별 비교 결과
└── summary.json          # split별 승패, 제한 생존 평균, 생존율, bootstrap CI
```

`summary.json`의 `reference`, `holdout`, `all`을 구분해 보고, **최종 일반화 판단은 holdout 수치**를 우선한다. 기존 기준선과의 비교에 reference를 사용하되 holdout 결과에 맞춰 파라미터를 계속 변경하면 홀드아웃의 독립성이 훼손될 수 있으므로 새 테스트 시드를 별도로 보관한다.

## 7. 주요 회귀 테스트 및 빠른 실행

가상환경에서 `numpy`, `pygame`, `gymnasium`, `kymnasium` 및 게임 관련 의존성이 설치돼 있어야 실제 환경 테스트를 실행할 수 있다. 설치 명령과 버전은 현 로컬 가상환경 구성에 맞춘다. 본 문서가 의존성 버전까지 검증한 것은 아니다.

```powershell
# 물리/미래 예측
.\.venv\Scripts\python.exe test_forward_simulator.py
.\.venv\Scripts\python.exe test_forward_simulator_robust.py

# 정책 및 사망 진단
.\.venv\Scripts\python.exe test_action_evaluator.py
.\.venv\Scripts\python.exe test_action_evaluator_v2.py
.\.venv\Scripts\python.exe test_failure_analyzer_v21.py

# 발판 안전성 및 일반화
.\.venv\Scripts\python.exe test_platform_safety_evaluator.py
.\.venv\Scripts\python.exe test_benchmark_generalization.py

# 동일 100개 시드의 v1/v2 정책 비교
.\.venv\Scripts\python.exe benchmark_action_evaluator_v2.py `
  --episodes 100 --steps 900 --seed 20261106 `
  --horizon-steps 8 --extended-horizon-steps 20 --audit-top-k 4 `
  --out logs/action_evaluator_v2_60s

# 엔진 사망 경로 감사 (진단용 내부 접근 포함)
.\.venv\Scripts\python.exe collect_failure_cases_v21.py `
  --episodes 100 --steps 900 --seed 20261106 `
  --out logs/failure_analyzer_v21
```

### API 사용 예시

```python
import numpy as np
from models.mario_motion_predictor import MarioMotionPredictor
from models.forward_simulator import ForwardSimulator
from models.action_evaluator_v2 import ActionEvaluatorV2

motion = MarioMotionPredictor.from_object_tiles_csv("map_data/object_tiles.csv")
forward = ForwardSimulator(motion)
policy = ActionEvaluatorV2(
    forward,
    horizon_steps=8,
    extended_horizon_steps=20,
    audit_top_k=4,
    robust_mode="none",
)

# obs: env.reset() 또는 env.step()의 최신 observation
result = policy.evaluate(obs)
action = np.asarray(result.best_action, dtype=np.float32)
# next_obs, reward, terminated, truncated, info = env.step(action)
```

실환경 시험에서 정책에는 **관측값만** 제공하며, FailureAnalyzer의 엔진 내부 정보는 **사후 진단 전용**으로 분리한다.

## 8. 데이터 해석 및 알려진 한계

1. **조건부 예측:** 미래 무작위 포탄의 생성, 일부 별 획득 및 숨겨진 상태는 완전 예측할 수 없다.
2. **양자화 경계:** `float32` 관측 정밀도 때문에 사망 프레임이 1프레임 차이 날 수 있다. Robust 9개 후보는 샘플이지 수학적 완전 보장이 아니다.
3. **측정 상한:** 20·60초까지 생존한 샘플은 관측 시간 상한에 의해 검열된다. 제한 평균을 실제 평균 사망 시간처럼 쓰지 않는다.
4. **동일 시드의 한계:** 정책 행동 차이가 RNG 소비에 영향을 준다면 미래 포탄 사건이 동일하지 않을 수 있다.
5. **플랫폼 진단 범위:** `PlatformSafetyEvaluator`의 가장자리 여유와 착지 보고서는 위험의 단서이지 플랫폼 복귀 가능성의 증명이 아니다.
6. **대안 행동의 불확실성:** 예측상 안전한 대안이 있더라도 실제로 그 행동을 실행했을 때 살았다는 뜻은 아니다.
7. **연산 지연:** 평균 ms/결정은 측정됐다. 최악 지연(p95/p99)과 환경·렌더링 비용을 포함한 실시간 보장은 별도 검증이 필요하다.
8. **10,000 시드 충돌 테스트의 범위:** 실제 포탄 상태를 매 내부 프레임 사용한 **동기화 이벤트 판정 정확도**다. 신규 포탄까지 포함한 독립 장기 미래 예측 정확도는 아니다.
9. **학습 모델 부재:** 현재 `.pt` 파일은 없고, 아직 BC/PPO 정책의 실환경 성능을 주장할 수 없다.

## 9. 다음 개발 계획

**현재 권장 우선순위**

1. **일반화 결과 수집 완료:** Holdout 100 → 1,000 시드에 대해 v2의 개선 폭·퇴행 비율·신뢰구간 확인.
2. **긴 벤치마크 안정화:** `benchmark_generalization.py`에 중간 CSV/JSON 저장, `--resume`, 실험 설정 및 시드 리스트 보존 추가.
3. **PlatformSafetyEvaluator의 실전 통합 검토:** 추락 방지 Override의 후보별 착지 안정성과 발판 지지 여유를 분석하여 잘못된 개입을 줄일 수 있는지 확인.
4. **ActionEvaluator v2.1/v3:** 23개 퇴행 시드 분석, Override 후 포탄 위험·발판 복귀 위험 재검증. 새로운 홀드아웃을 확보해 검증.
5. **학습용 상태 인코더·데이터 수집:** `state_encoder.py`, 교사 행동 데이터셋 수집기 작성.
6. **첫 PyTorch 모델:** BC 학습 `mario_bc_v1.pt` 저장 및 규칙 기반 교사와 동일/신규 시드 성능 비교.
7. **강화학습:** PPO 등으로 정책 미세조정 후 `mario_ppo_v1.pt`, 최종 `mario_best.pt` 평가.

MPC v1은 **성능 저하로 보류**한다. 향후 다단계 행동 탐색이 다시 필요해지면 첫 행동별 Beam 다양성, 발판 위험 점수, 기존 정책 폴백 및 행동 로그를 추가한 별도의 실험으로 재도입한다.

## 10. 실험 재현성 체크리스트

- Python/패키지 버전, 환경 ID, 지도 CSV 버전, Git 커밋을 기록한다.
- 시드 생성 알고리즘과 시드 리스트를 저장하고 reference/holdout을 섞지 않는다.
- `horizon_steps`, `extended_horizon_steps`, `audit_top_k`, `robust_mode`, `frame_skip`, 총 에피소드 스텝 상한을 기록한다.
- 보고서에는 평균 생존 시간뿐 아니라 **검열 수, 시간별 생존율, 동일 시드 승/패, 계산 지연, 조기 사망 사례**를 포함한다.
- 내부 엔진 접근은 사후 진단기에만 허용하고 정책 평가 중에는 관측 기반 입력을 유지한다.
- 실험 완료 전 중간 `PROGRESS` 출력은 **확정된 최종 결과가 아니며**, 특히 Reference 진행 수치를 Holdout 성능으로 인용하지 않는다.

---

### 문서 출처 및 범위

이 README는 **2026-11-06까지의 프로젝트 대화, 공유된 실제 Windows 실행 결과 및 현재 제공된 코드**를 종합한 개발 기록이다. 전체 사용자 로컬 프로젝트 파일을 직접 확인하거나 모든 테스트를 이 환경에서 재실행한 것은 아니다. 각 성능 수치는 명시된 **실험 조건과 공유된 결과**에 한정된다.

이전 초기 개발 단계 문서는 `README_2026-11-05_legacy.md`로 별도 보관했다.
