# models2 — Adaptive Altitude v4.1 (계승형 Closed-loop + Spawn-aware + Intervention Audit)

## 목적 및 실제 검증 상태

기존 v1→v2→v3→v4를 **변경하지 않고** 새 파일만 추가하는 v4.1 실험 패키지입니다.

- 최근 측정 기준: v1 5시드 평균 **77.227초**, v4-explore 5시드 평균 **78.330초**, v4 최고 **112.317초**.
- 지향점: 여러 **별도 홀드아웃 시드**에서 평균 생존 **150초 이상**. **아직 달성되지 않았으며 보장하지 않습니다.**
- 협업자 `agent_ppo_v10.py`의 148차원 PPO 모델과 `.pt` 체크포인트, `models/`, 기존 `models2/altitude_v2/v3/v4.py` **원본을 그대로 사용**합니다. 원본 파일을 수정하지 않습니다.
- Risk Guard는 관측 전용이고 액션을 교체하지 않습니다.
- 필요: 이전 v4 패키지를 이미 설치한 프로젝트. 이번 ZIP은 v4.1 **증분 패치**이며 기존 v4 파일을 덮어쓰지 않습니다.

## 추가 파일

```text
mario_pj/
├── agent_ppo_v10.py                    # 협업자 원본, 기존 상태
├── models/                             # 원본 그대로
├── models2/
│   ├── altitude_v2.py                  # 기존 유지
│   ├── altitude_v3.py                  # 기존 유지
│   ├── altitude_v4.py                  # 기존 유지
│   ├── altitude_v4_1.py                # [신규] 계승형 개입 승인 로직
│   ├── closed_loop_planner.py          # [신규] 가상 관측에서 PPO 재추론
│   ├── spawn_aware_risk.py             # [신규] 미래 포탄 스트레스 시나리오
│   ├── intervention_audit.py           # [신규] 한 스텝 예측 대 실제 관측 감사
│   └── test_altitude_v4_1.py           # [신규] 회귀 테스트
├── watch_models2_v10_v4_1.py           # [신규] 실행기: v1~v4.1 모두 지원
└── analyze_models2_v4_1.py            # [신규] 5/20/50 시드 성능 비교
```

## 기능의 구체적 정의

### 1. Closed-loop Trajectory Planner

`models2/closed_loop_planner.py`는 우선 v1/v4가 결정한 액션을 1스텝 실행한 것으로 물리 예측하고, **상상한 다음 관측**으로 PPO v10의 `FeatureExtractor`와 결정적 ActorCritic 정책을 다시 호출합니다. 이후에도 매 스텝 갱신해 12스텝(0.8초) 경로를 구성합니다. 고정 행동 시퀀스를 그대로 실행하는 v4와 달리 PPO의 재결정을 근사합니다.

- 각 경로는 **독립적인 FeatureExtractor**를 사용하며 실제 에이전트 내부 특징 추적기를 수정하지 않습니다.
- 협업자의 Numba `_run_ext`를 이용해 공중 물리, 충돌, 밟은 포탄의 제거, 별 획득을 가상 상태로 이어서 계산합니다.
- **한계**: 가상 관측에 나타나지 않은 새로운 포탄은 실제로 삽입하지 않습니다. 미래 PPO가 Shield 또는 Altitude에 의해 추가 변경되는 과정까지 재현하지는 않습니다. 모델/실제 상태의 차이도 남습니다. 따라서 *정확한 미래 폐루프 시뮬레이션이 아닌 근사 모델*입니다.

### 2. Spawn-aware Risk Evaluation

`models2/spawn_aware_risk.py`는 협업자의 공개 관측 기반 스폰 일정을 사용하고, 포탄 높이(마리오 top, top-48, top-96, top-144), 좌·우 출현, 프레임당 3·4·5px을 조합한 가상 출현 시나리오를 평가합니다.

- `exposure_fraction`은 **가상 시나리오의 접촉 비율**로, 통계적으로 보정된 실제 사망 확률이 아닙니다.
- 180초 이후의 스폰 규칙은 협업자의 제공된 180초 일정에서 검증할 수 없어 **외삽하지 않습니다**. 그 시점의 승인 로직은 v1/v4 기존 행동을 보존하도록 차단합니다. 실제 환경 소스의 180초 이후 생성 규칙이 확인되면 별도 확장해야 합니다.
- 보이지 않는 포탄을 `simulate_plans`에 확정적인 실포탄으로 주입하지 않습니다.

### 3. Intervention Audit

`models2/intervention_audit.py`는 실행된 **실제 액션**의 한 스텝 예측과 **실제 다음 관측**의 좌표/속도 오차, 예상치 못한 사망, 예측했으나 관측되지 않은 사망을 기록합니다.

- 모든 게임 스텝에 대해 `audit_ep_001_seed_*.csv` 등이 생성됩니다.
- v1 기준 행동, v4.1 제안 행동, **실제로 실행한 행동**, 예측 충돌 프레임, 스폰 노출도, 실제 위치 오차를 기록합니다.
- 대체 행동의 미래 결과는 실제로 실행하지 않는 한 **가상 예측값**이며 사후 정답(반사실적 결과)이 아닙니다.
- 무적 잔여 시간이 관측되지 않는 상태에서는 1스텝 예측을 불확실로 표기합니다.

## 실행 절차 (PowerShell)

프로젝트 루트 `mario_pj`에서 ZIP을 압축 해제합니다.

### 0. 단위 테스트

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe -m unittest `
  models2.test_altitude_v4_1 `
  models2.test_altitude_v4 `
  models2.test_altitude_v3 `
  models2.test_altitude_v2 -v
```

### 1. v4.1 Shadow — v1과 행동 동일성 검증

```powershell
.\.venv\Scripts\python.exe watch_models2_v10_v4_1.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v4.1 `
  --v41-mode shadow `
  --v41-base-mode shadow `
  --render none `
  --episodes 5 `
  --steps 9000 `
  --seed 20261110 `
  --out logs/models2_v41_shadow
```

기본 `--v41-base-mode shadow`에서는 원래 v4가 v1 액션을 바꾸지 않습니다. 이때 `v41_overrides=0`이고 시드별 생존 시간도 기존 v1과 일치해야 합니다. 단, 누락된 난수 결정성이나 실행 환경이 다르면 일치하지 않을 수 있으므로 차이를 조사해야 합니다.

### 2. v4.1 Active — 제한된 긴급 회피 개입 실험

```powershell
.\.venv\Scripts\python.exe watch_models2_v10_v4_1.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v4.1 `
  --v41-mode active `
  --v41-base-mode shadow `
  --render none `
  --episodes 5 `
  --steps 9000 `
  --seed 20261110 `
  --out logs/models2_v41_active
```

### 3. v4.1 Explore — 미래 스폰 노출도 감소로 전략적 개입

```powershell
.\.venv\Scripts\python.exe watch_models2_v10_v4_1.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v4.1 `
  --v41-mode explore `
  --v41-base-mode shadow `
  --render none `
  --episodes 5 `
  --steps 9000 `
  --seed 20261110 `
  --out logs/models2_v41_explore
```

**주의**: `explore`는 엄격한 시나리오 점수 차이가 확인되면 실제 행동을 교체합니다. 성능 개선은 미검증이며 v4나 v1보다 악화될 수도 있습니다.

### 4. 게임 화면 직접 보기

```powershell
.\.venv\Scripts\python.exe watch_models2_v10_v4_1.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v4.1 --v41-mode explore `
  --render human --episodes 1 --steps 9000 `
  --seed 20261112 `
  --out logs/models2_v41_visual_20261112
```

기존 v4에서 34.98초에 사망했던 `20261112` 시드를 우선 관찰할 수 있습니다. `Esc`나 게임 창 닫기로 중단합니다.

### 5. v1·v4·v4.1 시드별 비교

```powershell
.\.venv\Scripts\python.exe analyze_models2_v4_1.py `
  --v1 logs/models2_v1/episodes.csv `
  --v4 logs/models2_v4_explore/episodes.csv `
  --v41 logs/models2_v41_explore/episodes.csv `
  --out logs/models2_v41_comparison.json
```

5시드 개발 실험 이후 **튜닝에 사용하지 않은 추가 15~50시드** 평가가 필요합니다. 이전 5시드만 반복 조정해 얻은 150초 평균은 신뢰할 수 있는 일반화 근거가 아닙니다.

## 생성 로그와 진단 포인트

- `episodes.csv`: `v41_proposals`, `v41_overrides`, `v41_rollouts`, `v41_reasons`, `v41_audit` 포함.
- `shadow_ep_*.csv`: 검토된 후보와 승인/거절 이유, 기준/후보 스폰 노출도.
- `audit_ep_*.csv`: **모든 스텝**, 예측 대 실제 좌표/속도 오차, 예상치 못한 사망.
- `failure_ep_*.json`: 별도 사망 원인 분석 (엔진 감사 정보는 **사후 진단용만** 사용하며 행동 결정에는 사용하지 않음).
- `summary.json`: 평균/최고 생존, **120/150/180/300/600초 도달률**.

**경고**: 기존 FailureAnalyzer의 0-패딩 추적 로그는 이번에 직접 변경하지 않았습니다. v4.1 경로 예측과 새 스폰 분석은 유효 bounding box만 사용하도록 필터링합니다.

## 평가 기준

1. shadow 실행에서 v1 재현을 확인합니다.
2. active, explore를 동일 시드로 평가하고 **개입 횟수뿐 아니라** `model_divergences`, `unexpected_deaths`, `paired_delta_s`를 함께 확인합니다.
3. 성능이 감소하면 해당 모드 적용을 중단하고 `audit_ep_*.csv`로 **최초 행동 차이 및 예측 불일치**를 조사합니다.
4. 홀드아웃 평균 150초가 실제 측정될 때만 목표 달성으로 기록합니다. 실패 시에는 모델 재학습/환경 모델 재보정이 필요할 수 있습니다.
