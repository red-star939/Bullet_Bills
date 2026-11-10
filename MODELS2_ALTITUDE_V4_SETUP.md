# models2 v4 — PPO v10 + Spawn Shield + Adaptive Altitude 계승형 실험

## 목표 및 주의

- **목표 지표**: 이전과 동일한 5시드뿐 아니라 별도 홀드아웃 시드에서도 평균 생존 **150초 이상**. 이 수치는 **목표**이지 현재 실측 성능이 아닙니다.
- 기존 `agent_ppo_v10.py`(148차원), `models/`, `.pt` 체크포인트 **수정 없음**.
- v4는 v1을 기본 실행 정책으로 사용하며 v2·v3의 개선 시도와 실패 분석을 계승합니다. 위험 분석은 관측 전용입니다.
- PPO v10의 Numba `simulate_plans` / `simulate_trajectory`를 사용하는 1.6초 다중 경로 탐색. **새로 생성될 포탄은 정확히 예측하지 못하며**, 게임 엔진과 오차 가능성이 있습니다.
- `shadow`는 v1 행동 그대로 실행. `active`는 유한 예측에서 v1 대안이 사망·추락을 회피한 경우만 교체. `explore`는 고밀도 환경에서 예상 위치 이득이 클 때 전략적 고도 이동을 추가 허용하는 **실험 모드**.

## 설치

ZIP을 `mario_pj` 루트에 풀어 주세요. 기존 `models2` v2/v3도 포함되어 있지만 변경하지 않았습니다. 이 ZIP의 새 파일은 `models2/altitude_v4.py`, `models2/test_altitude_v4.py`, `watch_models2_v10_v4.py`, `benchmark_models2_v4.py`입니다. 기존 v1 모듈인 `models/adaptive_altitude_controller.py` 및 `models/v10_spawn_guidance.py`, 협업자의 `agent_ppo_v10.py`, 게임의 `kymnasium` 패키지가 설치되어 있어야 합니다.

실행 전:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe -m unittest models2.test_altitude_v4 models2.test_altitude_v3 models2.test_altitude_v2 -v
```

## 1. 무개입 v1 재현 및 v4 shadow 검증

```powershell
.\.venv\Scripts\python.exe watch_models2_v10_v4.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v4 --v4-mode shadow `
  --render none --episodes 5 --steps 9000 --seed 20261110 `
  --out logs/models2_v4_shadow
```

`logs/models2_v4_shadow/episodes.csv`의 `survival_s`는 v1 시드별 **96.633, 73.033, 62.100, 74.917, 79.450**초와 동일해야 합니다. 불일치하면 액션이나 환경 결정성 차이를 먼저 점검하세요. `v4_overrides`는 **반드시 0**이어야 합니다.

## 2. 보수적 긴급 회피 모드

```powershell
.\.venv\Scripts\python.exe watch_models2_v10_v4.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v4 --v4-mode active `
  --render none --episodes 5 --steps 9000 --seed 20261110 `
  --out logs/models2_v4_active
```

## 3. 전략적 고도 탈출 탐색 모드 (신규 실험)

```powershell
.\.venv\Scripts\python.exe watch_models2_v10_v4.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v4 --v4-mode explore `
  --render none --episodes 5 --steps 9000 --seed 20261110 `
  --out logs/models2_v4_explore
```

v4가 제안한 행동은 `shadow_ep_*.csv`에 기록됩니다.  `episodes.csv`에 `v4_proposals`, `v4_overrides`, `v4_plans_tested`, `v4_reasons`가 추가됩니다. `failure_ep_*.json`은 사망 사후 진단용이며 행동/보상에는 반영하지 않습니다.

## 4. v1 대비 시드 매칭 비교

```powershell
.\.venv\Scripts\python.exe compare_models2_results.py `
  --v1 logs/models2_v1/episodes.csv `
  --v2 logs/models2_v4_explore/episodes.csv
```

## 5. 자동 후보 탐색 + 홀드아웃 검증

```powershell
# 빠른 개발 점검: 기존 5시드에서 3개 프로필 비교
.\.venv\Scripts\python.exe benchmark_models2_v4.py --quick `
  --model models/bullet_bill_ppo_v10_best.pt `
  --out logs/v4_benchmark_quick

# 정식: 튜닝용 5시드 + 겹치지 않는 홀드아웃 15시드
.\.venv\Scripts\python.exe benchmark_models2_v4.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --train-seeds 5 --holdout-seeds 15 `
  --out logs/v4_benchmark_holdout
```

벤치마크는 `active_default`, `explore_conservative`, `explore_proactive`를 v1과 같은 시드로 비교합니다. 튜닝 5시드 평균 생존이 v1보다 나아진 설정만 홀드아웃에서 검증하고, 모두 나빠지면 v1을 선택합니다. 홀드아웃 결과를 보고 프로필을 다시 선택하면 테스트 세트 누수가 생기므로 별도 시드를 사용하세요.

## 결과 판정

- **성능 목표 충족**: 홀드아웃 평균 생존 150초 이상, 가능하면 v1 대비 양의 paired delta.
- 최솟값, 중앙값, 120/150/180초 도달률, death type, 개입 횟수도 함께 확인합니다.
- v4가 150초에 도달하지 못하면 목표 미달이며, **PPO 학습 재설계나 더 긴 단계적 학습이 필요**할 수 있습니다. 컨트롤러가 150초를 보장하는 것은 불가능합니다.

## 안전성·성능 한계

1. v4 평가는 실제 환경을 복제하는 완전한 카운터팩추얼 실험이 아닙니다. 행동 실행 뒤 시뮬레이션 매 스텝 새로 계획하며, 예측 밖의 포탄 출현이 영향을 줍니다.
2. `explore` 모드는 v1보다 나빠질 수 있습니다. 충분한 반복 검증 전 배포하지 마세요.
3. 별 근처에서는 무적 지속 시간 추적 불확실성 때문에 v1을 보존합니다.
4. Numba JIT 최초 실행은 시간이 길 수 있습니다. `render none`과 `torch_threads=1`을 권장합니다.
5. 관측 포탄 패딩 `[0,0,0,0,0]`은 v4 계획 단계에서 제거합니다. 기존 `FailureAnalyzerV2`의 추적 패딩 문제는 별도 수정해야 합니다.
