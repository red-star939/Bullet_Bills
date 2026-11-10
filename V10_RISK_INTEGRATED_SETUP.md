# PPO v10 + Spawn Shield + Adaptive Altitude + BulletTracker / RiskEstimator / LaneManager / FailureAnalyzerV2

## 설치

ZIP 파일을 Windows 프로젝트 `C:\Users\USER\Desktop\HONG_LB\mario_pj` 루트에 압축 해제합니다. 기존 `agent_ppo_v10.py`, `map_data/object_tiles.csv`, `models/v10_spawn_guidance.py`, `models/adaptive_altitude_controller.py` 및 해당 의존 모듈이 있어야 합니다. 이번 패키지의 `models/bullet_tracker.py`, `models/risk_estimator.py`, `models/lane_manager.py`, `models/failure_analyzer.py`, `models/failure_analyzer_v2.py`, `models/collision_event_audit.py`는 기존 파일을 프로젝트 `models` 경로에서 import할 수 있도록 구성되어 있습니다. 동명 파일을 수정한 적이 있다면 덮어쓰기 전에 비교하세요.

## 1. 기존 모델 동작 유지 + 분석 기능만 추가 (추천 첫 실행)

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe watch_ppo_v10_risk_integrated.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --risk-mode observe `
  --render human `
  --episodes 5 `
  --steps 9000 `
  --seed 20261110 `
  --fps 0 `
  --out logs/v10_risk_observe
```

`observe`는 트래커·TTC·레인 위험도를 계산하고 로그에 저장하지만 **최종 행동을 변경하지 않습니다**. 즉 기존 V10 + Spawn Shield + Altitude에 관측 전용 연결만 추가합니다.

## 2. 위험 경로 우회 후보도 선택적으로 실행 (실험)

```powershell
.\.venv\Scripts\python.exe watch_ppo_v10_risk_integrated.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --risk-mode guard `
  --render human `
  --episodes 5 `
  --steps 9000 `
  --seed 20261110 `
  --fps 0 `
  --out logs/v10_risk_guard
```

`guard`는 포탄 3발 이상 + 짧은 TTC일 때만 현재 행동과 대안을 ForwardSimulator에 넣어 검증합니다. 검증된 다른 후보가 레인 위험 노출을 줄이는 경우에만 1스텝 행동을 교체합니다. 미래 생성 포탄과 숨겨진 상태는 알 수 없으므로 `guard`의 성능 개선은 아직 입증되지 않았습니다. 기존 Shield/Altitude의 개입 뒤에 적용됩니다. 이 실험은 **새로운 정책**이므로 `observe`보다 나빠질 수 있습니다.

## 3. 시각화 없이 동일 시드 장기 비교

동일 시드로 관측만 하는 버전과 행동 개입 버전을 각각 실행합니다.

```powershell
.\.venv\Scripts\python.exe watch_ppo_v10_risk_integrated.py --model models/bullet_bill_ppo_v10_best.pt --risk-mode observe --render none --episodes 20 --steps 9000 --seed 20261110 --profile-every 0 --out logs/risk_base_20
.\.venv\Scripts\python.exe watch_ppo_v10_risk_integrated.py --model models/bullet_bill_ppo_v10_best.pt --risk-mode guard --render none --episodes 20 --steps 9000 --seed 20261110 --profile-every 0 --out logs/risk_guard_20
```

`--steps 9000`은 최대 게임시간 600초(10분)이며 60초에서 중단하지 않습니다. 300초까지만 관찰하려면 4500스텝입니다. 9000스텝 제한에 도달한 에피소드는 **사망이 아니라 중도 절단(censoring)**으로 해석해야 합니다. 실제 게임 자체의 제한은 별도로 존재할 수 있습니다.

## 출력과 진단

- `episodes.csv`: 생존 시간, 종료 이유, 개입 횟수, 상단/하단 체류 비율, 평균·최대 포탄 수, 포탄 신규 추적 수, 실패 라벨
- `failure_ep_XXX_seed_YYY.json`: FailureAnalyzerV2가 최근 4초 이력과 사망 원인을 기록. `EngineCollisionAudit`가 사용 가능하면 실제 엔진 충돌 관측을 포함. 게임의 내부 상태는 진단만 하며 정책에 제공하지 않음
- `summary.json`: 평균 관찰 생존, 최대 생존, 300초/600초 도달 횟수, 스텝 제한 사례 수
- `PERF`: 현재 포탄 수, 신규 추적, 상하단 분포, TTC 및 Shield 교정 횟수

**주의 사항**: `FailureAnalyzerV2`는 원래 ActionEvaluator의 행동 후보 순위를 받도록 만들어졌으므로 PPO v10의 실행기는 빈 후보 순위 진단 어댑터를 제공합니다. 따라서 '다른 후보가 살아남았을 것' 같은 반사실적 평가는 포함되지 않습니다. 엔진 감사가 비활성 또는 미지원이라면 일부 사망 원인은 미확인으로 남습니다. 벽 근처에 등장한 새 포탄과 정확한 충돌 포탄의 일치는 항상 보장되지 않습니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_v10_risk_integration.py
```

테스트는 추적·레인·TTC 연결, `observe`의 행동 불변, FailureAnalyzerV2 진단 어댑터를 확인합니다. 사용자 컴퓨터의 `.pt` 및 Pygame 환경에 대한 실제 평가 결과는 아직 없습니다.
