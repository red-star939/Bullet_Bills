# models2: PPO v10 + Spawn Shield + Adaptive Altitude v2

## 개요

- 기존 `agent_ppo_v10.py`, `models/` 및 `models/bullet_bill_ppo_v10_best.pt` **수정하지 않음**.
- 새 `models2/altitude_v2.py`는 **조건부 2단계 행동열**을 기존 `models.forward_simulator.ForwardSimulator`로 검사합니다.
- 위험 관측은 `models2/tracking.py`를 사용하며 패딩된 0 크기 포탄 행을 제외합니다.
- **Risk Guard의 행동 덮어쓰기는 사용하지 않습니다.**
- `FailureAnalyzerV2`의 사망 보고서와 엔진 충돌 감사는 사후 진단에만 활용됩니다.
- 미래 랜덤 포탄은 예측할 수 없으며, 기존 ForwardSimulator의 known-world 안전 판정은 실제 생존 보증이 아닙니다.
- **현재 단계는 학습이 아닌 시뮬레이션 비교(A/B)** 입니다.

## 설치

ZIP을 `C:\Users\USER\Desktop\HONG_LB\mario_pj`에 압축 해제하세요. 기존 프로젝트에 다음 파일이 있어야 합니다:

```
mario_pj/
  agent_ppo_v10.py
  map_data/object_tiles.csv
  models/                      # 기존 모듈, 변경하지 않음
    v10_spawn_guidance.py
    adaptive_altitude_controller.py
    v10_risk_integration.py
    failure_analyzer_v2.py
    failure_analyzer.py
    forward_simulator.py
    bullet_tracker.py
    lane_manager.py
    risk_estimator.py
    ...
  models/bullet_bill_ppo_v10_best.pt
  models2/                     # 새 폴더
    __init__.py
    altitude_v2.py
    tracking.py
    test_altitude_v2.py
  watch_models2_v10.py
  compare_models2_results.py
```

프로젝트 루트에서 실행하세요.

## 검사

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
.\.venv\Scripts\python.exe -m unittest models2.test_altitude_v2 -v
```

## 1차 성능 비교: 같은 5개 시드

```powershell
.\.venv\Scripts\python.exe watch_models2_v10.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v1 --render none --episodes 5 --steps 9000 `
  --seed 20261110 --out logs/models2_v1

.\.venv\Scripts\python.exe watch_models2_v10.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v2 --render none --episodes 5 --steps 9000 `
  --seed 20261110 --out logs/models2_v2

.\.venv\Scripts\python.exe compare_models2_results.py `
  --v1 logs/models2_v1/episodes.csv `
  --v2 logs/models2_v2/episodes.csv
```

시각화하고 싶으면 `--render human`을 지정합니다. 같은 시드를 쓰더라도 라이브러리 버전·기타 난수와 관측 처리 순서가 같아야 완전한 재현성을 보장할 수 있습니다.

## v2 동작 상세

- 하단 포탄이 8발 이상이고 상단보다 5발 이상 많으며 3스텝 지속된 경우에만 기본 탐색을 수행합니다.
- 5스텝 이상 간격을 두어 `18`스텝(1.2초) 예측을 수행합니다.
- 첫 4스텝 입력과 나머지 14스텝 입력을 달리하는 **2구간 행동열**을 최대 18개 평가합니다.
- 기존 행동이 예측상 종료되어도 대안을 검사합니다(기존 v1 결함 완화).
- 안전성 판정이 불확실한 행동열(별·중도 종료)은 성공으로 취급하지 않습니다.
- 경로의 현재 포탄 밟기 예측을 긍정적으로 평가합니다. 다만 미래 포탄 생성은 known-world 평가에 반영되지 않습니다.
- 실제 게임에는 매 스텝 첫 행동만 적용하며, 두 구간 계획을 강제로 끝까지 실행하지 않습니다.
- PPO v10의 148차원 특징 및 원래 모델 가중치는 변경하지 않습니다.

## 알려진 제한 및 다음 개발

1. 새로운 Guard를 더한 5개 시드 비교에서 성능이 떨어졌으므로, v2 역시 **v1보다 좋아진다는 보장이 없습니다**.
2. 현재 v2는 포탄이 밀집한 구간에서만 탐색하므로, 그 외의 위험 상황에서는 기존 PPO+Spawn Shield 행동을 존중합니다.
3. `agent_ppo_v10.py`는 `SPAWN_FRAMES`를 기본 180초까지만 만들어 장기 평가에서 생성 예측이 비게 되는 문제가 있습니다. **원본은 아직 수정하지 않았습니다.** 180초 이상의 장기 평가 전에 수정 및 엔진 규칙 확인이 필요합니다.
4. 포탄 추적 패딩 오류는 이 실행기의 RiskObserverV2 경로에 한해 관측 입력에서 방지합니다. 기존 `models/bullet_tracker.py` 원본은 변경하지 않습니다.
5. PPO 모델 체크포인트와 `kymnasium` 환경이 이 생성 환경에 없어 실제 게임 통합 실행·성능 증가는 아직 검증되지 않았습니다.
