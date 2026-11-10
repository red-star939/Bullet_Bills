# models2 — 계승형 Adaptive Altitude v3 / Shadow Planner

## 정책 계승 구조

- **v1:** PPO v10 + Spawn Shield + 기존 `AdaptiveAltitudeController` (실제 기준 정책)
- **v2:** 기존 `models2.altitude_v2.AdaptiveAltitudeV2`의 다단계 경로 제안
- **v3-shadow:** v1의 실제 행동을 *한 번도 수정하지 않고* v2가 제안했을 행동/이유를 기록
- **v3-active:** v1 우선, v1의 짧은 **관측 세계** 시뮬레이션에서 즉각적인 충돌/추락이 명시적으로 예측되고 v2 후보의 6스텝 전 구간이 완료/비치명적일 때만 제한적으로 대체

기존 Risk Guard의 행동 교체는 **적용하지 않습니다**. `agent_ppo_v10.py`, 기존 `models/` 및 v1/v2 실행기는 수정하지 않습니다. v2 계획 자체의 예측은 미래의 무작위 포탄 생성과 PPO의 후속 실제 행동을 확정할 수 없습니다.

## 설치

ZIP을 프로젝트 루트 `mario_pj/`에 압축 해제하세요. 기존 `models2/altitude_v2.py`, `models2/tracking.py`, 기존 `models/` 패키지, `agent_ppo_v10.py`, 맵 `map_data/object_tiles.csv`, 그리고 `models/bullet_bill_ppo_v10_best.pt`가 필요합니다.

ZIP에는 **새 파일 4개**만 포함합니다.

```
models2/altitude_v3.py
models2/shadow_planner.py
models2/test_altitude_v3.py
watch_models2_v10_v3.py
```

## 1. 단위 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest models2.test_altitude_v3 models2.test_altitude_v2 -v
```

## 2. v1 기준과 v3 shadow를 동일 시드로 비교

실제 행동은 v1 그대로인데 진단 계산만 추가되므로 생존 시간이 시드별로 일치해야 합니다. 일치하지 않으면 결정성/실행 경로 차이를 먼저 검사해야 합니다.

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe watch_models2_v10_v3.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v3-shadow --render none `
  --episodes 5 --steps 9000 --seed 20261110 `
  --out logs/models2_v3_shadow
```

비교 대상: `logs/models2_v1/episodes.csv` (기존 실험)과 `logs/models2_v3_shadow/episodes.csv`.

```powershell
.\.venv\Scripts\python.exe compare_models2_results.py `
  --v1 logs/models2_v1/episodes.csv `
  --v2 logs/models2_v3_shadow/episodes.csv
```

`shadow_ep_001_seed_20261110.csv` 등은 v2가 v1과 다른 행동을 제안한 순간을 기록합니다. `--shadow-all-steps`를 추가하면 행동 제안이 없는 시점까지 모든 스텝을 기록합니다.

## 3. 보수적인 active 제어 테스트

Shadow 일치 테스트를 확인한 후에만 별도 실행하십시오.

```powershell
.\.venv\Scripts\python.exe watch_models2_v10_v3.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --altitude v3-active --render none `
  --episodes 5 --steps 9000 --seed 20261110 `
  --out logs/models2_v3_active
```

```powershell
.\.venv\Scripts\python.exe compare_models2_results.py `
  --v1 logs/models2_v1/episodes.csv `
  --v2 logs/models2_v3_active/episodes.csv
```

- `v3_proposals`: v2의 행동 차이 제안 수
- `v3_overrides`: 실제 v3 추가 행동 교체 수 (shadow에서는 반드시 0)
- `v3_rejections`: 제안되었지만 안전 증거/조건 부족으로 기각된 수
- `v1_reason`, `v2_reason`, `v3_reason`: 행동 선택 근거
- `active_audit`: 짧은 생존 시뮬레이션 상태 (관측 세계 전용)

## 검증 경계

이 패키지는 Python `py_compile` 및 단위 테스트 9개를 통과했습니다. 제작 환경에는 게임 런타임 `gymnasium`/`kymnasium`이 없어 실제 5회 게임 시뮬레이션을 실행하지 못했습니다. 시드별 성능, 생존 시간, 포탄 밟기 향상은 **아직 주장할 수 없습니다**. 다음 단계는 Shadow=V1 결과 일치 검증과 첫 행동 제안 시점의 로그 확인입니다.
