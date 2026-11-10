# Spawn-Aware Safety Shield — 실험 버전

## 목적

PPO가 좌·우 벽의 포탄 생성 위치에 장시간 머물거나 더 다가가는 현상을 완화하는 **추론 시점 안전 보조장치**입니다. 기존 PPO 가중치(`.pt`), `ActionEvaluatorV2`, `ForwardSimulator`와 `PlatformSafetyEvaluator`의 소스는 바꾸지 않습니다.

- `models/spawn_risk_evaluator.py`: 실제 MarioMotionPredictor로 각 후보 행동의 12스텝(약 0.8초) 궤적을 예측하고, 벽 근처 체류 정도를 *노출 프록시 점수*로 환산합니다. 이 점수는 실제 포탄 생성 확률이나 사망 확률이 아닙니다.
- `models/spawn_aware_guidance.py`: 기존 Shield의 판단을 먼저 실행합니다. 벽 근접 후보가 있고, 다른 후보가 유의미하게 벽으로부터 멀어지며, **기존 ForwardSimulator의 알려진 세계에서 사망/추락 없이 예측을 완수했을 때만** 행동을 바꿉니다.
- `watch_ppo_spawn_aware.py`: 원본 `render_mode='human'` Pygame 시뮬레이션 재생기. `--guidance ppo`, `shield`, `teacher`, `spawn_shield` 지원.
- `benchmark_spawn_shield.py`: 동일 시드에서 세 모드 `ppo / shield / spawn_shield`를 각각 실행하고 에피소드별 CSV 즉시 저장, `--resume` 가능.
- `test_spawn_risk_evaluator.py`: 벽 좌·우, 중앙, 실모듈 연동 회귀 테스트.

## 설치

압축 내용을 `mario_pj` 루트에 기존 구조대로 추출하십시오. `models/ppo_policy.py`, `models/model_guidance.py`, `agent_env.py`, `map_data/object_tiles.csv` 등 기존 프로젝트 파일은 **유지**합니다. 신규 파일만 추가합니다.

```powershell
.\.venv\Scripts\python.exe test_spawn_risk_evaluator.py
```

## 원본 게임 화면 비교

새 모델 학습이 필요하지 않습니다. 이전 `.pt` 파일 그대로 사용합니다.

```powershell
.\.venv\Scripts\python.exe watch_ppo_spawn_aware.py `
  --model models/ppo_guided_30m_resume/mario_ppo_step_004403200.pt `
  --render human `
  --guidance spawn_shield `
  --episodes 5 `
  --seed 20261110
```

기존 정책과 비교하려면 `--guidance shield`를 사용하고 **같은 `--seed`를 지정**하십시오. 이 명령은 게임 화면을 재생하는 것이며 학습하지 않습니다. 별도의 `--guidance ppo`도 지원합니다.

## 동일 시드 벤치마크

```powershell
.\.venv\Scripts\python.exe benchmark_spawn_shield.py `
  --model models/ppo_guided_30m_resume/mario_ppo_step_004403200.pt `
  --episodes 30 `
  --steps 900 `
  --seed 20261110 `
  --out logs/spawn_shield_30
```

결과는 `logs/spawn_shield_30/episode_results.csv`, `summary.json`에 누적 기록됩니다. 기존 결과를 이어 받으려면 **동일한 인자**에 `--resume`을 추가하십시오. 평가 시드가 같아도 정책에 따라 미래 난수 호출 순서가 달라질 수 있습니다. `spawn_overrides`는 벽 위험을 이유로 기존 Shield가 선택한 행동을 추가로 교체한 횟수, `wall_step_fraction`은 각 행동 결정 이전에 마리오 AABB가 경계에서 180px 이내에 있었던 비율입니다.

## 초기 설정 및 주의

- `--caution-px 180`, `--critical-px 96`: **가설 기반 초기 값**입니다. 실제 게임의 포탄 출현 고도와 빈도를 측정해 조정해야 합니다.
- 고도별 생성 분포, 무적 만료 시점, 알려지지 않은 미래 포탄은 아직 모델링하지 못합니다. 따라서 벽에 가까운 것이 반드시 치명적인 상황이라고 주장하지 않습니다.
- 벽을 무조건 차단하면 별을 획득하거나 위험에서 탈출하는 행동까지 해칠 수 있습니다. 따라서 기본적으로 기존 `shield`를 유지하고 매우 근접하거나 벽으로 접근하는 경우만 검사합니다.
- `spawn_shield`는 **추론 시 행동 교체**입니다. 기존 PPO 학습 루프에 직접 적용하거나 교체한 행동을 PPO가 샘플링한 것으로 간주해서는 안 됩니다.
- 독립 Holdout에서 기존 Shield보다 생존 시간 및 벽면 사망률이 개선되는 것이 확인되기 전에는 기본 모드로 승격하지 마십시오.

실제 BulletBill 실행 테스트는 이 패키지 내에서 수행하지 않았습니다. 오프라인 모듈 회귀 테스트만 통과했습니다.
