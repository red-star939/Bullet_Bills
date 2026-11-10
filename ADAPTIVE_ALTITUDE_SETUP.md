# Adaptive Altitude Shield v1 — 실행 안내

기존 PPO 모델 가중치를 변경하지 않는 추론 전용 전략적 고도 제어기입니다. **실제 게임 생존 성능은 아직 검증되지 않았습니다.**

## 파일 설치

`bulletbill_adaptive_altitude_v1.zip`의 파일을 `C:\Users\USER\Desktop\HONG_LB\mario_pj`에 **폴더 구조를 유지**하며 압축 해제합니다. 기존 `models/forward_simulator.py`, `models/model_guidance.py`, `models/spawn_risk_evaluator.py`, `models/spawn_aware_guidance.py`, `models/v10_spawn_guidance.py`, `agent_ppo_v10.py` 및 PPO 학습 체크포인트는 그대로 두세요. **기존 파일을 덮어쓰지 않습니다.**

## 1. 협업자 v10 + 기존 Spawn Shield + 새 Altitude

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.\.venv\Scripts\python.exe watch_ppo_v10_altitude.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --guidance altitude_shield `
  --episodes 5 `
  --steps 9000 `
  --fps 0 `
  --seed 20261110
```

## 2. 우리 PPO 440만 스텝 + 기존 Spawn Shield + 새 Altitude

```powershell
.\.venv\Scripts\python.exe watch_ppo_altitude.py `
  --model models/ppo_guided_30m_resume/mario_ppo_step_004403200.pt `
  --render human `
  --guidance altitude_shield `
  --episodes 5 `
  --steps 9000 `
  --seed 20261110
```

`--guidance spawn_shield`로 같은 뷰어를 실행하면 기존 Shield만 적용합니다.

## 3. 동일 시드 비교 (렌더링 없음)

```powershell
.\.venv\Scripts\python.exe benchmark_ppo_v10_altitude.py `
  --model models/bullet_bill_ppo_v10_best.pt `
  --episodes 30 `
  --steps 9000 `
  --seed 20261110 `
  --out logs/v10_altitude_10min
```

```powershell
.\.venv\Scripts\python.exe benchmark_ppo_altitude.py `
  --model models/ppo_guided_30m_resume/mario_ppo_step_004403200.pt `
  --episodes 30 `
  --steps 9000 `
  --seed 20261110 `
  --out logs/our_ppo_altitude_10min
```

각 비교는 `spawn_shield` 대 `altitude_shield`이고 모델과 시작 시드는 동일하지만, 행동 차이로 이후 무작위 포탄 경로가 달라질 수 있습니다. **9000스텝=게임 시간 약 600초**로, 제한에 도달한 에피소드는 사망 관측이 아닌 right-censored 기록입니다. `--steps 18000`은 최대 약 20분입니다. 게임 엔진 자체 조기 종료가 있다면 별도로 확인해야 합니다.

## 4. 동작 원리와 한계

- 마리오가 상단 목표 높이보다 아래에 있고, 관측 포탄이 하단에 3발 이상이며 상단 대비 2발 이상 많아야 후보를 고려합니다.
- 3회 연속 혼잡 관측 후, 기존 행동과 점프 이동 후보의 약 1.47초 known-world 미래 예측을 수행합니다.
- 상단 이동량 최소 45px, 기존 행동 대비 추가 상승 24px, 예측 종료 위치의 벽면 이격 65px 이상이고 예측 중 포탄 사망/추락/미해결 별 획득이 없는 행동만 선택합니다.
- 스위칭 후 25 행동 스텝은 새 고도 개입을 하지 않습니다. **이것은 지속적인 경로 추종기가 아니라 단발 점프 이탈 개입입니다.**
- 기본 Spawn Shield가 먼저 행동을 검사합니다. Altitude 제어기는 마지막에 작동하며 자체 ForwardSimulator 검사를 통과한 후보만 사용할 수 있습니다.
- 상단·하단 혼잡은 관측된 포탄 수의 단순 분포입니다. 발사 확률 추정, 숨겨진 포탄/미래 생성 예측, 완전한 위치계획, 상단 안전 보장은 아닙니다.
- 원래 포탄 밟기(곡예) 행동은 혼잡 조건이 충족되지 않을 때 그대로 남고, 필요 시에도 안전한 이동 후보가 확인되지 않으면 원래 행동을 유지합니다.
- `agent_ppo_v10.py` 자체 추론이 느리다면 이 모듈은 계산을 더하므로 화면 버벅임을 완전히 해결하지는 못합니다.

## 5. 단위 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest test_adaptive_altitude_controller -v
```

정량 평가에서는 평균 생존 시간 외에 하단 체류비율, 곡예(포탄 stomp) 횟수, 탈출 후 실제 생존, 300초/600초 도달 비율을 별도로 추가 수집하는 것이 다음 단계입니다.
