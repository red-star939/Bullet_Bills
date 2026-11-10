# ActionEvaluator v3 — Model-based recovery controller (experimental)

## 목표와 실제 상태

장기 목표는 **독립적인 신규 시드에서 평균 생존 600초 이상**이다. 이 버전은 목표를 향한 **첫 번째 구조 변경**이며, 실제 BulletBill 게임에서 600초 생존을 달성했거나 v2보다 우수하다고 검증된 코드는 아니다. v2의 기존 60초 제한 평균 관측 생존 시간은 34.1343초(100시드)다.

## 구현한 설계

- 기존 v2의 행동 선택을 **기준선**으로 그대로 사용한다.
- ForwardSimulator에서 선택 행동을 20스텝(약 1.33초) 유지하는 미래를 검증하고 PlatformSafetyEvaluator로 지지/가장자리/낮은 고도 공중 상태를 분석한다.
- 기본 행동에서 **예측 사망/추락 또는 불안정한 착지 조건**이 감지된 경우에만 추가 탐색을 실행한다.
- 후보 행동 5개, 전환 시점 2개, 이후 행동 6개에 대해 `초기 행동 → 이후 다른 행동` 2단계 계획을 시뮬레이션한다(최대 60회 추가 예측 + 기준 예측 1회). 실제 게임에는 **첫 행동만 실행**, 다음 관측에서 재계획한다.
- 불확실한 별 획득 등 미래가 확인되지 않는 결과는 안전한 대안으로 승격하지 않는다.
- v2의 행동에 알려진 세계에서 명시적인 사망 위험이 없고 충분히 지지된 경우 **v2 행동을 유지**한다.
- 포탄 신규 생성, 무적 잔여시간, 관측 포탄의 숨겨진 상태는 아직 예측되지 않는다. 따라서 **안전성 보장이 아니다**.

## 프로젝트 배치

프로젝트 루트에서 압축을 해제하고 기존 `models/*.py`는 유지한다. 추가하는 파일:

- `models/action_evaluator_v3.py`
- `test_action_evaluator_v3.py`
- `benchmark_action_evaluator_v3.py`
- `watch_action_evaluator_v3.py`

`agent_env.py`는 기존 프로젝트에서 사용한 파일이며 벤치마크 실행에 필요하다. 원본 화면 뷰어는 `manual_play.py`와 같은 `gymnasium`, `kymnasium`, `pygame`을 사용한다.

## 테스트 (PowerShell)

```powershell
.\.venv\Scripts\python.exe test_action_evaluator_v3.py
.\.venv\Scripts\python.exe test_action_evaluator_v2.py
.\.venv\Scripts\python.exe test_platform_safety_evaluator.py
```

## 원본 게임 화면에서 직접 행동 보기

```powershell
.\.venv\Scripts\python.exe watch_action_evaluator_v3.py `
  --episodes 5 `
  --steps 900 `
  --fps 15
```

이 뷰어에는 PPO 또는 `.pt`가 필요 없다. v3의 행동을 직접 보는 용도이다. 렌더링은 직접 비용이 들므로 정확한 계산 성능 비교에는 쓰지 않는다.

## 1단계: 동일 시드 60초 비교

```powershell
.\.venv\Scripts\python.exe benchmark_action_evaluator_v3.py `
  --reference-episodes 20 `
  --holdout-episodes 20 `
  --steps 900 `
  --report-every 5 `
  --out logs/v3_smoke
```

**중요:** v3 탐색은 v2보다 훨씬 느릴 수 있다. 먼저 소수 시드에서 `v3_ms_decision`과 생존 시간 모두 확인하라.

## 2단계: 10분/20분 평가

환경이 600초 이상 지속 가능한지 먼저 확인하고 실행한다. 한 행동은 내부 물리 4프레임이므로 600초=9,000 행동, 1,200초=18,000 행동이다. `--steps 18000`은 측정 상한이지 자동으로 20분을 실행한다는 보장이 아니다.

```powershell
.\.venv\Scripts\python.exe benchmark_action_evaluator_v3.py `
  --reference-episodes 10 `
  --holdout-episodes 20 `
  --steps 18000 `
  --report-every 2 `
  --out logs/v3_long
```

각 에피소드 쌍을 끝낼 때마다 `episode_pairs.csv`를 덧붙여 저장한다. 작업 중단 후 동일한 명령에 `--resume`을 추가하면 저장된 시드를 건너뛴다. 동일 출력 경로에 서로 다른 설정을 섞으면 안 된다. `summary.json`은 완료 시에만 생성된다.

## 평가할 항목

1. 60초 제한 시드에서 v2 대비 **paired delta** (v3-v2), 퇴행 시드 수, 60초 생존율.
2. v3의 추가 미래 예측 수, 결정당 지연. v2보다 느려진 만큼 개선이 있는지.
3. 실제 포탄 충돌/추락 사망 원인: 기존 FailureAnalyzer를 별도로 연결해 확인.
4. 긴 평가에서 300/600/1200초 도달 에피소드 비율, 종료(`terminated`), 환경 자체 `truncated`, 스텝 상한 검열 분리.
5. 모델 변경·튜닝 과정에서 사용하지 않은 Holdout 집합에서 재검증.

## PPO 연결: 아직 자동 통합되지 않음

현재 파일은 **독립적인 v3 제어기**이며, 이전 `collect_ppo_teacher.py`와 `train_ppo_guided_parallel.py`는 여전히 **v2**를 사용할 수 있다. v3가 실환경 벤치마크에서 우수한 것을 확인한 다음 `models/model_guidance.py`의 교사 생성자를 `ActionEvaluatorV3`로 변경하고, 별도의 **v3 교사 데이터셋**을 만들어 PPO를 학습해야 한다. 기존 v2 데이터셋을 v3 데이터셋으로 이름만 바꾸면 안 된다.

## 미해결 한계

이산 8개 행동·2단계 전환 계획만 검색한다. 1.33초 시야 밖의 새 포탄 또는 장기적 위험은 예측하지 못한다. 과도한 보수적 개입이 성능을 떨어뜨릴 수 있다. 실제 600초 목표에는 긴 벤치마크, 위험 상황 회귀 시드, 확률적 포탄 시나리오 모델, 실시간 탐색 비용 최적화가 추가로 필요하다.
