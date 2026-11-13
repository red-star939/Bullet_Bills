# Model V4: 50만 스텝 정책 진단 및 최소 추가 학습

## 제공 내용

`real/model_v4/train_parallel_500k_diagnostic.py`는 원본 `train_parallel_500k.py`를 **수정하지 않는 별도 진단용 학습기**입니다. 원본 `model_v4_net.py`를 그대로 import하고 `model_v4_500k.pt`를 초기 가중치로 받아 아주 짧게 학습하며 세부 지표를 로깅합니다. PPO v10 의존성은 없습니다.

기존 보상 `0.1 - 10*death + 2*successful_terminal`은 유지합니다. 모델 행동 방식, 관측 인코더, 보상 함수를 변경하지 않았으므로 이 도구는 **원인 진단용**이며 수평 이동 문제를 해결했다고 주장하지 않습니다.

## 설치

ZIP의 `real/model_v4/train_parallel_500k_diagnostic.py`를 프로젝트 최상위 `mario_pj` 아래에 추출합니다. 기존 파일은 덮어쓰지 않습니다.

## 단기 학습 및 진단 (4096 스텝)

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe real/model_v4/train_parallel_500k_diagnostic.py `
  --init real/model_v4/model_v4_500k.pt `
  --total-steps 4096 `
  --envs 16 `
  --batch-steps 256 `
  --device cpu `
  --ppo-epochs 2 `
  --checkpoint-every 4096 `
  --checkpoint-dir real/model_v4/checkpoints `
  --out real/model_v4/model_v4_diagnostic_4k.pt
```

모델은 항상 학습된 50만 스텝 파일과 다른 이름으로 저장합니다. `--init`은 **추가 4096 스텝을 새 실행으로** 시작합니다. 최종 출력 `.pt`의 `steps`에는 이 실행의 4096이 들어가며 누적 총 504096은 기록하지 않습니다.

## 해석

- `mu_abs`: 결정적 정책의 수평 이동 평균 절댓값. `move_abs`는 **샘플링된** 행동의 절댓값입니다.
- `std_mean`: 수평 행동 정규분포 표준편차.
- `actual_dx_abs`: 게임에서 실제 측정된 수평 이동량.
- `approx_kl`: 행동 정책 변화량의 근사치.
- `actor_loss`, `value_loss`, `entropy`, `grad_norm`: 업데이트가 어떻게 진행되는지 확인.
- `deaths`: 해당 롤아웃에서 감지한 사망 수. `game_state == 'game_over'` 검출에 의존합니다.

관측 `mario` 필드가 [left, top, right, bottom] 형식이면 X 중앙 좌표가 기록됩니다. 다른 스키마라면 `get_mario_x()`를 게임 환경에 맞게 바꾸어야 합니다. 진단에서 `dx_available_pct`가 낮게 나오면 실제 위치 측정을 신뢰하지 마세요.

중간 재개 체크포인트는 `checkpoints/model_v4_diagnostic_resume.pt`로 분리 저장합니다.

## 발견된 구조적 원인

1. 수평 행동은 학습할 때 Gaussian sample, 화면 평가에서는 분포 평균값을 사용합니다. 따라서 무작위 이동만 학습 로그에 나타날 수 있습니다.
2. `obs_vector()`가 중첩 딕셔너리를 정렬/평탄화하여 256차원으로 자르므로 정보가 누락되거나 의미적 구분이 어려울 수 있습니다.
3. 생존/사망 보상만 사용하며 회피 방향의 중간 신호가 없습니다. 이 사실만으로 보상 오류로 단정할 수 없습니다.
4. `clip=0.0`은 학습 불능의 증거가 아닙니다. `approx_kl`, `grad_norm`을 같이 확인해야 합니다.

새 인코더/이산 수평 행동/위험 보상은 별도의 정책 버전 설계와 통제 실험이 필요한 **후속 수정**입니다.

## 검증

가짜 환경에서 8스텝의 행동 수집, 이동량 측정, PPO 업데이트 및 지표 계산을 수행했습니다. 실제 BulletBill 환경에서의 진단 결과는 사용자의 컴퓨터에서 확인해야 합니다.
