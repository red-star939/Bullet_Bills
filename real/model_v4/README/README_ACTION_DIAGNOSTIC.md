# Model V4 행동 진단 (학습 없음)

`real/model_v4/diagnose_actions_v4.py`는 기존 V4 모델의 가중치를 변경하지 않습니다. 동일한 게임 시드로 결정적 행동(`mu`, 점프 argmax)과 확률적 행동(정규분포 샘플, 점프 샘플)을 각각 평가합니다.

## 설치

ZIP을 `mario_pj` 루트에 풀어 `real/model_v4/diagnose_actions_v4.py`가 생성되는지 확인합니다. 기존 모델 또는 학습기를 덮어쓰지 않습니다.

## 실행 (권장: 화면 없이 5시드 × 2방식)

```powershell
.\.venv\Scripts\python.exe real/model_v4/diagnose_actions_v4.py `
  --model real/model_v4/model_v4_500k.pt `
  --episodes 5 `
  --seed 731209 `
  --max-steps 9000 `
  --render none `
  --out real/model_v4/eval_results/action_diagnostic
```

## 결과

- `summary.json`: 결정적/확률적 평가 평균 생존, `mu_abs_mean`, `std_mean`, `sent_action_abs_mean`, `actual_dx_abs_mean`, 이동 0 비율 및 점프 분포
- `episodes.csv`: 시드별 생존 시간/행동 지표
- `actions.csv`: 매 프레임 정책 평균, 분포 표준편차, 실제 전달 행동, 마리오 위치·속도·이동량

## 해석

1. `deterministic.mu_abs_mean`이 작고 `stochastic.raw_action_abs_mean`이 크면 이전 학습에서 관찰한 좌우 입력이 주로 탐색 노이즈일 가능성이 높습니다.
2. `sent_action_abs_mean`이 상당한데 `actual_dx_abs_mean`이 계속 0에 가깝다면 행동 반영 또는 물리 상태를 확인해야 합니다. 점프 도중/벽과 충돌 등도 함께 검토합니다.
3. `jump_fraction`으로 결정적 정책이 한 점프 값에 몰리는지 확인합니다.
4. 관측값은 실제 BulletBill `obs['mario']`의 `[left, top, right, bottom, vx, vy, ...]` 배열을 가정해 `x=(left+right)/2`, `vx=mario[4]`로 읽습니다. 다른 환경/버전에서는 위치 수치가 맞는지 확인하세요.
5. 평가 시각은 환경의 `info['time_elapsed']`가 있으면 이를 사용합니다. 없다면 `--game-fps`로 보간합니다.

단 5회는 기능 진단용입니다. 정책 성능을 비교하려면 시드를 늘리세요. `--render human` 옵션도 가능하지만, 모드별 5판씩 총 10판이 표시됩니다.
