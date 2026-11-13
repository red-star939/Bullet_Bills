# real/model_v4 — 독립 강화학습 연구 기반

**PPO v10 코드·체크포인트·148차원 특징 추출기·기존 모델 아키텍처를 포함하지 않습니다.**

이번 단계는 기존 연구 **방법론**을 새 프로젝트에 옮기는 작업입니다. 게임 실행기/학습 정책/환경 어댑터가 결정되지 않았으므로, 전체 학습이 실행되는 완제품으로 표시하지 않습니다.

## 폴더

- `methods/ppo_core.py`: 새 정책에 연결할 PPO clipped loss 및 GAE 수학 핵심
- `methods/teacher_student.py`: 교사 행동 증류 및 기존 행동 보존 손실
- `methods/curriculum.py`: 후반 시점 목표를 선택하는 적응형 backward-start 커리큘럼
- `methods/rewards.py`: 생존/종료/잠재함수 기반 보상 구성
- `methods/risk.py`: 다중 미래 시나리오 노출률과 개입 선택 기준
- `methods/parallel.py`: Windows spawn 방식 프로세스 컨텍스트
- `methods/device.py`: CUDA 선택 및 VRAM 측정
- `methods/evaluation.py`: 학습/검증/홀드아웃 시드 분리 및 생존 시간 통계
- `research_reference/v41/`: 이전 v4.1 회피·예측·감사 아이디어의 **연구 원본 사본**. 이전 코드 의존성이 있어 실행 모듈에 연결하지 않음.
- `tests/`: 독립 방법론 단위 테스트

## 제외한 것

`agent_ppo_v10.py`, 기존 PPO 모델 `.pt`, 기존 v10용 실행·평가 스크립트, 기존 특징 추출기, v10 이름을 가진 제어기 및 가중치를 제외했습니다. 원본 참조 파일 안에 이전 PPO를 언급하는 설명/인터페이스가 있을 수 있지만 실제 학습 경로에 import되지 않습니다.

## 설계상 핵심 제약

1. GPU VRAM 상주 모델/롤아웃/GAE/Adam 구현은 이후 새 모델에 통합할 설계 과제입니다. `device.py`는 장치 진단만 담당합니다. 환경 시뮬레이션까지 GPU에 올렸다고 주장하지 않습니다.
2. 커리큘럼은 시작 *시간*을 선택합니다. 그 시간의 물리적으로 유효한 환경 상태는 실제 warmup 또는 정확한 snapshot/restore로 확보해야 합니다.
3. 위험 시나리오 노출률은 **사망 확률이 아닙니다**.
4. PPO는 반드시 현재 정책에서 수집한 실제 행동과 로그확률로 업데이트해야 합니다. 교사 행동은 별도의 증류 손실로 사용합니다.
5. 평가 시드는 학습/검증/최종 홀드아웃을 분리하고 모델 선택은 검증 시드로만 수행합니다.

## 단위 테스트

프로젝트 루트에서:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

필요 패키지: `torch`. 실제 모델/게임 환경을 붙이는 단계는 별도로 진행합니다.

## 신규 Model V4 초기 50,000스텝 실험

이번 패키지에는 PPO v10과 **가중치/구조/특징 추출기 의존성이 없는** 새 정책이 포함됩니다.

- `model_v4_net.py` : 신규 공유 인코더(256D 입력, 192 hidden) + Jump categorical / Move Gaussian / Value. 관측값은 범용 수치 인코더를 통해 256D로 변환하므로 게임 상태 특징 추출의 최적화는 추후 과제입니다.
- `model_v4_initial_untrained.pt` : 로컬 학습 전에 생성된 **미학습 0스텝 초기 가중치**. 실전 학습 성능을 갖춘 모델이 아닙니다.
- `train_initial_50k.py` : 실행 시 독립 정책을 새로 만들고 **50,000 환경 스텝**을 학습합니다. 이전 `.pt`를 초기 정책으로 불러오지 않습니다.
- `model_v4_initial_50k.pt` : **Windows에서 실제 학습이 완료되면** `real/model_v4/` 바로 아래 생성되는 최종 학습 결과.
- `checkpoints/model_v4_resume.pt` : 학습 중간 저장(옵티마이저 포함). 재개 시 사용.
- `checkpoints/initial_50k_eval.json` : 학습 후 평가 결과.

### 실행 (PowerShell, 프로젝트 `mario_pj` 루트에서)

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

.\.venv\Scripts\python.exe real/model_v4/train_initial_50k.py `
  --total-steps 50000 `
  --envs 4 `
  --batch-steps 128 `
  --device cpu `
  --checkpoint-every 10000 `
  --eval-episodes 5
```

실험적으로 CUDA 신경망 학습을 사용하려면 `--device cuda`로 변경할 수 있습니다. 게임 환경은 여전히 CPU에서 실행됩니다. 4개 환경도 단일 프로세스에서 순차 진행되므로 16코어 병렬화를 주장하지 않습니다.

### 중단 후 이어 학습

```powershell
.\.venv\Scripts\python.exe real/model_v4/train_initial_50k.py `
  --resume real/model_v4/checkpoints/model_v4_resume.pt `
  --total-steps 50000 `
  --envs 4 `
  --batch-steps 128 `
  --device cpu `
  --checkpoint-every 10000
```

중간 저장 이후의 스텝은 다시 진행하며, 진행 중이던 게임 환경 자체는 복원하지 않고 에피소드를 재시작합니다. 현재 재개 기능은 모델/옵티마이저/중앙 난수 상태만 복원합니다. Worker 환경 에피소드별 시드 이력은 완전 복원되지 않아 비트 단위 동일 재현을 보장하지 않습니다.

### 유의 사항

이 컨테이너에는 `gymnasium`과 `kymnasium`이 없어 실제 게임의 50,000스텝 학습은 수행하지 않았습니다. 별도의 모의 환경 학습·중단 후 재개 테스트만 통과했습니다. 신규 인코더는 초기 실험용 범용 특징이므로 기존 특화 모델보다 생존 성능이 낮을 수 있습니다. 이 학습기는 행동을 실제 환경에 적용하기 전 이동 입력을 [-1,1]로 클리핑합니다. 극단적인 원시 가우시안 행동의 확률은 클리핑 전 행동 기준으로 계산되므로(기존 구현과 같은 방식) 경계 행동에 대한 엄밀한 행동분포 일치가 필요한 경우 tanh-squashed Gaussian으로 개선해야 합니다.
