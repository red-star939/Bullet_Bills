# Bullet Bills! — 레인 기반 포탄 추적 및 충돌 위험 예측

> **프로젝트 상태:** 환경 및 맵 분석 완료 · 포탄 추적과 1차 TTC/위험도 추정 구현 및 검증 · 강화학습 에이전트 개발 전  
> **기준일:** 2026-11-05  
> **환경 ID:** `kymnasium/BulletBill-Continuous-Hard-Stage-2`

## 1. 프로젝트 소개

이 프로젝트는 **Bullet Bills! (연속 행동, Hard, Stage 2)** 환경에서 마리오가 포탄을 회피하며 오래 생존하도록 하는 강화학습 에이전트를 개발하기 위한 연구·구현 프로젝트입니다.

현재까지는 학습 알고리즘을 바로 작성하기보다, **환경의 실제 물리 동작을 확인하고, 포탄의 위치 및 충돌 위험을 재현 가능하게 계산하는 분석 계층**을 먼저 구축했습니다.

핵심 파이프라인은 다음과 같습니다.

```text
Gymnasium / Kymnasium observation
             │
             ▼
    LaneManager (레인 분류)
             │
             ▼
    BulletTracker (포탄 식별 및 초기 상태 기록)
             │
             ▼
    risk_estimator (현재 TTC 및 위험도)
             │
             ▼
    [예정] Mario Motion Predictor → State Encoder → RL Agent
```

### 현재 구현 범위

- 18개 수평 레인 정의 및 객체의 점유 레인 계산
- 관측 배열에서 포탄 검출, 프레임 간 매칭, 고유 ID 부여 및 추적 종료 처리
- 포탄의 실제 정수 픽셀 이동 규칙을 반영하는 유효 수평 속도 계산
- 포탄 **최초 관측** 시점의 마리오·포탄 좌표, 중심 간 거리, 충돌 간격, 상대속도 및 초기 TTC 저장
- 프레임별 현재 TTC, 위험 점수, 가장 위험한 포탄 계산
- 단위 테스트와 실제 환경(`seed=42`, 정지 마리오)을 사용한 동작 검증

**아직 구현되지 않은 기능:** 마리오의 미래 운동 궤적 예측, 후보 행동별 2차원 충돌 예측, 강화학습 정책·보상 함수·학습 루프·학습된 가중치. 파일을 미리 생성했더라도 본 README의 *예정* 표기는 구현·검증 완료를 뜻하지 않습니다.

---

## 2. 개발 환경 및 설치

기존 개발 환경은 Windows + PowerShell + VS Code이며, 프로젝트 경로는 다음을 기준으로 했습니다.

```text
C:\Users\USER\Desktop\HONG_LB\mario_pj
```

사용 라이브러리는 `gymnasium`, `kymnasium`, `numpy`, `pygame`입니다. 개발 과정에서 `kymnasium 1.2.9`를 사용했습니다. Python 3.12 계열 가상환경(`.venv`)에서 실행을 검증했으며, 다른 버전과의 호환성은 별도 확인이 필요합니다.

### 기존 가상환경 실행

프로젝트 루트에서:

```powershell
.\.venv\Scripts\python.exe -c "import gymnasium, kymnasium, numpy, pygame; print('Imports OK')"
```

### 새 환경에 설치할 경우

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install gymnasium kymnasium==1.2.9 numpy pygame
```

> 위 명령은 프로젝트의 최소 의존성 예시입니다. 배포용 `requirements.txt` 및 모든 버전 고정은 아직 확정하지 않았습니다.

### 환경 생성 및 시드

```python
import gymnasium as gym

ENV_ID = "kymnasium/BulletBill-Continuous-Hard-Stage-2"
env = gym.make(ENV_ID, render_mode="none", bgm=False)
obs, info = env.reset(seed=42)

# ...
env.close()
```

**주의:** 이 환경에서는 시드를 `gym.make(..., seed=42)`가 아니라 **`env.reset(seed=42)`**에 전달해야 합니다.

---

## 3. 프로젝트 구조

아래는 실제 개발 과정에서 사용한 파일과 계획된 구성입니다. 개발자 로컬의 파일 존재 여부는 이 README가 자동으로 검증하지 않으며, 일부 분석용 파일은 생성 여부가 다를 수 있습니다.

```text
mario_pj/
├── .venv/                       # Python 가상환경
├── README.md                    # 이 문서
├── agent_env.py                 # 환경 생성 헬퍼 (로컬 프로젝트)
├── manual_play.py               # 수동 플레이 (환경 점검용)
├── test_env.py                  # 기초 환경 점검
├── extract_map.py               # 배경/오브젝트 타일 추출
├── build_collision_map.py       # 충돌 타일 맵 생성용 스크립트
│
├── map_data/                    # 맵 분석 결과
│   ├── map_info.json
│   ├── initial_entities.json
│   ├── background_tiles.csv
│   ├── object_tiles.csv
│   └── collision_map.json       # 생성했을 경우
│
├── models/
│   ├── __init__.py
│   ├── lane_manager.py          # 구현: 레인/수직 범위 관리
│   ├── bullet_tracker.py        # 구현: 다중 포탄 추적 + 최초 관측 상태
│   ├── risk_estimator.py        # 구현: 함수형 TTC/위험도 계산
│   ├── mario_motion_predictor.py # 예정
│   ├── state_encoder.py         # 예정
│   ├── reward.py                # 예정
│   ├── agent.py                 # 예정
│   └── checkpoints/             # 예정: 학습 가중치
│
├── test_lane_manager.py         # 레인 계산 확인
├── test_bullet_tracker.py       # 포탄 ID/매칭 단위 테스트
├── test_tracker_env.py          # 실제 환경에서 추적 검증
├── test_risk_estimator.py       # 함수형 TTC 계산 단위 테스트
├── test_initial_range.py        # 최초 거리/TTC 단위 테스트
├── test_risk_env.py             # 실제 환경에서 TTC/위험도 검증
├── train.py                     # 예정
└── evaluate.py                  # 예정
```

### 모듈 간 인터페이스

```python
from models.bullet_tracker import BulletTracker
from models.risk_estimator import (
    analyze_bullets,
    get_most_dangerous,
    get_total_risk,
)

tracker = BulletTracker(fps=60, frame_skip=4)
# 각 에피소드 reset 시 tracker.reset() 호출

# env.step(action) 이후:
bullets = tracker.update(
    obs,
    time_elapsed=float(info["time_elapsed"]),
)
risks = analyze_bullets(obs, tracked_bullets=bullets)
most_dangerous = get_most_dangerous(risks)
total_risk = get_total_risk(risks)
```

**중요한 API 결정:** 현재 `models/risk_estimator.py`는 **함수형 구현**입니다. 초기 개발 중 제안되었던 `RiskEstimator` 클래스는 현재 사용하지 않습니다. `from models.risk_estimator import RiskEstimator` 또는 `estimator.analyze(...)`는 현재 코드와 맞지 않습니다.

---

## 4. 환경 사양

### 4.1 화면과 시간

| 항목 | 값 | 비고 |
|---|---:|---|
| 플레이 영역 | 1296 × 864 px | 상태 표시줄 제외 |
| 표시줄 높이 | 50 px | `render_mode="human"` 화면에 추가 |
| 타일 크기 | 48 × 48 px | 스케일 반영 후 |
| 맵 크기 | 27열 × 18행 | 총 486개 격자 위치 |
| 내부 물리 FPS | 60 | `dt_internal = 1/60`초 |
| 기본 `frame_skip` | 4 | `env.step()`당 최대 4개 내부 프레임 |
| 기본 행동 간격 | 4/60 ≈ 0.066667초 | 약 15회/게임 시간 1초 |
| 최대 게임 시간 | 180초 | 환경 설정 기준 |

종료 직전에는 내부 프레임 루프가 조기 중단될 수 있으므로 마지막 `env.step()`의 경과 시간은 `4/60`보다 짧을 수 있습니다. 정밀한 시간 측정에는 `info["time_elapsed"]`를 사용합니다.

### 4.2 관측 구조

```python
observation = {
    "mario":        # shape=(8,)
        # [left, top, right, bottom, vx, vy, invincible, jumping]
    "bullet_bills": # shape=(30, 5)
        # 각 행: [left, top, right, bottom, vx]
    "stars":        # Stage 2에서 확인한 shape=(2, 4)
        # 각 행: [left, top, right, bottom]
}
```

`bullet_bills`의 비어 있는 행은 모두 0입니다. 포탄의 **배열 행 번호는 고유 ID가 아니므로** `BulletTracker`가 별도의 `bullet_id`를 부여합니다.

행동은 `numpy.float32` 2원소 연속 벡터입니다.

```python
action = np.array([horizontal_force, jump_force], dtype=np.float32)
# horizontal_force: -1.0 ~ 1.0
# jump_force:        0.0 ~ 1.0
```

현재 분석 단계의 테스트는 `action=[0.0, 0.0]`(정지)을 사용합니다. 원본 환경의 스텝 보상은 0이므로 향후 학습 시 보상 설계가 필요합니다.

### 4.3 맵 및 충돌 지형

`extract_map.py`를 이용해 Stage 2의 배경 및 오브젝트 레이어를 별도로 추출했습니다. 초기 상태에서 확인한 마리오의 바운딩 박스는 `[624, 720, 672, 768]`, 별은 좌우 `[0, 384, 48, 432]`, `[1248, 384, 1296, 432]`였습니다.

게임 소스의 스테이지 생성 로직상:

- **배경 레이어:** 시각적 타일
- **오브젝트 레이어:** 마리오/별 타일을 제외한 항목을 플랫폼 충돌 객체로 생성

즉, 보기에는 비어 있는 블록처럼 보이는 타일도 **오브젝트 레이어에 있으면 충돌 객체일 수 있습니다.** 맵 경로 및 점프 예측에는 배경 그림이 아닌 충돌 오브젝트를 사용해야 합니다.

---

## 5. 알고리즘 구현 상세

### 5.1 `LaneManager` — 레인 분류

게임 영역의 Y축을 48px 단위로 나누어 총 18개 레인을 관리합니다.

\[
L(y)=\left\lfloor \frac{y}{48}\right\rfloor,
\quad 0\le y<864
\]

- 레인 `0`: `0 ≤ y < 48`
- 레인 `1`: `48 ≤ y < 96`
- …
- 레인 `17`: `816 ≤ y < 864`

**대표 레인:** 객체 중심 Y좌표가 속한 레인.  
**점유 레인:** 객체 바운딩 박스 `[top, bottom)`이 실제로 교차하는 모든 레인.

예를 들어 포탄 `[left, 600, right, 648]`은 중심 Y가 `624`이므로 대표 레인은 `13`, 점유 레인은 `[12, 13]`입니다.

레인 분류는 객체 관리용이며, 실제 충돌 판정은 **픽셀 단위 바운딩 박스의 교차**로 수행합니다.

### 5.2 `BulletTracker` — 포탄 검출 및 다중 객체 추적

각 `env.step()` 후 새 관측을 받아 다음 순서로 처리합니다.

1. `bullet_bills`에서 모두 0인 행을 제외하고 포탄 검출
2. 포탄의 중심 좌표·Y레인·점유 레인 계산
3. 기존 포탄에 대한 다음 관측 위치 예측
4. 위치·Y좌표·속도·방향이 허용 오차 내인 쌍을 탐욕적(greedy) 매칭
5. 매칭된 포탄은 ID 유지, 매칭되지 않은 관측은 신규 ID 부여
6. 일정 횟수 이상 관측되지 않은 추적은 제거 (`lost_ids`)

주요 결과 필드:

| 필드 | 의미 |
|---|---|
| `bullet_id` | 추적기가 발급한 고유 ID |
| `bbox` | `[left, top, right, bottom]` |
| `vx` | 환경이 보고하는 원래 수평 속도 |
| `effective_vx` | 정수 픽셀 이동을 반영한 유효 속도 |
| `lane`, `occupied_lanes` | 대표 및 점유 레인 |
| `first_seen_frame`, `first_seen_time` | 최초 **관측** 시점 |
| `age_frames`, `missed_frames` | 추적 갱신 횟수 및 연속 미관측 횟수 |
| `is_new` | 이번 관측에서 신규 트랙인지 여부 |

**한계:** 현재 구현은 greedy 매칭입니다. 거의 같은 위치·높이·속도의 포탄이 겹쳐 등장하면 ID가 바뀔 수 있습니다. 또한 `age_frames`는 내부 물리 프레임 수가 아니라 `update()` 호출 기반의 추적 경과 횟수입니다.

### 5.3 속도 보정 — 엔진의 실제 픽셀 이동

처음에는 `vx`를 곧바로 픽셀/초 이동 속도로 가정했지만, 실제 로그에서 `vx≈285.86`, `vx≈297.56`인 두 포탄 모두 **측정 이동 속도 240px/s**를 보였습니다.

원본 `objs.py`에서 원인을 확인했습니다.

```python
# BulletBill.tick_frame 내부
self.rect.centerx += int(self._velocity * dt)
```

내부 `dt = 1/60`이므로 **매 내부 프레임의 소수 이동량이 0 방향으로 절삭되며 다음 프레임에 누적되지 않습니다.**

\[
\Delta x_{internal}=\operatorname{trunc}(v_x/60)
\]

\[
\boxed{v_x^{effective}=60\,\operatorname{trunc}(v_x/60)}
\]

`frame_skip=4`이고 포탄이 계속 활성 상태라면 스텝당 수평 이동량은:

\[
\Delta x_{step}=4\,\operatorname{trunc}(v_x/60)
\]

예: `vx=285.86` → 내부 프레임당 `+4px` → 스텝당 `+16px` → `effective_vx=+240px/s`. 음수 `vx`도 Python `int()`와 같이 0 방향 절삭으로 계산합니다.

**설계 원칙:** 원래 `vx`와 유효 속도 `effective_vx`를 구분하고, 포탄 위치 예측과 TTC에는 유효 속도를 사용합니다. 원본 물리 엔진 자체는 수정하지 않습니다.

### 5.4 최초 관측 거리 및 초기 TTC

포탄이 새로 발견되면 그 순간의 마리오·포탄 상태를 저장합니다. **포탄이 내부적으로 생성된 시점이나 화면 가장자리에 정확히 1px 진입한 순간과는 다릅니다.** `frame_skip=4`에서는 중간 내부 프레임을 직접 관측하지 못할 수 있습니다.

최초 관측 순간의 포탄 중심을 `x_b0`, 마리오 중심을 `x_m0`라고 할 때:

\[
D_0=|x_{m0}-x_{b0}|
\]

하지만 실제 바운딩 박스 접촉 시각은 중심 간 거리 대신 **경계 간 충돌 간격**을 사용해야 합니다.

왼쪽에서 오는 포탄:

\[
G_0=x_{m,left}-x_{b,right}
\]

\[
c_0=v_b^{effective}-v_m
\]

오른쪽에서 오는 포탄:

\[
G_0=x_{b,left}-x_{m,right}
\]

\[
c_0=v_m-v_b^{effective}
\]

현재 수직 바운딩 박스가 겹치고, 포탄이 접근 중이며(`c_0>0`), 간격이 양수이면:

\[
\boxed{TTC_0=G_0/c_0}
\]

이미 수평·수직으로 겹쳐 있으면 `TTC_0=0`, 현재 수직이 겹치지 않거나 접근하지 않으면 `TTC_0=inf`로 처리합니다.

추적기에 저장하는 초기값은 이후 프레임에서 덮어쓰지 않습니다.

```text
initial_bullet_x      최초 포탄 중심 X
initial_mario_x       최초 마리오 중심 X
initial_distance      최초 중심 간 거리 D0
initial_collision_gap 최초 바운딩 박스 간격 G0
initial_relative_vx   부호를 포함한 상대속도 (vb_effective - vm)
initial_closing_speed 접근 속도의 양의 방향 성분
initial_ttc           최초 TTC
first_seen_time       환경의 time_elapsed
```

**단위 테스트 사례:** 마리오 `[624, 720, 672, 768]`, 포탄 `[384, 720, 432, 768, 285.86]`일 때:

- 중심 간 거리 `D0 = 648 - 408 = 240px`
- 경계 간 거리 `G0 = 624 - 432 = 192px`
- 유효 포탄 속도 `240px/s`, 마리오 정지
- 최초 TTC `192 / 240 = 0.800초`
- 1회 `env.step()` 후 포탄이 16px 이동하면 현재 TTC `176 / 240 ≈ 0.733초`

### 5.5 `risk_estimator.py` — 현재 TTC와 위험 점수

현재 구현은 **함수형 API**이며 `analyze_bullets()`가 포탄별 **딕셔너리**를 반환합니다.

- `calculate_ttc(mario, bullet)`: 현재 바운딩 박스 기준 TTC
- `analyze_bullets(observation, tracked_bullets=None)`: 포탄별 위험도 계산
- `get_most_dangerous(results)`: 위험 점수가 가장 높은 포탄
- `get_total_risk(results)`: 모든 포탄 위험 점수의 최댓값

수직 범위가 겹치는 조건:

\[
y_{b,top}<y_{m,bottom}
\quad\land\quad
y_{b,bottom}>y_{m,top}
\]

현재 TTC는 해당 수직 범위가 겹친다는 전제에서 수평 경계 사이 간격을 접근 상대속도로 나누어 계산합니다.

위험 점수(기본 `tau=1.0`):

\[
R=\exp(-TTC/\tau)
\]

- 무적 상태 또는 `TTC=inf` → `R=0`
- 현재 바운딩 박스가 겹침 → `TTC=0`, `R=1` (무적 제외)
- `R ≥ 0.70`: `HIGH`
- `0.35 ≤ R < 0.70`: `MEDIUM`
- `0 < R < 0.35`: `LOW`
- `R = 0`: `NONE`

**주의:** 이 위험 점수는 임의 설계한 시간 기반 지표이며 **사망 확률이 아닙니다.** 현재 수직 범위가 겹치지 않아도 이후 점프·낙하로 충돌할 수 있습니다. 무적 상태일 때 `R=0`으로 설정하는 것도 무적 잔여 시간까지 예측한 것은 아닙니다.

---

## 6. 테스트 방법 및 확인된 결과

모든 명령은 **프로젝트 루트**에서 실행합니다.

### 6.1 단위 테스트

```powershell
.\.venv\Scripts\python.exe test_lane_manager.py
.\.venv\Scripts\python.exe test_bullet_tracker.py
.\.venv\Scripts\python.exe test_risk_estimator.py
.\.venv\Scripts\python.exe test_initial_range.py
```

검증 현황은 실제 제출된 콘솔 로그를 기준으로 기록했습니다.

| 테스트 | 확인된 결과 |
|---|---|
| `test_bullet_tracker.py` | `All BulletTracker tests passed!` — 신규 ID, 배열 재정렬 매칭, 소멸 처리 통과 |
| `test_initial_range.py` | `All initial range tests passed!` — 최초 D0·TTC0, 현재 TTC, 우측 접근, 초기값 보존 통과 |
| `test_lane_manager.py` | 구현 및 테스트 코드 작성; 본 README 기준 별도 최종 콘솔 결과 미확인 |
| `test_risk_estimator.py` | 함수형 테스트 코드 작성; 본 README 기준 별도 최종 콘솔 결과 미확인 |

### 6.2 실제 게임 환경 테스트

```powershell
.\.venv\Scripts\python.exe test_tracker_env.py
.\.venv\Scripts\python.exe test_risk_env.py
```

`test_risk_env.py`는 `seed=42`, `action=[0.0, 0.0]`으로 정지한 마리오를 대상으로 수행했습니다. 이 설정에서 포탄 최초 관측과 실제 게임 종료 로그를 비교했습니다.

**포탄 ID 1 최초 관측:**

```text
Time=1.467s
BulletX0=-16.0 | MarioX0=648.0
D0=664.0px | CollisionGap0=616.0px
RelativeVX0=240.0px/s | ClosingSpeed0=240.0px/s
TTC0=2.567s
```

**포탄 ID 1의 진행 경과:**

| 게임 시간(s) | 현재 TTC(s) | 위험도 |
|---:|---:|---:|
| 1.733 | 2.300 | 0.100 |
| 2.067 | 1.967 | 0.140 |
| 2.400 | 1.633 | 0.195 |
| 2.733 | 1.300 | 0.273 |
| 3.067 | 0.967 | 0.380 |
| 3.400 | 0.633 | 0.531 |
| 3.733 | 0.300 | 0.741 |
| 4.050 | 0.000 | 1.000 |

예측한 최초 수평 접촉 시간은 대략 `1.467 + 2.567 = 4.034초`였고 실제 게임 종료 로그는 `4.050초`였습니다. 차이는 **약 0.016초(물리 프레임 1개 수준)**였습니다.

이는 **seed=42, 정지 마리오, 해당 포탄**에서 일치한 결과이며, 이동·점프 중인 마리오 또는 다른 시드에서 같은 정확도가 확보되었다는 의미는 아닙니다. 실제 사망 판정은 엔진의 내부 프레임 충돌 검사에서 이루어집니다.

---

## 7. 구현 과정과 핵심 의사결정

| 순서 | 수행 내용 | 확인·결정된 사항 |
|---:|---|---|
| 1 | 환경 생성 및 관측 구조 확인 | `env.reset(seed=...)`, 연속 행동/상태 배열 확인 |
| 2 | 스테이지 맵 추출 | 27×18, 타일 48px, 배경/충돌 오브젝트 분리 |
| 3 | 18개 레인 등록 | 대표 레인 및 실제 점유 레인 별도 관리 |
| 4 | 포탄 추적기 구현 | 관측 행 번호 대신 포탄 고유 ID 사용 |
| 5 | 실제 게임 추적 테스트 | 신규 포탄 검출 및 다중 포탄 ID 유지 확인 |
| 6 | 시간 오차 조사 | `env.step()` 내부 `frame_skip=4` 확인 |
| 7 | 이동 속도 차이 조사 | `int(velocity * dt)`에 의한 픽셀 정수화 확인 |
| 8 | 포탄 유효 속도 도입 | 보고 속도 285.86·297.56 → 실제 240px/s와 일치 |
| 9 | 최초 관측 사거리 도입 | `D0`, 충돌 간격, 상대속도, 초기 TTC 고정 저장 |
| 10 | 실시간 위험도 분석 | `TTC`, `exp(-TTC)`, 가장 위험한 포탄 계산 |
| 11 | 단위 및 실환경 검증 | 초기 TTC 0.800→0.733, 정지 마리오 충돌 예측 확인 |

### 개발 중 해결한 호환성 이슈

- Kymnasium 수동 플레이 래퍼는 설치된 환경에서 `from kymnasium.envs.bullet_bill import ManualPlayWrapper`로 임포트했습니다.
- 일부 설치 버전의 수동 플레이 디버그 로그에서 NumPy 배열에 `action or default_action`을 적용하면 모호한 truth-value 오류가 발생할 수 있습니다. 개발 과정에서는 `action if action is not None else default_action` 방식으로 수정하여 테스트했습니다. **서드파티 패키지 소스를 로컬에서 수정했다면 다른 개발 환경에도 동일한 수정이 필요한지 확인해야 합니다.**
- 위험도 추정기 API는 개발 중 클래스형 설계안에서 **함수형 최종 구현**으로 정리했습니다. 따라서 예전 테스트 코드의 `RiskEstimator` 임포트나 `.analyze()` 호출은 함수 API로 변경해야 합니다.

---

## 8. 알려진 한계 및 후속 개발

### 현 단계의 가정과 한계

1. **포탄 최초 관측 ≠ 엔진 생성 순간 또는 경계에서 1px 진입한 순간.** `frame_skip=4`로 생기는 관측 공백은 존재합니다.
2. **마리오 미래 속도를 일정하게 가정.** 실제로는 수평 가속도, 마찰, 속도 제한, 정수화, 플랫폼 충돌이 있습니다.
3. **수직 운동을 아직 예측하지 않음.** 현재 Y구간이 겹치지 않으면 현 단계 TTC는 `inf`이나 점프 후 충돌 가능성은 여전히 있습니다.
4. **행동 사이 충돌 가능성.** 에이전트는 기본적으로 4개 내부 프레임마다 행동 결정을 하므로, 그 사이 발생하는 접촉도 고려해야 합니다.
5. **다중 포탄의 ID 매칭은 greedy 방식.** 유사한 포탄이 근접하면 재식별 실패 가능성이 있습니다.
6. **무적 잔여 시간 미예측.** 무적 여부만 즉시 반영합니다.
7. **성능 일반화 미검증.** 현재 정밀한 TTC 일치는 정지 마리오와 특정 시드 조건에서 확인했습니다.
8. **강화학습 성능 미평가.** 아직 학습 정책이 없으므로 생존 성능 향상 주장은 할 수 없습니다.

### 향후 작업 계획

| 우선순위 | 모듈 | 목표 |
|---|---|---|
| 1 | `models/mario_motion_predictor.py` | 가속·마찰·중력·점프·정수화·플랫폼 충돌을 반영한 미래 마리오 위치 예측 |
| 2 | 2차원 충돌 예측 | 행동 후보마다 포탄/마리오의 내부 프레임 단위 바운딩 박스 교차 탐지 |
| 3 | `models/state_encoder.py` | 추적 포탄, TTC, 레인, 마리오 운동 상태를 학습 입력으로 변환 |
| 4 | `models/reward.py` | 생존, 회피, 위험 감소 및 불필요한 행동 억제 보상 설계 |
| 5 | `models/agent.py` | 연속 행동 정책 구현 및 Kymnasium 평가 인터페이스 연결 |
| 6 | `train.py`, `evaluate.py` | 다중 시드 학습, 평가, 모델 저장/복원 및 실험 로그 |

### 강화학습 인터페이스 참고

최종 제출/평가 단계에서는 Kymnasium의 에이전트 인터페이스(`act`, `save`, `load`)와 공식 평가 함수를 구현·검증해야 합니다. 현재 README는 **위험 인지 모듈의 개발 상태를 설명**하며, 학습 완료 또는 대회 평가 완료를 나타내지 않습니다.

---

## 9. 빠른 시작 체크리스트

1. 프로젝트 루트에서 가상환경 및 `gymnasium`, `kymnasium`, `numpy`, `pygame` 임포트를 확인합니다.
2. `test_bullet_tracker.py`와 `test_initial_range.py`로 추적 및 초기 TTC 계산을 검사합니다.
3. `test_tracker_env.py`로 실제 포탄의 ID/속도 추적을 확인합니다.
4. `test_risk_env.py`로 최초 TTC와 매 프레임 위험도 증가를 확인합니다.
5. 기존 코드에서 `RiskEstimator` 클래스를 임포트하지 않는지 확인합니다. 현재 정식 인터페이스는 `analyze_bullets`, `get_most_dangerous`, `get_total_risk` 함수입니다.
6. 새 에피소드에서는 반드시 `tracker.reset()`을 호출하고, 가능하면 `tracker.update(obs, time_elapsed=...)`로 게임 시간을 전달합니다.
7. 마리오 움직임을 도입하면 현 TTC는 근사값이므로 2차원 운동 예측기 검증 전까지 안전 보장용으로 사용하지 않습니다.

---

## 10. 문서의 검증 범위

이 README는 2026-11-05까지 수행한 **코드 구현 대화, 실행 로그 및 게임 엔진 소스 분석 결과**를 정리한 것입니다. 실제 로컬 프로젝트 전체를 이 환경에서 자동으로 가져와 실행한 것은 아닙니다. 따라서 여기서 *검증 완료*로 표시한 항목은 **공유된 테스트 로그에서 확인된 범위**에 한정되며, 다른 시드·플랫폼·버전에서의 동작은 별도로 재확인해야 합니다.
