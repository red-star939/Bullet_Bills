# Bullet Bills

슈퍼 마리오 환경에서 킬러(Bullet Bill) 추적, 차선(Lane) 관리 및 위험도 추정(Risk Estimator)을 수행하는 강화학습/시뮬레이션 환경 프로젝트입니다.

## 프로젝트 구조

- `models/`: 핵심 로직 모듈
  - `bullet_tracker.py`: Bullet Bill(킬러) 인식 및 궤적 추적 모듈
  - `lane_manager.py`: 위험 구간/차선 분할 및 관리
  - `risk_estimator.py`: 상대 속도 및 거리를 바탕으로 한 충돌 위험도 평가
- `map_data/`: 맵 배경 및 타일/오브젝트 데이터 (JSON, CSV)
- `test_*.py`: 모듈별 단위 테스트 및 환경 테스트 스크립트
- `manual_play.py`: 수동 제어 플레이 스크립트

## 시작하기

### 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/Scripts/activate  # Windows (PowerShell: .venv\Scripts\Activate.ps1)

# 의존성 설치
pip install -r requirements.txt
```

### 테스트 실행

```bash
python test_agent_env.py
python test_bullet_tracker.py
python test_risk_env.py
```
