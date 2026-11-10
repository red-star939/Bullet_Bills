# Bullet Bills

슈퍼 마리오 환경(`kymnasium/BulletBill-Continuous-Hard-Stage-2`)에서 킬러(Bullet Bill) 추적, 레인(Lane) 관리 및 충돌 위험도 추정(Risk Estimator)을 수행하는 강화학습/시뮬레이션 환경 프로젝트입니다.

상세한 구현 현황, 물리 분석 및 아키텍처는 **[Development_Process.md](Development_Process.md)** 문서를 참고하세요.

## 📄 개발 및 구현 현황 문서
- **[Latest_Update.md](Latest_Update.md)**: `models2` 계승형 생존 제어 프로젝트, PPO v10 정책, Adaptive Altitude v1~v4.1 실험 결과 및 개발 문서.
- **[Development_Process.md](Development_Process.md)**: 레인 분류, 포탄 추적(BulletTracker), 정밀 물리/충돌 예측, 1차 TTC 및 위험도 추정 구현 검증 결과와 분석 내용이 상세히 정리되어 있습니다.

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

