"""2026-02 KNU Advanced RL Competition Round 2 - Bullet Bills
지침서(Instruction.pdf) 공식 평가 스크립트
"""
import os
import sys

# 현재 디렉터리를 모듈 탐색 경로에 추가
submission_dir = os.path.dirname(os.path.abspath(__file__))
if submission_dir not in sys.path:
    sys.path.insert(0, submission_dir)

import kymnasium as kym
from agent import YourAgent

def main():
    model_path = os.path.join(submission_dir, 'best_model.pt')
    print(f"Loading agent from: {model_path}")
    agent = YourAgent.load(model_path)

    # 지침서 5페이지 평가 코드와 동일
    print("Starting evaluation on 'kymnasium/BulletBill-Continuous-Hard-Stage-2'...")
    kym.evaluate(
        env_id='kymnasium/BulletBill-Continuous-Hard-Stage-2',
        agent=agent,
        render_mode='human',
        bgm=True,
        seed=42
    )

if __name__ == '__main__':
    main()

