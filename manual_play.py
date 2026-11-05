
import gymnasium as gym
import kymnasium
import numpy as np
import pygame

ENV_ID = "kymnasium/BulletBill-Continuous-Hard-Stage-2"

FPS = 60

# 키를 놓았을 때 남겨둘 상승 속도 비율
# 작을수록 짧게 눌렀을 때 점프가 낮아짐
JUMP_CUT_RATIO = 0.35


def main():
    env = gym.make(
        id=ENV_ID,
        render_mode="human",
        bgm=False
    )

    clock = pygame.time.Clock()

    try:
        obs, info = env.reset(seed=None)

        running = True
        jump_held = False
        jump_cut_pending = False

        print("=== Console-style Manual Play ===")
        print("Left / Right: 이동")
        print("Space: 점프")
        print("짧게 누르기: 낮은 점프")
        print("길게 누르기: 높은 점프")
        print("ESC: 종료")

        while running:
            jump_pressed = False

            # 키보드 이벤트 처리
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        if not jump_held:
                            jump_pressed = True
                            jump_held = True
                            jump_cut_pending = False

                elif event.type == pygame.KEYUP:

                    if event.key == pygame.K_SPACE:
                        jump_held = False
                        jump_cut_pending = True

            if not running:
                break

            keys = pygame.key.get_pressed()

            horizontal = (
                int(keys[pygame.K_RIGHT])
                - int(keys[pygame.K_LEFT])
            )

            # 점프는 누른 첫 프레임에 최대 강도로 실행
            jump_force = 1.0 if jump_pressed else 0.0

            action = np.array(
                [float(horizontal), jump_force],
                dtype=np.float32
            )

            obs, reward, terminated, truncated, info = env.step(action)

            # 수동 플레이에서만 적용되는 가변 점프 보정
            mario = env.unwrapped._mario

            if jump_cut_pending:
                velocity = mario._velocity

                # 상승 중일 때만 점프를 짧게 끊음
                if velocity.y < 0:
                    velocity.y *= JUMP_CUT_RATIO

                jump_cut_pending = False

            if terminated or truncated:
                print(
                    "게임 종료 |",
                    "생존 시간:",
                    round(info.get("time_elapsed", 0), 2),
                    "초"
                )
                break

            clock.tick(FPS)

    finally:
        env.close()


if __name__ == "__main__":
    main()
