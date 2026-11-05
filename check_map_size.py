
import gymnasium as gym
import kymnasium
import pygame

from kymnasium.envs.bullet_bill import consts

ENV_ID = "kymnasium/BulletBill-Continuous-Hard-Stage-2"


def main():
    env = gym.make(
        id=ENV_ID,
        render_mode="human",
        bgm=False
    )

    try:
        env.reset(seed=42)
        env.render()

        tile_size = consts.GAME_TILE_SIZE
        surface = pygame.display.get_surface()

        if surface is None:
            print("게임 화면 크기를 가져올 수 없습니다.")
            return

        width, height = surface.get_size()

        print("\n===== BULLET BILLS MAP =====")
        print(f"화면 가로: {width} px")
        print(f"화면 세로: {height} px")
        print(f"타일 크기: {tile_size} px")

        print("\n===== GRID SIZE =====")
        print(f"가로 칸 수: {width / tile_size:.2f}")
        print(f"세로 칸 수: {height / tile_size:.2f}")

        if width % tile_size == 0 and height % tile_size == 0:
            print(f"\n총 열: {width // tile_size}")
            print(f"총 행: {height // tile_size}")
            print(
                f"총 그리드 수: "
                f"{(width // tile_size) * (height // tile_size)}"
            )
        else:
            print("\n화면 크기가 타일 크기로 나누어떨어지지 않습니다.")
            print("맵 경계와 렌더링 영역을 추가 확인해야 합니다.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
