"""
맵을 글자로 출력:  # 블록, M 마리오 시작, * 별, ~ 물 높이(블록 없는 맨 아래 줄)
  python map_ascii.py
"""
import gymnasium as gym
import kymnasium as kym

env = gym.make(id='kymnasium/BulletBill-Continuous-Hard-Stage-2', render_mode='none', bgm=False)
env.reset(seed=0)
u = env.unwrapped
grid, T = u._platform_grid, 48
W, H = 27, 18
mx, my = u._mario_spawn_position
stars = {(x // T, y // T) for x, y in u._star_spawn_positions}

print("     열: " + ''.join(str(c % 10) for c in range(W)))
for r in range(H):
    line = ''
    for c in range(W):
        if (c, r) in grid:
            line += '#'
        elif (c, r) == (mx // T, my // T):
            line += 'M'
        elif (c, r) in stars:
            line += '*'
        elif r >= 16:
            line += '~'
        else:
            line += '.'
    print(f"{r:2d}행 y={r * T:3d} {line}")
print(f"\n블록 {len(grid)}개, 블록 좌표(열, 행) 목록:")
print(sorted(grid.keys(), key=lambda k: (k[1], k[0])))
env.close()