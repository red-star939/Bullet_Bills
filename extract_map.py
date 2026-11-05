
import csv
import json
from pathlib import Path

import gymnasium as gym
import kymnasium
import numpy as np

from kymnasium.envs.bullet_bill.consts import GAME_TILE_SIZE

ENV_ID = "kymnasium/BulletBill-Continuous-Hard-Stage-2"
OUTPUT_DIR = Path("map_data")


def extract_layer(base_env, layer, layer_name):
    """타일 레이어에서 좌표 및 타일 ID 추출."""
    records = []

    for identifier, grid_x, grid_y, sprite in base_env._iter_layer_tiles(layer):
        records.append({
            "layer": layer_name,
            "row": int(grid_y),
            "col": int(grid_x),
            "x_px": int(grid_x * GAME_TILE_SIZE),
            "y_px": int(grid_y * GAME_TILE_SIZE),
            "tile_id": int(identifier),
        })

    return records


def save_csv(filename, records):
    if not records:
        print(f"{filename}: 데이터 없음")
        return

    path = OUTPUT_DIR / filename

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"저장: {path}")


def extract_dynamic(obs):
    """관측값에서 실제 등장한 객체만 추출."""
    mario = obs["mario"]

    result = {
        "mario": {
            "bbox": mario[:4].tolist(),
            "vx": float(mario[4]),
            "vy": float(mario[5]),
            "invincible": bool(mario[6]),
            "jumping": bool(mario[7]),
        },
        "bullet_bills": [],
        "stars": [],
    }

    for bullet in obs["bullet_bills"]:
        if np.any(bullet != 0):
            result["bullet_bills"].append({
                "bbox": bullet[:4].tolist(),
                "vx": float(bullet[4]),
            })

    for star in obs["stars"]:
        if np.any(star != 0):
            result["stars"].append({
                "bbox": star[:4].tolist(),
            })

    return result


def print_grid(width, height, records):
    """레이어별 타일 존재 여부를 ASCII 그리드로 출력."""
    cols = width // GAME_TILE_SIZE
    rows = height // GAME_TILE_SIZE

    grid = [["." for _ in range(cols)] for _ in range(rows)]

    for item in records:
        r = item["row"]
        c = item["col"]

        if not (0 <= r < rows and 0 <= c < cols):
            continue

        symbol = "B" if item["layer"] == "background" else "O"

        if grid[r][c] == ".":
            grid[r][c] = symbol
        elif grid[r][c] != symbol:
            grid[r][c] = "X"

    print("\n===== TILE GRID =====")
    print("B: Background tile")
    print("O: Object-layer tile")
    print("X: Both layers")
    print(".: No extracted tile")
    print("   " + "".join(str(c % 10) for c in range(cols)))

    for r, line in enumerate(grid):
        print(f"{r:02d} " + "".join(line))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        ENV_ID,
        render_mode="none",
        bgm=False,
    )

    try:
        obs, info = env.reset(seed=42)
        base = env.unwrapped

        width = int(base.width)
        height = int(base.height)

        cols = width // GAME_TILE_SIZE
        rows = height // GAME_TILE_SIZE

        print("===== MAP INFORMATION =====")
        print(f"Width: {width}px")
        print(f"Height: {height}px")
        print(f"Tile size: {GAME_TILE_SIZE}px")
        print(f"Grid: {cols} columns x {rows} rows")

        background = extract_layer(
            base, base._background_layer, "background"
        )

        objects = extract_layer(
            base, base._object_layer, "object"
        )

        print(f"Background tiles: {len(background)}")
        print(f"Object tiles: {len(objects)}")

        save_csv("background_tiles.csv", background)
        save_csv("object_tiles.csv", objects)

        print_grid(width, height, background + objects)

        dynamic = extract_dynamic(obs)

        with (OUTPUT_DIR / "initial_entities.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(dynamic, f, ensure_ascii=False, indent=2)

        with (OUTPUT_DIR / "map_info.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump({
                "width_px": width,
                "height_px": height,
                "tile_size_px": GAME_TILE_SIZE,
                "rows": rows,
                "cols": cols,
                "background_tiles": len(background),
                "object_tiles": len(objects),
            }, f, ensure_ascii=False, indent=2)

        print("\n===== INITIAL ENTITIES =====")
        print(json.dumps(dynamic, indent=2, ensure_ascii=False))
        print("\n추출 완료!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
