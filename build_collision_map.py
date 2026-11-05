
import csv
import json
from pathlib import Path

MAP_DIR = Path("map_data")

ROWS = 18
COLS = 27
TILE_SIZE = 48

MARIO_ID = 0
STAR_ID = 26


def load_tiles():
    path = MAP_DIR / "object_tiles.csv"

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_map(tiles):
    grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
    collision = [[0 for _ in range(COLS)] for _ in range(ROWS)]

    for tile in tiles:
        row = int(tile["row"])
        col = int(tile["col"])
        tile_id = int(tile["tile_id"])

        if not (0 <= row < ROWS and 0 <= col < COLS):
            continue

        if tile_id == MARIO_ID:
            grid[row][col] = "M"

        elif tile_id == STAR_ID:
            grid[row][col] = "S"

        else:
            grid[row][col] = "#"
            collision[row][col] = 1

    return grid, collision


def main():
    tiles = load_tiles()
    grid, collision = build_map(tiles)

    print("===== COLLISION MAP =====")
    print("   " + "".join(str(i % 10) for i in range(COLS)))

    for row_idx, row in enumerate(grid):
        print(f"{row_idx:02d} " + "".join(row))

    output = {
        "width_px": COLS * TILE_SIZE,
        "height_px": ROWS * TILE_SIZE,
        "tile_size": TILE_SIZE,
        "rows": ROWS,
        "cols": COLS,
        "collision_grid": collision,
    }

    path = MAP_DIR / "collision_map.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n충돌 맵 저장 완료: {path}")
    print("충돌 타일 수:", sum(map(sum, collision)))


if __name__ == "__main__":
    main()
