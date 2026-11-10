"""Compare paired v1/v2 episodes.csv by seed; no RL environment needed."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import statistics

def read(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    return {int(r['seed']): float(r['survival_s']) for r in rows}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v1', default='logs/models2_v1/episodes.csv')
    p.add_argument('--v2', default='logs/models2_v2/episodes.csv')
    args = p.parse_args()
    a, b = read(args.v1), read(args.v2)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit('No matching seeds')
    print('SEED       V1(s)    V2(s)    V2-V1(s)')
    diffs = []
    for seed in common:
        d = b[seed] - a[seed]; diffs.append(d)
        print(f'{seed} {a[seed]:9.3f} {b[seed]:9.3f} {d:+10.3f}')
    print(f'N={len(common)}  V1_MEAN={statistics.mean(a[i] for i in common):.3f}  '
          f'V2_MEAN={statistics.mean(b[i] for i in common):.3f} '
          f'PAIRED_DELTA={statistics.mean(diffs):+.3f}  '
          f'WINS={sum(d>0 for d in diffs)}/{len(diffs)}')

if __name__ == '__main__':
    main()
