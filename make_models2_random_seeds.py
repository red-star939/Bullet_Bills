"""Generate unique random environment seeds; exclude every seed in prior splits."""
import argparse
import json
import random
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--exclude',nargs='*',default=[])
    p.add_argument('--master-seed',type=int,default=None,help='Optional deterministic seed list generation')
    a=p.parse_args()
    if a.count<1: p.error('count must be >=1')
    excluded=set()
    for file in a.exclude:
        excluded.update(int(n) for n in json.loads(Path(file).read_text(encoding='utf-8')))
    rng=random.SystemRandom() if a.master_seed is None else random.Random(a.master_seed)
    chosen=set()
    while len(chosen)<a.count:
        x=rng.randint(1,2_147_483_646)
        if x not in excluded:chosen.add(x)
    destination=Path(a.out);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(sorted(chosen),indent=2),encoding='utf-8')
    print(f'SEEDS {destination} count={len(chosen)} excludes={len(excluded)}')
if __name__=='__main__':main()
