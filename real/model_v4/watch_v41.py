"""Watch Model V4.1 in the real game window."""
import sys
from eval_v41 import parse_args,run

if __name__=='__main__':
    args=parse_args(['--mode','human',*sys.argv[1:]])
    run(args)
