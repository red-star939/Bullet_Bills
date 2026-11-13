"""Evaluate Model V4.1 without rendering."""
import sys
from eval_v41 import parse_args,run

if __name__=='__main__':
    args=parse_args(['--mode','none',*sys.argv[1:]])
    run(args)
