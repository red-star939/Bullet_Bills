"""Convenience entry point for on-screen Model V4 game simulation."""
from eval_model_v4 import parse_args, run

if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    if '--mode' not in argv:
        argv = ['--mode', 'human', *argv]
    run(parse_args(argv))
