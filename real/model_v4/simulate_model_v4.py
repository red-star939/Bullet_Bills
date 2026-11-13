"""Convenience entry point for headless Model V4 evaluation."""
from eval_model_v4 import parse_args, run

if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    if '--mode' not in argv:
        argv = ['--mode', 'none', *argv]
    run(parse_args(argv))
