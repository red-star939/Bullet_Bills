"""Extract collision-relevant blocks from the locally installed kymnasium.

Usage (from your mario_pj project root):
  .\\.venv\\Scripts\\python.exe inspect_bullet_collision_rules.py

No game state is changed. Review the generated text before implementing
lethal/invincibility rules: different kymnasium versions can differ.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

TARGETS = ('env.py', 'objs.py', 'consts.py')
KEYWORDS = ('collide', 'bullet', 'invinc', 'stomp', 'bounce', 'game_over',
            'jump', 'velocity', 'damage', 'dead', 'rect')


def source_lines(path: Path) -> list[str]:
    return path.read_text(encoding='utf-8-sig').splitlines()


def extract_functions(path: Path) -> list[tuple[int, int, str, list[str]]]:
    raw = source_lines(path)
    tree = ast.parse('\n'.join(raw), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = raw[node.lineno-1:node.end_lineno]
            content = '\n'.join(body).lower()
            if any(k in content for k in KEYWORDS):
                # Prioritize actual collision/step functions; avoid entire unrelated class.
                explicit = ('collide', 'collision', 'tick_frame', 'step_internal',
                            'apply_force', 'damage', 'invincible', 'die')
                if any(k in node.name.lower() for k in explicit) or any(
                    k in content for k in ('colliderect', "'game_over'", 'bounce')
                ):
                    found.append((node.lineno,node.end_lineno,node.name,body))
    # Descending extraction avoids overlapping nested scopes in ordinary cases.
    found.sort(key=lambda item: (item[0],item[1]))
    return found


def main():
    spec = importlib.util.find_spec('kymnasium')
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit('kymnasium is not installed in this Python interpreter.')
    base = Path(next(iter(spec.submodule_search_locations))) / 'envs' / 'bullet_bill'
    if not base.is_dir():
        raise SystemExit(f'Cannot find bullet_bill sources: {base}')
    report = [f'ENGINE SOURCE ROOT: {base}',
              'These are actual installed source excerpts, not model assumptions.']
    for filename in TARGETS:
        source = base / filename
        if not source.exists():
            continue
        report.append(f'\n===== {filename} =====')
        lines = source_lines(source)
        if filename == 'consts.py':
            for number, line in enumerate(lines, 1):
                if any(k in line.lower() for k in ('invinc', 'stomp', 'bounce',
                         'bullet', 'jump', 'mario', 'fps')):
                    report.append(f'{number:4d}: {line}')
        else:
            for start,end,name,body in extract_functions(source):
                report.append(f'\n--- {filename}:{start}-{end} {name} ---')
                for number, line in enumerate(body,start):
                    report.append(f'{number:4d}: {line}')
    out = Path('logs/collision_precision/engine_collision_rules.txt')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Saved source audit: {out}')
    print('Check whether the contact branch uses colliderect, descending vy,')
    print('Mario feet/top tests, invincible flags, and any bullet removal order.')


if __name__ == '__main__':
    main()
