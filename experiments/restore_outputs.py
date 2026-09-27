"""잘린 출력을 되살리고, 씨앗 없이 실린 수에 씨앗을 물린다.

`verify_outputs.py`가 어긋난 쪽을 **찾는다면** 이 스크립트는 그것을 **고친다**.
쪽마다 두 가지를 손본다.

1. **잘린 출력** — 끝이 `... (N lines omitted)`로 끊긴 출력. 이 표시는 읽는
   이를 코드를 다시 돌리게 만드는, 바로 그 물건이다. 다시 돌려 전부 싣는다.
   길면(`LONG`줄 넘으면) 접어서 넣는다 — 쪽이 출력에 뒤덮이지 않으면서도
   다시 돌릴 까닭은 없어진다.

2. **씨앗 없는 무작위** — `torch.randn`으로 뽑은 수를 실어 놓고 씨앗을
   고정하지 않은 쪽. 그 수는 누구의 기계에서도 다시 나오지 않는다.
   맨 위 import 무리 다음에 `torch.manual_seed(0)`을 끼운다.

    python experiments/restore_outputs.py ch07/mle/capture_recapture_mle.md
    python experiments/restore_outputs.py ch09/optimizers/*.md

경로는 `docs/` 기준이다. 코드는 **버리는 자리**에서 돈다 — 예제들이 그림과
저장 파일을 현재 자리에 쏟아 놓기 때문에, 저장소 뿌리에서 돌리면 뿌리가
찌꺼기로 덮인다. 자료(`data/`)만 심볼릭 링크로 빌려다 쓴다.

고치고 나면 반드시 `verify_outputs.py`로 되짚어 본다.
"""

import ast
import re
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LONG = 90                      # 이보다 긴 출력은 접는다

SEED_LINE = ("\n# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린\n"
             "# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다\n"
             "torch.manual_seed(0)\n")

USES_RANDOM = re.compile(r"torch\.randn|torch\.rand\b|torch\.randint|torch\.randperm|"
                         r"nn\.Linear|nn\.Conv|nn\.Sequential|np\.random|random\.")

CUT = re.compile(r"lines? omitted|^\s*\.\.\.\s*$", re.M)


def sandbox_cwd(stack):
    """코드를 돌릴 버리는 자리. 자료만 링크로 빌려 준다."""
    d = stack.enter_context(tempfile.TemporaryDirectory())
    for name in ("data", "figures"):
        src = ROOT / name
        if src.exists():
            (Path(d) / name).symlink_to(src)
    # 5장의 쪽들은 장 첫머리에서 만든 심판 CNN(mnist_judge.pt)을 함께 읽는다.
    # 뿌리에 그런 파일이 있으면 빌려 준다 — 없으면 그 쪽은 그냥 실패하고,
    # 그것이 곧 "먼저 만들어야 한다"는 알림이 된다.
    for art in list(ROOT.glob("*.pt")) + list(ROOT.glob("*.pth")):
        (Path(d) / art.name).symlink_to(art)
    return d


def insert_seed(code):
    """맨 위 import 무리 바로 다음에 씨앗 줄을 끼운다.

    자리는 **구문으로** 찾는다. 줄 생김새로 찾으면 여러 줄에 걸친 import

        from attention_visualization import (
            AttentionVisualizer,
            ...
        )

    의 첫 줄이 마지막 import 줄로 잡혀, 씨앗이 괄호 **안**에 들어가고
    쪽이 통째로 SyntaxError 가 된다.

    씨앗이 필요 없거나 끼울 자리가 없으면 코드를 그대로 돌려준다.
    """
    if "manual_seed" in code or "np.random.seed" in code or not USES_RANDOM.search(code):
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code                       # 이미 깨진 쪽은 건드리지 않는다

    # 무엇을 들였는지 **구문으로** 본다. `"import torch" in code` 처럼 글자로
    # 찾으면 주석이나 설명글 속의 그 말에 걸려, torch 를 들이지도 않은 쪽에
    # torch.manual_seed 를 끼우게 된다 — `ch06/autograd/autograd_from_scratch.md`
    # 가 numpy 만으로 자동 미분을 손수 짜면서 글 속에 그 말을 적어 둔 경우다.
    names, last = {}, 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, getattr(node, "end_lineno", node.lineno))
            if isinstance(node, ast.Import):
                for a in node.names:
                    names[a.name.split(".")[0]] = a.asname or a.name.split(".")[0]
            elif node.module:
                names.setdefault(node.module.split(".")[0],
                                 node.module.split(".")[0])
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue                      # 맨 앞 설명글은 건너뛴다
        else:
            break                         # 진짜 코드가 시작되면 멈춘다
    if last == 0:
        return code

    if "torch" in names:
        seed = f"{names['torch']}.manual_seed(0)"
    elif "numpy" in names:
        seed = f"{names['numpy']}.random.seed(0)"
    elif "random" in names:
        seed = f"{names['random']}.seed(0)"
    else:
        return code                       # 씨앗을 물릴 것이 없다

    line = ("\n# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린\n"
            "# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다\n"
            f"{seed}\n")
    lines = code.split("\n")
    return "\n".join(lines[:last]) + line + "\n".join(lines[last:])


def run(code, timeout=1200, page_dir=None):
    """코드를 버리는 자리에서 돌린다.

    `page_dir`을 주면 그 자리를 PYTHONPATH 앞에 붙인다. 쪽에 실린 코드가
    옆에 놓인 `.py`를 불러 쓰는 일이 있기 때문이다 —
    `ch40/gradient_methods/example_attention.md`가 같은 자리의
    `attention_visualization.py`를 부르는 것이 그런 경우다.
    """
    import os
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    env = dict(os.environ)
    if page_dir:
        env["PYTHONPATH"] = str(page_dir) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        with ExitStack() as stack:
            return subprocess.run([sys.executable, tmp], capture_output=True,
                                  text=True, timeout=timeout,
                                  cwd=sandbox_cwd(stack), env=env)
    finally:
        Path(tmp).unlink(missing_ok=True)


FOLD_HEAD = re.compile(r'\?\?\? note "전체 출력[^"]*"\n')


def find_output(md):
    """출력이 놓인 자리. 펼친 것과 접은 것을 둘 다 찾는다.

    한 번 접고 나면 `**출력:**` 표시가 사라지므로, 그것만 찾으면 접힌 쪽은
    다시 손볼 수 없게 된다.
    """
    m = re.search(r"\*\*출력[:：]?\*\*\s*\n+```[a-z]*\n(.*?)```", md, re.S)
    if m:
        return m.start(), m.end(), m.group(1)
    m = FOLD_HEAD.search(md)
    if not m:
        return None, None, None
    body, end = [], m.end()
    for line in md[m.end():].split("\n"):
        if line.strip() == "" or line.startswith("    "):
            body.append(line[4:] if line.startswith("    ") else "")
            end += len(line) + 1
        else:
            break
    return m.start(), end, "\n".join(body)


def fix(rel, timeout=1200):
    p = ROOT / "docs" / rel
    md = p.read_text()
    o_start, o_end, o_text = find_output(md)
    if o_start is None:
        return f"{rel}: 출력 블록이 없다"
    cm = list(re.finditer(r"(```python\n)(.*?)(```)", md[:o_start], re.S))
    if not cm:
        return f"{rel}: 코드 블록이 없다"
    block = max(cm, key=lambda m: len(m.group(2)))

    seeded = insert_seed(block.group(2))
    added_seed = seeded != block.group(2)

    try:
        r = run(seeded, timeout, page_dir=p.parent)
    except subprocess.TimeoutExpired:
        return f"{rel}: 시간초과 ({timeout}초) — 건드리지 않았다"
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or ["(까닭 없음)"])[-1]
        return f"{rel}: 실행 실패 — {last[:90]} (건드리지 않았다)"

    was_cut = bool(CUT.search(o_text))
    n_out = len(r.stdout.splitlines())
    if n_out == 0:
        return f"{rel}: 출력이 비었다 — 건드리지 않았다"

    if n_out > LONG:
        indented = "\n".join("    " + l if l else "" for l in r.stdout.split("\n"))
        new_out = f'??? note "전체 출력 ({n_out}줄)"\n\n    ```\n{indented}    ```\n'
    else:
        new_out = "**출력:**\n\n```\n" + r.stdout + "```"

    p.write_text(md[:block.start()] + block.group(1) + seeded + block.group(3)
                 + md[block.end():o_start] + new_out + md[o_end:])

    bits = []
    if was_cut:
        bits.append("잘린 것을 되살림")
    if added_seed:
        bits.append("씨앗 넣음")
    bits.append(("접어서 " if n_out > LONG else "") + f"출력 갈아 끼움 ({n_out}줄)")
    return f"{rel}: " + ", ".join(bits)


if __name__ == "__main__":
    args = sys.argv[1:]
    timeout = 1200
    if "--timeout" in args:
        i = args.index("--timeout")
        timeout = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        print(__doc__)
        sys.exit(0)
    for rel in args:
        print("  " + fix(rel, timeout))
