"""쪽에 실린 출력이 정말 그 쪽의 코드에서 나오는지 확인한다.

`mkdocs build --strict`는 링크와 문법만 본다. **실린 수가 아직 맞는지**는
보지 않으므로, 코드를 고치고 출력 블록을 그대로 두면 아무도 모른다.
이 스크립트가 그 틈을 메운다.

하는 일
-------
쪽마다 ```python 블록을 꺼내 돌리고, `**출력:**` 뒤의 블록에 적힌 줄이
실제 출력에 있는지 센다.

    python experiments/verify_outputs.py ch02/models/trees.md
    python experiments/verify_outputs.py ch01/libraries/*.md

경로는 `docs/` 기준이다. 코드는 **버리는 자리**에서 돈다 — 예제들이 그림과
저장 파일을 현재 자리에 쏟아 놓으므로 저장소 뿌리에서 돌리면 뿌리가 덮인다.
자료(`data/`)와 뿌리의 `.pt` 파일만 심볼릭 링크로 빌려 준다.

세 가지 함정
-----------
이 스크립트를 처음 쓸 때 셋 다 걸렸고, 셋 다 쪽의 잘못이 아니라
스크립트의 잘못이었다. 고쳐 두었으나 까닭은 적어 둔다.

1. **블록을 이어 붙이면 안 된다.** 한 쪽에 여러 블록이 있을 때 뒤의 것은
   대개 연습문제 풀이라, 이어 붙이면 그 쪽 출력에 없는 것이 섞인다.
   그래서 `**출력:**` **앞**의 블록만 본다.

2. **가장 긴 블록만 돌린다.** 설명하려고 넣은 몇 줄짜리 조각이 본
   스크립트 앞에 오기도 한다(`pad = (x == PAD)` 따위). 이어 붙이면
   정의되지 않은 이름에서 터진다.

3. **들여쓴 블록은 벗겨야 한다.** admonition(`!!! note`) 안의 코드는 네 칸
   들여써져 있어 그대로 돌리면 IndentationError가 난다.

맞을 수 없는 줄
--------------
시간·속도처럼 기계에 딸린 값은 다른 기계에서 같을 수 없다. 그런 줄은
세되 어긋남으로 치지 않고 따로 알린다.
"""

import re
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # 저장소 뿌리
DOCS = ROOT / "docs"

# 초·밀리초·배속처럼 기계마다 달라지는 값이 든 줄.
#
# 잰 시간끼리 **나눈 값**도 마찬가지다. `옮기는 값 / 계산 값: 0.07` 같은 줄은
# 단위가 없어 눈에 잘 띄지 않지만 기계가 바뀌면 함께 바뀐다 — 그런 쪽의
# 주장은 대개 "1보다 크면" 처럼 문턱으로 적혀 있지 값으로 적혀 있지 않다.
TIMING = re.compile(r"\d+\.?\d*\s*(m?s\b|us\b|sec|초|x\b|배)|elapsed|time=|GB/s|MB"
                    r"|옮기는 값|계산 값|처리량|초당|비율")

# 출력이 길어 줄인 자리에 글쓴이가 손으로 적어 넣은 표시. 나올 리가 없다.
ELIDED = re.compile(r"\(\s*\d+\s*lines? omitted\s*\)|^\s*\.\.\.\s*$|생략")


def code_blocks(md_text):
    """```python 블록을 뽑되, admonition 안의 들여쓰기를 벗긴다 (함정 3)."""
    out = []
    for b in re.findall(r"```python\n(.*?)```", md_text, re.S):
        lines = b.split("\n")
        pad = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        out.append("\n".join(l[pad:] if l.strip() else l for l in lines))
    return out


FOLD_HEAD = re.compile(r'\?\?\? note "전체 출력[^"]*"\n')


def find_output(md_text):
    """출력을 찾는다. 펼쳐 놓은 것과 접어 놓은 것을 **둘 다** 읽는다.

    긴 출력은 `??? note "전체 출력 (186줄)"` 안에 접혀 들어간다. 접히면
    `**출력:**` 표시가 사라지므로, 그것만 찾으면 접힌 쪽은 통째로 확인
    대상에서 빠져 버린다 — 고쳐 놓고 확인은 못 하게 되는 셈이다.

    돌려주는 것: (시작 자리, 끝 자리, 출력 내용)
    """
    m = re.search(r"\*\*출력[:：]?\*\*\s*\n+```[a-z]*\n(.*?)```", md_text, re.S)
    if m:
        return m.start(), m.end(), m.group(1)

    m = FOLD_HEAD.search(md_text)
    if not m:
        return None, None, None
    body, end = [], m.end()
    for line in md_text[m.end():].split("\n"):
        if line.strip() == "" or line.startswith("    "):
            body.append(line[4:] if line.startswith("    ") else "")
            end += len(line) + 1
        else:
            break
    text = "\n".join(body)
    text = re.sub(r"^\s*```[a-z]*\n", "", text)         # 안쪽 울타리를 벗긴다
    text = re.sub(r"```\s*$", "", text)
    return m.start(), end, text


def output_block(md_text):
    return find_output(md_text)[2]


def numeric_lines(text):
    """수가 든 줄만 남긴다. 견주어 뜻이 있는 것은 그것들이다."""
    return [l.strip() for l in text.splitlines()
            if len(l.strip()) > 3 and re.search(r"\d", l)
            and not ELIDED.search(l)]



def sandbox_cwd(stack):
    """쪽의 코드를 돌릴 **버리는 자리**를 만든다.

    예제들은 그림(.png)과 저장 파일(.pth)을 현재 자리에 쏟아 놓는다.
    저장소 뿌리에서 돌리면 그것들이 뿌리에 쌓이므로, 빈 자리를 만들고
    자료만 심볼릭 링크로 빌려다 쓴다. 자리째 지우면 찌꺼기도 함께 간다.
    """
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

def check(md_path, timeout=1800):
    md = md_path.read_text()

    start, _, want = find_output(md)
    if start is None:
        return None
    blocks = code_blocks(md[:start])                 # 함정 1: 출력 앞의 것만
    if not blocks or not want:
        return None

    # 함정 2: 어느 한 가지로 정할 수 없다.
    #   쪽에 따라 블록 여럿이 이어져 하나의 프로그램이기도 하고
    #   (ch02/models/trees.md — 트리 다음에 신경망을 견준다),
    #   설명용 조각이 본 스크립트 앞에 끼어 있기도 하다
    #   (imdb/05_attention.md — safe_mask 를 보이는 네 줄).
    # 그래서 이어 붙인 것을 먼저 돌려 보고, 터지면 가장 긴 것만 돌린다.
    candidates = ["\n".join(blocks)]
    if len(blocks) > 1:
        candidates.append(max(blocks, key=lambda b: len(b.splitlines())))

    # 옆에 놓인 .py 를 불러 쓰는 쪽이 있다 — 그 자리를 PYTHONPATH 에 얹는다
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(md_path.parent) + os.pathsep + env.get("PYTHONPATH", "")

    r = None
    with ExitStack() as stack:
        cwd = sandbox_cwd(stack)
        for code in candidates:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp = f.name
            try:
                r = subprocess.run([sys.executable, tmp], capture_output=True,
                                   text=True, timeout=timeout, cwd=cwd, env=env)
            except subprocess.TimeoutExpired:
                return ("시간초과", 0, 0, [], 0)
            finally:
                Path(tmp).unlink(missing_ok=True)
            if r.returncode == 0:
                break

    if r is None or r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-2:] if r else []
        return ("실행실패", 0, 0, tail, 0)

    wanted = numeric_lines(want)
    missing = [l for l in wanted if l not in r.stdout]
    hard = [l for l in missing if not TIMING.search(l)]
    return ("확인", len(wanted) - len(missing), len(wanted), hard, len(missing) - len(hard))


def main(rels, timeout=1800):
    bad = 0
    for rel in rels:
        res = check(DOCS / rel, timeout=timeout)
        if res is None:
            print(f"  건너뜀   {rel}   (코드나 출력 블록이 없다)")
            continue
        status, ok, tot, detail, n_time = res
        if status != "확인":
            print(f"  {status} {rel}")
            for d in detail:
                print(f"      {d}")
            bad += 1
        else:
            note = f"   (기계에 딸린 줄 {n_time}개는 셈에서 뺐다)" if n_time else ""
            print(f"  {'일치  ' if not detail else '어긋남'} {rel}   {ok}/{tot} 줄{note}")
            for d in detail:
                print(f"      실린 줄이 안 나온다: {d[:90]}")
            bad += bool(detail)
    return bad


if __name__ == "__main__":
    args = sys.argv[1:]
    # --timeout <초> : 오래 걸리는 쪽(IMDB 사다리 따위)을 기다려 주려면 늘린다
    timeout = 1800
    if "--timeout" in args:
        i = args.index("--timeout")
        timeout = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        print(__doc__)
        sys.exit(0)
    sys.exit(1 if main(args, timeout=timeout) else 0)
