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
시간·속도처럼 기계에 딸린 값은 다른 기계에서 같을 수 없다. 메모리 번지와
`id()`, 갈무리한 파일의 바이트 수도 그렇다. 그런 줄은 세되 어긋남으로
치지 않고 따로 알린다.

아직 못 하는 것
--------------
**한 줄에 성한 값과 기계에 딸린 값이 섞여 있으면 가리지 못한다.** 견주기를
줄 단위로 하기 때문이다. `ch16/continual_learning/03_comprehensive_comparison.md`
의 표가 그렇다 — 한 줄에 정확도 셋과 걸린 시간이 함께 있어, 정확도 열두 개가
모두 그대로 나오는데도 그 네 줄이 어긋남으로 뜬다. 그런 쪽은 열을 따로
떼어 손으로 견주는 수밖에 없다(그렇게 해서 확인해 두었다).
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
                    r"|옮기는 값|계산 값|처리량|초당|비율"
                    # 시간에서 끌어낸 백분율·배수도 기계에 딸린 값이다.
                    # `Time Savings: 36.4%` 는 %만 붙어 있어 시간처럼 보이지
                    # 않지만, 두 시간을 견준 값이라 기계가 바뀌면 바뀐다.
                    r"|[Ss]avings|[Ss]peedup|[Tt]ime [Pp]er|아낀"
                    # 갈무리한 그림의 바이트 수. matplotlib 판과 글꼴에 따라
                    # 달라지므로 다른 기계에서 같을 수 없다.
                    r"|파일 크기|[Ff]ile size|바이트|bytes")

# 출력이 길어 줄인 자리에 글쓴이가 손으로 적어 넣은 표시. 나올 리가 없다.
ELIDED = re.compile(r"\(\s*\d+\s*lines? omitted\s*\)|^\s*\.\.\.\s*$|생략")

# 메모리 번지와 id. 프로세스마다 달라 어느 기계에서도 다시 나오지 않는다.
#
# `ch06/tensor_attrs/memory_layout_strides.md`는 같은 번지를 네 번 찍어 두 텐서가
# 자료를 함께 쓴다는 것을 보인다. 값 자체는 아무 뜻이 없고 **같다는 것**이 뜻이라,
# 번지를 싣는 것은 그 쪽에서는 올바른 가르침이다. 다만 견줄 수는 없다.
# (`x.data_ptr() == y.data_ptr()` 를 찍으면 견줄 수 있게 되지만, 그것은 코드를
# 바꾸는 일이다.)
ADDRESS = re.compile(r"object at 0x|0x[0-9a-fA-F]{6,}|"
                     r"\b(id|pointer|ptr|address|번지)\b\s*[:=]|^\s*\d{9,}\s*$",
                     re.I | re.M)


def code_blocks(md_text):
    """```python 블록을 뽑되, admonition 안의 들여쓰기를 벗긴다 (함정 3)."""
    out = []
    for b in re.findall(r"```python\n(.*?)```", md_text, re.S):
        lines = b.split("\n")
        pad = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        out.append("\n".join(l[pad:] if l.strip() else l for l in lines))
    return out


FOLD_HEAD = re.compile(r'\?\?\? note "전체 출력[^"]*"\n')


PLAIN_OUT = re.compile(r"\*\*출력[:：]?\*\*\s*\n+```[a-z]*\n(.*?)```", re.S)


def find_output(md_text, start=0):
    """`start` 뒤에 **가장 먼저** 나오는 출력 블록.

    펼쳐 놓은 것(`**출력:**`)과 접어 놓은 것(`??? note "전체 출력 …"`)을 함께
    보고, 둘 중 앞에 있는 것을 고른다. 접히면 `**출력:**` 표시가 사라지므로
    그것만 찾으면 접힌 쪽이 확인에서 빠지고, 어느 한쪽만 찾으면 둘이 섞여
    있는 쪽에서 차례가 뒤집힌다.

    돌려주는 것: (시작 자리, 끝 자리, 출력 내용)
    """
    plain = PLAIN_OUT.search(md_text, start)
    fold = FOLD_HEAD.search(md_text, start)
    if plain and (not fold or plain.start() < fold.start()):
        return plain.start(), plain.end(), plain.group(1)
    if not fold:
        return None, None, None

    body, end = [], fold.end()
    for line in md_text[fold.end():].split("\n"):
        if line.strip() == "" or line.startswith("    "):
            body.append(line[4:] if line.startswith("    ") else "")
            end += len(line) + 1
        else:
            break
    text = "\n".join(body)
    text = re.sub(r"^\s*```[a-z]*\n", "", text)         # 안쪽 울타리를 벗긴다
    text = re.sub(r"```\s*$", "", text)
    return fold.start(), end, text


def all_outputs(md_text):
    """쪽에 있는 출력 블록을 앞에서부터 모두 찾는다.

    한 쪽에 출력이 여럿인 곳이 54쪽 있다(`ch06/tensor_attrs/memory_layout_strides.md`
    는 22개다). 첫 블록만 보면 나머지는 확인도 손질도 받지 못한다.
    """
    outs, pos = [], 0
    while True:
        s, e, t = find_output(md_text, pos)
        if s is None:
            return outs
        outs.append((s, e, t))
        pos = e


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

    outs = all_outputs(md)
    if not outs:
        return None

    # 블록이 여럿이면 한 번만 돌리고 **모든** 출력을 그 하나에 견준다.
    # 뒤 블록은 앞 블록에서 만든 이름을 쓰는 일이 많아 따로 돌릴 수 없고,
    # 블록마다 돌리면 같은 학습을 여러 번 하게 된다. 이어 붙여 한 번 돌리면
    # 모든 출력이 차례로 나오므로, 실린 줄이 그 안에 있는지만 보면 된다.
    blocks = code_blocks(md[:outs[-1][0]])           # 함정 1: 마지막 출력 앞까지만
    want = "\n".join(t for _, _, t in outs)
    if not blocks or not want.strip():
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
    # plt.show() 는 화면이 있는 백엔드에서 **영원히 멈춘다**. 확인하는 자리에는
    # 화면이 없으니 기다릴 사람도 없다. Agg 로 묶으면 그림은 파일로만 가고
    # 코드는 그대로 지나간다 — 이 한 줄이 14쪽을 시간초과에서 90초로 바꿨다.
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = str(md_path.parent) + os.pathsep + env.get("PYTHONPATH", "")

    r = None
    timed_out = False
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
                # 이어 붙인 것이 오래 걸리면 가장 긴 블록만으로 한 번 더 해 본다.
                # 이어 붙이면 연습문제 풀이까지 함께 돌아 훨씬 오래 걸리는데,
                # 여기서 바로 포기하면 되돌림을 써 보지도 못하고 끝난다.
                r = None
                timed_out = True
                continue
            finally:
                Path(tmp).unlink(missing_ok=True)
            if r.returncode == 0:
                break

    # 시간초과를 실행실패와 한 칸에 넣으면 안 된다. 시간초과는 stderr 가 비어 있어
    # 까닭 없는 "실행실패" 한 줄로만 보이는데, 정작 고칠 것은 쪽이 아니라 --timeout 이다.
    if r is None and timed_out:
        return ("시간초과", 0, 0,
                [f"{timeout}초 안에 못 끝냈다. --timeout <초> 로 늘려서 다시 돌려라."], 0)

    if r is None or r.returncode != 0:
        tail = r.stderr.strip().splitlines()[-2:] if r else []
        return ("실행실패", 0, 0, tail or ["stderr 가 비어 있다"], 0)

    wanted = numeric_lines(want)

    # 기계에 딸린 **자리**만 지우고 나머지는 그대로 견준다. 줄째로 빼면 안 된다 --
    # 4.2 의 결과 줄은 "... 72.76%  퍼짐 0.89 ... (328s)" 처럼 정확도와 시간이
    # 한 줄에 같이 있어서, 시간을 핑계로 줄을 빼면 정확도까지 안 보고 넘어간다.
    # 그 쪽이 실제로 "일치 0/5 줄"로 통과했었다.
    def blur(s):
        return re.sub(r"\s+", " ", ADDRESS.sub(" ", TIMING.sub(" ", s))).strip()

    out_blurred = "\n".join(blur(l) for l in r.stdout.splitlines())

    missing, soft = [], 0
    for l in wanted:
        if l in r.stdout:
            continue
        b = blur(l)
        if b and b in out_blurred:
            soft += 1                      # 시간이나 번지만 다르다. 값은 맞다
            continue
        missing.append(l)

    hard = [l for l in missing
            if not TIMING.search(l) and not ADDRESS.search(l)]
    # (값은 맞고 시간만 다른 줄, 값까지 안 맞지만 봐 준 줄) 을 따로 돌려준다.
    # 둘을 한 수로 합치면 "5개는 셈에서 뺐다"가 되어, 실제로는 값이 다 맞은
    # 경우까지 안 본 것처럼 읽힌다.
    return ("확인", len(wanted) - len(missing), len(wanted), hard,
            (soft, len(missing) - len(hard)))


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
            n_soft, n_skip = n_time
            bits = []
            if n_soft:
                bits.append(f"줄 {n_soft}개는 시간·번지만 다르고 값은 같다")
            if n_skip:
                bits.append(f"줄 {n_skip}개는 값이 달라도 기계 탓으로 보고 봐 줬다")
            note = f"   ({', '.join(bits)})" if bits else ""
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
