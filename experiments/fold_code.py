"""참고용으로 길게 늘어놓은 코드를 접는다.

코드는 웬만하면 펼쳐 둔다. 이 책에서 코드는 곁들인 것이 아니라 설명 그 자체라,
접으면 쪽이 뒤집힌다. 브라우저의 쪽 안 찾기(Ctrl+F)도 사파리·파이어폭스에서는
접힌 `<details>` 안까지 들어가지 못하므로, 찾을 만한 이름을 숨기는 셈이 된다.

다만 **글이 거의 없는 쪽**은 다르다. 코드가 400줄을 넘고 글이 25%에 못 미치면
그 쪽은 가르치는 쪽이 아니라 늘어놓는 쪽이다 — 끊어 줄 설명이 아예 없으니
접어도 잃을 것이 없다. `ch27/flow_architectures/flow_utils.md`가 그런 쪽으로,
코드 1,348줄에 글이 78줄이다.

    python experiments/fold_code.py --list          # 해당하는 쪽만 보여 준다
    python experiments/fold_code.py <쪽> [<쪽> ...]  # 접는다

경로는 `docs/` 기준이다. 접은 뒤에도 `verify_outputs.py`는 그대로 돈다 —
그쪽은 코드 블록을 울타리(```python)로 찾지 접힘 여부로 찾지 않는다.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

MIN_CODE = 400          # 이보다 짧으면 접지 않는다
MAX_PROSE = 0.25        # 글이 이보다 많으면 가르치는 쪽이다 — 접지 않는다


def measure(md):
    """코드·출력·글의 줄 수를 센다.

    글의 몫을 잴 때 출력을 빼는 것이 핵심이다. 빼지 않으면 긴 출력을
    되살린 쪽이 '글이 많은 쪽'으로 잘못 보여 접기에서 빠져나간다 —
    출력은 글이 아니다.
    """
    total = len(md.splitlines())
    code = sum(len(b.splitlines()) for b in re.findall(r"```python\n(.*?)```", md, re.S))
    out = sum(len(b.splitlines()) for b in
              re.findall(r"\*\*출력[:：]?\*\*\s*\n+```[a-z]*\n(.*?)```", md, re.S))
    out += sum(len(b.splitlines()) for b in
               re.findall(r'\?\?\? note "전체 출력[^"]*"\n\n((?:    .*\n|\n)*)', md))
    prose = max(total - code - out, 0)
    return code, total, (prose / total if total else 1.0)


def is_dump(md):
    code, total, prose = measure(md)
    return total >= 20 and code >= MIN_CODE and prose < MAX_PROSE


def fold(rel):
    """그 쪽의 **모든** 바깥쪽 코드 블록을 접는다.

    가장 긴 것 하나만 접으면 안 된다. 늘어놓는 쪽은 대개 블록이 여럿으로
    쪼개져 있어(`ch15/metric_learning/siamese.md`가 그렇다), 하나만 접으면
    나머지가 그대로 남아 접은 보람이 없다.
    """
    p = DOCS / rel
    md = p.read_text()
    if not is_dump(md):
        code, total, prose = measure(md)
        return f"{rel}: 해당 없음 (코드 {code}줄, 글 {prose*100:.0f}%)"

    # 줄 맨 앞에서 시작하는 블록만 고른다. 들여쓴 것은 이미 admonition
    # 안에 있다는 뜻이라, 또 감싸면 상자가 겹쳐 깨진다.
    #
    # 앞자리가 **비었는지**를 본다. strip() 으로 견주면 "    " 도 빈 것으로
    # 읽혀서, 한 번 접어 들여쓴 블록을 다시 바깥쪽으로 세고 접힘 안에
    # 접힘을 또 만든다.
    outer = [m for m in re.finditer(r"```python\n(.*?)```", md, re.S)
             if md[md.rfind("\n", 0, m.start()) + 1:m.start()] == ""]
    if not outer:
        return f"{rel}: 접을 바깥쪽 블록이 없다"

    many = len(outer) > 1
    folded = 0
    for i, m in reversed(list(enumerate(outer, 1))):     # 뒤에서부터 — 자리가 안 밀린다
        n = len(m.group(1).splitlines())
        title = f"코드 {i} ({n}줄)" if many else f"코드 ({n}줄)"
        # 울타리째 네 칸 들여쓴다. 빈 줄은 빈 줄로 두어야 details 가 끊기지 않는다
        body = "\n".join("    " + l if l.strip() else ""
                         for l in m.group(0).split("\n"))
        md = md[:m.start()] + f'??? note "{title}"\n\n' + body + "\n" + md[m.end():]
        folded += n

    p.write_text(md)
    return f"{rel}: 블록 {len(outer)}개, 코드 {folded}줄을 접었다"


def candidates():
    out = []
    for p in sorted(DOCS.rglob("*.md")):
        if p.name.endswith("_score.md") or ".v" in p.name:
            continue
        md = p.read_text()
        if is_dump(md):
            code, total, prose = measure(md)
            out.append((code, int(prose * 100), str(p.relative_to(DOCS))))
    return sorted(out, reverse=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
    elif args[0] == "--list":
        rows = candidates()
        print(f"  접을 쪽: {len(rows)}개  (코드 {MIN_CODE}줄 이상 + 글 {int(MAX_PROSE*100)}% 미만)")
        for c, pf, n in rows:
            print(f"    코드 {c:>4}줄, 글 {pf:>2}%   {n}")
    else:
        for rel in args:
            print("  " + fold(rel))
