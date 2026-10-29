"""「그림으로 보는 AI」 상자가 어느 쪽에 있고 어느 쪽에 없는지 센다.

목표는 쪽마다 하나다(CLAUDE.md). 1,238쪽을 한 번에 돌 수 없으므로 장별로
어디까지 했는지 보여 주는 것이 이 스크립트의 일이다.

    python experiments/box_coverage.py              # 장별 요약
    python experiments/box_coverage.py ch24         # 그 장에서 빠진 쪽 이름
    python experiments/box_coverage.py --missing    # 빠진 쪽 전체

쪽 수는 발행되는 .md 만 센다 — `*_score.md` 와 `*.v[0-9]*.md` 는 mkdocs 가
빼는 파일이라(exclude_docs) 세면 분모가 틀린다.
"""

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

BOX = re.compile(r'<div class="storybox"')
SNAPSHOT = re.compile(r"\.v[0-9]")


# === 쪽 모으기 ===============================================================

def pages():
    """발행되는 .md 를 장 이름과 함께 돌려준다."""
    for p in sorted(DOCS.rglob("*.md")):
        if p.name.endswith("_score.md") or SNAPSHOT.search(p.name):
            continue
        rel = p.relative_to(DOCS)
        yield rel.parts[0], rel, BOX.search(p.read_text(encoding="utf-8")) is not None


# === 세기 ====================================================================

def main(argv):
    want = [a for a in argv if not a.startswith("-")]
    show_missing = "--missing" in argv

    chapters = {}
    missing = []
    for chapter, rel, has_box in pages():
        if want and chapter not in want:
            continue
        done, total = chapters.get(chapter, (0, 0))
        chapters[chapter] = (done + has_box, total + 1)
        if not has_box:
            missing.append(rel)

    if not chapters:
        print(f"해당하는 쪽이 없다: {' '.join(want)}")
        return 1

    done = sum(d for d, _ in chapters.values())
    total = sum(t for _, t in chapters.values())

    print(f"{'장':<12}{'상자':>6}{'쪽':>6}{'비율':>8}")
    print("-" * 32)
    for chapter in sorted(chapters):
        d, t = chapters[chapter]
        bar = "완료" if d == t else ""
        print(f"{chapter:<12}{d:>6}{t:>6}{d / t * 100:>7.0f}%  {bar}")
    print("-" * 32)
    print(f"{'합계':<12}{done:>6}{total:>6}{done / total * 100:>7.1f}%")
    print(f"\n남은 쪽 {total - done}")

    if show_missing or (want and len(missing) <= 60):
        print()
        for rel in missing:
            print(f"  {rel}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
