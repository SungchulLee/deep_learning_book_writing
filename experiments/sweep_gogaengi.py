"""고갱이 -> 핵심. 조사를 받침에 맞게 갈아 준다.

고갱이는 홀소리로 끝나고(이) 핵심은 닿소리로 끝난다(ㅁ). 그래서 뒤에 붙는
조사가 달라진다 — 가/이, 는/은, 를/을, 로/으로, 와/과, 다/이다.
그냥 바꿔치기하면 "핵심가", "핵심는" 같은 것이 900군데 생긴다.
"""
import pathlib, re, sys, collections

SUB = {"가":"이", "는":"은", "를":"을", "로":"으로", "와":"과",
       "다":"이다", "란":"이란", "라":"이라", "나":"이나", "든":"이든",
       "의":"의", "에":"에", "도":"도", "만":"만", "":""}
PAT = re.compile("고갱이(" + "|".join(sorted((k for k in SUB if k), key=len, reverse=True)) + ")?")
DRY = "--apply" not in sys.argv

hits = collections.Counter(); files = 0
for p in sorted(pathlib.Path("docs").rglob("*.md")):
    if re.search(r"\.v[0-9]+\.md$", p.name):
        continue
    t = orig = p.read_text()
    def rep(m):
        j = m.group(1) or ""
        hits[f"고갱이{j} -> 핵심{SUB[j]}"] += 1
        return "핵심" + SUB[j]
    t = PAT.sub(rep, t)
    if t != orig:
        files += 1
        if not DRY:
            p.write_text(t)
print(("[미리보기] " if DRY else "[적용] ") + f"{files}개 파일, {sum(hits.values())}군데")
for k, v in hits.most_common(14):
    print(f"   {v:5d}  {k}")
