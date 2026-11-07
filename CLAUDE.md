# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

MkDocs Material documentation site for a mathematical textbook. Content uses LaTeX (MathJax)
and includes Python example scripts. Three specialized review/write agents live in `agents/`;
their full instructions are in `agents/SKILL.md`.

## Build Commands

```bash
mkdocs serve          # local dev with live reload
mkdocs build          # build static site
mkdocs build --strict # used in CI — all warnings are errors
pip install -r requirements.txt
```

## Deployment

GitHub Actions (`.github/workflows/deploy-mkdocs.yml`) auto-deploys to GitHub Pages on push
to `main`. Build uses `--strict` mode.

## Repository Structure

```
book_name/
├── CLAUDE.md
├── README.md
├── mkdocs.yml
├── requirements.txt
├── agents/
│   ├── SKILL.md            ← agent orchestration hub (read this first)
│   ├── MATH_REVIEWER.md
│   ├── WRITING_REVIEWER.md
│   └── WRITER.md
├── .github/workflows/deploy-mkdocs.yml
└── docs/
    ├── index.md
    ├── assets/favicon.ico
    ├── stylesheets/extra.css
    ├── javascripts/mathjax.js
    └── chapter_name/
        ├── index.md
        └── section_name/
            ├── topic.md
            └── figures/make_figures.py
```

**No `.py` beside a page.** A page's whole program and whole output live in its
`.md`; see "Add a Python example" below. `figures/make_figures.py` is the only
`.py` to create under `docs/`.

## Agent Commands (Quick Reference)

> Full command semantics, file conventions, and agent prompts: **read `agents/SKILL.md` first**.

| Command | What it does |
|---|---|
| `review <file\|folder\|all>` | Freeze vN snapshot, run both reviewers, print to stdout. No writes. |
| `write <file\|folder\|all> [if score < N]` | Requires prior `review`. Runs reviewers in-memory, writes improved file, updates score. |
| `update <file\|folder\|all> [if score < N]` | `review` + `write` in one step. |

**Path convention**: paths are relative to `docs/` — omit the `docs/` prefix.

**Execution rules**: sequential only · batch size = 1 · commit after each file.

## File Management

| File | Git | GitHub Pages |
|---|---|---|
| `<name>.md` | ✅ committed | ✅ published |
| `<name>_score.md` | ✅ committed | ❌ excluded via `mkdocs.yml` |
| `<name>.v[0-9]*.md` | ❌ gitignored | ❌ never built |

`.gitignore` entry: `docs/**/*.v[0-9]*.md`

> The snapshot separator is a **dot**, not an underscore. An underscore pattern
> (`*_v[0-9]*.md`) also matches real content pages such as `mobilenet_v2.md`,
> `inception_v3.md`, and `yolo_v3.md`, which silently excludes them from git and
> breaks the `--strict` CI build.

`mkdocs.yml` exclusion:
```yaml
exclude_docs: |
  *_score.md
  *.v[0-9]*.md
  # Real content pages that collide with the *_score.md pattern above.
  !inception_score.md
  !01_inception_score.md
```

`*_score.md` has the same collision hazard: it also matches content pages such as
`inception_score.md`, which drops them from the site silently (mkdocs logs this at
INFO, so `--strict` still passes and the page 404s in production). Any new content
page ending in `_score.md` needs a `!` negation line here.

Commit after `update` — stage only `<name>.md` and `<name>_score.md`:
```bash
git add docs/path/to/<name>.md docs/path/to/<name>_score.md
git commit -m "update: <name>"
```

## Navigation Structure

Nav hierarchy: **Parts → Chapters → Sections → Pages**. Nav entries point to `.md` files.

A section with many pages may add **one grouping level** between Section and
Page. Only `1.6 자동 미분` does (27 pages in 7 groups); every other section is
flat. Add a level only when the flat list is long enough to be unreadable —
the median section has 3 pages.

> **Nothing but `.md` goes in nav.** The `figures/make_figures.py` scripts are build tooling,
> not pages; mkdocs copies them into `site/` and that is fine, but they are never navigated.

```yaml
nav:
  - I Part Title:
    - 1 Chapter Title:
      - Chapter Overview: ch01/index.md
      - 1.1 Section Title:
        - Topic Title: ch01/section_title/topic_title.md
      - 1.6 Long Section:          # grouped — see above
        - Group Title:
          - Topic Title: ch01/section_title/topic_title.md
```

YAML quoting: quote any title containing `:`, `#`, `*`, `&`.

## 용어는 전문어로, 설명은 낮게

**이 책의 가장 중요한 글쓰기 규칙이다.** 두 가지를 따로 다룬다.

| | 기준 |
|---|---|
| **용어** | 언제나 **표준 전문어**. 순우리말로 바꾸어 짓지 않는다 |
| **설명** | 중·고등학생이나 학부생이 따라올 수 있는 수준 |

용어를 쉬운 말로 바꾸면 **쉬워지지 않고 못 알아보게 된다.** `time embedding` 을
「때 묻힘」으로, `reconstruction` 을 「되살리기」로, `embedding` 을 「묻힘」으로 적으면
독자는 그 말이 무엇을 가리키는지 알 수 없고, **이 책을 읽고 나서 다른 책·논문·
라이브러리 문서를 못 읽게 된다.** 용어를 배우는 것도 공부의 일부다.

대신 **처음 나오는 자리에서 한 문장으로 풀어 준다.**

```markdown
두 벡터가 얼마나 닮았는지는 보통 **내적**으로 잰다 — 같은 자리끼리 곱해서
모두 더하는 것이다. 그런데 원핫은 1이 선 자리가 단어마다 달라서 …
```

용어(`내적`)는 그대로 두고 그 자리에서 뜻만 붙였다. 「속곱」 따위로 바꾸지 않는다.

- **남긴다**: 둘레 · 몫 · 자리 · 갈래 — 못 알아볼 위험이 없고 글이 좋아지는 말
- **수식 기호와 라이브러리 이름은 늘 원래대로**: `nn.Embedding`, $D_{\mathrm{KL}}$,
  `log_softmax`

### 이미 쓸어 낸 조어

2026-11-07 에 전수 조사로 바꾼 것들이다. **되살리지 말 것.**

| 조어 | 표준어 | 조어 | 표준어 |
|---|---|---|---|
| 넷트웍 | 네트워크 | 밑금 | 베이스라인 |
| 변환기 | 트랜스포머 | 알아내기 | 탐지 |
| 묻힘 | 임베딩 | 첫자리매김 | 초기화 |
| 그림점·낱그림점 | 픽셀 | 거르개 | 필터 |
| 잣수 | 스케일 | 이어 배우기 | 연속 학습 |
| 자리바꿈 | 순열 | 덧댐 | 패딩 |
| 앞가지 | 프리픽스 | 물음 집합 | 쿼리 집합 |
| 가림막 | 마스크 | 잡음 지우기 | 디노이징 |
| 마음결 | 감성 | 세쌍 | 트리플렛 |
| 받침 집합 | 서포트 집합 | 풀어 헤치기 | 분리 |
| 닻 | 앵커 | 본디 꼴 | 원래 형태 |
| 겹선형 | 쌍선형 | 깊은 러닝 | 딥러닝 |
| 꼴 (shape 뜻만) | 형태 | 낱말 | 단어 |
| 아래낱말 | 서브워드 | | |
| 되살리기·되살림 | 복원 (ch16 은 리플레이) | 심판 | 평가 분류기 |

> **이 쓸이에서 세 번 어겼다** — `넷트웍`→`네트워크`, `가림막`→`마스크`,
> `표`→`행렬`. 앞의 둘은 조사 표를 만들어 놓고도 마크다운 강조(`**를`)에 가려
> 놓쳤고, 셋째는 아예 스캔을 안 돌렸다. **받침이 바뀌는 치환은 사후 스캔이
> 선택이 아니다.**
>
> ```python
> # 새 말 + 조사 + (한글 또는 줄끝). 받침 있으면 이/을/은/과, 없으면 가/를/는/와
> re.finditer(r"행렬(를|가|는|와)([^가-힣]|$)", text)   # ㄹ받침인데 모음형이 붙은 것
> ```

**받침이 바뀌면 조사도 바꾼다.** `넷트웍`(ㄱ) → `네트워크`(없음)처럼 받침 유무가
달라지면 이/가 · 을/를 · 은/는 · 과/와 · 으로/로가 전부 따라 움직인다. ㄹ 받침은
`으로`가 아니라 `로`를 받는다. 일괄 치환한 뒤 **새 말 + 조사 + 한글 한 자**를 다시
훑어 비문을 찾는다 — 이 쓸이에서 그렇게 여섯 곳이 나왔다(`네트워크가며`,
`마스크가고` 따위).

### 아직 남은 것 — 뜻이 여럿이라 일괄 치환하면 안 되는 말

| 말 | 왜 못 바꾸나 |
|---|---|
| `고름` | 고르다의 명사형 · 균일함 · 공정성 · 균등 · 선택 · 평균 — 일곱 뜻 |
| `잦기` | 위치 부호화에서는 주파수, 보정에서는 빈도 |
| `알갱이` | SMC 의 입자, 그 밖에서는 알갱이 크기 |
| `눌러 담기` | JPEG 압축과 tanh 포화가 같은 말을 쓴다 |
| `자리표` | URL 이기도 하고 bbox 좌표이기도 하며 `자리표시자`의 부분 문자열이다 |
| `쏠림` | 치우침 · 뾰족함 · 불균형 |

`낱말`은 **882곳을 모두** `단어`로 바꿨다. 한때 지키는 말에 들어 있었으나
**한국에서 더 흔히 쓰이는 쪽이 `단어`**라 바꾸기로 했다. ㄹ받침이 모음으로 바뀌어
조사가 전부 뒤집히고(이→가, 을→를, 은→는, 과→와), `로`만 그대로다. 서술격은
`꼴`→`형태`와 같은 관례를 따른다 — `낱말이다`→`단어다`, `낱말이라`→`단어라`,
`낱말이나`→`단어나`로 줄이고 `이며`·`이고`·`이면`·`이지`는 `이`를 남긴다.
`이므로`를 빠뜨려 `단어가므로`가 한 곳 나왔으니, 서술격 목록을 **넉넉히** 적어야
한다.

**복합어는 저절로 풀린다** — `낱말집`→`단어집`, `낱말판`→`단어판`이 다 말이 된다.
다만 `아래낱말`은 **subword** 라 따로 `서브워드`로 바꿨다(책이 이미 27곳 쓰던 말).

**6장의 두 대상에는 `단어`를 앞에 붙인다.**

| 대상 | 이름 |
|---|---|
| 단어 × 차원, 배우는 수 | **단어 임베딩 표** |
| 단어 × 단어, 같이 나온 횟수 | **단어 동시 출현 행렬** |

앞엣것이 `행렬`이 아니라 `표`인 것은 일부러다 — `6.1` 의 주장이 「조회는 원핫에
**행렬**을 곱한 것이다」라서, **표(조회) ↔ 행렬 곱**이라는 대비가 걸려 있다.
영어도 lookup table 과 matmul 로 가른다.

`꼴`은 **홀로 쓰인 1,601곳을 모두** `형태`로 바꿨다 — shape 뜻이든
(`꼴 (batch, dim)`) form 뜻이든(`닫힌 꼴`, `수학 꼴`, `여러 꼴을 띤다`) 가리지
않는다. 한국어 `형태`가 둘 다 덮는다.

**복합어는 건드리지 않는다.** `꼴찌`·`사다리꼴`·`세모꼴`·`마름모꼴`·`글꼴`·
`사인꼴`·`지수꼴`은 그대로다. 앞에 한글이 오면 빼고, 뒤가 `찌`면 뺀다.

**서술형은 책의 관례를 따른다.** `꼴`은 ㄹ받침이고 `형태`는 받침이 없으므로
`꼴이다`→`형태다`, `꼴이라`→`형태라`, `꼴이나`→`형태나`로 줄이되
`꼴이며`→`형태이며`, `꼴이고`→`형태이고`, `꼴이면`→`형태이면`은 `이`를 남긴다
(책이 이미 그렇게 쓴다: 형태다 26 · 형태라 8 · 형태이며 5). 주격 `꼴이`는
`형태가`다 — 서술격 `이`와 갈라야 하므로 **뒤 두 글자를 보고 판단한다.**

이것들은 쪽을 고칠 때 **그 자리를 읽고** 바꾼다. `*.v[0-9]*.md` 스냅샷과
`*_score.md` 는 쓸이에서 뺀다 — 앞엣것은 빌드도 커밋도 안 되고, 뒤엣것은 그때의
검토 기록이다.

## Content Conventions (Summary)

Full MathJax/admonition rules are in `agents/SKILL.md`. Key points:

- `$...$` inline math, `$$...$$` display — always blank lines above and below display math
- No blank lines *inside* `$$...$$` blocks
- **Blank line between consecutive `$$` blocks too** — without it Markdown merges them and
  the delimiters leak through as literal `$$`. For a chain of equalities prefer one
  `$$\begin{aligned} … \end{aligned}$$` block over several `$$` blocks in a row.
- MathJax loads only `ams` and `boldsymbol` (see `docs/javascripts/mathjax.js`). Environments
  from other packages — `psmallmatrix`, anything from `mathtools` — silently fail to render.
- No LaTeX in `#` headings (breaks TOC)
- `\$` for currency, never bare `$`
- QED: `$\square$`
- Every content page ends with `## Exercises` (interleaved solutions, collapsible)
- Python: module docstring · `# ===` dividers · `if __name__ == "__main__":` guard
- Korean headings get positional anchors (`#_6`), not slugs, so intra-page `](#…)` links
  are fragile. Name the section in the link text instead of linking to it.

## Code and Output

A content page carries the **whole** program and the **whole** output it produced, so
that reading the page is enough and nobody has to rerun anything.

- **Never truncate output.** A block ending in `... (N lines omitted)` sends the reader
  straight back to the interpreter, which is the one thing this rule exists to prevent.
  If the output runs long, put it in a collapsible block rather than cutting it:

  ```text
  ??? note "전체 출력 (186줄)"

      ```
      ...the complete output, indented four spaces...
      ```
  ```

- **Seed anything random that reaches the page.** `torch.randn`, `nn.Linear` init,
  `np.random`, shuffling dataloaders — if a printed number depends on it, the page needs
  `torch.manual_seed(...)` / `np.random.seed(...)` before it. An unseeded number is not
  reproducible for any reader, and it cannot be checked later either.
- **One run is not a measurement.** When a page compares two models, vary the seed and
  report the spread; a gap smaller than the spread is not a gap. `docs/ch02/models/trees.md`
  shows the short form of this, and the IMDB chapter builds on it.
- **Seeding is not always enough.** Iterating a `set` prints in hash order, and Python
  randomizes string hashing per process — such a page varies run to run no matter how it is
  seeded. Sort the set. `ch21/detection/basic_object_detection.md` had this.

### Code stays open; output may fold

Code is the explanation, not an appendix to it, so it is **not** collapsed on a normal page.
Find-in-page does not reach inside a closed `<details>` on Safari or Firefox, so folding
code hides exactly the identifiers a reader hunts for. (Site search is unaffected — mkdocs
indexes the source either way.)

The exception is a page that is a **reference listing rather than a lesson**: 400+ lines of
code with under 25% prose. There is no explanation there for the code to interrupt, so it
folds into `??? note "코드 (N줄)"`. About 118 pages qualify; `experiments/fold_code.py --list`
names them, and folds every top-level block on a page when run against one.

Measure prose as `total − code − output`. Counting output as prose makes a page with a long
restored output look discursive and slip out of the rule.

### Checking that the page still tells the truth

`mkdocs build --strict` validates links and syntax. It cannot tell whether the numbers
printed on a page still come out of the code above them, so editing code without rerunning
it drifts silently.

```bash
python experiments/verify_outputs.py ch02/models/trees.md   # runs it, diffs the numbers
python experiments/restore_outputs.py <page>                # reruns and writes output back
```

**Both default to a 1800-second timeout.** A page that trains anything — `ch04/02_depth.md`
is 25 trainings, about 45 minutes — will blow through it, so pass `--timeout <seconds>` for
those. A timeout used to report as a bare `실행실패` with no reason under it, because the
killed subprocess leaves no stderr; it now reports `시간초과` and names the flag. If you see
`실행실패` with nothing under it on a heavy page, suspect the timeout before the page.

Paths are relative to `docs/`. Both run the page's code in a throwaway directory with
`data/` linked in — examples write `.png` and `.pth` next to themselves, and running them
from the repo root litters it.

**절 번호 참조는 링크 없는 것까지 훑는다.** 링크가 달린 `[6.1절의 …](02_context.md)`
은 목표 파일과 대조할 수 있지만, 산문·독스트링·표 머리의 **맨 번호**는 아무것도
붙들어 주지 않는다. 절을 다시 나누면 그쪽만 남는다 — `embeddings/04_glove.md` 가
6.2=스킵그램 · 6.3=세기 · 6.4=GloVe 라는 **지금 없는 체계**로 열 자리를 적고
있었고(「경사 하강법으로 맞춘다(6.2처럼)」이 자기 자신을 가리켰다), 링크 기반
검사는 0건을 냈다.

```python
# 링크가 안 달린 N.N 참조만 뽑아 눈으로 읽는다
for m in re.finditer(r"[^\n]{0,24}6\.\d[^\n]{0,30}", text):
    if "](" not in m.group(0): print(m.group(0))
```

소수점이 든 수(6.93 ms, 6.5549 nat)가 같이 걸리므로 걸러 읽는다. 그리고 절을
합쳤다면 **번호만으로는 못 가른다** — 옛 6.2 와 6.3 이 둘 다 6.1 이 되면
「6.1 의 배우는 길」·「6.1 의 세는 길」처럼 **이름으로** 적어야 한다.

**검증기는 산문을 안 본다.** `verify_outputs` 가 대조하는 것은 **출력 블록의 줄이
stdout 에 나오는가** 하나뿐이고, **산문·표·절 제목이 그 출력과 맞는지는 보지 않는다.**
게다가 시간이 든 줄은 「기계 탓」으로 봐 준다. 그래서 쪽이 **11/11 로 통과하면서**
들머리·표·제목 다섯 곳이 출력에 없는 수를 말할 수 있다 —
`embeddings/03_counting.md` 가 「0.5초 · 14.3초 · 스물여덟 배」라고 적고 출력은
「0.7초 · 14.6초」(21배)였다. 쪽을 고칠 때 **산문의 수를 출력 블록과 손으로 대조**한다.

**「산문에만 있는 수」로 찾으면 안 된다.** 전수로 돌려 보니 263쪽 2,394곳이
나왔고 거의 다 정상이었다 — 쪽이 요약한 표, 파생값(%포인트·차이·비), 다른 쪽에서
인용한 수가 전부 걸린다. 읽을 수가 없는 양이다.

**쓸 수 있는 신호는 「같은 이름표에 다른 수」다.** 「말 + 수 + 단위」 짝을 산문과
블록에서 각각 뽑아, 같은 이름표인데 값이 다르고 **그 값이 서로 가까운**(0.3~3배)
자리만 본다. 같은 조사를 1,238쪽에 돌리면 2곳으로 줄고 그중 하나가 진짜였다.

```python
UNIT = r"(초|배|%|퍼센트|밀리초|ms|MB|GB)"
PAIR = re.compile(r"([가-힣A-Za-z_][가-힣A-Za-z_0-9 ]{1,14}?)\s*[:은는이가]?\s*\**"
                  r"(\d+(?:\.\d+)?)\**\s*" + UNIT)
# PAIR 로 산문과 블록에서 {(이름표, 단위): {값}} 을 만들고,
# 같은 열쇠에서 산문에만 있는 값 중 블록 값과 0.3~3배 안인 것을 고른다
```

남는 거짓 양성은 두 갈래다. **반올림**(「파라미터 1 MB」 ↔ 출력 1.05 MB)과
**같은 이름표를 쓰는 다른 양**(ch05 의 컨볼루션 VAE 쪽이 코드에서 「견줄 자리」로
앞 쪽 모델의 정체 71.28% 를 찍고, 출력에는 이 쪽 모델의 65.91% 가 있다). 둘 다
읽으면 바로 갈린다.

**한 번 재어 소수점까지 적지 않는다.** `0.7초` 는 반올림 폭이 커서 참값의 비가
19.4~22.5배 어디든 된다. 그런 자리에 「스물여덟 배」를 박아 두면 출력이 한 번만
바뀌어도 쪽이 거짓말을 시작한다. **「스무 배 남짓」처럼 폭을 담은 말**로 적는다.

**검증기는 쪽의 `python` 블록을 전부 이어 붙여 한 번 돌린다** — 마지막 출력 블록 앞까지다.
한 쪽이 여러 블록으로 이어진 하나의 프로그램인 경우가 많기 때문이다. 그런데 **이어 붙인
것이 터지면 말없이 「가장 긴 블록 하나」로 물러난다.** 그러면 나머지 블록의 출력은 아무도
안 재는데 화면에는 그대로 `일치`라고 뜬다.

그래서 **설명하려고 끼워 넣는 조각 블록이 쪽 전체의 검증을 끌 수 있다.** 본문에 네 줄짜리
클래스를 ` ```python ` 으로 넣었더니 그 조각이 `nn` 을 임포트하지 않아 이어 붙인 것이
`NameError` 로 터졌고, 쪽이 통째로 폴백으로 내려갔다. 고치는 길은 둘이다.

- **조각이 이어 붙여도 돌아가게 적는다** — 본 블록의 임포트를 조각이 쓰는 이름에 맞춘다
  (`import torch` 만 있는 블록에 `nn.Module` 조각을 붙이지 말고 본 블록을
  `import torch.nn as nn` 로 바꾼다). 그러면 조각이 **베껴 적은 것이 아니라 그 프로그램의
  그 줄**이 되기도 한다.
- 또는 조각을 ` ```python ` 이 아닌 보통 블록으로 둔다.

조각을 넣은 뒤에는 **`python` 블록을 손으로 이어 붙여 돌려 보고** 검증 점수가 줄지 않았는지
확인한다. 점수가 갑자기 작아졌으면 폴백으로 내려간 것이다.

> `??? success "풀이"` 안의 코드는 네 칸 들여써 있어 이어 붙이면 `IndentationError` 가
> 난다. 다만 그런 블록은 대개 **마지막 출력 블록보다 뒤**에 있어 이어 붙이는 대상이
> 아니다. 연습문제 풀이를 출력 블록 앞으로 옮기면 그 쪽의 검증이 폴백으로 내려간다.

## Figures

Figures live in `docs/<chapter>/<section>/figures/` and are embedded with plain
`![alt](figures/name.svg)`.

- **SVG, not PNG** — text-based, diffs and compresses, stays sharp at any zoom.
- `svg.fonttype='path'` so glyphs become outlines. With the default `'none'` the file
  depends on fonts the browser may not have, and since matplotlib positions each glyph
  absolutely, a fallback font shifts the labels.
- `transparent=True` — the theme configures no palette, so Material renders light-only and
  a transparent background sits cleanly on the page. Adding a dark-mode toggle later would
  require dark variants of every figure.
- **All figure text must be ASCII.** matplotlib's default font has no Hangul, so Korean
  labels render as tofu boxes. `fonttype='path'` hides this from a text search — the string
  survives only in an XML comment while the glyphs are already baked to outlines — so check
  the rendered image, or grep the SVG for `[가-힣]`.
- Alt text is plain prose. `$...$` inside alt gets converted to `\(...\)` by arithmatex and
  is then read aloud as markup.
- Generate figures with the page's own code where possible, and make sure the numbers in
  the figure match the numbers in the prose.

## Block Scheme (정의 · 정리 · 증명 · 보기 · 문제 · 연습문제 · 풀이)

Ported from `~/Desktop/book/high_school_math`. The required extensions
(`admonition`, `pymdownx.details`, `attr_list`, `md_in_html`) are already enabled;
the box styles live in `docs/stylesheets/extra.css`.

**모든 블록은 상자에 담는다** — 저마다 클래스를 가진 `<div>` 로 감싼다.

| 블록 | 클래스 | 색 |
|---|---|---|
| 정의 | `<div class="defn" markdown>` | 회색 |
| 정리 (제목 + 주장) | `<div class="thmbox" markdown>` | 보라 — 증명과 같은 색 |
| 보기 | `<div class="exbox" markdown>` | 초록 — 풀이와 같은 색 |
| 문제 | `<div class="probox" markdown>` | 초록 |
| 연습문제 | `<div class="drillbox" markdown>` | 초록 |
| 그림으로 보는 AI | `<div class="storybox" markdown>` | 호박 — 점선 테두리 |
| 들어가는 이야기 | `<div class="leadbox" markdown>` | 자두 — 왼쪽 굵은 띠 |

색이 곧 갈래다 — **보라는 책이 증명하는 것, 초록은 독자가 손을 대는 것**이다. 호박빛 `storybox`
는 그 어느 쪽도 아닌 **세 번째 갈래**다.

**「그림으로 보는 AI」** — 이 쪽의 생각이 처음 나온 자리를 그림으로 보이는 상자다.
`~/Documents/book/high_school_math` 의 「그림으로 보는 수학」에서 꼴을 가져왔다.

- **자리는 그 생각을 설명한 블록 바로 뒤**다. 절 전체의 이야기일 때만 절 끝에 둔다.
- **그림은 사진을 빌려 오지 않고 직접 그린다.** 장치나 실험 배치는 자유 이미지를
  기대하기 어렵고, 형식만 빌려 이 책의 그림체로 다시 그리면 라이선스 문제가 없다
  (그 책의 「지폐·묘비·기념물은 직접 그린다」와 같은 규칙이다). 그리는 코드는 그 쪽의
  `figures/make_figures.py` 에 둔다.
- **지어낸 수를 그림에 넣지 않는다.** 꼴만 옮긴 것이면 그림 안에 그렇게 적는다
  (`schematic - not recorded counts`).
- **마지막은 한계로 닫는다.** 놀라움으로 닫으면 광고가 된다 — 그 관찰이 **무엇까지**
  말해 주고 무엇은 말해 주지 않는지를 적는다.
- 끝에 `_확인한 곳 — …_` 로 출처를 적는다.
- **쉽게 쓰되 용어는 바꾸지 않는다** — 아래 「용어는 전문어로, 설명은 낮게」를 따른다.
  상자는 그 쪽을 아직 읽지 않은 사람이 들어오는 자리이므로 **설명을 낮추고**, 비유를
  하나 두면 좋다. 그러나 `경사 하강법`을 「조금씩 내려가기」로 **바꾸지는 않는다.**
  처음 나오는 자리에서 한 문장으로 풀어 줄 뿐이다.
- **번호는 쪽 안에서 1부터 매긴다.** 책 전체를 꿰는 번호가 아니다 — `**그림으로 보는 AI 1.**`
  은 연재의 이름이지 그 편의 대상이 아니므로, 무엇을 다루는지는 **제목 줄에 적어** 알린다.
  한 쪽에 둘을 두면 그 쪽에서 `1.`, `2.` 가 된다. `statistics_in_korean` 이 같은 방식이다
  (그 책은 517쪽 가운데 490쪽에 상자가 있고, 338쪽이 `1.` 을 단 쪽이다).

**목표는 쪽마다 하나다.** 다만 **정직하게 맞는 쪽에만 단다** — 확인한 자리가 없으면
그 쪽은 비워 둔다. 지어낸 유래 하나가 확인한 상자 열의 값어치를 깎는다.

### 「들어가는 이야기」 — 연재가 아니라 역할이다

`<div class="leadbox" markdown>` · 자두 · 왼쪽 굵은 띠.
`~/Documents/book/statistics_in_korean` 에서 꼴을 가져왔다.

**storybox 와 갈래가 다르다.** storybox 는 연재라 쪽 안에서 1부터 번호를 매기고
그 쪽의 **어느 블록에 딸린다.** 이것은 **역할**이다 — 쪽마다 하나뿐이고 늘 맨 앞이라
번호가 언제나 1이 되어 아무것도 가리지 못하므로 **번호를 매기지 않는다.**

```markdown
<div class="leadbox" markdown>

**들어가는 이야기** — ⟨그 이야기의 제목⟩

⟨들머리 본문⟩

_확인한 곳 — ⟨확인한 것⟩_

</div>
```

- **자리는 쪽의 맨 앞**, `#` 제목 바로 아래이며 기존 들머리 문단보다 **위**다.
  특정 정의·정리에 딸리지 않고 **쪽 전체가 푸는 물음**을 연다.
- **여는 방식은 둘이다.** (가) 바깥 세상의 어느 자리에서 시작하는 이야기,
  또는 (나) **이 쪽이 뒤에서 스스로 뒤집을 물음** 하나. 뒤엣것을 함께 쓰는 것이
  이 책이 `statistics_in_korean` 과 갈리는 점이다 — 이 책의 쪽은 1,238쪽 가운데
  90.7%가 개념 진술로 바로 들어가고 바깥 일화를 가진 쪽이 드문 반면, **재어서
  뒤집을 거리는 거의 모든 쪽에 있다.** 지어낸 일화보다 잰 물음이 낫다.
- **출처 줄은 바깥 사실을 들었을 때만 단다.** 상자와 출처 줄은 서로 독립이다 —
  이 쪽의 수만으로 여는 들머리에는 출처 줄이 없는 것이 맞다.
- **storybox 와 같은 이야기를 쓰지 않는다.** 한 쪽에 둘이 다 있으면 독자는
  서로 다른 두 가지를 얻어야 한다. 이미 storybox 가 쓴 논문·일화는 들머리에서
  다시 꺼내지 말고, 들머리는 **그 쪽이 뒤에 내놓을 수**로 열어 둔다.
- **들머리는 본문의 전문 용어를 쓴다.** 중학생 규칙이 여기에도 걸리지만,
  **낮추는 것은 설명이지 용어가 아니다.** 그 쪽을 아직 읽지 않은 사람이 처음
  만나는 글이므로 **설명을 낮추되**, 용어는 storybox 와 똑같이 **그대로 쓰고
  그 자리에서 한 문장으로 풀어 준다.** 용어를 돌려 말하면 독자는 본문에서
  그것을 다시 만나도 **같은 것인 줄 모른다** — 이 책이 가장 먼저 금지한 일이다.

  ```markdown
  ✗ 「이것은 학습에서 빼 달라」고 말하는 길을 넷이나 준다. … 따라가지 않는다는
     표시가 꺼져 있다. … 하나는 되돌릴 수 없는 표를 찍는다.
  ✓ 경사 추적을 끄는 길을 넷이나 준다 — 그 텐서를 역전파가 더는 따라가지 않게
     하는 것이다. `requires_grad_(False)` · `no_grad()` · `detach()` ·
     `inference_mode()`. … `inference_mode` 는 추론 텐서라는 표를 박는다.
  ```

  왼쪽은 넷 가운데 **어느 것도 이름을 안 불러** 독자가 본문에서 그것을 다시
  만나도 같은 것인 줄 모른다. 오른쪽은 이름을 부르고 뜻을 바로 붙였다.
- **마지막은 한계나 물음으로 닫는다.** 답을 다 주면 쪽을 읽을 까닭이 없어진다.

**증명과 풀이는 상자 밖**에 둔다 (`??? proof "증명"`, `??? success "풀이"`) — 이미
테두리를 가진 블록이라 안에 넣으면 상자가 겹친다. 반대로 **문항을 설명하는 그림은
상자 안**에 넣는다. 여는 태그와 본문 사이, 본문과 `</div>` 사이에는 빈 줄을 둔다
(`md_in_html` 이 안쪽을 마크다운으로 읽게 하는 조건이다).

정리는 오른쪽 목차에 뜨도록 `### 정리 N. 제목 { .thm }` 으로 적는다.

```text
<div class="thmbox" markdown>          <div class="drillbox" markdown>

### 정리 1. ... { .thm }               **연습문제 1.** <span class="diff easy" …></span> …

주장 ...                                </div>

</div>                                  ??? success "풀이"   ← 풀이는 상자 밖

??? proof "증명"   ← 증명은 상자 밖
```

**난이도 점** — 보기·문제·연습문제의 굵은 도입어 바로 뒤에 색 점 하나를 붙인다.
글자는 넣지 않고 **초록(쉬움) → 노랑(중간) → 빨강(어려움)** 색만으로 나타내며,
마우스를 올리면 뜨도록 `title` 을 적는다. 쪽마다 범례를 달지 않는다.

```html
**보기 2.** <span class="diff easy" title="쉬움"></span> ...
**연습문제 4.** <span class="diff hard" title="어려움"></span> ...
```

보기는 언제나 쉬움, 문제는 중간·어려움으로 둔다. 연습문제는 **새로 쓰는 쪽에서** 쉬움 → 중간 → 어려움 순서로 둔다.

> 이미 있는 쪽에는 이 순서가 강제되지 않는다. 난이도 점을 쓰는 1,144쪽 가운데
> 521쪽(46%)이 이 순서를 따르지 않으며, 대개 연습문제가 **난이도가 아니라 주제**로
> 묶여 있기 때문이다(`ch03/mnist/03_mlp.md`가 그 보기다 — 이론 문제 넷이 어려움으로
> 붙어 있고 그 뒤에 코드 문제 넷이 중간으로 이어진다). 되돌려 번호를 매기려면
> 본문의 상호 참조까지 함께 고쳐야 하므로(그 쪽만 12군데), 순서를 맞추자고
> 기존 쪽을 다시 매기지 말 것. 주제 묶음이 난이도 순서보다 읽기에 낫다면 그대로 둔다.

## Common Tasks

### Add a new section
1. Create `docs/chapter_name/section_name/` with `.md` files
2. Add nav entries to `mkdocs.yml` for them

### Add a new chapter
1. Create `docs/chapter_name/` with `index.md` and section subdirectories
2. Add chapter block to `mkdocs.yml` under the correct Part

### Add a Python example

**The code goes in the `.md`. Do not create a `.py` beside the page.**

1. Put the **whole** program in one `python` block and the **whole** output it produced in
   the block below it, so the page can be read without running anything.
2. Educational style still applies: module docstring, `# ===` dividers,
   `if __name__ == "__main__":` guard. The listing should be a real program, not a fragment.
3. Paths inside it are relative to the repo root (`./data/...`), which is where it is meant
   to be run from. Say so in the docstring when it matters. Do **not** write
   `실행: python topic.py` — there is no such file; write
   `실행: 이 쪽의 코드를 파일로 갈무리해 돌린다`.

Why: a sibling `.py` and its page drift apart, and the page is the artifact. It has already
happened here — `b12724e1` retermed the markdown and left the extracted copies printing the
old words. `verify_outputs.py` only reads the `.md`, so the drift is invisible to CI.

> **No page has a `.py` beside it any more.** The 43 that other pages used to `import` as
> shared modules were inlined into their importers and removed. If you find yourself wanting
> a module two pages share, put the code in both — duplication across pages is the price of
> each page standing alone, and it is the price this book pays.

Separate from all of this, a few directories hold genuine importable packages with no page of
their own (`ch26/training/models/vae.py`, `ch12/pretrained_models/utils/positional_encoding.py`,
`*/\_\_init\_\_.py`). Those are libraries, not page companions, and are out of scope here.
