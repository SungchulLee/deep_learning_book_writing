# Snorkel을 쓰는 약지도 학습

약지도 학습은 비싼 수작업 주석을 어림 규칙과 분야 지식을 담은 프로그램적 레이블링 함수로 대신한다. Snorkel 같은 틀은 불완전하고 서로 겹치는 레이블링 함수를 여럿 모아, 손으로 붙인 레이블 하나 없이 확률적 학습 레이블을 만들어 낸다. 레이블 데이터가 드물거나 얻는 비용이 클 때 특히 값어치가 있다.

## 코드

```python
"""Snorkel을 쓰는 약지도 학습."""
# ---
# title: "Snorkel을 쓰는 약지도 학습"
# description: "레이블링 함수, 레이블 집계, 잡음을 고려한 학습으로 하는
#               프로그램적 데이터 레이블링 — 수작업 주석이 필요 없다"
# ---
#
# 수작업 주석은 비싸다. 약지도 학습은 이를 어림 규칙, 패턴, 분야 지식을
# 담은 프로그램적 레이블링 함수(LF)로 대신한다. Snorkel은 이 잡음 섞인
# 레이블을 모아 확률적 학습 레이블로 만든다.
#
#
#   1부 – 레이블링 함수라는 추상
#   2부 – 레이블링 함수의 종류 (핵심어, 정규식, 자연어 처리, 모델 기반)
#   3부 – 레이블 집계: 다수결과 레이블 모델
#   4부 – 레이블링 함수 분석: 적용률, 겹침, 충돌
#   5부 – 처음부터 끝까지의 파이프라인: 레이블링 함수 → 레이블 모델 → 분류기
#   6부 – 금융에서의 약지도 학습 사례
#
# 출처: O'Reilly "Practical NLP" 2장에서 고쳐 씀

import numpy as np
import re
from typing import List, Callable, Dict, Optional
from collections import Counter


# =====================================================================
# 1부 – 레이블링 함수라는 추상
# =====================================================================
print("=" * 60)
print("Part 1: Labeling Functions — The Core Abstraction")
print("=" * 60)

# 레이블링 함수(LF)는 다음과 같은 파이썬 함수이다:
#   - 데이터 점 하나를 입력으로 받는다
#   - 레이블 또는 ABSTAIN(-1)을 돌려준다
#
# 핵심: 각 레이블링 함수가 완벽하거나 모든 데이터를 다룰 필요는 없다.
# 불완전한 레이블링 함수를 여럿 모아 품질 좋은 레이블을 만든다.
#
# ABSTAIN = -1  (레이블링 함수가 모른다는 뜻)
# NEGATIVE = 0
# POSITIVE = 1

ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1

# 예: 유튜브 스팸 탐지
# 댓글 2000개에 손으로 레이블을 붙이는 대신 어림 규칙 10개를 쓴다.

print("""
  # Snorkel API:
  from snorkel.labeling import labeling_function

  @labeling_function()
  def lf_contains_link(x):
      return SPAM if "http" in x.text.lower() else ABSTAIN

  @labeling_function()
  def lf_subscribe(x):
      return SPAM if "subscribe" in x.text.lower() else ABSTAIN

  @labeling_function()
  def lf_short_comment(x):
      return HAM if len(x.text.split()) < 5 else ABSTAIN
""")


# 순수 파이썬 구현 (Snorkel에 기대지 않는다)
class DataPoint:
    """판다스의 행을 흉내 낸 간단한 데이터 담개."""
    def __init__(self, text: str, **kwargs):
        self.text = text
        for k, v in kwargs.items():
            setattr(self, k, v)


# 레이블링 함수를 보통의 파이썬 함수로 정의한다
def lf_contains_link(x: DataPoint) -> int:
    """스팸 댓글에는 링크가 들어 있는 일이 많다."""
    return POSITIVE if "http" in x.text.lower() else ABSTAIN


def lf_subscribe(x: DataPoint) -> int:
    """스팸 댓글은 구독을 권한다."""
    return POSITIVE if "subscribe" in x.text.lower() else ABSTAIN


def lf_check_out(x: DataPoint) -> int:
    """스팸 댓글은 '내 채널을 보라'고 한다."""
    return POSITIVE if re.search(r"check.*out", x.text, re.I) else ABSTAIN


def lf_my_channel(x: DataPoint) -> int:
    """스팸 댓글은 자기 채널을 홍보한다."""
    return POSITIVE if "my channel" in x.text.lower() else ABSTAIN


def lf_please(x: DataPoint) -> int:
    """스팸 댓글은 자주 부탁한다 ('please', 'plz')."""
    text_lower = x.text.lower()
    return POSITIVE if "please" in text_lower or "plz" in text_lower else ABSTAIN


def lf_short_comment(x: DataPoint) -> int:
    """진짜 댓글은 짧은 반응인 경우가 많다."""
    return NEGATIVE if len(x.text.split()) < 5 else ABSTAIN


def lf_song_mention(x: DataPoint) -> int:
    """노래에 대한 댓글은 진짜일 가능성이 크다."""
    return NEGATIVE if "song" in x.text.lower() else ABSTAIN


def lf_sentiment_positive(x: DataPoint) -> int:
    """아주 긍정적인 감성은 진짜 참여를 시사한다."""
    positive_words = {"love", "amazing", "awesome", "great", "best", "beautiful"}
    words = set(x.text.lower().split())
    if len(words & positive_words) >= 2:
        return NEGATIVE  # 진짜 팬의 댓글
    return ABSTAIN


# 시연
sample_comments = [
    DataPoint(text="Check out my channel for more videos http://youtube.com/spam"),
    DataPoint(text="This song is amazing! Love it!"),
    DataPoint(text="Please subscribe to my channel plz"),
    DataPoint(text="lol"),
    DataPoint(text="Great performance, one of the best live shows"),
    DataPoint(text="Subscribe for daily content http://bit.ly/xyz"),
]

lfs = [lf_contains_link, lf_subscribe, lf_check_out, lf_my_channel,
       lf_please, lf_short_comment, lf_song_mention, lf_sentiment_positive]

print("  Labeling function outputs (ABSTAIN=-1, NEG=0, POS=1):")
print(f"  {'Comment':<55} ", end="")
for lf in lfs:
    print(f"{lf.__name__[3:]:>10}", end="")
print()
print("  " + "-" * 135)

for dp in sample_comments:
    text_short = dp.text[:53] + ".." if len(dp.text) > 55 else dp.text
    print(f"  {text_short:<55} ", end="")
    for lf in lfs:
        label = lf(dp)
        symbol = "·" if label == ABSTAIN else str(label)
        print(f"{symbol:>10}", end="")
    print()
print()


# =====================================================================
# 2부 – 레이블링 함수의 종류
# =====================================================================
print("=" * 60)
print("Part 2: Types of Labeling Functions")
print("=" * 60)

# 1. 핵심어 기반: 특정 낱말이 있는지 본다
# 2. 패턴 기반: 정규식으로 맞춘다
# 3. 어림 규칙: 길이, 구조, 메타데이터
# 4. 자연어 처리 기반: 개체명 인식, 품사 태그, 감성
# 5. 외부 모델: 이미 있는 (약한) 분류기를 쓴다
# 6. 지식 베이스: 사전이나 온톨로지에서 찾는다

print("""
  1형: 키워드 레이블링 함수(가장 단순)
  ──────────────────────────────
    @labeling_function()
    def lf_keyword(x):
        return SPAM if "subscribe" in x.text.lower() else ABSTAIN

  2형: 패턴 레이블링 함수(정규식)
  ──────────────────────────────
    @labeling_function()
    def lf_regex(x):
        return SPAM if re.search(r"check.*out", x.text, re.I) else ABSTAIN

  3형: 어림 규칙 레이블링 함수(메타데이터 / 구조)
  ──────────────────────────────────────────────
    @labeling_function()
    def lf_short(x):
        return HAM if len(x.text.split()) < 5 else ABSTAIN

  4형: 자연어 처리 기반 레이블링 함수(전처리가 필요하다)
  ────────────────────────────────────────────────
    from snorkel.preprocess.nlp import SpacyPreprocessor
    spacy = SpacyPreprocessor(text_field="text", doc_field="doc")

    @labeling_function(pre=[spacy])
    def lf_has_person(x):
        if any(ent.label_ == "PERSON" for ent in x.doc.ents):
            return HAM
        return ABSTAIN

  5형: 외부 모델 레이블링 함수
  ──────────────────────────────
    from textblob import TextBlob

    @labeling_function()
    def lf_textblob_positive(x):
        polarity = TextBlob(x.text).sentiment.polarity
        return HAM if polarity > 0.9 else ABSTAIN

  6형: 프로그램으로 만드는 레이블링 함수 공장(규모 확장용)
  ─────────────────────────────────────────────────
    def make_keyword_lf(keywords, label):
        def lf(x):
            if any(w in x.text.lower() for w in keywords):
                return label
            return ABSTAIN
        lf.__name__ = f"keyword_{'_'.join(keywords[:2])}"
        return lf

    lf_money = make_keyword_lf(["earn", "money", "income"], SPAM)
    lf_music = make_keyword_lf(["song", "music", "lyrics"], HAM)
""")


# 레이블링 함수를 찍어 내는 공장 구현
def make_keyword_lf(keywords: List[str], label: int) -> Callable:
    """핵심어 목록으로 레이블링 함수를 만든다."""
    def lf(x):
        text_lower = x.text.lower()
        if any(kw in text_lower for kw in keywords):
            return label
        return ABSTAIN
    lf.__name__ = f"keyword_{'_'.join(keywords[:2])}"
    return lf


# =====================================================================
# 3부 – 레이블 집계
# =====================================================================
print("=" * 60)
print("Part 3: Label Aggregation — Majority Vote vs Label Model")
print("=" * 60)

# 레이블 행렬 L(n_samples × n_lfs)이 주어지면, 잡음 섞이고 겹치며
# 서로 어긋날 수도 있는 레이블을 표본마다 하나의 확률적 레이블로
# 모아야 한다.


def apply_lfs(data_points: List[DataPoint], lfs: List[Callable]) -> np.ndarray:
    """모든 데이터 점에 모든 레이블링 함수를 적용한다.

    반환값:
        모양이 (n_samples, n_lfs)인 레이블 행렬 L.
        L[i,j]은 표본 i에 대해 레이블링 함수 j가 준 레이블이다.
    """
    n = len(data_points)
    m = len(lfs)
    L = np.full((n, m), ABSTAIN, dtype=int)
    for i, dp in enumerate(data_points):
        for j, lf in enumerate(lfs):
            L[i, j] = lf(dp)
    return L


def majority_vote(L: np.ndarray, tie_break: int = ABSTAIN) -> np.ndarray:
    """다수결로 레이블을 모은다.

    표본마다 (ABSTAIN을 뺀) 표를 세어 가장 많은 레이블을 고른다.
    동점이면 tie_break을 쓴다.
    """
    n = L.shape[0]
    preds = np.full(n, tie_break, dtype=int)
    for i in range(n):
        votes = L[i][L[i] != ABSTAIN]
        if len(votes) == 0:
            continue
        counts = Counter(votes)
        most_common = counts.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            preds[i] = most_common[0][0]
    return preds


# Snorkel의 레이블 모델 (확률적)
print("""
  다수결: 단순하지만 레이블링 함수마다 정확도가 다르다는 점을 무시한다.

  레이블 모델(Snorkel): 레이블링 함수마다 정확도와 상관을 학습하여
  확률적 레이블을 만들어 낸다.

    from snorkel.labeling.model import LabelModel

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, seed=42)

    # 확률적 레이블 얻기
    probs = label_model.predict_proba(L=L_train)
    # → shape: (n_samples, n_classes) with values in [0, 1]

    # 딱딱한 레이블 얻기
    preds = label_model.predict(L=L_train)
""")

# 예시 데이터로 시연
L = apply_lfs(sample_comments, lfs)
preds = majority_vote(L, tie_break=NEGATIVE)

print("  Majority vote predictions:")
for dp, pred in zip(sample_comments, preds):
    label = "SPAM" if pred == POSITIVE else "HAM" if pred == NEGATIVE else "ABSTAIN"
    n_votes = (L[sample_comments.index(dp)] != ABSTAIN).sum()
    print(f"    {label:>7} ({n_votes} LFs voted): {dp.text[:60]}")
print()


# =====================================================================
# 4부 – 레이블링 함수 분석: 적용률, 겹침, 충돌
# =====================================================================
print("=" * 60)
print("Part 4: LF Analysis Metrics")
print("=" * 60)


def analyze_lfs(L: np.ndarray, lf_names: List[str]) -> None:
    """레이블링 함수의 품질 지표를 계산하여 보인다.

    적용률:  레이블링 함수가 기권하지 않은 표본의 비율
    겹침:   다른 레이블링 함수 하나 이상과 일치하는 표본의 비율
    충돌:  다른 레이블링 함수와 어긋나는 표본의 비율
    """
    n, m = L.shape

    print(f"  {'LF Name':<25} {'Coverage':>10} {'Overlaps':>10} {'Conflicts':>10}")
    print("  " + "-" * 55)

    for j in range(m):
        # 적용률
        labeled = L[:, j] != ABSTAIN
        coverage = labeled.mean()

        # 겹침: 이 레이블링 함수와 다른 하나 이상이 같은 표본에 레이블을 붙인 경우
        overlaps = 0
        conflicts = 0
        for i in range(n):
            if L[i, j] == ABSTAIN:
                continue
            other_labels = L[i, np.arange(m) != j]
            other_active = other_labels[other_labels != ABSTAIN]
            if len(other_active) > 0:
                overlaps += 1
                if any(other_active != L[i, j]):
                    conflicts += 1

        overlap_rate = overlaps / n
        conflict_rate = conflicts / n

        print(f"  {lf_names[j]:<25} {coverage:>10.1%} {overlap_rate:>10.1%} {conflict_rate:>10.1%}")

    # 전체 적용률
    any_label = (L != ABSTAIN).any(axis=1).mean()
    print(f"\n  Overall coverage: {any_label:.1%} of samples have ≥1 LF label")


lf_names = [lf.__name__ for lf in lfs]
analyze_lfs(L, lf_names)
print()

# Snorkel의 LFAnalysis
print("""
  Snorkel에 내장된 분석:
    from snorkel.labeling import LFAnalysis

    analysis = LFAnalysis(L=L_train, lfs=lfs)
    df = analysis.lf_summary()
    # 다음 열을 갖는 DataFrame을 돌려준다:
    #   j, Polarity, Coverage, Overlaps, Conflicts, Correct, Incorrect, Emp. Acc.
""")


# =====================================================================
# 5부 – 처음부터 끝까지의 파이프라인
# =====================================================================
print("=" * 60)
print("Part 5: End-to-End Weak Supervision Pipeline")
print("=" * 60)

# 전체 파이프라인:
#   1. 레이블링 함수를 쓴다 (어림 규칙, 패턴, 모델)
#   2. 레이블 없는 데이터에 레이블링 함수를 적용한다 → 레이블 행렬 L
#   3. 레이블링 함수를 분석한다 (적용률, 충돌, 개발 집합에서의 정확도)
#   4. 집계를 위해 레이블 모델을 학습시킨다 → 확률적 레이블
#   5. 확신이 낮은 표본을 걸러 낸다
#   6. 확률적 레이블로 최종 모델(예: BERT)을 학습시킨다

print("""
  Snorkel 전체 파이프라인:

  # 1단계: 레이블링 함수 정의
  lfs = [lf_contains_link, lf_subscribe, lf_check_out,
         lf_my_channel, lf_please, lf_short_comment,
         lf_song_mention, lf_textblob_positive, lf_has_person]

  # 2단계: 레이블 없는 데이터에 적용
  from snorkel.labeling import PandasLFApplier
  applier = PandasLFApplier(lfs=lfs)
  L_train = applier.apply(df=df_train)  # 모양: (n_train, n_lfs)
  L_test = applier.apply(df=df_test)

  # 3단계: 분석
  from snorkel.labeling import LFAnalysis
  LFAnalysis(L=L_train, lfs=lfs).lf_summary()

  # 4단계: 레이블 모델 학습
  from snorkel.labeling.model import LabelModel
  label_model = LabelModel(cardinality=2, verbose=True)
  label_model.fit(L_train=L_train, n_epochs=500, seed=42)

  # 레이블 모델 평가
  label_model_acc = label_model.score(
      L=L_test, Y=Y_test, tie_break_policy="random"
  )["accuracy"]
  print(f"Label Model Accuracy: {label_model_acc:.1%}")

  # 5단계: 확률적 레이블을 얻어 거르기
  from snorkel.labeling import filter_unlabeled_dataframe
  from snorkel.utils import probs_to_preds

  probs_train = label_model.predict_proba(L=L_train)
  df_filtered, probs_filtered = filter_unlabeled_dataframe(
      X=df_train, y=probs_train, L=L_train
  )
  preds_filtered = probs_to_preds(probs=probs_filtered)

  # 6단계: 최종 분류기 학습
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.linear_model import LogisticRegression

  vectorizer = CountVectorizer(ngram_range=(1, 5))
  X_train = vectorizer.fit_transform(df_filtered.text.tolist())
  X_test = vectorizer.transform(df_test.text.tolist())

  clf = LogisticRegression(C=1e3, solver="liblinear")
  clf.fit(X=X_train, y=preds_filtered)
  print(f"End Model Accuracy: {clf.score(X=X_test, y=Y_test):.1%}")

  # 원래 노트북의 결과 (유튜브 스팸):
  #   다수결:           89.6%
  #   레이블 모델:      92.4%
  #   최종 모델 (LR):   93.5%
  # 손으로 레이블을 붙인 학습 예가 하나도 없이!
""")


# 간략한 시연
print("  Simplified demo with synthetic data:")
np.random.seed(42)

# 합성 레이블 데이터 생성 (이 레이블을 모른다고 치자)
n_samples = 500
true_labels = np.random.randint(0, 2, n_samples)
texts = []
for label in true_labels:
    if label == POSITIVE:  # 스팸
        templates = [
            "Check out my channel http://link.com",
            "Subscribe to my page please",
            "Visit http://spam.com for great deals",
            "Check out my new video plz subscribe",
        ]
    else:  # 정상
        templates = [
            "Great song love it",
            "This is amazing",
            "lol",
            "Nice performance really enjoyed this",
        ]
    texts.append(np.random.choice(templates))

data = [DataPoint(text=t) for t in texts]

# 레이블링 함수 적용
L_demo = apply_lfs(data, lfs)

# 적용률 통계
coverage = (L_demo != ABSTAIN).any(axis=1).mean()
avg_lfs = (L_demo != ABSTAIN).sum(axis=1).mean()
print(f"  Coverage: {coverage:.1%} of samples have ≥1 label")
print(f"  Avg LFs per sample: {avg_lfs:.1f}")

# 다수결의 정확도
preds_mv = majority_vote(L_demo, tie_break=NEGATIVE)
# 예측이 있는 곳에서만 평가
labeled_mask = preds_mv != ABSTAIN
mv_acc = (preds_mv[labeled_mask] == true_labels[labeled_mask]).mean()
print(f"  Majority vote accuracy (on labeled): {mv_acc:.3f}")
print(f"  Labeled: {labeled_mask.sum()}/{n_samples} samples")
print()


# =====================================================================
# 6부 – 금융에서의 약지도 학습 사례
# =====================================================================
print("=" * 60)
print("Part 6: Financial Weak Supervision Use Cases")
print("=" * 60)

print("""
  약지도 학습은 다음과 같은 금융 분야에서 특히 값어치가 있다.
  - 레이블 데이터가 드물다(독점적이고 주석 비용이 크다)
  - 분야 전문가는 어림 규칙은 말할 수 있으나 1만 개를 일일이 레이블할 수는 없다
  - 패턴이 시간에 따라 바뀐다(빠른 재레이블링이 필요하다)

  예: 실적 감성 분류
  ───────────────────────────────────────────
  @labeling_function()
  def lf_beat_estimates(x):
      return BULLISH if re.search(r"beat.*estimates?", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_missed_estimates(x):
      return BEARISH if re.search(r"miss(ed)?.*estimates?", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_raised_guidance(x):
      return BULLISH if re.search(r"rais(ed|ing).*guidance", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_lowered_guidance(x):
      return BEARISH if re.search(r"lower(ed|ing).*guidance", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_revenue_growth(x):
      match = re.search(r"revenue.*(grew|increased|up)\s+(\d+)%", x.text, re.I)
      if match and int(match.group(2)) > 10:
          return BULLISH
      return ABSTAIN

  @labeling_function()
  def lf_analyst_upgrade(x):
      return BULLISH if re.search(r"upgrade[sd]?.*to.*buy", x.text, re.I) else ABSTAIN

  @labeling_function()
  def lf_finbert_positive(x):
      # 미리 학습된 FinBERT를 약한 레이블러로 쓰기
      score = finbert_pipeline(x.text)[0]
      if score["label"] == "positive" and score["score"] > 0.8:
          return BULLISH
      return ABSTAIN

  그 밖의 금융 응용:
  ─────────────────────────────
  1. SEC 공시 위험 분류
     레이블링 함수: 위험 범주별 키워드 목록, 규제 인용을 잡는 정규식,
     기관 언급을 잡는 개체명 인식

  2. 뉴스 사건 탐지
     레이블링 함수: 동사 패턴("acquired", "merged"), 개체 동시 출현,
     날짜 근접성, 출처 신뢰도

  3. 신용 위험 평가
     레이블링 함수: 재무 비율 임계값, 업종 기준치,
     경영진 교체 신호, 부채 약정 키워드

  4. ESG 분류
     레이블링 함수: 환경 키워드, 지배구조 패턴,
     규제 준수 언급, 지속가능성 지표

  수작업 레이블링에 견준 이점:
  ┌──────────────────────┬─────────────┬───────────────────┐
  │ 지표                 │ 수작업      │ 약지도            │
  ├──────────────────────┼─────────────┼───────────────────┤
  │ 레이블링 시간        │ 며칠~몇 주  │ 몇 시간           │
  │ 레이블 하나당 비용   │ \$0.50~\$5.00│ 거의 \$0(코드뿐)  │
  │ 확장성               │ 선형        │ 한 번 짜면 O(1)   │
  │ 적응성               │ 재레이블링  │ 함수 갱신         │
  │ 일관성               │ 들쭉날쭉    │ 결정적            │
  │ 분야 지식            │ 사라짐      │ 코드에 담김       │
  └──────────────────────┴─────────────┴───────────────────┘
""")

print("Done.")


if __name__ == "__main__":
    pass
```

## 논의

레이블링 함수라는 추상은 약지도 학습의 바탕이다. 각 함수는 링크, 핵심어, 패턴을 확인하는 어림 규칙 하나를 담고, 레이블을 돌려주거나 확신이 없으면 ABSTAIN을 돌려준다. 핵심은 어느 한 레이블링 함수도 정확하거나 빠짐없을 필요가 없다는 것이다. 이들의 집단 지혜를 제대로 모으면 품질 좋은 학습 레이블이 나온다.

레이블을 모으는 방법은 다수결처럼 단순할 수도 있고, 레이블링 함수마다의 정확도와 상관 구조를 배우는 Snorkel의 확률적 레이블 모델처럼 정교할 수도 있다. 레이블 모델은 부드러운 확률적 레이블을 만들어 잡음을 고려한 학습에 바로 쓸 수 있으며, 딱딱한 다수결 레이블보다 낫다.

레이블링 함수 분석 지표, 곧 적용률(레이블이 붙은 표본의 비율), 겹침(여러 함수가 일치하는 비율), 충돌(함수들이 어긋나는 비율)은 레이블링 함수를 되풀이해 다듬는 길잡이가 된다. 적용률이 낮으면 어림 규칙이 모자란다는 뜻이고, 충돌이 크면 서로 어긋나는 규칙을 손봐야 한다는 뜻이다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

