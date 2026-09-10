# 갈래별 마음결

갈래별 마음결.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 글 가르기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""갈래별 마음결."""
# ---
# title: "갈래별 마음결 살피기"
# description: "VADER로 것/갈래 수준의 결이 고운 마음결,
#               PyTorch 눈길 바탕 모델, 변환기 방식"
# ---
#
# 보통의 마음결 살피기는 글월마다 이름표 하나를 준다.
# 갈래별 마음결 살피기(ABSA)는 글 속의 특정한
# 갈래나 것에 대한 마음결을 뽑아낸다.
#
# 보기: "The phone has an amazing camera but terrible battery life."
#   → 사진기: 양성
#   → 배터리 수명: 음성
#
#   1부 – VADER로 하는 규칙 바탕 갈래별 마음결 살피기
#   2부 – 갈래 뽑기(이름씨 마디 덩이 짓기)
#   3부 – 눈길 바탕 갈래별 마음결 모델(PyTorch)
#   4부 – 곁딸린 월을 쓴 변환기 갈래별 마음결 살피기
#   5부 – 금융 갈래별 마음결 살피기 쓰임새
#
# 바탕: O'Reilly "Practical NLP" 9장

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import re


# =====================================================================
# 1부 – VADER로 하는 규칙 바탕 갈래별 마음결 살피기
# =====================================================================
print("=" * 60)
print("Part 1: Rule-Based ABSA with VADER")
print("=" * 60)

# VADER는 사회 그물 글에 맞춰 다듬은
# 낱말집 바탕 마음결 도구이다.
# 다루는 것: 대문자 쓰기, 문장 부호 강조, 정도 꾸밈말,
# 부정, 이음씨.
#
# 갈래별 마음결 살피기에서는 다음을 한다:
#   1. 글을 마디로 쪼갠다(문장 부호나 이음씨 기준)
#   2. 마디마다 갈래 낱말을 가려낸다
#   3. 마디마다 VADER로 점수를 매긴다
#   4. 마디의 마음결을 그 갈래에 매긴다

# 간추린 VADER 비슷한 점수 매기기(실제 VADER는 낱말집 항목 약 7500개를 쓴다)
SENTIMENT_LEXICON = {
    # 양성 낱말
    "amazing": 3.0, "excellent": 3.0, "great": 2.5, "good": 2.0,
    "strong": 2.0, "impressive": 2.5, "solid": 1.5, "beat": 2.0,
    "exceeded": 2.5, "growth": 1.5, "profit": 1.5, "surged": 2.5,
    "bullish": 2.0, "outperform": 2.0, "upgrade": 2.0, "record": 2.0,
    # 음성 낱말
    "terrible": -3.0, "poor": -2.0, "bad": -2.0, "weak": -2.0,
    "disappointing": -2.5, "declined": -2.0, "missed": -2.5,
    "bearish": -2.0, "downgrade": -2.5, "loss": -2.0, "risk": -1.5,
    "concern": -1.5, "slowed": -1.5, "underperform": -2.0, "cut": -1.5,
    # 꾸밈말
    "very": 1.5, "extremely": 2.0, "slightly": 0.5, "somewhat": 0.7,
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "hardly", "barely"}
CONJUNCTION_SPLIT = re.compile(r'\b(but|however|although|though|yet|while)\b', re.IGNORECASE)


def vader_score_clause(clause: str) -> float:
    """간추린 VADER 방식으로 글 마디에 점수 매기기."""
    words = clause.lower().split()
    score = 0.0
    negate = False
    modifier = 1.0

    for word in words:
        # 부정 살피기
        if word in NEGATION_WORDS:
            negate = True
            continue

        # 정도 꾸밈말 살피기
        if word in SENTIMENT_LEXICON and abs(SENTIMENT_LEXICON[word]) < 2.0 and word in {"very", "extremely", "slightly", "somewhat"}:
            modifier = SENTIMENT_LEXICON[word]
            continue

        # 마음결 낱말에 점수 매기기
        if word in SENTIMENT_LEXICON:
            word_score = SENTIMENT_LEXICON[word] * modifier
            if negate:
                word_score *= -0.75  # 부정은 누그러뜨리되 완전히 뒤집지는 않는다
                negate = False
            score += word_score
            modifier = 1.0  # 쓴 뒤 꾸밈말 되돌리기

    # [-1, 1]로 고르게 맞추기
    return np.tanh(score / 3.0)


def extract_aspects_and_sentiment(text: str) -> List[Dict]:
    """글에서 갈래와 그 마음결 뽑기.

    전략:
    1. 이음씨(but, however 등)에서 쪼갠다
    2. 마디마다 대문자나 따옴표로 된 낱말을 갈래로 찾는다
    3. 마디마다 점수 매기기
    """
    # 이음씨에서 쪼개기
    clauses = CONJUNCTION_SPLIT.split(text)
    clauses = [c.strip() for c in clauses if c.strip() and c.lower() not in
               {"but", "however", "although", "though", "yet", "while"}]

    results = []
    for clause in clauses:
        score = vader_score_clause(clause)
        # 갈래 후보 뽑기(단순 어림짐작: 이름씨나 것 찾기)
        # 실전에서는 이름씨 마디 뽑기에 자연어 도구를 쓴다
        words = clause.split()
        aspect_candidates = []
        for w in words:
            if w[0] == '$' or (w[0].isupper() and w.lower() not in SENTIMENT_LEXICON
                               and w.lower() not in NEGATION_WORDS
                               and len(w) > 2):
                aspect_candidates.append(w)

        label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        results.append({
            "clause": clause,
            "aspects": aspect_candidates,
            "score": score,
            "label": label,
        })

    return results


# 시연
texts = [
    "$AAPL reported amazing revenue growth but iPhone margins were disappointing",
    "Tesla has impressive delivery numbers however valuation risk remains a concern",
    "Strong cloud revenue from AWS but advertising growth slowed significantly",
]

for text in texts:
    print(f"\n  Text: {text}")
    aspects = extract_aspects_and_sentiment(text)
    for a in aspects:
        asp_str = ", ".join(a["aspects"]) if a["aspects"] else "(general)"
        print(f"    {a['label']:>8} ({a['score']:+.2f}): [{asp_str}] — {a['clause'][:50]}")

print()

# VADER 온전히 쓰기
print("  Using the vaderSentiment library:")
print("""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    text = "The product is absolutely amazing but the price is too high"
    scores = analyzer.polarity_scores(text)
    # → {'neg': 0.15, 'neu': 0.52, 'pos': 0.33, 'compound': 0.34}

    # 갈래별 살피기: "but"에서 쪼개어 부분마다 점수 매기기
    parts = text.split(" but ")
    for part in parts:
        s = analyzer.polarity_scores(part)
        print(f"  '{part}' → compound: {s['compound']:.3f}")
    # → 'The product is absolutely amazing' → compound: 0.62
    # → 'the price is too high' → compound: -0.19
""")


# =====================================================================
# 2부 – 갈래 낱말 뽑기
# =====================================================================
print("=" * 60)
print("Part 2: Aspect Term Extraction")
print("=" * 60)

# 갈래 뽑기는 다음으로 할 수 있다:
# 1. 달림 뜯어 읽기(nsubj, dobj 관계)
# 2. 잦기 바탕(그 분야에서 가장 흔한 이름씨)
# 3. 익힌 모델의 눈길 무게
# 4. 차례 이름표 붙이기(갈래에 대한 BIO 이름표)

print("""
  방식 1: spaCy 달림 뜯어 읽기
  ─────────────────────────────────────
    import spacy
    nlp = spacy.load("en_core_web_sm")

    text = "The camera quality is excellent but battery drains quickly"
    doc = nlp(text)

    aspects = []
    for token in doc:
        # 주어나 목적어인 이름씨 → 갈래일 가능성이 높다
        if token.dep_ in ("nsubj", "dobj", "attr") and token.pos_ == "NOUN":
            aspects.append(token.text)
    # → ["quality", "battery"]

  방식 2: 차례 이름표 붙이기(BIO 이름표)
  ─────────────────────────────────────────
    # 표시한 갈래별 마음결 자료로 토막 갈래 매개 익히기
    # 이름표: B-ASP, I-ASP, O
    # "The [camera quality]_ASP is excellent"
    # → O  B-ASP  I-ASP  O  O

    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(
        "yangheng/deberta-v3-base-absa-v1.1"
    )
""")


# =====================================================================
# 3부 – 눈길 바탕 갈래별 마음결 살피기(PyTorch)
# =====================================================================
print("=" * 60)
print("Part 3: Attention-Based ABSA Model (PyTorch)")
print("=" * 60)

# 핵심 생각: 월과 목표 갈래가 주어질 때 눈길을 써서
# 그 갈래와 맞닿는 낱말에 초점을 둔다.
#
# 얼개:
#   [낱말 묻힘] → 두 방향 LSTM → 눈길(aspect_emb) → 갈래 매기기


class AspectAttentionClassifier(nn.Module):
    """눈길 바탕 갈래별 마음결 모델.

    갈래 묻힘을 물음으로 삼아 맥락 낱말에 눈길을 준 뒤
    눈길을 준 나타냄에 갈래를 매긴다.

    인수:
        vocab_size:  낱말 곳간 크기
        embed_dim:   묻힘 차원
        hidden_dim:  LSTM 숨은 차원
        num_classes: 마음결 갈래(양성/음성/가운데)
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        # 눈길: 갈래와 맥락을 같은 공간으로 내리쬐기
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, context_ids, aspect_ids, context_mask=None):
        """
        인수:
            context_ids: 온 월의 토막 번호 [B, L]
            aspect_ids:  갈래 낱말의 토막 번호 [B, A]
            context_mask: 덧대기 마스크 [B, L]

        반환값:
            logits: 마음결 로짓 [B, num_classes]
        """
        # 맥락 부호화
        ctx_emb = self.embedding(context_ids)       # [B, L, E]
        ctx_out, _ = self.lstm(ctx_emb)             # [B, L, 2H]

        # 갈래 부호화(묻힘의 평균)
        asp_emb = self.embedding(aspect_ids)         # [B, A, E]
        asp_repr = asp_emb.mean(dim=1, keepdim=True) # [B, 1, E]

        # 갈래를 헤아리는 눈길
        # 맥락을 내리쬐고 눈길 점수 셈하기
        ctx_proj = self.attn_proj(ctx_out)           # [B, L, 2H]
        # 내리쬔 맥락과의 점곱으로 갈래를 물음으로 쓰기
        # 묻힘으로 갈래를 숨은 차원까지 부풀린 뒤 그대로 쓰기
        attn_scores = torch.bmm(ctx_proj, asp_repr.expand(-1, -1, ctx_proj.size(-1)).transpose(1, 2))
        attn_scores = attn_scores.squeeze(-1)        # [B, L]

        if context_mask is not None:
            attn_scores = attn_scores.masked_fill(context_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L]

        # 맥락의 무게 합
        weighted = torch.bmm(attn_weights.unsqueeze(1), ctx_out).squeeze(1)  # [B, 2H]

        return self.classifier(weighted)


# 인공 자료로 보이기
torch.manual_seed(42)
VOCAB = 200
model = AspectAttentionClassifier(VOCAB, embed_dim=32, hidden_dim=32, num_classes=3)

# 인공 들임
ctx = torch.randint(1, VOCAB, (4, 20))
asp = torch.randint(1, VOCAB, (4, 3))
mask = torch.ones(4, 20)

logits = model(ctx, asp, mask)
print(f"  Model output shape: {logits.shape}")
print(f"  Predictions: {F.softmax(logits, dim=-1).detach().numpy().round(3)}")
print(f"  Labels: negative=0, neutral=1, positive=2")
print()


# =====================================================================
# 4부 – 곁딸린 월을 쓴 변환기 갈래별 마음결 살피기
# =====================================================================
print("=" * 60)
print("Part 4: Transformer ABSA (Auxiliary Sentence Approach)")
print("=" * 60)

# 요즘 방식: 갈래별 마음결 살피기를 월 짝 갈래 매기기로 바꾼다.
# 들임: "[CLS] 월 [SEP] 갈래 낱말 [SEP]"
# 이러면 BERT가 엇갈린 눈길로 갈래에 눈길을 줄 수 있다.

print("""
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  model_name = "yangheng/deberta-v3-base-absa-v1.1"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)

  # 들임을 월 짝으로 세우기
  sentence = "The restaurant has great food but terrible service"
  aspect = "food"

  inputs = tokenizer(
      sentence, aspect,        # 월 짝!
      return_tensors="pt",
      truncation=True,
      max_length=128,
  )

  with torch.no_grad():
      logits = model(**inputs).logits
      # 이름표: 0=음성, 1=가운데, 2=양성
      pred = logits.argmax(-1).item()
      labels = ["negative", "neutral", "positive"]
      print(f"Aspect '{aspect}': {labels[pred]}")
  # → 갈래 'food': 양성

  # 이제 "service" 살피기
  inputs = tokenizer(sentence, "service", return_tensors="pt")
  with torch.no_grad():
      pred = model(**inputs).logits.argmax(-1).item()
      print(f"Aspect 'service': {labels[pred]}")
  # → 갈래 'service': 음성
""")


# =====================================================================
# 5부 – 금융 갈래별 마음결 살피기 쓰임새
# =====================================================================
print("=" * 60)
print("Part 5: Financial ABSA Use Cases")
print("=" * 60)

print("""
  금융 글은 흔히 한 월에서 여러 갈래를 이야기한다:

  "Revenue grew 15% driven by cloud services,
   but hardware margins compressed due to supply constraints."
  → 클라우드 서비스: 양성
  → 하드웨어 이익률: 음성

  "Management is bullish on AI initiatives but cautious about
   the regulatory environment in Europe."
  → 인공지능 계획: 양성
  → 규제 환경: 음성

  좇아야 할 핵심 금융 갈래:
  ┌──────────────────┬────────────────────────────┐
  │ 갈래 묶음        │ 갈래 보기                  │
  ├──────────────────┼────────────────────────────┤
  │ 매출             │ 총매출, 판매, 수주         │
  │ 수익성           │ 이익률, 주당순이익, EBITDA │
  │ 성장             │ 전년 대비 성장, 전망, TAM  │
  │ 재무 상태        │ 부채, 현금, 지렛대         │
  │ 운영             │ 효율, 인원                 │
  │ 시장 자리        │ 시장 점유율, 경쟁          │
  │ 위험 요인        │ 규제, 소송                 │
  │ 경영             │ 지도력, 전략               │
  └──────────────────┴────────────────────────────┘

  갈래별 마음결 살피기에서 얻는 거래 신호:
    실적 발표나 뉴스 글마다:
    1. 갈래 낱말 뽑기(이름 알아보기 + 이름씨 마디)
    2. 갈래마다 마음결에 점수 매기기
    3. 그 종목에 대한 갈래의 중요도로 무게 주기
    4. 모으기: 신호 = Σ (갈래 무게 × 갈래 마음결)

    갈래의 무게는 업종마다 다르다:
    - 기술: 성장 (0.3), 이익률 (0.2), 클라우드 (0.2), 인공지능 (0.2)
    - 은행: 순이자수익 (0.3), 신용 질 (0.3), 수수료 (0.2)
    - 소매: 동일 점포 매출 (0.3), 이익률 (0.2), 재고 (0.2)
""")

print("Done.")


if __name__ == "__main__":
    pass
```

## 2. 논의

`AspectAttentionClassifier` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다. `forward` 메서드가 셈 그래프를 정하므로 익히는 동안 PyTorch의 자동 미분 체계가 기울기 셈을 알아서 다룬다. 이 단원별 꾸밈 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 끼워 넣기가 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
붙박이 첫자리매김일 때 `AspectAttentionClassifier`의 배울 수 있는 매개변수 전체 개수를 셈하여라. 무게와 치우침을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `AspectAttentionClassifier`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = AspectAttentionClassifier(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 갈래별 마음결

`AspectAttentionClassifier` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다.

고갱이 갈래는 `AspectAttentionClassifier`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
