# 측면 기반 감성 분석

측면 기반 감성 분석(ABSA)은 문장 전체가 아니라 특정 대상이나 속성에 대한 감성을 구분한다. ‘음식은 좋지만 서비스는 나쁘다’에서 음식과 서비스는 서로 다른 감성을 갖는다. 측면 표현의 추출과, 주어진 측면에 대한 감성 분류는 구분되는 작업이며 이 페이지는 규칙·어텐션 구조·사전 학습 모델 사용법을 예시로 다룬다.

## 1. 코드

```python
"""측면 기반 감성 분석."""
# ---
# title: "측면 기반 감성 분석"
# description: "VADER로 하는 개체/측면 수준의 세밀한 감성 분석,
#               PyTorch 어텐션 기반 모델, 트랜스포머 방식"
# ---
#
# 보통의 감성 분석은 텍스트마다 라벨 하나를 준다.
# 측면 기반 감성 분석(ABSA)은 글 속의 특정한
# 측면이나 개체에 대한 감성을 추출한다.
#
# 보기: "The phone has an amazing camera but terrible battery life."
#   → 카메라: 긍정
#   → 배터리 수명: 부정
#
#   1부 – VADER로 하는 규칙 기반 측면 감성 분석
#   2부 – 측면 추출(명사구 청킹)
#   3부 – 어텐션 기반 측면 감성 모델(PyTorch)
#   4부 – 보조 문장을 쓴 트랜스포머 측면 감성 분석
#   5부 – 금융 측면 감성 분석 활용 사례
#
# 출처: O'Reilly "Practical NLP" 9장

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
import re


# =====================================================================
# 1부 – VADER로 하는 규칙 기반 측면 감성 분석
# =====================================================================
print("=" * 60)
print("Part 1: Rule-Based ABSA with VADER")
print("=" * 60)

# VADER는 소셜 미디어 글에 맞춰 조정한
# 감성 사전(lexicon) 기반 감성 도구이다.
# 다루는 것: 대문자 쓰기, 문장 부호 강조, 정도 부사,
# 부정, 접속사.
#
# 측면 기반 감성 분석에서는 다음을 한다:
#   1. 글을 절로 나눈다(문장 부호나 접속사 기준)
#   2. 절마다 측면 단어를 찾는다
#   3. 절마다 VADER로 점수를 매긴다
#   4. 절의 감성을 그 측면에 부여한다

# 단순화한 VADER 방식 점수 매기기(실제 VADER는 감성 사전 항목 약 7500개를 쓴다)
SENTIMENT_LEXICON = {
    # 긍정 단어
    "amazing": 3.0, "excellent": 3.0, "great": 2.5, "good": 2.0,
    "strong": 2.0, "impressive": 2.5, "solid": 1.5, "beat": 2.0,
    "exceeded": 2.5, "growth": 1.5, "profit": 1.5, "surged": 2.5,
    "bullish": 2.0, "outperform": 2.0, "upgrade": 2.0, "record": 2.0,
    # 부정 단어
    "terrible": -3.0, "poor": -2.0, "bad": -2.0, "weak": -2.0,
    "disappointing": -2.5, "declined": -2.0, "missed": -2.5,
    "bearish": -2.0, "downgrade": -2.5, "loss": -2.0, "risk": -1.5,
    "concern": -1.5, "slowed": -1.5, "underperform": -2.0, "cut": -1.5,
    # 정도 부사
    "very": 1.5, "extremely": 2.0, "slightly": 0.5, "somewhat": 0.7,
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "hardly", "barely"}
CONJUNCTION_SPLIT = re.compile(r'\b(but|however|although|though|yet|while)\b', re.IGNORECASE)


def vader_score_clause(clause: str) -> float:
    """단순화한 VADER 방식으로 절에 점수 매기기."""
    words = clause.lower().split()
    score = 0.0
    negate = False
    modifier = 1.0

    for word in words:
        # 부정어(negation) 확인하기
        if word in NEGATION_WORDS:
            negate = True
            continue

        # 정도 부사 확인하기
        if word in SENTIMENT_LEXICON and abs(SENTIMENT_LEXICON[word]) < 2.0 and word in {"very", "extremely", "slightly", "somewhat"}:
            modifier = SENTIMENT_LEXICON[word]
            continue

        # 감성 단어에 점수 매기기
        if word in SENTIMENT_LEXICON:
            word_score = SENTIMENT_LEXICON[word] * modifier
            if negate:
                word_score *= -0.75  # 부정어는 감성을 약화하되 완전히 뒤집지는 않는다
                negate = False
            score += word_score
            modifier = 1.0  # 쓴 뒤 정도 배율 초기화

    # [-1, 1] 범위로 정규화하기
    return np.tanh(score / 3.0)


def extract_aspects_and_sentiment(text: str) -> List[Dict]:
    """글에서 측면과 그 감성 추출하기.

    전략:
    1. 접속사(but, however 등)에서 나눈다
    2. 절마다 대문자나 따옴표로 된 단어를 측면으로 찾는다
    3. 절마다 점수 매기기
    """
    # 접속사에서 나누기
    clauses = CONJUNCTION_SPLIT.split(text)
    clauses = [c.strip() for c in clauses if c.strip() and c.lower() not in
               {"but", "however", "although", "though", "yet", "while"}]

    results = []
    for clause in clauses:
        score = vader_score_clause(clause)
        # 측면 후보 추출하기(단순 휴리스틱: 명사나 개체 찾기)
        # 실전에서는 명사구 추출에 NLP 도구를 쓴다
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

# 실제 VADER 라이브러리 쓰기
print("  Using the vaderSentiment library:")
print("""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    text = "The product is absolutely amazing but the price is too high"
    scores = analyzer.polarity_scores(text)
    # → {'neg': 0.15, 'neu': 0.52, 'pos': 0.33, 'compound': 0.34}

    # 측면별 분석: "but"에서 나누어 부분마다 점수 매기기
    parts = text.split(" but ")
    for part in parts:
        s = analyzer.polarity_scores(part)
        print(f"  '{part}' → compound: {s['compound']:.3f}")
    # → 'The product is absolutely amazing' → compound: 0.62
    # → 'the price is too high' → compound: -0.19
""")


# =====================================================================
# 2부 – 측면 단어 추출
# =====================================================================
print("=" * 60)
print("Part 2: Aspect Term Extraction")
print("=" * 60)

# 측면 추출은 다음으로 할 수 있다:
# 1. 의존 구문 분석(nsubj, dobj 관계)
# 2. 빈도 기반(그 분야에서 가장 흔한 명사)
# 3. 학습한 모델의 어텐션 웨이트
# 4. 시퀀스 라벨링(측면에 대한 BIO 라벨)

print("""
  방식 1: spaCy 의존 구문 분석
  ─────────────────────────────────────
    import spacy
    nlp = spacy.load("en_core_web_sm")

    text = "The camera quality is excellent but battery drains quickly"
    doc = nlp(text)

    aspects = []
    for token in doc:
        # 주어나 목적어인 명사 → 측면일 가능성이 높다
        if token.dep_ in ("nsubj", "dobj", "attr") and token.pos_ == "NOUN":
            aspects.append(token.text)
    # → ["quality", "battery"]

  방식 2: 시퀀스 라벨링(BIO 라벨)
  ─────────────────────────────────────────
    # 라벨이 붙은 ABSA 데이터로 토큰 분류 모델 학습하기
    # 라벨: B-ASP, I-ASP, O
    # "The [camera quality]_ASP is excellent"
    # → O  B-ASP  I-ASP  O  O

    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(
        "yangheng/deberta-v3-base-absa-v1.1"
    )
""")


# =====================================================================
# 3부 – 어텐션 기반 측면 감성 분석(PyTorch)
# =====================================================================
print("=" * 60)
print("Part 3: Attention-Based ABSA Model (PyTorch)")
print("=" * 60)

# 핵심 생각: 문장과 목표 측면이 주어질 때 어텐션을 써서
# 그 측면과 관련된 단어에 초점을 둔다.
#
# 구조:
#   [단어 임베딩] → 양방향 LSTM → 어텐션(aspect_emb) → 분류


class AspectAttentionClassifier(nn.Module):
    """어텐션 기반 측면 감성 모델.

    측면 임베딩을 쿼리로 삼아 문맥 단어에 어텐션을 준 뒤
    어텐션으로 가중합한 표현을 분류한다.

    인수:
        vocab_size:  어휘 크기
        embed_dim:   임베딩 차원
        hidden_dim:  LSTM 은닉 차원
        num_classes: 감성 클래스(긍정/부정/중립)
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        # 어텐션: 측면과 문맥을 같은 공간으로 사영하기
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, context_ids, aspect_ids, context_mask=None):
        """
        인수:
            context_ids: 전체 문장의 토큰 ID [B, L]
            aspect_ids:  측면 단어의 토큰 ID [B, A]
            context_mask: 패딩 마스크 [B, L]

        반환값:
            logits: 감성 로짓 [B, num_classes]
        """
        # 문맥 인코딩
        ctx_emb = self.embedding(context_ids)       # [B, L, E]
        ctx_out, _ = self.lstm(ctx_emb)             # [B, L, 2H]

        # 측면 인코딩(임베딩의 평균)
        asp_emb = self.embedding(aspect_ids)         # [B, A, E]
        asp_repr = asp_emb.mean(dim=1, keepdim=True) # [B, 1, E]

        # 측면을 고려한 어텐션
        # 문맥을 사영하고 어텐션 점수 계산하기
        ctx_proj = self.attn_proj(ctx_out)           # [B, L, 2H]
        # 사영한 문맥과의 내적으로 측면을 쿼리로 쓰기
        # 측면 임베딩을 은닉 차원까지 확장한 뒤 그대로 쓰기
        attn_scores = torch.bmm(ctx_proj, asp_repr.expand(-1, -1, ctx_proj.size(-1)).transpose(1, 2))
        attn_scores = attn_scores.squeeze(-1)        # [B, L]

        if context_mask is not None:
            attn_scores = attn_scores.masked_fill(context_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L]

        # 문맥의 가중합
        weighted = torch.bmm(attn_weights.unsqueeze(1), ctx_out).squeeze(1)  # [B, 2H]

        return self.classifier(weighted)


# 합성 데이터로 시연
torch.manual_seed(42)
VOCAB = 200
model = AspectAttentionClassifier(VOCAB, embed_dim=32, hidden_dim=32, num_classes=3)

# 합성 입력
ctx = torch.randint(1, VOCAB, (4, 20))
asp = torch.randint(1, VOCAB, (4, 3))
mask = torch.ones(4, 20)

logits = model(ctx, asp, mask)
print(f"  Model output shape: {logits.shape}")
print(f"  Predictions: {F.softmax(logits, dim=-1).detach().numpy().round(3)}")
print(f"  Labels: negative=0, neutral=1, positive=2")
print()


# =====================================================================
# 4부 – 보조 문장을 쓴 트랜스포머 측면 감성 분석
# =====================================================================
print("=" * 60)
print("Part 4: Transformer ABSA (Auxiliary Sentence Approach)")
print("=" * 60)

# 최근 방식: 측면 기반 감성 분석을 문장 쌍 분류로 바꾼다.
# 입력: "[CLS] 문장 [SEP] 측면 단어 [SEP]"
# 이러면 BERT가 두 문장에 걸친 어텐션으로 측면에 어텐션을 줄 수 있다.

print("""
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  model_name = "yangheng/deberta-v3-base-absa-v1.1"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)

  # 입력을 문장 쌍으로 구성하기
  sentence = "The restaurant has great food but terrible service"
  aspect = "food"

  inputs = tokenizer(
      sentence, aspect,        # 문장 쌍!
      return_tensors="pt",
      truncation=True,
      max_length=128,
  )

  with torch.no_grad():
      logits = model(**inputs).logits
      # 라벨: 0=부정, 1=중립, 2=긍정
      pred = logits.argmax(-1).item()
      labels = ["negative", "neutral", "positive"]
      print(f"Aspect '{aspect}': {labels[pred]}")
  # → 측면 'food': 긍정

  # 이제 "service" 분석하기
  inputs = tokenizer(sentence, "service", return_tensors="pt")
  with torch.no_grad():
      pred = model(**inputs).logits.argmax(-1).item()
      print(f"Aspect 'service': {labels[pred]}")
  # → 측면 'service': 부정
""")


# =====================================================================
# 5부 – 금융 측면 감성 분석 활용 사례
# =====================================================================
print("=" * 60)
print("Part 5: Financial ABSA Use Cases")
print("=" * 60)

print("""
  금융 글은 흔히 한 문장에서 여러 측면을 이야기한다:

  "Revenue grew 15% driven by cloud services,
   but hardware margins compressed due to supply constraints."
  → 클라우드 서비스: 긍정
  → 하드웨어 이익률: 부정

  "Management is bullish on AI initiatives but cautious about
   the regulatory environment in Europe."
  → 인공지능 계획: 긍정
  → 규제 환경: 부정

  추적할 핵심 금융 측면:
  ┌──────────────────┬────────────────────────────┐
  │ 측면 범주        │ 측면 예시                  │
  ├──────────────────┼────────────────────────────┤
  │ 매출             │ 총매출, 판매, 수주         │
  │ 수익성           │ 이익률, 주당순이익, EBITDA │
  │ 성장             │ 전년 대비 성장, 전망, TAM  │
  │ 재무 상태        │ 부채, 현금, 레버리지       │
  │ 운영             │ 효율, 인원                 │
  │ 시장 자리        │ 시장 점유율, 경쟁          │
  │ 위험 요인        │ 규제, 소송                 │
  │ 경영             │ 리더십, 전략               │
  └──────────────────┴────────────────────────────┘

  측면 기반 감성 분석에서 얻는 거래 신호:
    실적 발표나 뉴스 글마다:
    1. 측면 단어 추출(개체명 인식 + 명사구)
    2. 측면마다 감성에 점수 매기기
    3. 그 종목에 대한 측면의 중요도로 가중하기
    4. 집계: 신호 = Σ (측면 웨이트 × 측면 감성)

    측면의 웨이트는 업종마다 다르다:
    - 기술: 성장 (0.3), 이익률 (0.2), 클라우드 (0.2), 인공지능 (0.2)
    - 은행: 순이자수익 (0.3), 신용 질 (0.3), 수수료 (0.2)
    - 소매: 동일 점포 매출 (0.3), 이익률 (0.2), 재고 (0.2)
""")

print("Done.")


if __name__ == "__main__":
    pass
```

## 2. 절 분리와 사전 점수의 범위

vader_score_clause는 직접 작성한 작은 감성 사전과 tanh 점수다. VADER 라이브러리의 구현이나 원래 compound 점수와 같지 않다. 코드가 실제로 호출하는 것은 이 함수이며, vaderSentiment·spaCy·트랜스포머 사용 부분은 문자열로 출력하는 안내다.

but·however 등의 접속 표현에서 절을 나누고 절마다 하나의 점수를 준다. 대문자나 달러 기호로 시작하는 토큰을 측면 후보로 삼으므로 문장 첫 단어·기업 이름과 실제 속성을 혼동하고 소문자 측면을 놓칠 수 있다. 한 절에 여러 측면이 있으면 각각에 다른 감성을 연결하는 기능도 없다.

공백 분리만 하므로 구두점이 붙은 감성어가 사전에 맞지 않을 수 있다. 부정은 다음에 만나는 감성어까지 남고 이중 부정·범위를 제대로 분석하지 않는다. modifier 조건은 절댓값이 2보다 작은 경우만 허용해 extremely=2.0이 강조 배율이 아니라 독립 감성어로 처리된다. 이런 규칙의 결과를 정교한 언어 이해로 해석하지 않는다.

## 3. 측면 조건부 어텐션과 차원

의도한 뉴럴 구조는 문맥의 BiLSTM 출력에 측면 임베딩을 쿼리로 사용해 가중합한 뒤 감성을 분류하는 것이다. 문맥 길이가 T, 단방향 은닉 크기가 H라면 BiLSTM 출력은 B×T×2H, 측면 평균 임베딩은 B×1×E다. 두 표현을 내적하려면 같은 차원으로 사영해야 한다.

현재 시연은 E=32, H=32이므로 측면 표현의 마지막 차원은 32, 문맥 사영은 64다. expand로 이미 크기가 32인 축을 64로 늘릴 수 없으므로 logits를 얻기 전에 실패한다. 현재 코드가 학습된 측면 감성을 예측하거나 뒤의 모든 안내까지 정상 실행했다고 설명할 수 없다.

측면 평균은 padding 개수까지 분모에 넣고, 문맥의 어텐션 마스크는 LSTM이 padding을 처리한 영향을 제거하지 않는다. 모든 문맥 위치를 −∞로 마스킹하면 소프트맥스가 NaN이 될 수 있다. 측면·문맥의 실제 길이와 유효한 후보를 맞춰야 한다. 무작위 ID와 학습하지 않은 웨이트로 낸 출력은 고쳐서 돌린다 해도 성능 평가는 아니다.

## 4. 사전 학습 모델과 평가

문장·측면 쌍을 입력하는 감성 분류 체크포인트와 BIO로 측면 구간을 찾는 토큰 분류 모델은 다르다. 안내 문자열에 나온 ABSA 시퀀스 분류 체크포인트를 이름만 보고 토큰 분류기로 사용할 수 있다고 가정하지 않는다. head·학습 과제·라벨 ID는 모델 설정과 학습 데이터에서 확인해야 한다.

평가에서는 정답 측면을 제공한 감성 분류와 측면 추출부터 수행한 전체 시스템을 구분한다. 유형별 F1, 측면 구간 및 감성의 공동 일치, 부정·대조·여러 측면의 오류를 분석한다. 텍스트 감성을 가중합한 값은 설계한 지표일 뿐 거래 성과나 경제적 의미가 검증된 신호가 아니다. 본문의 고정 업종별 웨이트도 학습·검증된 값이 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
E=32, H=32, V=200, 클래스 3개일 때 AspectAttentionClassifier의 파라메터 수와 순전파 차원 문제를 계산하라.

</div>

??? success "풀이"

    임베딩 200×32=6,400개, 양방향 LSTM은 2×(4HE+4H²+8H)=16,896개다. attn_proj는 64×64+64=4,160개, classifier는 64×3+3=195개로 총 27,651개다. 그러나 측면 평균의 차원 32를 expand로 문맥 차원 64에 맞출 수 없어 현재 순전파는 실패한다. 파라메터 수가 정의된다는 것과 올바른 계산이 가능하다는 것은 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
절 단위 감성 점수를 측면별 감성과 동일하게 볼 수 없는 이유를 설명하라.

</div>

??? success "풀이"

    한 절에 두 측면과 서로 다른 감성이 있을 수 있고 대조 접속사 없이도 대상이 바뀔 수 있다. 대문자 후보 추출도 속성을 완전히 찾지 못한다. 측면 구간과 감성 표현의 관계, 부정 범위와 문맥을 따로 모델링하고 정답 측면 조건의 평가와 전체 추출 평가를 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
현재 어텐션과 전체 셀프 어텐션의 계산 구조가 어떻게 다른지 설명하라.

</div>

??? success "풀이"

    현재 의도는 측면 쿼리 하나가 T개 문맥 위치에 점수를 주는 방식으로 T×T 쌍을 모두 만들지 않는다. 문맥 사영·LSTM 비용도 포함해야 하며 이를 곧바로 O(T²) 셀프 어텐션이라고 부르지 않는다. padding을 제외한 측면 평균과 문맥 길이, 전부 마스킹된 입력의 정책도 필요하다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 자리표시자로 현재 차원 문제나 레이어 수 확장이 완성되는지 검토하라.

</div>

??? success "풀이"

    YourBlock과 dim이 정의되지 않았고 현재 생성자에 num_layers도 없으므로 완성된 확장이 아니다. 측면 사영 차원을 먼저 정한 뒤 반복할 모듈과 각 입력·출력 크기, forward 연결을 설계해야 한다. LSTM의 num_layers를 늘리는 방식과 임의 블록 목록을 추가하는 방식도 다르다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

ABSA는 측면의 위치와 그 대상에 대한 감성을 연결하는 작업이다. 절 단위 규칙·조건부 분류·전체 추출을 구분하고, 현재 뉴럴 예제의 차원 불일치와 padding 계약을 확인한 뒤 결과를 해석해야 한다.
