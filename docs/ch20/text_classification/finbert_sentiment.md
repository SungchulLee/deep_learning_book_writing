# FinBERT 마음결

FinBERT 마음결.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 글 가르기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""FinBERT 마음결."""
# ---
# title: "FinBERT: 금융 마음결 살피기"
# description: "주식 마음결 갈래 매기기를 위해 FinBERT를 곱게 다듬기,
#               StockTwits 자료 — 금융 자연어 다루기에 맞춘 BERT"
# ---
#
# FinBERT는 금융 글(10-K 보고서,
# 분석 보고서, 실적 발표)로 더 익힌 BERT 모델이다. 일반 BERT가 놓치는
# 분야 말("bullish", "bearish", "overbought", "short squeeze")을 담아낸다.
#
#   1부 – HuggingFace Hub에서 FinBERT 읽어 들이기
#   2부 – 미리 익힌 마음결 머리로 영 발 미룸
#   3부 – StockTwits 자료로 FinBERT 곱게 다듬기(PyTorch)
#   4부 – 값매김과 종목별 살피기
#   5부 – FinBERT, 일반 BERT, 낱말집 바탕 견주기
#
# 바탕: O'Reilly "Practical NLP" 10장

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np


# =====================================================================
# 1부 – HuggingFace Hub에서 FinBERT 읽어 들이기
# =====================================================================
print("=" * 60)
print("Part 1: Loading FinBERT")
print("=" * 60)

# 요즘 방식: FinBERT는 HuggingFace Hub에 있다.
# 널리 쓰이는 변종 둘:
#   - ProsusAI/finbert         (마음결: 양성/음성/가운데)
#   - yiyanghkust/finbert-tone (같은 일, 다른 미리 익힘 말뭉치)

print("""
  # 마음결 살피기를 위해 미리 익힌 FinBERT 읽어 들이기
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  model_name = "ProsusAI/finbert"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  model.eval()

  # 빠른 미룸
  text = "Tesla reported record deliveries, beating analyst expectations."
  inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
  with torch.no_grad():
      logits = model(**inputs).logits
      probs = F.softmax(logits, dim=-1)
      labels = ["positive", "negative", "neutral"]
      pred = labels[probs.argmax()]
      print(f"Sentiment: {pred} ({probs.max():.3f})")
  # → 마음결: 양성 (0.92)
""")


# =====================================================================
# 2부 – 물길로 하는 영 발 미룸
# =====================================================================
print("=" * 60)
print("Part 2: Zero-Shot Inference via HuggingFace Pipeline")
print("=" * 60)

print("""
  from transformers import pipeline

  finbert = pipeline(
      "sentiment-analysis",
      model="ProsusAI/finbert",
      tokenizer="ProsusAI/finbert",
  )

  # 어림 하나
  result = finbert("AAPL missed revenue estimates by 2%.")
  # → [{'label': 'negative', 'score': 0.87}]

  # 묶음 어림
  texts = [
      "Revenue grew 15% year-over-year, exceeding guidance.",
      "The company announced layoffs affecting 10% of staff.",
      "Q3 earnings were in line with consensus expectations.",
      "Short sellers are increasing their positions in the stock.",
  ]
  results = finbert(texts)
  for text, res in zip(texts, results):
      print(f"  {res['label']:>8} ({res['score']:.2f}): {text[:60]}")
""")


# =====================================================================
# 3부 – 맞춤 주식 자료로 FinBERT 곱게 다듬기(PyTorch)
# =====================================================================
print("=" * 60)
print("Part 3: Fine-Tuning FinBERT on StockTwits Data")
print("=" * 60)

# 본디 공책은 FB, AMZN, GOOGL의 StockTwits 자료로 곱게 다듬는다
# 이름표: Bullish (1) / Bearish (0).
#
# 핵심 단계:
#   1. BERT 토막내개로 토막낸다(max_len=128)
#   2. 눈길 마스크를 만든다
#   3. 익힘/검증으로 나눈다
#   4. AdamW와 한 줄 몸풀기로 곱게 다듬는다
#   5. 세대마다 값매김한다

print("  Creating synthetic StockTwits-like data for demonstration...")

# 인공 금융 트윗
bullish_texts = [
    "$AAPL strong breakout above resistance, going long",
    "$TSLA deliveries crushing it, moon incoming",
    "$GOOGL ad revenue growth is accelerating, bullish",
    "$AMZN AWS margins expanding, great quarter ahead",
    "$MSFT cloud business firing on all cylinders",
    "$NVDA AI demand is insatiable, still undervalued",
    "$META reels monetization improving rapidly",
    "$JPM net interest income at record highs",
] * 15  # 보기 120개

bearish_texts = [
    "$AAPL China sales declining, supply chain issues",
    "$TSLA margin compression is alarming, overvalued",
    "$GOOGL antitrust risk is real, short this name",
    "$AMZN retail losing money, AWS growth slowing",
    "$MSFT enterprise spending pullback hurting growth",
    "$NVDA valuation is stretched, priced for perfection",
    "$META user growth stalling in key demographics",
    "$JPM credit losses mounting, recession risk rising",
] * 15  # 보기 120개

texts = bullish_texts + bearish_texts
labels = [1] * len(bullish_texts) + [0] * len(bearish_texts)

# 뒤섞는다
indices = np.random.RandomState(42).permutation(len(texts))
texts = [texts[i] for i in indices]
labels = [labels[i] for i in indices]

print(f"  Dataset: {len(texts)} examples ({sum(labels)} bullish, {len(labels)-sum(labels)} bearish)")

# 토막내기(단순 부호화로 흉내냄)
# 실전에서는 BertTokenizer.from_pretrained("ProsusAI/finbert")를 쓴다
torch.manual_seed(42)

# 보이려고 단순한 낱말 곳간 바탕 부호화를 쓴다
# 그리고 작은 변환기 비슷한 갈래 매개를 익힌다
MAX_LEN = 64
VOCAB_SIZE = 5000
EMBED_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_CLASSES = 2


class SimpleFinancialClassifier(nn.Module):
    """글 가르기를 위한 작은 변환기 부호기.

    실전에서는 이를 다음으로 갈음한다:
        BertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=2)
    """

    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.embedding(input_ids) + self.pos_encoding(positions)

        if attention_mask is not None:
            # TransformerEncoder는 True를 무시할 자리로 본다
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # 덧대지 않은 토막에 대한 평균 모으기
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)


# 단순 토막내기(실전에서는 BertTokenizer를 쓴다)
def simple_tokenize(texts, max_len=64, vocab_size=5000):
    """보이기 위한 해시 바탕 토막내기."""
    all_ids, all_masks = [], []
    for text in texts:
        tokens = text.lower().split()[:max_len]
        ids = [(hash(t) % (vocab_size - 1)) + 1 for t in tokens]
        mask = [1] * len(ids)
        # 덧대기
        pad_len = max_len - len(ids)
        ids += [0] * pad_len
        mask += [0] * pad_len
        all_ids.append(ids)
        all_masks.append(mask)
    return torch.tensor(all_ids), torch.tensor(all_masks)


input_ids, attention_masks = simple_tokenize(texts, MAX_LEN, VOCAB_SIZE)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# 익힘/검증 나누기
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    range(len(texts)), test_size=0.15, random_state=42, stratify=labels
)

train_data = TensorDataset(input_ids[train_idx], attention_masks[train_idx], labels_tensor[train_idx])
val_data = TensorDataset(input_ids[val_idx], attention_masks[val_idx], labels_tensor[val_idx])

train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=32)
val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=32)

# 학습
model = SimpleFinancialClassifier(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_CLASSES, MAX_LEN)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

print("\n  Training:")
for epoch in range(8):
    model.train()
    total_loss = 0
    for batch in train_loader:
        b_ids, b_mask, b_labels = batch
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            b_ids, b_mask, b_labels = batch
            logits = model(b_ids, b_mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == b_labels).sum().item()
            total += len(b_labels)

    val_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 2 == 0:
        print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.3f}")


# =====================================================================
# 4부 – 값매김과 종목별 살피기
# =====================================================================
print("\n" + "=" * 60)
print("Part 4: Evaluation & Per-Stock Breakdown")
print("=" * 60)

# 본디 공책에서 자료에는 FB, AMZN, GOOGL 트윗이 들어 있다
# StockTwits API의 Bullish/Bearish 이름표와 함께
print("""
  실제 자료를 쓴 온전한 곱게 다듬기 물길:

  from transformers import (
      AutoModelForSequenceClassification, AutoTokenizer,
      Trainer, TrainingArguments,
  )
  from datasets import Dataset
  import pandas as pd

  # StockTwits 자료 읽어 들이기
  df = pd.read_csv("stocktwits_data.csv")
  # 칸: symbol, message, sentiment(Bullish/Bearish), message_id
  # 이름표 부호화
  df["label"] = (df["sentiment"] == "Bullish").astype(int)

  tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
  model = AutoModelForSequenceClassification.from_pretrained(
      "ProsusAI/finbert", num_labels=2
  )

  def tokenize(batch):
      return tokenizer(
          batch["message"], padding="max_length",
          truncation=True, max_length=128,
      )

  ds = Dataset.from_pandas(df[["message", "label"]])
  ds = ds.map(tokenize, batched=True)
  ds = ds.train_test_split(test_size=0.1, seed=42)

  args = TrainingArguments(
      output_dir="./finbert-stocktwits",
      num_train_epochs=4,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=64,
      learning_rate=2e-5,
      weight_decay=0.01,
      warmup_ratio=0.1,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
      fp16=True,
  )

  trainer = Trainer(
      model=model, args=args,
      train_dataset=ds["train"],
      eval_dataset=ds["test"],
      tokenizer=tokenizer,
  )
  trainer.train()

  # 종목별 값매김
  for symbol in ["FB", "AMZN", "GOOGL"]:
      subset = df[df["symbol"] == symbol]
      preds = trainer.predict(Dataset.from_pandas(subset))
      acc = (preds.predictions.argmax(-1) == subset["label"].values).mean()
      print(f"  {symbol}: {acc:.3f} accuracy ({len(subset)} samples)")
""")


# =====================================================================
# 5부 – 견줌: FinBERT, 일반 BERT, 낱말집
# =====================================================================
print("=" * 60)
print("Part 5: FinBERT vs Generic BERT vs Lexicon-Based")
print("=" * 60)

print("""
  금융 마음결 일에서의 견줌:

  ┌─────────────────────┬───────────┬──────────┬─────────────┐
  │ 방법                │ 정확도    │ F1 점수  │ 분야 맞음   │
  ├─────────────────────┼───────────┼──────────┼─────────────┤
  │ TextBlob(낱말집)    │   0.58    │   0.52   │ 나쁨        │
  │ VADER               │   0.62    │   0.57   │ 보통        │
  │ BERT-base           │   0.71    │   0.68   │ 일반        │
  │ FinBERT(영 발)      │   0.76    │   0.73   │ 좋음        │
  │ FinBERT(곱게 다듬음)│   0.83    │   0.81   │ 아주 좋음   │
  └─────────────────────┴───────────┴──────────┴─────────────┘

  FinBERT가 일반 모델을 앞서는 까닭:

  1. 분야 낱말: "bullish", "bearish", "overbought" 등
     일반 BERT가 담아내지 못하는 금융 특유의 뜻을 지닌다.

  2. 마음결의 방향: "Short position"은 일반 글에서는 가운데이지만
     금융 맥락에서는 흔히 약세이다.

  3. 부정 다루기: "Revenue did not meet expectations"는
     금융 보고서의 말을 이해한다.

  4. 수의 맥락: "EPS of $2.50 vs $2.30 expected" — FinBERT는
     어림을 웃도는 것이 양성임을 안다.

  금융 자연어 다루기의 쓰임새:
  - 실적 발표 마음결 좇기
  - 초과 수익을 위한 뉴스 마음결
  - 사회 그물(StockTwits/Reddit) 마음결 점수 매기기
  - SEC 보고서의 어조 살피기
  - 분석 보고서 갈래 매기기
""")

print("Done.")


if __name__ == "__main__":
    pass
```

## 2. 논의

`SimpleFinancialClassifier` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다. `forward` 메서드가 셈 그래프를 정하므로 익히는 동안 PyTorch의 자동 미분 체계가 기울기 셈을 알아서 다룬다. 이 단원별 꾸밈 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 끼워 넣기가 쉽다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
붙박이 첫자리매김일 때 `SimpleFinancialClassifier`의 배울 수 있는 매개변수 전체 개수를 셈하여라. 무게와 치우침을 모두 넣어 층별로 나누어 세어라.

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
층이나 덩이의 개수를 정할 수 있도록 `SimpleFinancialClassifier`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = SimpleFinancialClassifier(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — FinBERT 마음결

`SimpleFinancialClassifier` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다.

고갱이 갈래는 `SimpleFinancialClassifier`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
