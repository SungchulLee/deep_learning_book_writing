# FinBERT와 금융 텍스트 감성 분류

[ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert)는 금융 텍스트로 추가 사전 학습하고 감성 분류에 맞춘 BERT 기반 모델이다. 이 페이지에는 그 사용 안내와 별도로, 작은 무작위 초기화 트랜스포머를 합성 문장에 학습하는 실제 코드가 들어 있다. 두 모델과 각각의 레이블·평가 범위를 구분해서 읽는다.

## 1. 코드와 기존 출력

```python
"""FinBERT 마음결."""
# ---
# title: "FinBERT: 금융 마음결 살피기"
# description: "주식 마음결 클래스 매기기를 위해 FinBERT를 곱게 다듬기,
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
#   4부 – 밸류와 종목별 살피기
#   5부 – FinBERT, 일반 BERT, 낱말집 바탕 견주기
#
# 바탕: O'Reilly "Practical NLP" 10장

import zlib          # 어디서 돌려도 같은 값을 주는 해시 (아래 토막내기에 쓴다)
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

  # 배치 어림
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
#   2. 어텐션 마스크를 만든다
#   3. 익힘/검증으로 나눈다
#   4. AdamW와 한 줄 몸풀기로 곱게 다듬는다
#   5. 세대마다 밸류한다

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
# 그리고 작은 변환기 비슷한 클래스 매개를 익힌다
MAX_LEN = 64
VOCAB_SIZE = 5000
EMBED_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_CLASSES = 2


class SimpleFinancialClassifier(nn.Module):
    """글 가르기를 위한 작은 변환기 인코더.

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
        # 붙박이 hash() 를 쓰면 안 된다. 파이썬은 글자열 해시에 프로세스마다
        # 다른 소금을 섞으므로, 씨앗을 아무리 고정해도 토막마다 붙는 번호가
        # 돌릴 때마다 달라진다 — 여기 실린 로스값이 다시 나오지 않는 까닭이
        # 그것이었다. crc32 는 어디서 돌려도 같은 값을 준다
        ids = [(zlib.crc32(t.encode()) % (vocab_size - 1)) + 1 for t in tokens]
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
# 4부 – 밸류와 종목별 살피기
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

  # 종목별 밸류
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
  - 사회 넷트웍(StockTwits/Reddit) 마음결 점수 매기기
  - SEC 보고서의 어조 살피기
  - 분석 보고서 클래스 매기기
""")

print("Done.")


if __name__ == "__main__":
    pass
```

**기존 출력:**

    ```
    ============================================================
    Part 1: Loading FinBERT
    ============================================================

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

    ============================================================
    Part 2: Zero-Shot Inference via HuggingFace Pipeline
    ============================================================

      from transformers import pipeline

      finbert = pipeline(
          "sentiment-analysis",
          model="ProsusAI/finbert",
          tokenizer="ProsusAI/finbert",
      )

      # 어림 하나
      result = finbert("AAPL missed revenue estimates by 2%.")
      # → [{'label': 'negative', 'score': 0.87}]

      # 배치 어림
      texts = [
          "Revenue grew 15% year-over-year, exceeding guidance.",
          "The company announced layoffs affecting 10% of staff.",
          "Q3 earnings were in line with consensus expectations.",
          "Short sellers are increasing their positions in the stock.",
      ]
      results = finbert(texts)
      for text, res in zip(texts, results):
          print(f"  {res['label']:>8} ({res['score']:.2f}): {text[:60]}")

    ============================================================
    Part 3: Fine-Tuning FinBERT on StockTwits Data
    ============================================================
      Creating synthetic StockTwits-like data for demonstration...
      Dataset: 240 examples (120 bullish, 120 bearish)

      Training:
        Epoch 2: loss=0.5276, val_acc=1.000
        Epoch 4: loss=0.2557, val_acc=1.000
        Epoch 6: loss=0.0389, val_acc=1.000
        Epoch 8: loss=0.0058, val_acc=1.000

    ============================================================
    Part 4: Evaluation & Per-Stock Breakdown
    ============================================================

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

      # 종목별 밸류
      for symbol in ["FB", "AMZN", "GOOGL"]:
          subset = df[df["symbol"] == symbol]
          preds = trainer.predict(Dataset.from_pandas(subset))
          acc = (preds.predictions.argmax(-1) == subset["label"].values).mean()
          print(f"  {symbol}: {acc:.3f} accuracy ({len(subset)} samples)")

    ============================================================
    Part 5: FinBERT vs Generic BERT vs Lexicon-Based
    ============================================================

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
      - 사회 넷트웍(StockTwits/Reddit) 마음결 점수 매기기
      - SEC 보고서의 어조 살피기
      - 분석 보고서 클래스 매기기

    Done.
    ```

## 2. 실제 FinBERT와 안내 문자열

첫 부분의 from_pretrained·pipeline과 뒤의 실제 자료 파인튜닝은 print 문자열 안에 있다. 이 파일의 실행 코드가 FinBERT를 다운로드하거나 파인튜닝하지는 않는다. 안내의 예측 확률과 모델 비교 표도 실행해 측정한 결과가 아니다.

ProsusAI 체크포인트의 레이블은 0=positive, 1=negative, 2=neutral이다. 본문의 합성 이진 분류 레이블은 0=bearish, 1=bullish로 서로 다르다. 미리 감성 레이블로 학습된 모델을 새 자료에 추가 학습 없이 적용하는 것을 이 코드에서 ‘zero-shot’이라고 부르지만, 사전의 감성 지도 학습 자체가 없었다는 뜻은 아니다.

감성은 텍스트의 태도·표현에 대한 분류다. softmax 최댓값이 실제 정답 확률로 보정되어 있거나 주가 상승 확률이라는 보장은 없다. 금융 용어·수치 비교·부정의 이해 정도는 독립 자료에서 확인해야 하며 모든 문맥에서 일반 모델을 능가한다고 단정하지 않는다.

## 3. 실제 학습하는 작은 모델

SimpleFinancialClassifier는 CRC32 해시 단어 ID와 학습 가능한 위치 임베딩, 두 트랜스포머 인코더 레이어, 마스크 평균과 이진 head를 사용한다. 사전 학습 웨이트를 가져오지 않는 모델이며 FinBERT의 축약 체크포인트가 아니다. 해시 충돌로 서로 다른 단어가 같은 ID를 가질 수 있고 문장 길이는 64로 잘린다.

padding_idx=0인 임베딩과 src_key_padding_mask, 평균의 attention_mask는 각각 입력·어텐션·pooling에서 다른 역할을 한다. 위치 ID는 max_len 범위 안이어야 하며 모든 토큰이 padding인 입력은 평균 분모를 clamp하는 것만으로 encoder의 전부 마스킹된 계산까지 안전해지지 않는다.

학습은 AdamW와 클리핑으로 8에포크 수행하도록 작성되어 있다. 출력 로스는 배치 평균을 다시 평균한 값으로 마지막 배치 크기에 따라 전체 표본 가중 평균과 다를 수 있다. 이 편집에서는 해당 학습이나 출력을 다시 실행하지 않았다.

## 4. 중복 문장과 검증의 한계

합성 자료는 긍정 8문장·부정 8문장을 각각 15번 반복한 240개다. 반복 항목을 무작위로 분할하므로 같은 문장이 학습·검증에 걸쳐 들어가는 누출이 생길 수 있다. 높은 정확도가 새로운 기업·표현·시기의 금융 텍스트로 일반화한다는 근거는 아니다.

원문·재게시·유사 문장의 그룹을 분리하고 시간·출처·주석 기준을 고려해 검증·시험 자료를 구성한다. 정확도 외에 유형별 정밀도·재현율·F1과 중립·부정·숫자·복합 감성의 오류를 확인한다. 시기별 평가에는 공개된 당시의 정보만 사용해야 한다.

## 5. 실제 자료로 확장할 때

안내의 3클래스 체크포인트를 num_labels=2로 가져오는 예는 출력 head의 크기 불일치를 자동으로 해결하는 완성 코드가 아니다. 새 head 초기화와 올바른 label2id·id2label이 필요하고 라이브러리 버전의 인터페이스도 맞춰야 한다. 종목별 predict에 원래 데이터프레임을 바로 넘기면 필요한 토큰화 입력이 없으며 학습에 쓴 문장까지 평가에 포함될 수 있다.

출력 표의 TextBlob·VADER·BERT·FinBERT 고정 정확도와 F1은 비교 실험 결과로 취급하지 않는다. 같은 데이터 분할·레이블·전처리·튜닝 예산에서 실제 측정해야 한다. 감성 분류 성능과 거래 성과의 평가는 별개다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
본문 설정의 SimpleFinancialClassifier 파라메터 수를 계산하고 FinBERT와 구분하라.

</div>

??? success "풀이"

    단어 임베딩 5,000×128=640,000개, 위치 임베딩 64×128=8,192개다. encoder 레이어마다 어텐션 66,048개, FFN 65,920개, 두 LayerNorm 512개로 132,480개이며 두 레이어는 264,960개다. 분류 head 128×2+2=258개를 더해 총 913,410개다. 무작위 초기화 모델이며 사전 학습한 FinBERT의 파라메터 수가 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
반복한 합성 문장을 항목 단위로 무작위 분할하는 검증의 한계를 설명하라.

</div>

??? success "풀이"

    16개의 고유 문장을 반복한 자료라 동일 문자열이 학습·검증 양쪽에 나타날 수 있다. 모델이 문장을 외운 결과와 새로운 문맥의 일반화를 구분할 수 없다. 원문·유사 문장 그룹을 나누고 새로운 시기·출처의 독립 시험 자료로 평가해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
FinBERT의 3클래스 레이블과 합성 이진 레이블이 다른 이유 및 head 변경 시 필요한 처리를 설명하라.

</div>

??? success "풀이"

    체크포인트는 0=positive, 1=negative, 2=neutral이고 합성 모델은 0=bearish, 1=bullish다. 번호를 그대로 옮기면 의미가 뒤집히거나 중립 처리가 사라진다. 이진 과제로 바꿀 때는 출력 크기·새 head 학습·레이블 매핑과 중립 자료의 포함/제외 폴리시를 명시해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 자리표시자가 모델 깊이를 바꾸는 완성 코드인지 검토하라.

</div>

??? success "풀이"

    YourBlock·dim은 정의되지 않았고 현재 생성자는 num_layers를 받지 않는다. 실제 깊이는 TransformerEncoder 생성의 num_layers=2에 고정되어 있다. 깊이를 인자로 노출하면 연결·초기화·학습 설정과 파라메터 증가를 함께 검토해야 한다. ModuleList는 모듈 등록 도구일 뿐 그 자체로 forward나 파인튜닝을 완성하지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

FinBERT 사용 안내와 합성 자료의 작은 트랜스포머 학습은 다르다. 레이블·체크포인트 head·중복 누출을 확인하고 같은 조건의 독립 평가로 분류 성능을 판단해야 한다.
