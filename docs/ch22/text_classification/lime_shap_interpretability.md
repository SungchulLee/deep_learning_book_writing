# LIME과 SHAP를 이용한 모델 해석

텍스트 분류의 설명은 특정 예측이 어떤 입력 피처에 의존하는지 살펴보는 도구다. [LIME](https://arxiv.org/abs/1602.04938)은 예측 주변의 교란 표본에 해석 가능한 대리모델을 맞춘다. [SHAP](https://arxiv.org/abs/1705.07874)은 선택한 피처 게임에서 기여도를 배분한다. 둘 다 모델이 옳거나 설명이 현실의 인과관계를 증명한다는 보장은 아니다.

## 1. 코드와 기존 출력

```python
"""LIME과 SHAP로 읽어 내기."""
# ---
# title: "자연어 모델 읽어 내기를 위한 LIME과 SHAP"
# description: "LIME으로 하는 낱낱의 풀이와 전체/낱낱의 풀이,
#               글 가르기 모델에 SHAP 쓰기"
# ---
#
# 금융 자연어 다루기에서 읽어 냄은 결정적이다. 규제 기관과 위험
# 살림꾼은 모델이 왜 어떤 글월을 알맞다고 보거나 마음결을 약세로
# 매기는지 이해해야 하기 때문이다.
#
#   1부 – 글 가르기를 위한 LIME(모델에 매이지 않음)
#   2부 – LinearExplainer로 하는 SHAP(로지스틱 되돌리기 / SVM용)
#   3부 – PyTorch 모델을 위한 SHAP DeepExplainer
#   4부 – 견줌: LIME과 SHAP
#   5부 – 금융 자연어 다루기의 읽어 내기 쓰임새
#
# 바탕: O'Reilly "Practical NLP" 4장

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)


# =====================================================================
# 1부 – 글 가르기를 위한 LIME
# =====================================================================
print("=" * 60)
print("Part 1: LIME (Local Interpretable Model-Agnostic Explanations)")
print("=" * 60)

# LIME은 들임을 흔들어 보며 낱낱의 어림을 풀이한다
# 어림이 어떻게 바뀌는지 살핀다.
#
# 글에서는 LIME이 낱말을 마구잡이로 없애 흔든 판을 여럿 만들고,
# 그 모두의 어림을 얻은 뒤 그 보기 둘레에서 분류기의 몸짓을
# 어림하는 자리에 매인 선형 모델을 맞춘다.
#
# 핵심 눈썰미: 선형 모델의 계수가 어떤 낱말이
# 클래스마다 어림을 어느 쪽으로 미는지 보여 준다.

# 보기 자료 뭉치 만들기
texts = [
    "Revenue grew 15% beating analyst expectations for the quarter",
    "The company reported strong earnings and raised full year guidance",
    "Profit margins expanded as cost cutting measures took effect",
    "Stock buyback program signals management confidence in growth",
    "Dividend increase reflects strong cash flow generation",
    "Revenue declined sharply missing consensus estimates by wide margin",
    "The company warned of deteriorating demand and lowered guidance",
    "Profit margins compressed due to rising input costs and competition",
    "Management announced restructuring and significant layoffs planned",
    "Debt levels are concerning as interest coverage ratio declined",
] * 20
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 20  # 1은 긍정, 0은 부정

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 단순한 TF-IDF + 로지스틱 되돌리기 물길 익히기
vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = LogisticRegression(class_weight="balanced", random_state=42)
clf.fit(X_train_tfidf, y_train)
print(f"  Classifier accuracy: {accuracy_score(y_test, clf.predict(X_test_tfidf)):.3f}")

# LIME 풀이
print("""
  # 글 풀이에 LIME 쓰기:
  from lime.lime_text import LimeTextExplainer
  from sklearn.pipeline import make_pipeline

  # 물길 만들기(벡터로 생성기 + 클래스 매개)
  pipe = make_pipeline(vectorizer, clf)

  # LIME 풀이개 첫자리매김
  class_names = ["negative", "positive"]
  explainer = LimeTextExplainer(class_names=class_names)

  # 어림 하나 풀이하기
  text = "Revenue declined sharply missing consensus estimates"
  exp = explainer.explain_instance(
      text,
      pipe.predict_proba,
      num_features=6,       # 가장 중요한 낱말 6개
      num_samples=1000,     # 흔든 횟수
  )

  # 풀이 보기
  print(exp.as_list())
  # → [('declined', -0.42),    # 부정 쪽으로 민다
  #    ('missing', -0.31),     # 음성 쪽으로 민다
  #    ('revenue', 0.08),      # 살짝 양성(아리송함)
  #    ('consensus', -0.12),   # 음성 쪽으로 민다
  #    ('sharply', -0.19),     # 음성 쪽으로 민다
  #    ('estimates', 0.05)]    # 가운데

  # 그려 보기(공책에서)
  exp.as_pyplot_figure()  # 가로 막대 그림
  exp.show_in_notebook()  # 주고받는 HTML
""")

# LIME 라이브러리 없이 개념 보이기
print("  Manual LIME-style explanation (without library):")
print("  " + "-" * 50)

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(vectorizer, clf)

sample_text = "Revenue declined sharply missing consensus estimates"
words = sample_text.split()
base_prob = pipe.predict_proba([sample_text])[0]

print(f"  Original prediction: neg={base_prob[0]:.3f}, pos={base_prob[1]:.3f}")
print(f"  Word importance (leave-one-out):")

importances = []
for i, word in enumerate(words):
    # 낱말을 하나씩 없애기
    perturbed = " ".join(w for j, w in enumerate(words) if j != i)
    perturbed_prob = pipe.predict_proba([perturbed])[0]
    # 이 낱말을 없애면 양성 확률이 얼마나 바뀌는가?
    delta = base_prob[1] - perturbed_prob[1]
    importances.append((word, delta))
    print(f"    {word:>15}: {delta:+.4f} (removing it changes P(pos) by this much)")

print()


# =====================================================================
# 2부 – LinearExplainer로 하는 SHAP
# =====================================================================
print("=" * 60)
print("Part 2: SHAP — LinearExplainer for Sklearn Models")
print("=" * 60)

# SHAP(섀플리 더하기 풀이)은 협력 게임 이론의 섀플리 값을
# 셈한다. 피처마다 공정한 몫을 받아
# 어림을
# 평균 어림에서 얼마나 밀어내는지를 나타낸다.
#
# LIME보다 나은 점:
#   - 이론으로 뒷받침됨(섀플리 값은 하나뿐이다)
#   - 전체 풀이와 낱낱의 풀이 모두
#   - 한결같음: 어떤 피처의 이바지가 늘면 그 SHAP 값도
#     결코 줄지 않는다

print("""
  import shap

  # 선형 모델에는 SHAP에 정확한 디코더가 있다
  explainer = shap.LinearExplainer(
      clf,
      X_train_tfidf,
      feature_perturbation="interventional",
  )
  shap_values = explainer.shap_values(X_test_tfidf)

  # 전체 간추림: 모든 어림에 걸쳐 어떤 피처가 가장 중요한가?
  shap.summary_plot(
      shap_values,
      X_test_tfidf.toarray(),
      feature_names=vectorizer.get_feature_names_out(),
  )
  # → 시험 뭉치에 걸친 피처마다의 SHAP 값 분포를 보여 준다
  # 빨간 점 = 피처 값이 큼, 파란 점 = 피처 값이 작음
  # x축 = SHAP 값(모델 내놓음에 미치는 영향)

  # 낱낱의 풀이: 이 보기가 왜 이렇게 클래스 매겨졌는가?
  shap.force_plot(
      explainer.expected_value,
      shap_values[0, :],
      X_test_tfidf[0, :].toarray(),
      feature_names=vectorizer.get_feature_names_out(),
  )
""")

# 모델 계수를 써서 손수 하는 SHAP 비슷한 살피기
print("  Top features by logistic regression coefficient (SHAP-adjacent):")
feature_names = vectorizer.get_feature_names_out()
coefs = clf.coef_[0]
top_positive = np.argsort(coefs)[-5:][::-1]
top_negative = np.argsort(coefs)[:5]

print("  Most positive (bullish indicators):")
for idx in top_positive:
    print(f"    {feature_names[idx]:>20}: {coefs[idx]:+.4f}")

print("  Most negative (bearish indicators):")
for idx in top_negative:
    print(f"    {feature_names[idx]:>20}: {coefs[idx]:+.4f}")
print()


# =====================================================================
# 3부 – PyTorch 모델을 위한 SHAP DeepExplainer
# =====================================================================
print("=" * 60)
print("Part 3: SHAP DeepExplainer for PyTorch Models")
print("=" * 60)

# 깊은 러닝 모델에는 SHAP이 (DeepLIFT에 바탕한) DeepExplainer를 쓴다
# 또는 (쌓은 그래디언트에 바탕한) GradientExplainer를 쓴다.


class TextLSTM(nn.Module):
    """글 가르기를 위한 단순 LSTM."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        return self.fc(h_n.squeeze(0))


# 단순한 낱말 곳간을 만들고 글을 부호화하기
all_words = set()
for t in texts:
    all_words.update(t.lower().split())
word2idx = {w: i + 1 for i, w in enumerate(sorted(all_words))}

MAX_LEN = 15


def encode_texts(text_list, max_len=MAX_LEN):
    encoded = []
    for t in text_list:
        ids = [word2idx.get(w.lower(), 0) for w in t.split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


X_encoded = encode_texts(texts)
y_tensor = torch.tensor(labels, dtype=torch.long)

# LSTM 익히기
model = TextLSTM(len(word2idx) + 1, embed_dim=32, hidden_dim=32, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(20):
    logits = model(X_encoded)
    loss = F.cross_entropy(logits, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(X_encoded).argmax(dim=-1)
    acc = (preds == y_tensor).float().mean()
print(f"  LSTM accuracy: {acc:.3f}")

print("""
  # PyTorch LSTM에 SHAP DeepExplainer 쓰기:
  import shap

  # 익힘 자료의 일부를 배경으로 쓰기
  background = X_encoded[:20]

  explainer = shap.DeepExplainer(model, background)

  # 시험 어림 풀이하기
  test_samples = X_encoded[180:185]
  shap_values = explainer.shap_values(test_samples)

  # shap_values는 목록이다(클래스마다 하나)
  # shap_values[1] = 양성 클래스 쪽 몫
  # 꼴: (num_samples, seq_len)

  # 그려 보려고 낱말로 되돌려 대응시키기
  idx2word = {v: k for k, v in word2idx.items()}
  sample_words = [
      [idx2word.get(idx.item(), "PAD") for idx in sample]
      for sample in test_samples
  ]

  shap.initjs()
  shap.force_plot(
      explainer.expected_value[1],
      shap_values[1][0],
      sample_words[0],
  )
""")
print()


# =====================================================================
# 4부 – 견줌: LIME과 SHAP
# =====================================================================
print("=" * 60)
print("Part 4: LIME vs SHAP Comparison")
print("=" * 60)

print("""
  ┌─────────────────┬──────────────────────┬──────────────────────┐
  │ 클래스            │ LIME                 │ SHAP                 │
  ├─────────────────┼──────────────────────┼──────────────────────┤
  │ 이론            │ 자리에 매인 선형 어림 │ 섀플리 값            │
  │ 범위            │ 자리만               │ 자리 + 전체          │
  │ 한결같음        │ 보장 없음            │ 수학으로 증명됨      │
  │ 빠르기          │ 더 빠름(표집)        │ 더 느림(정확)        │
  │ 모델 받치기     │ 아무 깜깜이 상자나   │ 다듬은 풀이개        │
  │ 든든함          │ 들쭉날쭉(마구잡이)   │ 늘 같음              │
  │ 더해짐          │ 아니다               │ 그렇다(값이 더해진다) │
  │ 알맞은 곳       │ 빠른 자리 눈썰미     │ 엄밀한 몫 나누기     │
  └─────────────────┴──────────────────────┴──────────────────────┘

  언제 무엇을 쓸까:
  - LIME: 빠른 벌레잡기, 모델에 매이지 않음, 아무 클래스 매개나
  - SHAP (Linear): 선형 모델에 정확한 값(빠르다)
  - SHAP (Deep):   뉴럴 넷트웍 몫 나누기
  - SHAP (Kernel): 아무 모델이나, 가장 비싸지만 유연하다
""")


# =====================================================================
# 5부 – 금융 자연어 다루기의 읽어 내기 쓰임새
# =====================================================================
print("=" * 60)
print("Part 5: Financial NLP Interpretability Use Cases")
print("=" * 60)

print("""
  1. 규제 지킴:
     - 어떤 낱말이 "높은 위험" 클래스를 일으켰는지 보이기
     - 자동 글월 가려내기의 감사 자취

  2. 마음결 바탕 거래:
     - 뉴스 글이 왜 약세로 클래스 매겨졌는지 이해하기
     - 참된 마음결과 잡음 낱말 가리기

  3. 신용 위험 재기:
     - 연차 보고서의 어떤 마디가 경고 신호를 냈는가?
     - 신용도 점수 매기기의 피처 몫 나누기

  4. 이상 알아채기:
     - 어떤 보고서가 왜 이상하다고 표시됐는지 풀이하기
     - 이상 점수를 끌어올리는 핵심 마디 가려내기

  5. 모델 벌레잡기:
     - 헛된 얽힘 찾기(보기로 회사 이름 → 마음결)
     - 피처 중요도로 자료 새어 나감 알아채기

  보기: FinBERT의 어림 풀이하기

    text = "Tesla missed delivery targets amid supply chain disruptions"
    # FinBERT의 어림: 음성 (0.89)
    # LIME 풀이:
    #   "missed"       → -0.35 (센 음성 신호)
    #   "disruptions"  → -0.22 (음성 맥락)
    #   "delivery"     → -0.08 (이 맥락에서는 살짝 음성)
    #   "targets"      → -0.05 ("missing"과 얽힘)
    #   "Tesla"        →  0.02 (가운데 — 좋다! 상표 치우침 없음)
""")

print("Done.")


if __name__ == "__main__":
    pass
```

**기존 출력:**

    ```
    ============================================================
    Part 1: LIME (Local Interpretable Model-Agnostic Explanations)
    ============================================================
      Classifier accuracy: 1.000

      # 글 풀이에 LIME 쓰기:
      from lime.lime_text import LimeTextExplainer
      from sklearn.pipeline import make_pipeline

      # 물길 만들기(벡터로 생성기 + 클래스 매개)
      pipe = make_pipeline(vectorizer, clf)

      # LIME 풀이개 첫자리매김
      class_names = ["negative", "positive"]
      explainer = LimeTextExplainer(class_names=class_names)

      # 어림 하나 풀이하기
      text = "Revenue declined sharply missing consensus estimates"
      exp = explainer.explain_instance(
          text,
          pipe.predict_proba,
          num_features=6,       # 가장 중요한 낱말 6개
          num_samples=1000,     # 흔든 횟수
      )

      # 풀이 보기
      print(exp.as_list())
      # → [('declined', -0.42),    # 부정 쪽으로 민다
      #    ('missing', -0.31),     # 음성 쪽으로 민다
      #    ('revenue', 0.08),      # 살짝 양성(아리송함)
      #    ('consensus', -0.12),   # 음성 쪽으로 민다
      #    ('sharply', -0.19),     # 음성 쪽으로 민다
      #    ('estimates', 0.05)]    # 가운데

      # 그려 보기(공책에서)
      exp.as_pyplot_figure()  # 가로 막대 그림
      exp.show_in_notebook()  # 주고받는 HTML

      Manual LIME-style explanation (without library):
      --------------------------------------------------
      Original prediction: neg=0.808, pos=0.192
      Word importance (leave-one-out):
                Revenue: +0.0228 (removing it changes P(pos) by this much)
               declined: -0.0556 (removing it changes P(pos) by this much)
                sharply: -0.0236 (removing it changes P(pos) by this much)
                missing: -0.0236 (removing it changes P(pos) by this much)
              consensus: -0.0236 (removing it changes P(pos) by this much)
              estimates: -0.0236 (removing it changes P(pos) by this much)

    ============================================================
    Part 2: SHAP — LinearExplainer for Sklearn Models
    ============================================================

      import shap

      # 선형 모델에는 SHAP에 정확한 디코더가 있다
      explainer = shap.LinearExplainer(
          clf,
          X_train_tfidf,
          feature_perturbation="interventional",
      )
      shap_values = explainer.shap_values(X_test_tfidf)

      # 전체 간추림: 모든 어림에 걸쳐 어떤 피처가 가장 중요한가?
      shap.summary_plot(
          shap_values,
          X_test_tfidf.toarray(),
          feature_names=vectorizer.get_feature_names_out(),
      )
      # → 시험 뭉치에 걸친 피처마다의 SHAP 값 분포를 보여 준다
      # 빨간 점 = 피처 값이 큼, 파란 점 = 피처 값이 작음
      # x축 = SHAP 값(모델 내놓음에 미치는 영향)

      # 낱낱의 풀이: 이 보기가 왜 이렇게 클래스 매겨졌는가?
      shap.force_plot(
          explainer.expected_value,
          shap_values[0, :],
          X_test_tfidf[0, :].toarray(),
          feature_names=vectorizer.get_feature_names_out(),
      )

      Top features by logistic regression coefficient (SHAP-adjacent):
      Most positive (bullish indicators):
                      strong: +1.2797
                        year: +0.9374
                      raised: +0.9374
                    earnings: +0.9374
                    reported: +0.9374
      Most negative (bearish indicators):
                    declined: -1.1870
               deteriorating: -1.0194
                      warned: -1.0194
                      demand: -1.0194
                     lowered: -1.0194

    ============================================================
    Part 3: SHAP DeepExplainer for PyTorch Models
    ============================================================
      LSTM accuracy: 0.700

      # PyTorch LSTM에 SHAP DeepExplainer 쓰기:
      import shap

      # 익힘 자료의 일부를 배경으로 쓰기
      background = X_encoded[:20]

      explainer = shap.DeepExplainer(model, background)

      # 시험 어림 풀이하기
      test_samples = X_encoded[180:185]
      shap_values = explainer.shap_values(test_samples)

      # shap_values는 목록이다(클래스마다 하나)
      # shap_values[1] = 양성 클래스 쪽 몫
      # 꼴: (num_samples, seq_len)

      # 그려 보려고 낱말로 되돌려 대응시키기
      idx2word = {v: k for k, v in word2idx.items()}
      sample_words = [
          [idx2word.get(idx.item(), "PAD") for idx in sample]
          for sample in test_samples
      ]

      shap.initjs()
      shap.force_plot(
          explainer.expected_value[1],
          shap_values[1][0],
          sample_words[0],
      )


    ============================================================
    Part 4: LIME vs SHAP Comparison
    ============================================================

      ┌─────────────────┬──────────────────────┬──────────────────────┐
      │ 클래스            │ LIME                 │ SHAP                 │
      ├─────────────────┼──────────────────────┼──────────────────────┤
      │ 이론            │ 자리에 매인 선형 어림 │ 섀플리 값            │
      │ 범위            │ 자리만               │ 자리 + 전체          │
      │ 한결같음        │ 보장 없음            │ 수학으로 증명됨      │
      │ 빠르기          │ 더 빠름(표집)        │ 더 느림(정확)        │
      │ 모델 받치기     │ 아무 깜깜이 상자나   │ 다듬은 풀이개        │
      │ 든든함          │ 들쭉날쭉(마구잡이)   │ 늘 같음              │
      │ 더해짐          │ 아니다               │ 그렇다(값이 더해진다) │
      │ 알맞은 곳       │ 빠른 자리 눈썰미     │ 엄밀한 몫 나누기     │
      └─────────────────┴──────────────────────┴──────────────────────┘

      언제 무엇을 쓸까:
      - LIME: 빠른 벌레잡기, 모델에 매이지 않음, 아무 클래스 매개나
      - SHAP (Linear): 선형 모델에 정확한 값(빠르다)
      - SHAP (Deep):   뉴럴 넷트웍 몫 나누기
      - SHAP (Kernel): 아무 모델이나, 가장 비싸지만 유연하다

    ============================================================
    Part 5: Financial NLP Interpretability Use Cases
    ============================================================

      1. 규제 지킴:
         - 어떤 낱말이 "높은 위험" 클래스를 일으켰는지 보이기
         - 자동 글월 가려내기의 감사 자취

      2. 마음결 바탕 거래:
         - 뉴스 글이 왜 약세로 클래스 매겨졌는지 이해하기
         - 참된 마음결과 잡음 낱말 가리기

      3. 신용 위험 재기:
         - 연차 보고서의 어떤 마디가 경고 신호를 냈는가?
         - 신용도 점수 매기기의 피처 몫 나누기

      4. 이상 알아채기:
         - 어떤 보고서가 왜 이상하다고 표시됐는지 풀이하기
         - 이상 점수를 끌어올리는 핵심 마디 가려내기

      5. 모델 벌레잡기:
         - 헛된 얽힘 찾기(보기로 회사 이름 → 마음결)
         - 피처 중요도로 자료 새어 나감 알아채기

      보기: FinBERT의 어림 풀이하기

        text = "Tesla missed delivery targets amid supply chain disruptions"
        # FinBERT의 어림: 음성 (0.89)
        # LIME 풀이:
        #   "missed"       → -0.35 (센 음성 신호)
        #   "disruptions"  → -0.22 (음성 맥락)
        #   "delivery"     → -0.08 (이 맥락에서는 살짝 음성)
        #   "targets"      → -0.05 ("missing"과 얽힘)
        #   "Tesla"        →  0.02 (가운데 — 좋다! 상표 치우침 없음)

    Done.
    ```

## 2. 실제로 계산하는 설명

첫 모델은 TF-IDF와 로지스틱 회귀다. LIME 라이브러리 호출은 안내 문자열로 출력하고 실제 코드는 단어를 하나씩 삭제한 뒤 positive 확률 차이를 계산한다. 이 leave-one-out 민감도는 국소 가중 대리모델을 학습하는 LIME 알고리즘과 같지 않다.

Δ_i=P(pos|원문)−P(pos|단어 i 삭제)이므로 양수면 삭제 후 positive 확률이 낮아졌다는 뜻이다. 삭제가 문법·의미를 바꾸거나 TF-IDF 전체 정규화를 바꾸기 때문에 원래 단어 하나의 독립적 인과 효과로 해석하지 않는다. 반복 단어의 위치를 삭제하는 규칙과 LIME의 피처 표현도 다를 수 있다.

훈련·시험 자료는 10개의 고유 문장을 반복한 뒤 항목별로 나누므로 같은 문장이 양쪽에 들어갈 수 있다. 해당 정확도는 새로운 자료에 대한 신뢰성의 근거가 아니다. 설명이 그럴듯하다는 이유로 이 평가 누출을 무시할 수도 없다.

## 3. 회귀 계수와 SHAP의 차이

실제 실행 부분이 출력하는 상위 양·음 피처는 로지스틱 회귀 계수다. 계수는 해당 피처 변화에 대한 log-odds의 방향·크기이며 특정 문서의 SHAP 값이나 확률 변화량과 같지 않다. 입력 값·배경 분포·출력 단위를 함께 정해야 개별 기여도를 해석할 수 있다.

[LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html)의 독립 피처 가정에서는 선형 출력의 기여도가 β_i(x_i−E[X_i])다. 로지스틱 회귀에서는 보통 log-odds 단위이며 확률에 그대로 더하는 값이 아니다. 상관을 고려하는 설정은 기여도 배분과 계산이 달라진다. 안내의 feature_perturbation 옵션과 반환 모양은 설치 버전의 API도 확인해야 한다.

SHAP의 가산성은 정의한 출력과 배경에 대한 성질이지 현실 인과관계의 보장이나 모든 근사 계산의 정확성 보장이 아니다. SHAP가 항상 결정적이고 느리며 LIME가 항상 빠르다는 비교도 맞지 않는다. 설명기·샘플 수·배경·난수와 모델에 따라 비용·안정성이 달라진다.

## 4. LSTM과 DeepExplainer 안내

TextLSTM은 단어 임베딩과 단층 LSTM의 마지막 상태를 분류한다. 어휘는 전체 자료에서 만들고 같은 전체 자료로 학습·정확도 계산을 수행하므로 출력은 학습 정확도다. PAD와 미등록 단어를 모두 ID 0으로 매핑하며 padding 길이를 LSTM에 전달하지 않아 마지막 상태가 padding을 처리한 영향을 받는다.

[DeepExplainer](https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html)는 미분 가능한 경로에 대한 근사 설명을 사용한다. 정수 단어 ID를 일반 연속 입력처럼 직접 미분할 수 없으므로 안내의 DeepExplainer(model, X_encoded)만으로 유효한 텍스트 설명이 완성된다고 볼 수 없다. 임베딩 레이어의 입력·출력에 대한 설명이나 텍스트 마스킹 등 설명 대상을 명확히 해야 한다. 지원 연산과 반환 차원도 확인해야 한다.

이 파일의 SHAP 호출·그림 코드는 모두 출력 문자열에 있으며 실제 SHAP 값을 계산하거나 시각화하지 않는다. 선택한 출력 클래스·로짓/확률·배경에 따라 설명이 달라지므로 ‘단어별 기여도’라는 말만으로 단위를 생략하지 않는다.

## 5. 설명의 검토와 활용

설명은 회사 이름 같은 우연한 단서나 누출을 찾는 데 도움을 줄 수 있다. 한 문장의 회사 이름 기여도가 작다는 사실만으로 전체 모델에 브랜드 편향이 없다고 결론 내릴 수는 없다. 여러 표본·반사실적 교체·교란 방식과 설명 안정성을 함께 살핀다.

설명과 모델 정확도·보정·공정성·업무 타당성은 각각 평가해야 한다. 안내의 FinBERT 확률과 단어별 숫자는 측정 결과가 아니며 실제로 실행한 모델도 FinBERT가 아니다. 이 편집에서는 학습·설명 호출·그림 생성을 실행하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
TextLSTM의 어휘 크기를 V, E=H=32, 클래스 수를 2라 할 때 파라메터 수를 구하라.

</div>

??? success "풀이"

    임베딩은 32V개, 단층 LSTM은 4HE+4H²+8H=8,448개, 분류 레이어는 H×2+2=66개다. 총 32V+8,514개이며 padding_idx의 행도 Parameter 크기에 포함된다. V는 실제 word2idx 항목 수에 PAD/UNK 공용 행 하나를 더한 모델 어휘 크기다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
단어 삭제 확률 차이를 LIME나 SHAP 값으로 부를 수 없는 이유를 설명하라.

</div>

??? success "풀이"

    삭제 전후의 모델 출력 차이만 계산하며 LIME의 여러 교란 표본·거리 가중 대리모델이나 SHAP의 부분집합 기여도 배분을 수행하지 않는다. 삭제는 문맥과 TF-IDF 정규화도 바꾸므로 단어 하나의 독립 효과나 현실 인과 효과로 단정하지 않는다. 설명 방식·출력 클래스·단위를 명시해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
계수·확률 변화·SHAP의 출력 단위와 배경 분포를 구분하라.

</div>

??? success "풀이"

    로지스틱 회귀 계수는 피처 한 단위 변화의 log-odds 방향과 크기다. 삭제 확률 차이는 해당 입력과 교란 규칙의 확률 민감도다. SHAP는 선택한 출력 함수와 배경에 대한 기여도이며 선형 log-odds 설명을 확률에 그대로 더할 수 없다. 배경·상관 가정이 달라지면 기여도도 달라질 수 있다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 예가 설명 방법을 구현하거나 정수 입력의 미분 문제를 해결하는지 검토하라.

</div>

??? success "풀이"

    YourBlock·dim은 자리표시자이고 TextLSTM 생성자에 num_layers도 없어 완성된 확장이 아니다. 깊이를 늘리는 것과 설명기를 연결하는 것은 별개다. 텍스트 마스킹 또는 임베딩 수준의 설명 대상을 정하고 지원 연산·출력·배경을 확인해야 하며, 모듈 등록만으로 정수 ID에 대한 미분이 가능해지지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

실제 코드의 단어 삭제 민감도·회귀 계수와 안내의 LIME·SHAP는 다르다. 설명의 대상·단위·배경·교란을 명시하고 데이터 누출·모델 정확도·설명 안정성을 따로 검토해야 한다.
