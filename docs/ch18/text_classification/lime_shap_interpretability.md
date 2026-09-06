# LIME과 SHAP로 읽어 내기

LIME과 SHAP로 읽어 내기.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 글 가르기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 코드

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
# 갈래마다 어림을 어느 쪽으로 미는지 보여 준다.

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

  # 물길 만들기(벡터로 만들개 + 갈래 매개)
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
  # → [('declined', -0.42),    # pushes toward negative
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
# 셈한다. 특징마다 공정한 몫을 받아
# 어림을
# 평균 어림에서 얼마나 밀어내는지를 나타낸다.
#
# LIME보다 나은 점:
#   - 이론으로 뒷받침됨(섀플리 값은 하나뿐이다)
#   - 전체 풀이와 낱낱의 풀이 모두
#   - 한결같음: 어떤 특징의 이바지가 늘면 그 SHAP 값도
#     결코 줄지 않는다

print("""
  import shap

  # 선형 모델에는 SHAP에 정확한 풀개가 있다
  explainer = shap.LinearExplainer(
      clf,
      X_train_tfidf,
      feature_perturbation="interventional",
  )
  shap_values = explainer.shap_values(X_test_tfidf)

  # 전체 간추림: 모든 어림에 걸쳐 어떤 특징이 가장 중요한가?
  shap.summary_plot(
      shap_values,
      X_test_tfidf.toarray(),
      feature_names=vectorizer.get_feature_names_out(),
  )
  # → 시험 뭉치에 걸친 특징마다의 SHAP 값 분포를 보여 준다
  # 빨간 점 = 특징 값이 큼, 파란 점 = 특징 값이 작음
  # x축 = SHAP 값(모델 내놓음에 미치는 영향)

  # 낱낱의 풀이: 이 보기가 왜 이렇게 갈래 매겨졌는가?
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

# 깊은 배움 모델에는 SHAP이 (DeepLIFT에 바탕한) DeepExplainer를 쓴다
# 또는 (쌓은 기울기에 바탕한) GradientExplainer를 쓴다.


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

  # shap_values는 목록이다(갈래마다 하나)
  # shap_values[1] = 양성 갈래 쪽 몫
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
  │ 갈래            │ LIME                 │ SHAP                 │
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
  - LIME: 빠른 벌레잡기, 모델에 매이지 않음, 아무 갈래 매개나
  - SHAP (Linear): 선형 모델에 정확한 값(빠르다)
  - SHAP (Deep):   신경망 몫 나누기
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
     - 어떤 낱말이 "높은 위험" 갈래를 일으켰는지 보이기
     - 자동 글월 가려내기의 감사 자취

  2. 마음결 바탕 거래:
     - 뉴스 글이 왜 약세로 갈래 매겨졌는지 이해하기
     - 참된 마음결과 잡음 낱말 가리기

  3. 신용 위험 재기:
     - 연차 보고서의 어떤 마디가 경고 신호를 냈는가?
     - 신용도 점수 매기기의 특징 몫 나누기

  4. 이상 알아채기:
     - 어떤 보고서가 왜 이상하다고 표시됐는지 풀이하기
     - 이상 점수를 끌어올리는 핵심 마디 가려내기

  5. 모델 벌레잡기:
     - 헛된 얽힘 찾기(보기로 회사 이름 → 마음결)
     - 특징 중요도로 자료 새어 나감 알아채기

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

## 논의

`TextLSTM` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다. `forward` 메서드가 셈 그래프를 정하므로 익히는 동안 PyTorch의 자동 미분 체계가 기울기 셈을 알아서 다룬다. 이 단원별 꾸밈 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 끼워 넣기가 쉽다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
붙박이 첫자리매김일 때 `TextLSTM`의 배울 수 있는 매개변수 전체 개수를 셈하여라. 무게와 치우침을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `TextLSTM`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = TextLSTM(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
