# 영 예시 글 분류

이름표 달린 데이터가 모자라거나 없을 때에도 영 예시와 소수 예시 기법으로 쓸 만한 글 분류기를 만들 수 있다. 이 모듈은 자연어 추론에 바탕한 영 예시 분류에서 임베딩 기반 소수 예시 방법을 거쳐 분야에 맞춘 미세 조정에 이르는 흐름을 보여 주어, 자원이 적은 자연어 처리 상황을 위한 실용 연장을 준다.

## 1. 코드

```python
import numpy as np
import torch
import random
import re
from typing import List, Dict

# 1부: 자연어 추론으로 하는 영 예시 분류
try:
    from transformers import pipeline

    zero_shot_clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )

    text = ("The stock market rallied today after the Federal Reserve "
            "announced it would hold interest rates steady.")
    candidate_labels = ["finance", "politics", "sports", "technology", "science"]

    result = zero_shot_clf(text, candidate_labels)
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label:12s}: {score:.3f}")

    # 여러 이름표
    result_ml = zero_shot_clf(text, candidate_labels, multi_label=True)
    for label, score in zip(result_ml["labels"], result_ml["scores"]):
        print(f"  {label:12s}: {score:.3f}")

except (ImportError, OSError) as e:
    print(f"  Skipping: {e}")


# 2부: 가린 언어 모형 프롬프트
try:
    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    text = "The researchers discovered a new species of butterfly in the Amazon."
    prompt = f"{text} This text is about [MASK]."
    candidates = ["science", "sports", "finance", "nature"]
    output = fill_mask(prompt, targets=candidates)
    for item in output:
        print(f"  {item['token_str']:12s}: {item['score']:.4f}")
except (ImportError, OSError) as e:
    print(f"  Skipping: {e}")


# 3부: 임베딩 기반 소수 예시
try:
    from transformers import AutoTokenizer, AutoModel
    from numpy.linalg import norm

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def mean_pooling(model_output, attention_mask):
        token_embs = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        return torch.sum(token_embs * mask_expanded, 1) / torch.clamp(
            mask_expanded.sum(1), min=1e-9
        )

    def encode_texts(texts):
        inputs = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()

    few_shot_examples = [
        ("Apple reported record quarterly earnings of \\$83B.", "finance"),
        ("The Fed raised interest rates by 25 basis points.", "finance"),
        ("NVIDIA announced a new GPU architecture for AI.", "technology"),
        ("Python 3.12 brings significant performance improvements.", "technology"),
    ]
    texts = [t for t, _ in few_shot_examples]
    labels = [l for _, l in few_shot_examples]
    embeddings = encode_texts(texts)

    query = encode_texts(["Tesla stock surged 15% after strong delivery numbers."])
    sims = [np.dot(query[0], e) / (norm(query[0]) * norm(e)) for e in embeddings]
    best_idx = np.argmax(sims)
    print(f"  Predicted: {labels[best_idx]} (sim={sims[best_idx]:.3f})")

except (ImportError, OSError) as e:
    print(f"  Skipping: {e}")


# 4부: 데이터 불리기
def synonym_replace(text, n=1):
    synonyms = {
        "good": ["great", "excellent", "fine"],
        "bad": ["poor", "terrible", "awful"],
        "big": ["large", "huge", "massive"],
    }
    words = text.split()
    for _ in range(n):
        for i, w in enumerate(words):
            if w.lower() in synonyms:
                words[i] = random.choice(synonyms[w.lower()])
                break
    return " ".join(words)


def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) <= 1:
        return text
    remaining = [w for w in words if random.random() > p]
    return " ".join(remaining) if remaining else random.choice(words)


def random_swap(text, n=1):
    words = text.split()
    for _ in range(n):
        if len(words) >= 2:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
    return " ".join(words)
```

## 2. 논의

이름표 달린 데이터가 없을 때 자연어 추론을 쓰는 영 예시 분류가 가장 손쉬운 길이다. 전제-가설 쌍으로 학습한 자연어 추론 모형이 가설("This text is about finance")이 입력 글에서 따라 나오는지를 따진다. 그 함의 점수가 후보 이름표마다의 분류 확신도가 된다. 한 이름표 방식은 이름표들이 겨루도록 소프트맥스를 쓰고, 여러 이름표 방식은 시그모이드로 이름표마다 따로 점수를 매긴다.

임베딩 기반 소수 예시 분류는 부류마다 이름표 달린 예가 몇 개만 있어도 된다. 사전 학습된 모형으로 모든 글을 인코딩하고 질의 임베딩과 받침 임베딩을 코사인 비슷함으로 견주어, 가장 가까운 이웃을 바탕으로 새 글을 분류한다. 평균 풀링이 토큰 수준 표현을 문장 수준 임베딩으로 모은다. 이 방식은 빠르고 학습이 필요 없으며 더 튼튼한 예측을 위해 K-최근접 이웃으로 넓힐 수 있다.

비슷한말 바꾸기, 무작위 지우기, 무작위 자리바꿈 같은 데이터 불리기 기법은 작은 이름표 데이터셋을 인위로 넓힌다. 간단하지만 이 방법들로 실효 학습 자료를 두세 배로 늘릴 수 있다. 더 센 불리기가 필요하면 가린 언어 모형(이를테면 DistilBERT)을 쓰는 맥락 낱말 바꾸기가 문법 짜임과 뜻을 지키는 더 자연스러운 대치를 낸다.

## 연습문제

**연습문제 1.**
자연어 추론 영 예시 파이프라인으로 "The quantum computer solved the problem in 3 minutes"라는 문장을 후보 이름표 `["physics", "computer science", "mathematics", "engineering"]`에 대해 분류하라. 코드를 돌리기 전에 어느 이름표가 가장 높은 점수를 받을지 맞혀 보아라.

??? success "연습문제 1 풀이"
    ```python
    from transformers import pipeline
    zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = zs(
        "The quantum computer solved the problem in 3 minutes",
        candidate_labels=["physics", "computer science", "mathematics", "engineering"]
    )
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label}: {score:.3f}")
    ```
    "quantum computer"가 두 분야 모두와 관련되므로 대체로 "computer science"나 "physics"가 가장 높은 점수를 받는다. 자연어 추론 모형은 전제가 주어졌을 때 "This text is about computer science"의 함의를 따지는데, "quantum computer"와 컴퓨터 과학의 강한 연결이 대개 이긴다.

---

**연습문제 2.**
$k = 3$에 다수결을 쓰는 K-최근접 이웃 소수 예시 분류기를 구현하라. 주어진 소수 예시로 시험하고 $k$이 편향-분산 맞바꿈에 어떤 영향을 주는지 설명하라.

??? success "연습문제 2 풀이"
    ```python
    from collections import Counter

    def knn_classify(query_emb, support_embs, support_labels, k=3):
        sims = [np.dot(query_emb, e) / (norm(query_emb) * norm(e))
                for e in support_embs]
        top_k_idx = np.argsort(sims)[-k:]
        votes = [support_labels[i] for i in top_k_idx]
        return Counter(votes).most_common(1)[0][0]
    ```
    $k$이 작으면(이를테면 1) 예 하나하나의 잡음에 민감하다(분산이 크고 편향이 작다). $k$이 크면 예측이 매끄러워지지만 관련 없는 이웃이 낄 수 있다(분산이 작고 편향이 크다). 부류마다 예가 둘뿐일 때 $k = 3$이면 적어도 두 부류가 표에 참여하게 되어 어느 한 극단값의 영향이 준다.

---

**연습문제 3.**
세 가지 불리기 기법(비슷한말 바꾸기, 무작위 지우기, 무작위 자리바꿈)을 모두 적용하여 "The big company reported good quarterly results." 문장의 불린 판 다섯 개를 만들어라. 어느 불리기가 본디 뜻을 가장 잘 지키는지 논하라.

??? success "연습문제 3 풀이"
    ```python
    original = "The big company reported good quarterly results"
    for i in range(5):
        print(f"  Synonym:   {synonym_replace(original)}")
        print(f"  Deletion:  {random_deletion(original, p=0.2)}")
        print(f"  Swap:      {random_swap(original, n=1)}")
        print()
    ```
    비슷한말 바꾸기는 낱말을 가까운 말로 갈아 넣으므로("big"을 "large"로, "good"을 "great"로) 대체로 뜻을 가장 잘 지킨다. 무작위 지우기는 핵심 낱말을 없애 뜻을 바꿀 수 있다(이를테면 "good"이 빠지면 긍정의 느낌이 사라진다). 무작위 자리바꿈은 대체로 낱말 주머니로서의 뜻은 지키지만 문법에 맞지 않는 문장이 나올 수 있다.

## 정리하며

**다룬 것** — 영 예시 글 분류

이름표 달린 데이터가 없을 때 자연어 추론을 쓰는 영 예시 분류가 가장 손쉬운 길이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
