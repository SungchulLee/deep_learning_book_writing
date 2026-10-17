# 제로샷 텍스트 분류

추가 과제별 학습 없이 NLI나 마스크 예측으로 레이블을 점수화하는 제로샷 분류와, 소수의 예시 임베딩으로 분류하는 퓨샷 방식을 살펴본다. 어느 방식도 새 분야에서 정확도를 보장하지 않으며 후보 레이블·프롬프트·평가 자료를 함께 설계해야 한다.

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

NLI 기반 파이프라인은 입력을 전제, 후보 레이블을 넣은 템플릿을 가설로 만든다. 일반적인 단일 레이블 모드는 후보들의 entailment 로짓끼리 정규화한다. multi_label=True는 후보마다 contradiction과 entailment 로짓의 두 값으로 정규화해 entailment 점수를 낸다. 단순히 각 entailment 로짓에 독립 시그모이드를 적용하는 것과 다르다. 구현 규약은 [제로샷 파이프라인 소스](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py)를 참고한다.

단일 레이블 점수는 후보 집합이 바뀌면 함께 달라지고, 다중 레이블 점수도 보정된 실제 정답 확률로 간주할 수 없다. 영어 MNLI 모델의 범위와 새로운 분야·언어에서의 성능을 구분한다. fill-mask 예제는 후보 단어의 마스크 토큰 확률을 이용하는 다른 방식이며 단어가 여러 서브워드가 되면 targets 처리도 확인해야 한다.

퓨샷 예제는 DistilBERT의 마스크 평균 임베딩과 코사인 유사도로 가장 가까운 예시를 고른다. 모델을 추가 학습하지 않는 것이지 레이블 예시를 쓰지 않는 제로샷 방식은 아니다. 기본 언어 모델의 평균 표현이 문장 유사도에 최적화되어 있다는 보장도 없다. no_grad와 별개로 평가 모드를 명시하고 영벡터의 노름·빈 입력을 처리한다. max_length=128 절단은 중요한 정보를 버릴 수 있다.

데이터 어그멘테이션은 원래 레이블이 유지되는지 확인해야 한다. 동의어라도 문맥·극성이 달라질 수 있고 부정어 삭제나 어순 변경은 정답을 바꿀 수 있다. 이 코드의 synonym_replace는 작은 사전에서 처음 일치하는 단어를 바꾸며, 반복 횟수만큼 서로 다른 단어를 바꾸는 구현은 아니다. 사본 수를 늘리는 것이 독립적인 정보량 증가나 정확도 개선을 뜻하지 않는다. 어그멘테이션은 분할 후 학습셋에 적용해 원문·변형이 평가 분할에 섞이지 않게 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 문장의 제로샷 점수 순위를 실행 전에 확정할 수 있는가? 후보 레이블과 템플릿의 영향을 설명하라.

</div>

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
    computer science나 physics가 관련 있을 것으로 예상할 수 있지만 실제 점수 순위는 체크포인트·템플릿·후보 표현에 따라 달라진다. 서로 겹치는 후보를 단일 레이블로 경쟁시키는 것이 원하는 과제인지 먼저 정해야 한다. 예제 한 문장은 성능 검증을 대신하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
아래 k-NN의 유효 입력, 동률 처리와 예시 수의 한계를 설명하라.

</div>

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
    1≤k≤예시 수를 확인하고 코사인 유사도의 0 노름을 처리해야 한다. Counter 동률은 예시의 순서에 영향을 받을 수 있어 명시적인 폴리시가 필요하다. 이 예시처럼 두 클래스에 각 두 개뿐이면 k=3의 다수결이 적어도 두 클래스의 표를 포함하지만, 데이터가 늘거나 클래스가 달라지면 같은 성질을 일반화할 수 없다. k의 효과는 독립 검증셋에서 비교한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 어그멘테이션 조각이 만드는 출력 수와 레이블 보존을 검토하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    original = "The big company reported good quarterly results"
    for i in range(5):
        print(f"  Synonym:   {synonym_replace(original)}")
        print(f"  Deletion:  {random_deletion(original, p=0.2)}")
        print(f"  Swap:      {random_swap(original, n=1)}")
        print()
    ```
    반복 다섯 번마다 세 방법의 결과를 각각 출력하므로 총 15개 출력이며, 세 방법을 연속 적용한 변형 다섯 개는 아니다. 결과가 중복되거나 원문과 같을 수도 있다. 동의어·삭제·자리 바꾸기 중 어느 방법도 의미 보존을 보장하지 않으므로 문맥과 정답을 확인해야 한다.

## 정리하며

제로샷 점수의 정규화 기준과 후보·템플릿 의존성을 이해하고, 퓨샷의 레이블 예시 활용과 구분한다. 평균 임베딩·최근접 이웃·어그멘테이션은 간단한 출발점이며 독립 평가와 레이블 보존 검토가 필요하다.
