# GloVe 임베딩

사전 학습한 GloVe 벡터를 불러와 단어 유사도, 최근접 이웃과 벡터 유추를 계산한다. 이 예제는 GloVe를 새로 학습하지 않으며 순환 신경망도 사용하지 않는다. 단어 사전과 벡터의 대응, 검색 후보와 어휘 밖 단어의 처리를 중심으로 살펴본다.

## 1. 코드

```python
"""
GloVe 임베딩 — 사전 학습 벡터 불러오기와 질의하기
==========================================================

다음을 보인다:
    1. torchtext로 사전 학습 GloVe 임베딩 불러오기
    2. 낱말 벡터 꺼내기
    3. 낱말 사이의 코사인 유사도 계산하기
    4. 가장 가까운 이웃 찾기
    5. 낱말 유추 과제 풀기 (king - man + woman ≈ queen)

필요한 것:
    pip install torchtext torch

Note:
    처음 돌리면 GloVe 벡터를 내려받는다 (6B/100d는 약 862MB).
"""

import torch
import torchtext.vocab as vocab

# ========================================================================
# 메인
# ========================================================================


def load_glove(name: str = "6B", dim: int = 100) -> vocab.GloVe:
    """사전 학습 GloVe 벡터를 불러온다."""
    print(f"Loading GloVe (name={name}, dim={dim})...")
    glove = vocab.GloVe(name=name, dim=dim)
    print(f"  Loaded {len(glove.itos)} word vectors of dimension {dim}.")
    return glove


def find_closest_words(
    word_vector: torch.Tensor, glove: vocab.GloVe, top_k: int = 5
) -> list[tuple[str, float]]:
    """코사인 유사도로 가장 가까운 낱말 k개를 돌려준다."""
    sims = torch.nn.functional.cosine_similarity(
        word_vector.unsqueeze(0), glove.vectors, dim=1
    )
    values, indices = torch.topk(sims, k=top_k)
    return [(glove.itos[idx], val.item()) for idx, val in zip(indices, values)]


def analogy(
    a: str, b: str, c: str, glove: vocab.GloVe, top_k: int = 5
) -> list[tuple[str, float]]:
    """
    푼다: a와 b의 관계는 c와 무엇의 관계와 같은가?
    벡터 산술: 결과 ≈ b - a + c

    예: analogy("man", "king", "woman") ≈ "queen"
    """
    vec = glove[b] - glove[a] + glove[c]
    # 결과에서 입력 낱말 빼기
    exclude = {a, b, c}
    results = find_closest_words(vec, glove, top_k=top_k + len(exclude))
    return [(w, s) for w, s in results if w not in exclude][:top_k]


def main():
    glove = load_glove(name="6B", dim=100)

    # --- 낱말 벡터 ---
    print("\n--- Word vector for 'hello' ---")
    hello_vec = glove["hello"]
    print(f"  Shape: {hello_vec.shape}")
    print(f"  First 5 dims: {hello_vec[:5].tolist()}")

    # --- 가장 가까운 이웃 ---
    print("\n--- Words most similar to 'hello' ---")
    for word, sim in find_closest_words(glove["hello"], glove, top_k=5):
        print(f"  {word:15s}  cosine={sim:.4f}")

    # --- 쌍마다의 유사도 ---
    print("\n--- Pairwise cosine similarities ---")
    pairs = [("king", "queen"), ("king", "apple"), ("cat", "dog"), ("good", "bad")]
    for w1, w2 in pairs:
        sim = torch.nn.functional.cosine_similarity(
            glove[w1].unsqueeze(0), glove[w2].unsqueeze(0)
        ).item()
        print(f"  sim({w1}, {w2}) = {sim:.4f}")

    # --- 유추 ---
    print("\n--- Analogy: man → king, woman → ? ---")
    for word, sim in analogy("man", "king", "woman", glove, top_k=5):
        print(f"  {word:15s}  cosine={sim:.4f}")

    print("\n--- Analogy: paris → france, tokyo → ? ---")
    for word, sim in analogy("paris", "france", "tokyo", glove, top_k=5):
        print(f"  {word:15s}  cosine={sim:.4f}")


if __name__ == "__main__":
    main()
```

## 2. 논의

`load_glove`는 TorchText의 GloVe 로더로 벡터를 가져온다. 처음 사용할 때는 데이터 다운로드가 필요할 수 있으며, 캐시와 네트워크 환경에 따라 동작이 달라진다. [TorchText 0.18 릴리스 안내](https://github.com/pytorch/text/releases/tag/v0.18.0)는 개발 중단을 명시하므로, 현재 PyTorch 환경과의 호환성을 확인해야 한다.

`find_closest_words`는 입력 벡터와 모든 단어 벡터의 코사인 유사도를 계산한 뒤 상위 $k$개를 반환한다. 같은 단어의 벡터를 질의하면 그 단어 자신도 후보에 포함된다. 따라서 main의 hello 검색은 자기 자신을 제외한 이웃 다섯 개가 아니다. 유사도 값은 정답 확률도, 의미가 같은 정도의 직접적인 척도도 아니다.

`analogy`는 $v_b-v_a+v_c$와 가까운 후보를 찾고 입력 단어 세 개를 제외한다. 이는 유추를 평가하는 한 방식이지 정확한 의미 등식이 아니다. 결과는 학습 데이터와 벡터 파일, 평가 방식에 따라 달라지므로 특정 정답이 반드시 1위라고 단정할 수 없다.

기본 로더에서 어휘 밖 단어가 영벡터로 처리될 수 있으므로 조회 전에 사전 포함 여부를 확인해야 한다. 영벡터의 코사인 점수는 의미 있는 관계가 아니며 임의의 동점 후보가 나올 수 있다. 이 코드는 입력 차원, 유한한 값과 top_k 범위를 검증하지 않는다. 벡터가 $(V,d)$라면 질의는 길이 $d$의 벡터여야 하고 검색 후보 수는 $V$를 넘지 않아야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
find_closest_words와 analogy의 검색 후보가 어떻게 다른지 설명하라.

</div>

??? success "연습문제 1 풀이"
    일반 이웃 검색은 어휘 전체를 후보로 하므로 질의 단어 자신도 포함될 수 있다. 유추 함수는 상위 후보를 더 많이 가져온 뒤 입력 단어 a, b, c를 제외한다. 두 함수의 결과 개수를 비교할 때 제외 규칙을 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
질의 벡터, 단어 입력과 top_k에 필요한 검사를 설명하라.

</div>

??? success "연습문제 2 풀이"
    질의 벡터는 임베딩 차원 $d$와 같은 길이의 유한한 실수 벡터여야 한다. 영벡터는 별도로 처리한다. 단어는 사전에 있는지 확인한다. top_k는 유효한 정수여야 하고 실제 topk 호출의 후보 수가 어휘 크기를 넘지 않아야 한다. 유추 함수는 제외 단어 수만큼 더 요청하므로 이 조건도 반영한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
good과 bad의 코사인 유사도가 높으면 임베딩 학습이 잘못된 것인가?

</div>

??? success "연습문제 3 풀이"
    반대 의미의 단어도 비슷한 문맥에 쓰이므로 가까운 벡터를 가질 수 있다. 코사인 유사도는 감성의 동일성을 뜻하지 않는다. 과제에 필요한 구별은 해당 과제의 검증 지표로 평가하고, 예시 몇 쌍으로 전체 품질을 판정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_glove embedding():
        model = Glove Embedding(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 손실이나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

GloVe 벡터의 이웃 검색과 유추는 학습된 관계를 탐색하는 도구다. 자기 자신과 입력 단어의 제외 규칙, OOV 처리와 검색 범위를 명확히 하고 결과를 확률이나 보편적인 의미 등식으로 해석하지 않는다.
