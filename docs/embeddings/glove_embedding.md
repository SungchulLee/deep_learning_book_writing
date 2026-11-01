# GloVe 임베딩

사전 학습한 GloVe 벡터를 불러와 단어 유사도, 최근접 이웃과 벡터 유추를 계산한다. 이 예제는 GloVe를 새로 학습하지 않으며 순환 뉴럴 넷트웍도 사용하지 않는다. 단어 사전과 벡터의 대응, 검색 후보와 어휘 밖 단어의 처리를 중심으로 살펴본다.

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

## 3. 가져다 쓸 때 걸리는 것

위 코드는 GloVe를 **읽어 보기만** 한다. 실제로 모델에 얹으려면 더 따질 것이 있다.

사전 학습 벡터를 사용할 때는 현재 어휘의 각 인덱스에 맞춰 행을 복사해야 한다. 벡터 파일의 행 순서와 현재 어휘의 인덱스가 같다고 가정하면 잘못된 단어에 벡터가 연결된다. 임베딩 차원, 토큰화와 대소문자 처리도 맞춰야 한다.

어휘 밖(OOV) 단어에는 무작위 벡터나 공통 미등록 단어 벡터를 줄 수 있다. 모든 사전 학습 벡터의 평균을 쓰면 처음에는 OOV 단어들의 표현이 같아진다. 이를 고정하면 그 단어들을 구분할 수 없다. FastText의 하위 단어 벡터를 쓰는 방법은 해당 모델과 하위 단어 정보가 필요하며, GloVe 파일만 읽어서 같은 처리를 할 수는 없다.

웨이트를 고정하면 갱신할 파라메터를 줄이고 초기 표현을 유지할 수 있지만, 나머지 모델의 과적합까지 막지는 않는다. 미세 조정은 과제에 적응할 여지를 주는 대신 러닝레이트와 데이터 규모에 따라 표현이 훼손될 수 있다. 고정 학습 후 일부 또는 전체를 해제하는 방식도 검증 성능으로 선택한다. 옵티마이저를 만들 때 고정 파라메터를 제외했다면 해제 후 갱신 대상에 추가해야 한다.

GloVe는 단어 동시 출현 횟수의 로그를 내적과 바이어스로 설명하는 가중 최소제곱 목적 함수를 사용한다. 단순한 행렬 분해나 모든 단어에 성립하는 의미 연산으로 이해해서는 안 된다. 벡터 유추의 성공 여부는 단어와 학습 데이터에 따라 다르며, 사전 학습 벡터에는 데이터의 바이어스도 반영될 수 있다. 원래 벡터와 설명은 [GloVe 프로젝트](https://nlp.stanford.edu/projects/glove/)에서 확인할 수 있다.

실제로 얹어서 **재어 보는 것**은 [7.3절 「빌려 온 벡터」](../imdb/03_frozen.md)가 한다. 거기서는 IMDB 사다리 전체를 GloVe로 다시 오르며 얼린 것과 미세 조정한 것을 견주는데, **부호가 뒤집히는 자리**가 나온다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
find_closest_words와 analogy의 검색 후보가 어떻게 다른지 설명하라.

</div>

??? success "풀이"
    일반 이웃 검색은 어휘 전체를 후보로 하므로 질의 단어 자신도 포함될 수 있다. 유추 함수는 상위 후보를 더 많이 가져온 뒤 입력 단어 a, b, c를 제외한다. 두 함수의 결과 개수를 비교할 때 제외 규칙을 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
질의 벡터, 단어 입력과 top_k에 필요한 검사를 설명하라.

</div>

??? success "풀이"
    질의 벡터는 임베딩 차원 $d$와 같은 길이의 유한한 실수 벡터여야 한다. 영벡터는 별도로 처리한다. 단어는 사전에 있는지 확인한다. top_k는 유효한 정수여야 하고 실제 topk 호출의 후보 수가 어휘 크기를 넘지 않아야 한다. 유추 함수는 제외 단어 수만큼 더 요청하므로 이 조건도 반영한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
good과 bad의 코사인 유사도가 높으면 임베딩 학습이 잘못된 것인가?

</div>

??? success "풀이"
    반대 의미의 단어도 비슷한 문맥에 쓰이므로 가까운 벡터를 가질 수 있다. 코사인 유사도는 감성의 동일성을 뜻하지 않는다. 과제에 필요한 구별은 해당 과제의 검증 지표로 평가하고, 예시 몇 쌍으로 전체 품질을 판정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

GloVe 벡터의 이웃 검색과 유추는 학습된 관계를 탐색하는 도구다. 자기 자신과 입력 단어의 제외 규칙, OOV 처리와 검색 범위를 명확히 하고 결과를 확률이나 보편적인 의미 등식으로 해석하지 않는다.
