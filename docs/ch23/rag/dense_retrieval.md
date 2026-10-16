# 밀집 검색

밀집 검색(dense retrieval)은 질문과 문서를 벡터로 표현하고 유사한 후보를 찾는 방식이다. [DPR](https://arxiv.org/abs/2004.04906)은 개방형 질의응답에서 질문·문서 인코더를 사용하는 대표 연구다. 검색 결과는 모델의 관련성 추정이며 정답·사실·최신성을 보장하지 않는다. RAG에서는 그 결과를 생성의 근거로 사용하는 별도 단계가 이어진다.

## 1. 두 인코더와 유사도

질문 q와 문서 d에 s(q,d)=sim(E_q(q),E_d(d))를 계산한다. 두 인코더는 별도 웨이트일 수도, 공유 웨이트에 역할별 입력 형식을 쓸 수도 있다. 문서를 미리 인코딩할 수 있어 많은 후보 검색에 유리하지만 encoder·토크나이저·정규화가 바뀌면 색인도 일치시켜야 한다.

내적과 코사인 유사도는 벡터 노름이 1일 때 같다. 정규화하지 않으면 방향뿐 아니라 크기도 점수에 영향을 준다. 작은 말뭉치는 전체 내적을 계산할 수 있고 큰 자료는 ANN 색인을 사용해 검색 속도·메모리와 후보 재현율을 절충할 수 있다. ANN이 모든 밀집 검색에 필수인 것은 아니다.

## 2. 대조 학습과 음성 후보

한 정답 문서 d⁺와 음성 집합 N의 대조 로스 예는 다음과 같다.

$$\mathcal L=-\log\frac{\exp[s(q,d^+)/\tau]}{\exp[s(q,d^+)/\tau]+\sum_{d^-\in N}\exp[s(q,d^-)/\tau]},\qquad\tau>0.$$

같은 배치의 다른 문서나 검색으로 얻은 유사 후보를 음성으로 사용할 수 있다. 하지만 질문의 정답·관련 문서가 여러 개라면 다른 정답을 음성으로 잘못 다루는 false negative가 생길 수 있다. 어려운 음성이 항상 성능을 높이는 것도 아니다.

```python
def mine_hard_negatives(query_embedding, document_embeddings, positive_idx, k=10):
    """어려운 아닌 보기를 찾는다: 비슷하나 관련 없는 문서."""
    similarities = query_embedding @ document_embeddings.T
    # 맞는 문서를 뺀다
    similarities[positive_idx] = -float('inf')
    # 가장 비슷하면서 관련 없는 문서 위 k개
    hard_neg_indices = similarities.argsort(descending=True)[:k]
    return hard_neg_indices
```

mine_hard_negatives는 1차원 질문 벡터와 문서 행렬의 내적 순위를 가정한다. 정답 하나의 점수를 −∞로 바꾸지만 k가 문서 수 이상이면 그 인덱스가 다시 반환될 수 있다. 여러 정답·중복 문서, k 범위·유효 인덱스·정규화와 장치를 검증해야 한다. 1×d 배치 벡터를 그대로 넣으면 positive_idx가 문서 축이 아닌 첫 축에 적용될 수도 있다.

## 3. 임베딩 모델 선택

DPR·E5·BGE·GTE는 각각 여러 체크포인트와 크기·언어·학습 목표를 가진 계열이다. 계열 이름만으로 차원 1024·다국어 지원·학습 자료·품질 순위를 고정하지 않는다. 벤치마크 이름이 모델의 전체 학습 데이터라는 뜻도 아니다.

사용할 체크포인트의 질문/문서 지시 형식·pooling·길이 한도·정규화·언어와 도메인을 확인한다. 실제 자료의 관련성 정답으로 recall·순위·비용을 평가해야 하며, 다른 모델의 벡터를 같은 색인에 섞거나 차원만 맞는다고 호환된다고 가정해서는 안 된다.

## 4. BGE 검색 예제

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 미리 익힌 빽빽한 찾개를 불러온다
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 문서를 부호화한다(한 번만 하고 목록에 담는다)
documents = [
    "AAPL reported Q3 revenue of $81.8B, beating estimates by \$1.2B",
    "The Federal Reserve held rates steady at 5.25-5.50%",
    "NVDA guidance implies 170% YoY data center revenue growth",
]
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# 물음을 부호화하고 찾아온다
query = "Which company beat revenue estimates?"
query_embedding = model.encode([query], normalize_embeddings=True)

# 코사인 닮음(묻힘은 고르게 되어 있다)
scores = query_embedding @ doc_embeddings.T
top_idx = scores[0].argsort()[::-1]
print(f"Most relevant: {documents[top_idx[0]]}")
```

[bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)는 영어 임베딩 체크포인트다. 코드는 문서 3개와 질문을 정규화된 벡터로 만들고 전체 내적·정렬을 수행한다. 학습·ANN 색인·재순위화·생성은 없으며 완성된 RAG 시스템도 아니다.

모델 설명은 짧은 질문 검색에서 질문용 지시 접두사를 사용할 수 있음을 안내한다. 현재 예는 질문에 그 지시를 붙이지 않는다. 항상 실패한다는 뜻은 아니지만 입력 규칙을 고정하고 해당 자료에서 비교해야 한다. 기존 금융 문장은 검색용 예시이며 현재 사실을 검증한 데이터가 아니다. 이 편집에서는 모델 다운로드·인코딩·검색을 실행하지 않았다.

## 5. 검색과 생성의 전체 흐름

RAG는 문서를 검색해 질문과 함께 생성기에 제공할 수 있다. 자료를 갱신해도 모델 웨이트를 다시 학습할 필요가 없는 경우가 있지만 색인·권한·시점·중복·폐기 문서 처리를 갱신해야 한다. 검색이 환각을 없애거나 인용이 자동으로 정확해지는 것은 아니다.

BM25는 용어 빈도·문서 빈도와 길이 정규화의 어휘 신호를 사용하고 밀집 검색은 학습한 벡터 신호를 사용한다. 정확한 코드·기호·용어와 의미적 의역에서 장단점이 달라질 수 있다. 혼합 검색의 점수 결합·순위 융합도 실제 후보 재현율과 비용으로 검증한다.

재순위화 모델은 처음 검색한 후보의 질문·문서를 함께 보고 더 세밀한 점수를 줄 수 있다. 후보 수에 따른 비용이 추가되고 누락된 문서를 되살리지는 못한다. 후보를 batch로 처리할 수도 있으므로 k후보를 반드시 개별 순전파 k번 호출해야 하는 것은 아니다. 어떤 cross-encoder도 모든 과제에서 더 정확하다고 보장하지 않는다.

## 6. 평가 단위

문서와 청크의 관련성·정답 기준을 정하고 상위 k의 recall, MRR·NDCG, 검색 지연·색인 크기를 평가한다. 검색 근거가 맞는지와 생성 답이 그 근거를 충실히 사용했는지, 실제 질문의 답이 맞는지는 별도다. 같은 문서의 청크·중복이 학습과 평가에 걸쳐 누출되지 않도록 분리한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
밀집 검색 예제와 완성된 RAG 시스템의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    예제는 질문과 세 문서의 임베딩 내적을 정렬할 뿐 생성기를 호출하지 않는다. RAG에는 후보·근거 구성과 생성, 출처 연결·유보·정답 평가가 더 필요하다. 검색한 문서가 틀리거나 무관하면 답도 잘못될 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정규화한 내적과 코사인 유사도의 관계 및 모델 변경 시 색인 갱신을 설명하라.

</div>

??? success "연습문제 2 풀이"

    두 벡터의 노름이 1이면 내적이 코사인과 같다. 그렇지 않으면 벡터 크기도 내적에 영향을 준다. encoder·토크나이저·정규화·역할 지시가 바뀌면 벡터 공간이 달라질 수 있어 차원이 같아도 이전 문서 벡터와 섞으면 안 된다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
mine_hard_negatives에서 정답이 반환되거나 잘못된 축을 처리할 수 있는 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"

    문서 수 이상으로 k를 주면 −∞로 낮춘 정답도 정렬 끝에서 다시 포함될 수 있다. 질문이 1×d이면 점수가 1×N이어서 positive_idx의 첫 축 인덱싱도 맞지 않는다. 문서 축·k 범위·모든 정답과 중복을 명시적으로 제외해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
재순위화의 이점과 최초 검색 recall의 한계를 설명하라.

</div>

??? success "연습문제 4 풀이"

    질문과 후보 문서의 토큰 상호작용으로 순서를 더 정밀하게 평가할 수 있지만 추가 비용이 있고 최초 후보에 없는 정답을 복구할 수는 없다. 후보 수·batch·모델을 같게 하고 recall·최종 순위·전체 답과 지연을 함께 평가해야 한다.



## 정리하며

밀집 검색은 벡터 공간·입력 규칙·정답 후보·색인을 일치시켜야 한다. 전체 검색과 ANN·재순위화·RAG를 구분하고 관련성·근거·생성 정확도를 각각 평가한다.

**참고 문헌**

1. Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain QA." *EMNLP*.
2. Wang, L., et al. (2022). "Text Embeddings by Weakly-Supervised Contrastive Pre-training." *arXiv*.
