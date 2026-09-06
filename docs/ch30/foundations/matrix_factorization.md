# 행렬 인수 분해
## 학습 목표

- 낮은 계수 어림에서 행렬 인수 분해의 목표를 이끌어 낸다
- 행렬 인수 분해를 특잇값 분해와 잇고 성긴 행렬에서 곧바른 특잇값 분해가 왜 안 되는지 밝힌다
- 박아 넣기로 보는 풀이와 PyTorch 짜기를 이해한다
- 매김 헤아리기의 치우침 쪼갬을 이끌어 내고 치우침 항이 왜 헤아림 어긋남을 크게 줄이는지 밝힌다
- `nn.Embedding`으로 기본 행렬 인수 분해와 치우침 있는 판을 PyTorch로 짠다
- 모임 성질과 박아 넣기 차원의 효과를 살핀다

## 낮은 계수 어림에서 행렬 인수 분해로

### 이상적인 경우: 온전한 행렬

매김 행렬 $R \in \mathbb{R}^{m \times n}$을 빠짐없이 보았다면 가장 좋은 계수 $d$ 어림은 **잘라 낸 특잇값 분해**로 주어진다:

$$R \approx U_d \Sigma_d V_d^\top$$

여기서 $U_d \in \mathbb{R}^{m \times d}$, $\Sigma_d \in \mathbb{R}^{d \times d}$, $V_d \in \mathbb{R}^{n \times d}$은 위 $d$개 성분이다. 에카르트–영–미르스키 정리에 따라 이것이 다음을 가장 작게 한다:

$$\min_{\text{rank}(X) = d} \|R - X\|_F^2$$

$P = U_d \Sigma_d^{1/2}$과 $Q = V_d \Sigma_d^{1/2}$으로 두면 $R \approx P Q^\top$이 된다.

### 실제 경우: 성긴 행렬

실제로 $R$의 칸은 대부분 비어 있다. 곧바른 특잇값 분해가 안 되는 까닭은 다음과 같다:

1. **특잇값 분해에는 온전한 행렬이 필요하다.** 빈 칸을 0으로 다루면 분해가 0 쪽으로 치우친다. 쓰는 이가 매기지 않은 영화는 0으로 매긴 영화와 같지 않다.
2. **채워 넣기는 돌고 돈다.** 특잇값 분해 전에 빈 값을 채우려면 헤아리려는 바로 그것을 알아야 한다.

풀이: **본 칸에 대해서만** 가장 좋게 한다.

## 행렬 인수 분해의 목표

### 정식화

본 매김 $\Omega = \{(u, i) : R_{ui} \text{를 보았음}\}$이 주어질 때 다음을 가장 작게 하여 쓰는 이 박아 넣기 $\mathbf{p}_u \in \mathbb{R}^d$과 물건 박아 넣기 $\mathbf{q}_i \in \mathbb{R}^d$을 배운다:

$$\mathcal{L}(\theta) = \sum_{(u,i) \in \Omega} \bigl(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\bigr)^2$$

이는 **볼록하지 않은** 가장 좋게 하기 문제이지만($P$과 $Q$에 대해 겹선형) 기울기 바탕 방법이 실제로 잘 듣는다.

### 기울기의 유도

본 매김 $(u, i)$ 하나에 대해 헤아림 어긋남을 다음과 같이 뜻매김한다:

$$e_{ui} = R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i$$

이 짝의 손실은 $\ell_{ui} = e_{ui}^2$이다. 기울기는 다음과 같다:

$$\frac{\partial \ell_{ui}}{\partial \mathbf{p}_u} = -2\, e_{ui}\, \mathbf{q}_i$$

$$\frac{\partial \ell_{ui}}{\partial \mathbf{q}_i} = -2\, e_{ui}\, \mathbf{p}_u$$

**이끌어 냄:** 사슬 규칙으로:

$$\frac{\partial \ell_{ui}}{\partial \mathbf{p}_u} = 2(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i) \cdot \frac{\partial}{\partial \mathbf{p}_u}\bigl(-\mathbf{p}_u^\top \mathbf{q}_i\bigr) = -2\, e_{ui}\, \mathbf{q}_i$$

$\frac{\partial}{\partial \mathbf{p}_u}(\mathbf{p}_u^\top \mathbf{q}_i) = \mathbf{q}_i$이기 때문이다($\mathbf{q}_i$을 $\mathbf{p}_u$에 대해 상수로 본다).

확률 기울기 내려가기의 고침 규칙은 다음과 같다:

$$\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta \cdot e_{ui} \cdot \mathbf{q}_i$$

$$\mathbf{q}_i \leftarrow \mathbf{q}_i + \eta \cdot e_{ui} \cdot \mathbf{p}_u$$

여기서 $\eta$은 학습률이다.

### L2 다잡기와 함께

(특히 성긴 자료에서) 지나친 맞춤을 막으려 무게 스러짐을 더한다:

$$\mathcal{L}_{\text{reg}}(\theta) = \sum_{(u,i) \in \Omega} \bigl(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\bigr)^2 + \lambda \Bigl(\sum_u \|\mathbf{p}_u\|^2 + \sum_i \|\mathbf{q}_i\|^2\Bigr)$$

다잡은 기울기는 다음이 된다:

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \mathbf{p}_u} = -2\, e_{ui}\, \mathbf{q}_i + 2\lambda\, \mathbf{p}_u$$

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \mathbf{q}_i} = -2\, e_{ui}\, \mathbf{p}_u + 2\lambda\, \mathbf{q}_i$$

!!! note "Adam의 무게 스러짐"
    PyTorch에서 `torch.optim.Adam`에 `weight_decay=wd`을 넘기면 L2 다잡기가 걸린다. Adam의 무게 스러짐은 맞춰 가는 배움 빠르기와 서로 얽히므로, 떼어 놓은 무게 스러짐을 쓰려면 `AdamW`을 쓴다(2장 참고).

## 박아 넣기로 보는 풀이

행렬 $P \in \mathbb{R}^{m \times d}$은 **박아 넣기 표**로 볼 수 있다. 줄 $u$은 쓰는 이 $u$의 $d$차원 나타냄이다. 마찬가지로 $Q \in \mathbb{R}^{n \times d}$은 물건 박아 넣기 표이다.

PyTorch에서 `nn.Embedding(num_users, d)`이 바로 이 표를 담는다. 쓰는 이 $u$과 물건 $i$의 앞으로 가기는:

```python
p_u = self.user_emb(u)   # 꼴: (batch, d) — P에서 줄 u을 찾는다
q_i = self.item_emb(i)   # 꼴: (batch, d) — Q에서 줄 i을 찾는다
rating = (p_u * q_i).sum(1)  # 원소마다 곱한 뒤 더한다 → 안쪽 곱
```

**왜 `torch.dot` 대신 원소마다 곱한 뒤 더하는가?** 들임이 묶여 있다. `p_u`과 `q_i`의 꼴은 `(batch_size, d)`이다. 원소마다 곱한 뒤 `sum(1)`을 하면 묶음의 **표본마다 따로** 안쪽 곱을 셈해 꼴이 `(batch_size,)`인 텐서가 나온다.

!!! warning "박아 넣기의 첫 값이 중요하다"
    이 짜기는 박아 넣기의 첫 값을 `uniform_(0, 0.05)`으로 둔다. 작고 양인 첫 값을 보장한다. 흔한 다른 고름으로는 자비에르 첫 값 두기나 $\mathcal{N}(0, 0.01)$에서 뽑기가 있다. 첫 값이 나쁘면(보기로 큰 값) 가장 좋게 하기가 발산할 수 있다.

## PyTorch 짜기: 기본 행렬 인수 분해

```python
class MF(nn.Module):
    """
    함께 거르기를 위한 행렬 인수 분해.
    
    Predicts rating as: r_hat = p_u^T q_i
    여기서 p_u과 q_i은 배운 박아 넣기 벡터이다.
    """
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        # 작은 양의 첫 값
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, u, v):
        u = self.user_emb(u)          # (batch, emb_size)
        v = self.item_emb(v)          # (batch, emb_size)
        return (u * v).sum(1)         # (batch,) — 표본마다 안쪽 곱
```

### 하나씩 짚기

1. **`__init__`**: 꼴이 `(num_users, emb_size)`과 `(num_items, emb_size)`인 박아 넣기 표 둘을 만든다.
2. **`forward(u, v)`**: 정수 텐서 `u`(쓰는 이 번호)과 `v`(물건 번호)를 받아 박아 넣기를 찾고 안쪽 곱을 헤아린 매김으로 돌려준다.
3. **잡 개수**: $(m + n) \times d$. $m = 610$, $n = 9{,}724$, $d = 100$이면 잡이 약 103만 개이다.

## 특잇값 분해와의 이음

행렬 인수 분해와 특잇값 분해는 가깝지만 같지는 않다:

| 갈래 | 잘라 낸 특잇값 분해 | 배운 행렬 인수 분해 |
|--------|--------------|-----------|
| **목표** | $\min \|R - PQ^\top\|_F^2$(모든 칸) | $\min \sum_{(u,i) \in \Omega}(R_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2$(본 칸만) |
| **풀이** | 닫힌 꼴(에카르트–영) | 되풀이(기울기 내려가기) |
| **직교성** | $P^\top P = I$, $Q^\top Q = I$ | 매임 없음 |
| **빈 자료** | 본디 다루지 못함 | 자연스럽게 다룸 |
| **하나뿐임** | 하나뿐(부호를 빼고) | 최적이 여럿 |

배운 행렬 인수 분해를 (Simon Funk의 넷플릭스 상 방식을 따라) "Funk SVD"라 부르기도 하지만 참된 특잇값 분해는 아니다.

## 박아 넣기 차원의 효과

박아 넣기 차원 $d$이 모델의 담는 힘을 다스린다:

- **$d$이 너무 작음**: 모자란 맞춤. 모델이 숨은 얼개를 담지 못한다.
- **$d$이 너무 큼**: 지나친 맞춤. 자료가 성기면 모델이 잡소리를 외운다.

**어림 규칙**: $d \in \{50, 100, 200\}$으로 시작해 살피기 손실로 손본다. MovieLens-Small 자료 뭉치(매김 약 10만 개)에서는 $d = 100$이 알맞은 기본값이다.

자료에 실제로 있는 숨은 인수의 개수는 (채워 넣은) 매김 행렬의 특잇값 스펙트럼을 살펴 어림할 수 있다. 특잇값이 빠르게 스러지면 작은 $d$으로 넉넉하다.

## 기본 행렬 인수 분해의 한계

기본 행렬 인수 분해에는 중요한 한계가 있다. 매김을 오로지 쓰는 이와 물건의 주고받음으로만 나타내어 다음과 같은 몸에 밴 효과를 무시한다:

- **쓰는 이 치우침**: 어떤 이는 무엇이든 높게 매기고 어떤 이는 박하다.
- **물건 치우침**: 어떤 영화는 두루 사랑받고 어떤 것은 소수 취향이다.
- **온 자리 평균**: 모든 봄을 통틀은 평균 매김.

보기로 온 자리 평균 매김이 3.5이고 쓰는 이 $u$이 평균보다 0.5 높게 매기며 영화 $i$이 평균보다 0.3 낮다면, 숨은 인수의 주고받음을 셈에 넣기 전 밑금 헤아림은 \$3.5 + 0.5 - 0.3 = 3.7$이어야 한다.

---

## 치우침을 더한 행렬 인수 분해

### 까닭: 매김에 몸에 밴 효과

MovieLens 자료 뭉치를 보자. 눈에 띄는 어떤 무늬는 쓰는 이와 물건의 궁합과 아무 상관이 없다:

- **쓰는 이의 버릇**: A은 평균 4.2로, B은 평균 2.8로 매긴다. 이는 특정 영화에 대한 취향이 아니라 성향을 비춘다.
- **물건 품질**: "쇼생크 탈출"은 모든 이를 통틀어 평균 4.5이고 저예산 영화는 평균 2.1이다. 이는 개인 취향이 아니라 품질에 대한 뜻 모임을 비춘다.
- **온 자리 밑금**: 자료 뭉치의 전체 평균 매김이 3.5쯤일 수 있다.

기본 행렬 인수 분해 $\hat{R}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$은 이런 몸에 밴 효과까지 **모든** 흔들림을 숨은 인수가 담게 만든다. 이는 모델의 담는 힘을 낭비하고 숨은 인수를 풀이하기 어렵게 한다.

### 치우침 쪼갬

Koren, Bell, Volinsky(2009)을 따라 매김마다 다음과 같이 쪼갠다:

$$R_{ui} = \underbrace{\mu}_{\text{global mean}} + \underbrace{b_u}_{\text{user bias}} + \underbrace{b_i}_{\text{item bias}} + \underbrace{\mathbf{p}_u^\top \mathbf{q}_i}_{\text{interaction}} + \underbrace{\epsilon_{ui}}_{\text{noise}}$$

헤아린 매김은 다음이 된다:

$$\hat{R}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$$

여기서 각 기호는 다음과 같다.

- $\mu = \frac{1}{|\Omega|}\sum_{(u,i) \in \Omega} R_{ui}$은 온 자리 평균이다(미리 셈할 수 있다)
- $b_u \in \mathbb{R}$은 쓰는 이 $u$의 치우침이다(배운다)
- $b_i \in \mathbb{R}$은 물건 $i$의 치우침이다(배운다)
- $\mathbf{p}_u^\top \mathbf{q}_i$은 남은 쓰는 이-물건 주고받음을 담는다

**풀이**: 밑금 어림 $\mu + b_u + b_i$은 "쓰는 이가 누구이고 물건이 무엇인지에 바탕한 기대 매김"을 담는다. 주고받음 항 $\mathbf{p}_u^\top \mathbf{q}_i$은 "바로 이 쓰는 이-물건 짝이 밑금에서 얼마나 벗어나는지"를 담는다.

### 가장 좋은 치우침 값(닫힌 꼴)

숨은 인수를 붙박아 두고 치우침만 가장 좋게 하면 가장 좋은 값이 다음을 가장 작게 한다:

$$\min_{b_u, b_i} \sum_{(u,i) \in \Omega} \bigl(R_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\bigr)^2 + \lambda\bigl(\sum_u b_u^2 + \sum_i b_i^2\bigr)$$

$b_u$에 대해 미분해 0으로 놓으면:

$$\frac{\partial}{\partial b_u}: \quad -2 \sum_{i:(u,i) \in \Omega} \bigl(R_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\bigr) + 2\lambda b_u = 0$$

풀면:

$$b_u^* = \frac{\sum_{i:(u,i) \in \Omega} (R_{ui} - \mu - b_i - \mathbf{p}_u^\top \mathbf{q}_i)}{|I_u| + \lambda}$$

여기서 $|I_u|$은 쓰는 이 $u$이 매긴 물건의 개수이다. 마찬가지로:

$$b_i^* = \frac{\sum_{u:(u,i) \in \Omega} (R_{ui} - \mu - b_u - \mathbf{p}_u^\top \mathbf{q}_i)}{|U_i| + \lambda}$$

실제로는 이 닫힌 꼴 풀이를 쓰지 않고 기울기 내려가기로 모든 잡을 함께 배운다. 그래도 닫힌 꼴은 통찰을 준다. 분모의 다잡기 항 $\lambda$이 치우침을 0 쪽으로 **줄이며**, 매김이 적은 쓰는 이나 물건일수록 더 세게 줄인다.

### 치우침을 넣은 목표 함수

치우침 있는 행렬 인수 분해의 다잡은 손실은 다음과 같다:

$$\mathcal{L} = \sum_{(u,i) \in \Omega} \bigl(R_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\bigr)^2 + \lambda\Bigl(\sum_u \|\mathbf{p}_u\|^2 + \sum_i \|\mathbf{q}_i\|^2 + \sum_u b_u^2 + \sum_i b_i^2\Bigr)$$

남은 값 $e_{ui} = R_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i$을 뜻매김한다. 봄 $(u, i)$ 하나의 기울기는:

$$\frac{\partial \mathcal{L}}{\partial b_u} = -2\, e_{ui} + 2\lambda\, b_u$$

$$\frac{\partial \mathcal{L}}{\partial b_i} = -2\, e_{ui} + 2\lambda\, b_i$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{p}_u} = -2\, e_{ui}\, \mathbf{q}_i + 2\lambda\, \mathbf{p}_u$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{q}_i} = -2\, e_{ui}\, \mathbf{p}_u + 2\lambda\, \mathbf{q}_i$$

치우침 기울기는 낱값이고 박아 넣기 기울기는 $d$차원 벡터임에 유의한다.

## PyTorch 짜기: 치우침 있는 행렬 인수 분해

```python
class MF_bias(nn.Module):
    """
    쓰는 이와 물건의 치우침을 넣은 행렬 인수 분해.
    
    Predicts rating as: r_hat = p_u^T q_i + b_u + b_i
    
    유의: 온 자리 평균 mu은 치우침 항에 빨려 들어가거나
    during training (PyTorch optimizes b_u and b_i to include it),
    붙박인 상수로 드러나게 더할 수 있다.
    """
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF_bias, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        # 숨은 인수: 작은 양의 값
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        # 치우침: 0을 가운데 둔 작은 값
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, u, v):
        U = self.user_emb(u)                # (batch, emb_size)
        V = self.item_emb(v)                # (batch, emb_size)
        b_u = self.user_bias(u).squeeze()   # (배치,)
        b_v = self.item_bias(v).squeeze()   # (배치,)
        return (U * V).sum(1) + b_u + b_v   # (배치,)
```

### 짜기의 세부

**`nn.Embedding(n, 1)`으로 둔 치우침**: 치우침은 쓰는 이나 물건마다 낱값 하나다. 내놓기 차원이 1인 `nn.Embedding`을 쓰면 숨은 인수처럼 이 낱값을 찾기 표에 담는다. `.squeeze()`이 뒤따르는 차원을 없앤다. `(batch, 1)` → `(batch,)`.

**왜 `nn.Parameter`이 아닌가?** `self.user_bias = nn.Parameter(torch.zeros(num_users))`으로 두고 손으로 번호를 짚어도 *된다*. `nn.Embedding` 쪽이 깔끔한 까닭은 묶음 찾기를 저절로 다루고 가장 좋게 하개의 무게 스러짐과 잘 어울리기 때문이다.

**잡 개수**: $(m + n) \times d + (m + n)$. 치우침 항은 잡을 $m + n$개만 더한다. 박아 넣기에 견주면 하찮다.

!!! note "온 자리 평균 $\mu$은 어디에 있는가"
    위 짜기는 $\mu$을 드러나게 담지 않는다. 익히는 동안 치우침 항 $b_u$과 $b_i$이 온 자리 평균을 빨아들인다. $\mu$을 드러내고 싶으면 미리 셈해 앞으로 가기에서 더한다:
    ```python
    return (U * V).sum(1) + b_u + b_v + self.global_mean
    ```
    치우침이 가장 좋은 값에 더 가까이서 시작하므로 모임이 좋아질 수 있다.

### 첫 값의 어긋남

숨은 인수는 $[0, 0.05]$(양)에서, 치우침은 $[-0.01, 0.01]$(가운데 맞춤)에서 첫 값을 잡는다. 이는 맡은 몫이 다름을 비춘다:

- **숨은 인수**: 양의 첫 값은 첫 안쪽 곱이 음이 아니게 하며 이는 양의 매김 잣대에 알맞다.
- **치우침**: 가운데 맞춘 첫 값은 대부분의 쓰는 이와 물건이 평균에 가깝다는 사전 믿음을 비춘다.

## 치우침이 왜 도움이 되는가: 수량으로 살피기

무엇이든 평균보다 1점 높게 매기는 쓰는 이를 보자. 치우침이 없으면:

- 숨은 인수가 이 버릇을 담아야 하므로 주고받음을 담을 힘이 줄어든다.
- 같은 맞음을 얻으려면 더 큰 $d$이 필요하다.

치우침이 있으면:

- $b_u \approx 1.0$이 그 버릇을 곧바로 담는다.
- 숨은 인수는 오로지 쓰는 이와 물건의 궁합에 집중한다.
- 더 작은 $d$으로 넉넉해져 널리 쓰임이 좋아진다.

**겪어 본 효과**: MovieLens 자료 뭉치에서 치우침을 더하면 대개 살피기 평균 제곱 어긋남이 10~20% 줄고, 평균 매김이 극단인 쓰는 이나 물건에서 가장 크게 좋아진다.

## 행렬 인수 분해 변형 견주기

| 모델 | 헤아림 | 잡 | 담는 것 |
|-------|-----------|-----------|----------|
| 기본 행렬 인수 분해 | $\mathbf{p}_u^\top \mathbf{q}_i$ | $(m+n)d$ | 주고받음만 |
| 행렬 인수 분해 + 치우침 | $b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$ | $(m+n)(d+1)$ | 치우침 + 주고받음 |
| 행렬 인수 분해 + 치우침 + $\mu$ | $\mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$ | $(m+n)(d+1) + 1$ | 온 자리 + 치우침 + 주고받음 |

## 학습할 때 살필 점

### 배움 빠르기 차례표

원본 코드는 **단계별 배움 빠르기** 셈속을 쓴다:

```python
train_epochs(model, df_train, df_val, epochs=10, lr=0.1)   # 1단계: 거칠게
train_epochs(model, df_train, df_val, epochs=15, lr=0.01)  # 2단계: 다듬기
train_epochs(model, df_train, df_val, epochs=15, lr=0.001) # 3단계: 매끄럽게
```

이는 배움 빠르기 스러짐을 손으로 어림한 것이다. 처음의 높은 빠르기가 좋은 자리를 빨리 찾고, 낮은 빠르기가 그 안에서 곱게 다듬는다.

### 온 묶음 익히기와 작은 묶음 익히기

이 짜기는 **온 묶음** 기울기 내려가기를 쓴다(걸음마다 익히기 자료 전체). MovieLens-Small(익히기 매김 약 8만 개)에서는 할 만하지만 더 큰 자료 뭉치에서는 작은 묶음 익히기로 바꾸어야 한다:

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(
    torch.LongTensor(df_train.userId.values),
    torch.LongTensor(df_train.movieId.values),
    torch.FloatTensor(df_train.rating.values)
)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)
```

## 온전한 익히기 흐름

다음은 자료 불러오기, 담기, 익히기의 온전한 흐름을 보여 준다:

```python
def train_epochs(model, df_train, df_val, epochs=10, lr=0.01, wd=0.0):
    """밝힌 판수만큼 모델을 익힌다."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        model.train()
        users = torch.LongTensor(df_train.userId.values)
        items = torch.LongTensor(df_train.movieId.values)
        ratings = torch.FloatTensor(df_train.rating.values)

        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 때때로 살피기
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_users = torch.LongTensor(df_val.userId.values)
                val_items = torch.LongTensor(df_val.movieId.values)
                val_ratings = torch.FloatTensor(df_val.rating.values)
                val_hat = model(val_users, val_items)
                val_loss = F.mse_loss(val_hat, val_ratings).item()
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train MSE: {loss.item():.4f} | Val MSE: {val_loss:.4f}")
```

!!! tip "번호 담기"
    MovieLens의 쓰는 이 번호와 영화 번호는 이어져 있지 않다(보기로 쓰는 이 번호가 3에서 7로 건너뛸 수 있다). `nn.Embedding`을 쓰기 전에 번호를 $[0, N)$의 이어진 정수로 다시 옮긴다. 원본 코드는 `proc_col`과 `encode_data` 함수로 이를 다룬다. 살피기 자료는 익히기 자료와 **같은** 옮김으로 담아야 하며 본 적 없는 쓰는 이나 물건이 든 칸은 버려야 한다.

## 요약

행렬 인수 분해는 안쪽 곱이 본 매김을 어림하도록 쓰는 이와 물건의 낮은 차원 박아 넣기를 배운다. (특잇값 분해와 달리) 본 칸에 대해서만 가장 좋게 하고, 기울기 내려가기를 써서 작은 묶음 익히기를 할 수 있으며, PyTorch에서 `nn.Embedding`으로 자연스럽게 적힌다. 치우침 항을 더하면 몸에 밴 효과(쓰는 이가 누구인지, 물건이 무엇인지)를 주고받음 효과(이 짝이 얼마나 잘 맞는지)에서 떼어 놓아, 잡을 거의 늘리지 않고도 헤아림 품질을 뜻있게 높인다. 다음 마디에서는 나타냄 힘을 더 키우려 안쪽 곱을 신경망으로 바꾼다.

---

## 연습문제

1. **기울기 확인**: 행렬 인수 분해 모델을 짜고 `torch.autograd.gradcheck`으로 손으로 구한 기울기를 PyTorch의 자동 미분과 맞대어 확인하여라.

2. **계수 살피기**: 계수가 2인 $5 \times 5$ 매김 행렬에서 $d = 2$의 행렬 인수 분해가 모든 칸을 완벽히 되짓는다는 것을 보여라. $d = 1$이면 어떻게 되는가?

3. **첫 값 실험**: 첫 값 두는 방식 셋(`uniform_(0, 0.05)`, `uniform_(-1, 1)`, `normal_(0, 1)`)으로 행렬 인수 분해 모델을 익혀라. 모이는 빠르기와 마지막 살피기 손실을 견주어라.

4. **박아 넣기 차원 훑기**: MovieLens-Small에서 $d \in \{10, 50, 100, 200, 500\}$으로 익혀라. $d$에 대한 익히기와 살피기 평균 제곱 어긋남을 그려라. 어느 지점에서 지나친 맞춤이 시작되는가?

5. **수학의 같음**: $R$을 빠짐없이 보았고 다잡기 없이 $\|R - PQ^\top\|_F^2$을 가장 작게 하면 온 자리 최소가 $PQ^\top = U_d \Sigma_d V_d^\top$(잘라 낸 특잇값 분해)을 만족함을 밝혀라.

6. **치우침 되찾기**: 치우침을 알고 숨은 인수가 없는 $R_{ui} = 3.0 + b_u + b_i + \epsilon_{ui}$의 지어낸 자료 뭉치를 만들어라. $d = 1$으로 치우침 있는 행렬 인수 분해를 익혀라. 배운 치우침이 참값을 되찾는가?

7. **떼어 보기**: MovieLens에서 모델 셋을 익혀라. (가) 치우침 없는 행렬 인수 분해, (나) 치우침만 있는 모델($d = 0$으로 $\mu + b_u + b_i$뿐), (다) 치우침 있는 행렬 인수 분해. 살피기 평균 제곱 어긋남을 알려라. 헤아리는 힘 가운데 치우침만으로 얻는 몫은 얼마인가?

8. **다잡기 민감도**: (Adam의 `weight_decay`으로) $\lambda \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$을 두어 치우침 있는 행렬 인수 분해를 익혀라. $\lambda$에 대한 살피기 평균 제곱 어긋남을 그려라. 가장 좋은 $\lambda$이 치우침과 박아 넣기에서 같은가?

9. **풀이**: 익힌 뒤 배운 치우침 $b_u$이 가장 높은 이와 가장 낮은 이를 찾아라. 그들의 매김 지난 일은 어떠한가? 치우침이 느낌과 맞는가?
