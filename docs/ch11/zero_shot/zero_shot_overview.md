# 영 예시 학습의 기초
## 문제 정식화

영 예시 학습은 여느 기계 학습의 근본 한계, 곧 학습 중에 보지 못한 부류를 알아보지 못한다는 점을 다룬다. 이 절은 ZSL 문제를 정식화하고 모든 ZSL 방법의 수학적 바탕을 세운다.

### 형식적 정의

$\mathcal{X}$을 입력 공간(이를테면 그림), $\mathcal{Y}$을 이름표 공간이라 하자. ZSL에서 이름표 공간은 다음으로 나뉜다.

- **본 부류**: $\mathcal{Y}^s = \{y_1^s, y_2^s, \ldots, y_K^s\}$ — 학습 보기가 있는 부류
- **못 본 부류**: $\mathcal{Y}^u = \{y_1^u, y_2^u, \ldots, y_L^u\}$ — 학습 보기가 없는 부류

근본 제약은 다음과 같다.

$$\mathcal{Y}^s \cap \mathcal{Y}^u = \emptyset$$

### 학습과 시험의 규약

**학습 단계:**

- 이름표 붙은 데이터를 쓸 수 있다: $\mathcal{D}^{tr} = \{(\mathbf{x}_i, y_i) : y_i \in \mathcal{Y}^s\}_{i=1}^{N}$
- 모든 부류의 뜻 설명을 쓸 수 있다: $\{\mathbf{s}_c\}_{c \in \mathcal{Y}^s \cup \mathcal{Y}^u}$

**시험 단계(여느 ZSL):**

- 시험 사례는 못 본 부류에만 든다
- 예측 공간: $\mathcal{Y}^u$

**시험 단계(일반화된 ZSL):**

- 시험 사례는 어느 부류에나 들 수 있다
- 예측 공간: $\mathcal{Y}^s \cup \mathcal{Y}^u$

## 뜻 공간

ZSL의 열쇠는 본 부류와 못 본 부류에 두루 쓰이는 표현을 주는 뜻 공간 $\mathcal{S}$이다.

### 속성 기반 표현

부류 $c$마다 속성 벡터 $\mathbf{a}_c \in \mathbb{R}^M$으로 그려지며 $M$은 속성의 개수이다.

$$\mathbf{a}_c = [a_c^{(1)}, a_c^{(2)}, \ldots, a_c^{(M)}]$$

속성은 다음일 수 있다.

- **이진**: 있고 없음을 나타내는 $a_c^{(m)} \in \{0, 1\}$
- **이어짐**: 세기와 관련도를 나타내는 $a_c^{(m)} \in [0, 1]$

**보기: 동물의 속성**

| 동물 | has_fur | has_stripes | is_large | has_4_legs | can_fly |
|--------|---------|-------------|----------|------------|---------|
| 개 | 1 | 0 | 0.3 | 1 | 0 |
| 고양이 | 1 | 0 | 0.1 | 1 | 0 |
| 얼룩말 | 1 | 1 | 0.9 | 1 | 0 |
| 독수리 | 0 | 0 | 0.6 | 0 | 1 |

### 낱말 묻힘 표현

부류 이름은 미리 학습된 언어 모델로 이어진 벡터 공간에 옮겨진다.

$$\mathbf{s}_c = \text{Embed}(\text{classname}_c) \in \mathbb{R}^D$$

흔한 묻힘 방법은 다음과 같다.

- Word2Vec(스킵그램, CBOW)
- GloVe(전역 벡터)
- FastText(낱말 아래 정보)
- BERT, GPT 묻힘

**속성보다 나은 점:**

1. 손 표시가 필요 없다
2. 풍부한 뜻의 관계를 알아서 담는다
3. 엄청난 글 뭉치로 미리 학습되어 있다
4. 말에 담긴 앎을 시각 영역으로 옮긴다

### 뜻의 닮음

뜻 공간은 닮음 관계를 자아낸다.

$$\text{sim}(c_1, c_2) = f(\mathbf{s}_{c_1}, \mathbf{s}_{c_2})$$

흔한 닮음 함수는 다음과 같다.

**코사인 닮음:**

$$\text{sim}_{\cos}(\mathbf{s}_1, \mathbf{s}_2) = \frac{\mathbf{s}_1 \cdot \mathbf{s}_2}{\|\mathbf{s}_1\| \|\mathbf{s}_2\|}$$

**유클리드 거리(닮음으로 바꾼 것):**

$$\text{sim}_{\text{euc}}(\mathbf{s}_1, \mathbf{s}_2) = \exp(-\|\mathbf{s}_1 - \mathbf{s}_2\|^2)$$

**점곱:**

$$\text{sim}_{\text{dot}}(\mathbf{s}_1, \mathbf{s}_2) = \mathbf{s}_1^\top \mathbf{s}_2$$

## 어울림 함수

어울림 함수 $F: \mathcal{V} \times \mathcal{S} \rightarrow \mathbb{R}$은 시각 특징이 뜻 표현과 얼마나 잘 맞는지를 잰다.

### 선형 어울림

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

여기서 $W \in \mathbb{R}^{d_v \times d_s}$은 배운 쏘아 넣기 행렬이다.

### 겹선형 어울림

더 풍부한 어울림을 위한 것이다.

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

여기서 $W$은 시각 차원과 뜻 차원 사이의 쌍마다의 어울림을 모두 담는다.

### 신경 어울림

비선형 어울림에 신경망을 쓴다.

$$F(\mathbf{v}, \mathbf{s}) = g_\theta(\mathbf{v}, \mathbf{s})$$

흔한 구조는 다음과 같다.

- 이어 붙인 뒤 MLP
- 교차 주의 장치
- 잔차 어울림 망

### 묻힘 기반 어울림

두 갈래를 모두 함께 쓰는 공간으로 쏘아 넣는다.

$$F(\mathbf{v}, \mathbf{s}) = f_\phi(\mathbf{v})^\top g_\psi(\mathbf{s})$$

또는 코사인 닮음을 쓴다.

$$F(\mathbf{v}, \mathbf{s}) = \frac{f_\phi(\mathbf{v}) \cdot g_\psi(\mathbf{s})}{\|f_\phi(\mathbf{v})\| \|g_\psi(\mathbf{s})\|}$$

## ZSL의 손실 함수

### 교차 엔트로피 손실

ZSL을 본 부류 위의 가려내기로 다룬다.

$$\mathcal{L}_{CE} = -\sum_{(\mathbf{x}, y) \in \mathcal{D}^{tr}} \log \frac{\exp(F(\phi(\mathbf{x}), \mathbf{s}_y))}{\sum_{c \in \mathcal{Y}^s} \exp(F(\phi(\mathbf{x}), \mathbf{s}_c))}$$

### 순위 손실(경첩 손실)

맞는 부류가 틀린 부류보다 더 잘 어울리도록 북돋운다.

$$\mathcal{L}_{rank} = \sum_{(\mathbf{x}, y)} \sum_{c \neq y} \max(0, \Delta + F(\phi(\mathbf{x}), \mathbf{s}_c) - F(\phi(\mathbf{x}), \mathbf{s}_y))$$

여기서 $\Delta$은 여백 매개변수이다.

### 세쌍 손실

묻힘의 거리를 곧바로 최적화한다.

$$\mathcal{L}_{triplet} = \sum_{(\mathbf{x}, y)} \max(0, \|f(\mathbf{x}) - \mathbf{s}_y\|^2 - \|f(\mathbf{x}) - \mathbf{s}_{y^-}\|^2 + \alpha)$$

여기서 $y^-$은 음의 부류이고 $\alpha$은 여백이다.

### 대조 손실

쌍으로 닮음을 배우기 위한 것이다.

$$\mathcal{L}_{cont} = \sum_{i,j} y_{ij} D_{ij}^2 + (1 - y_{ij}) \max(0, m - D_{ij})^2$$

여기서 쌍이 같은 부류에서 오면 $y_{ij} = 1$이고 그렇지 않으면 0이다.

## 시각 특징 뽑기

오늘날의 ZSL 방법은 미리 학습된 CNN 특징을 쓴다.

### CNN 등뼈

| 모델 | 출력 차원 | 미리 학습 | 흔히 쓰는 층 |
|-------|------------|--------------|--------------|
| VGG-19 | 4096 | 이미지넷 | fc7 |
| ResNet-101 | 2048 | 이미지넷 | avg_pool |
| Inception-v3 | 2048 | ImageNet | pool3 |
| CLIP | 512/768 | 4억 쌍 | 마지막 층 |

### 특징 뽑기 파이프라인

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# 미리 학습된 ResNet을 불러온다
resnet = models.resnet101(pretrained=True)
# 가려내기 머리를 없앤다
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# 표준 앞손질
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 특징을 뽑는다
def extract_features(image):
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0)
        features = feature_extractor(x)
        return features.squeeze()  # 꼴: (2048,)
```

## 시험 때의 예측

### 뜻 공간에서의 최근접 이웃

시각 특징이 $\mathbf{v} = \phi(\mathbf{x})$인 시험 사례 $\mathbf{x}$에 대해 다음과 같다.

1. 쏘아 넣은 특징이나 어울림 점수를 셈한다
2. 가장 잘 어울리는 부류를 찾는다

**여느 ZSL:**

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} F(\mathbf{v}, \mathbf{s}_c)$$

**일반화된 ZSL:**

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^s \cup \mathcal{Y}^u} F(\mathbf{v}, \mathbf{s}_c)$$

### 구현

```python
import numpy as np

def predict_zsl(visual_features, class_embeddings, class_names, 
                compatibility_fn='cosine'):
    """
    영 예시 학습으로 부류를 맞힌다.
    
    인수:
        visual_features: (n_samples, d_v) 시각 특징 행렬
        class_embeddings: (n_classes, d_s) 뜻 묻힘 행렬
        class_names: 부류 이름 목록
        compatibility_fn: 'cosine', 'dot' 또는 'euclidean'
    
    반환값:
        predictions: 맞힌 부류 이름 목록
    """
    if compatibility_fn == 'cosine':
        # 코사인 닮음을 위해 고른다
        v_norm = visual_features / np.linalg.norm(visual_features, axis=1, keepdims=True)
        s_norm = class_embeddings / np.linalg.norm(class_embeddings, axis=1, keepdims=True)
        scores = v_norm @ s_norm.T
    elif compatibility_fn == 'dot':
        scores = visual_features @ class_embeddings.T
    elif compatibility_fn == 'euclidean':
        # 거리의 음수를 점수로 쓴다
        scores = -np.sum((visual_features[:, None, :] - class_embeddings[None, :, :]) ** 2, axis=2)
    
    pred_indices = np.argmax(scores, axis=1)
    predictions = [class_names[idx] for idx in pred_indices]
    
    return predictions, scores
```

## 핵심 어려움

### 영역 이동

본 부류의 시각 특징은 못 본 부류와 분포가 다를 수 있다.

- **부류 안 흩어짐**: 같은 부류의 사례끼리도 달라 보인다
- **부류 사이 닮음**: 못 본 부류끼리가 본 부류와보다 더 닮았을 수 있다
- **쏘아 넣기 영역 이동**: 배운 옮김이 잘 일반화되지 않을 수 있다

### 중심 쏠림 문제

차원이 높은 공간에서는 어떤 점(중심점)이 다른 많은 점의 최근접 이웃이 되기 쉽다.

$$\text{중심쏠림}(c) = \sum_{\mathbf{x} \in \text{test}} \mathbb{1}[\text{NN}(\mathbf{x}) = c]$$

그래서 어떤 부류가 너무 자주 예측된다.

**누그러뜨리는 전략:**

- 국소 눈금 조절(CSLS)을 쓴다
- 중심 쏠림 줄이기 기법을 쓴다
- 묻힘 학습에 벌주기를 쓴다

### 뜻의 틈

뜻 공간이 시각의 갈림을 온전히 담지 못할 수 있다.

- 눈으로 닮은 부류가 서로 다른 뜻 표현을 가질 수 있다
- 눈으로 다른 부류가 닮은 뜻 표현을 가질 수 있다
- 촘촘한 갈림은 속성이나 낱말 묻힘으로 담기지 않을 수 있다

### GZSL의 치우침

본 부류로만 익힌 모델은 본 부류를 내놓는 쪽으로 크게 치우친다.

- 본 부류의 특징은 분포 안에 있다
- 어울림 점수가 본 부류에서 자연스레 더 높다
- 못 본 부류는 거의 예측되지 않는다

이는 뒤 절에서 다루는 눈금 맞춤과 균형 잡기 기법으로 다룬다.

## 요약

ZSL 틀은 다음으로 이루어진다.

1. **입력 공간** $\mathcal{X}$: 그림이나 그 밖의 데이터 갈래
2. **시각 부호기** $\phi$: 입력을 시각 특징으로 옮긴다
3. **뜻 공간** $\mathcal{S}$: 부류 표현을 준다
4. **어울림 함수** $F$: 시각과 뜻이 얼마나 맞는지를 잰다
5. **예측 규칙**: 가장 잘 어울리는 부류를 고른다

ZSL을 가능하게 하는 핵심 통찰은 뜻 표현이 본 부류와 못 본 부류를 이어 주어, 나누어 쓰는 속성이나 묻힘의 닮음을 거쳐 앎을 옮길 수 있다는 것이다.

## 연습문제

**연습문제 1.**
영 예시 학습을 정의하고 소수 예시 학습과 어떻게 다른지 설명하라.

??? success "연습문제 1 풀이"
    영 예시 학습은 학습 중에 한 번도 보지 못한 부류의 사례를, 본 부류와 못 본 부류를 잇는 딸린 정보(속성, 글 설명, 낱말 묻힘)를 써서 가려낸다. 소수 예시 학습은 이름표 붙은 보기를 몇 개 쓴다. 영 예시 학습은 대상 부류의 보기가 아예 없어도 되며 오로지 뜻 표현을 거친 앎의 옮김에 기댄다.

---

**연습문제 2.**
영 예시 학습의 중심 쏠림 문제와 그 다루는 법을 설명하라.

??? success "연습문제 2 풀이"
    차원이 높은 공간에서는 어떤 점('중심점')이 참 부류와 상관없이 다른 많은 점의 최근접 이웃이 된다. 그래서 최근접 이웃 기반 영 예시 분류에서 어떤 부류가 너무 자주 예측된다. 해법: 묻힘을 고르거나, 눈금 맞춘 점수 매기기를 쓰거나, 전용 거리 재기를 배운다.

---

**연습문제 3.**
영 예시 학습의 속성 기반 접근법과 묻힘 기반 접근법을 견주어라.

??? success "연습문제 3 풀이"
    속성 기반: 부류마다 이진이나 이어진 속성 벡터로 그려진다(이를테면 '줄무늬가 있다', '털이 있다'). 모델은 속성을 맞히는 법을 배운 다음 부류 원형과 맞춘다. 묻힘 기반: 시각 특징과 부류 설명(이를테면 부류 이름의 word2vec)을 함께 쓰는 공간으로 옮긴다. 묻힘 방법이 규모를 키우기 쉽고 손 표시가 덜 든다.

---

**연습문제 4.**
일반화된 영 예시 학습(GZSL)의 치우침 문제란 무엇인가?

??? success "연습문제 4 풀이"
    GZSL에서는 시험 때 본 부류와 못 본 부류가 함께 나온다. 모델은 본 부류로 익혔으므로 그쪽으로 치우친다. 해법: 눈금 맞춘 쌓기(본 부류 점수에서 치우침을 빼기), 본 부류와 못 본 부류를 가르는 분포 밖 알아채기, 또는 못 본 부류의 특징을 지어내는 생성 접근법.
