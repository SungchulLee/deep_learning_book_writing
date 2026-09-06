# 도메인 적응을 위한 최대 평균 불일치
## 들어가며

최대 평균 불일치(MMD)는 원천 도메인과 목표 도메인 사이의 분포 차이를 재고 가장 작게 하는, 원칙에 바탕한 이론적 방법이다. 원천과 목표의 분포가 가장 적게 어긋나는 표현을 배움으로써 MMD 기반 도메인 적응은 이름표 붙은 목표 데이터 없이도 튼튼한 전이 학습을 가능케 한다.

MMD가 커널 이론과 확률 거리에 바탕한다는 점은 분포 맞추기에 대한 이론적 보장이 필요한 실무자에게 특히 매력적이다. 계량 금융에서 MMD는 적응이 잘되었는지를 거리로 해석할 수 있게 재면서 시장 국면과 자산 갈래를 넘나드는 적응을 가능케 한다.

## 핵심 개념

- **분포 불일치**: 확률 분포 사이의 차이를 재는 것
- **최대 평균 불일치**: 이론적 성질을 지닌 커널 기반 발산 척도
- **커널 임베딩**: 분포를 재생 커널 힐베르트 공간으로 은근히 잇대기
- **도메인 불변 특징**: 원천과 목표의 분포가 맞물리는 표현
- **비지도 도메인 적응**: 목표 도메인의 이름표 없이 배우기

## 수학적 틀

### MMD의 정의

분포 $P$과 $Q$ 사이의 최대 평균 불일치는 다음으로 정의한다.

$$\text{MMD}^2(P, Q) = \left\| \mathbb{E}_{\mathbf{x} \sim P}[\phi(\mathbf{x})] - \mathbb{E}_{\mathbf{y} \sim Q}[\phi(\mathbf{y})] \right\|_H^2$$

여기서 각 기호는 다음과 같다.

- $\phi(\cdot)$은 재생 커널 힐베르트 공간(RKHS)으로 가는 특징 잇댐이다
- $\|\cdot\|_H$은 RKHS의 노름이다
- $\mathbb{E}$은 기댓값을 나타낸다

### 커널 표현

커널 함수 $k(\mathbf{x}, \mathbf{y}) = \langle \phi(\mathbf{x}), \phi(\mathbf{y}) \rangle_H$을 쓰면 MMD가 다음으로 펼쳐진다.

$$\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(\mathbf{x}, \mathbf{x}')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(\mathbf{x}, \mathbf{y})] + \mathbb{E}_{y,y' \sim Q}[k(\mathbf{y}, \mathbf{y}')]$$

### 경험적 MMD 셈하기

$P$에서 뽑은 유한 표본 $\{\mathbf{x}_i\}_{i=1}^m$과 $Q$에서 뽑은 $\{\mathbf{y}_j\}_{j=1}^n$이 주어지면 다음과 같다.

$$\widehat{\text{MMD}}^2 = \frac{1}{m(m-1)} \sum_{i \neq i'} k(\mathbf{x}_i, \mathbf{x}_{i'}) - \frac{2}{mn} \sum_{i,j} k(\mathbf{x}_i, \mathbf{y}_j) + \frac{1}{n(n-1)} \sum_{j \neq j'} k(\mathbf{y}_j, \mathbf{y}_{j'})$$

## 이론적 성질

### 보편성

!!! tip "보편 커널"
    (RBF나 라플라스 같은) 보편 커널을 쓰면 MMD가 제대로 된 거리이다(대칭성, 음이 아님, 삼각 부등식을 만족한다).

보편 커널에서는 $\text{MMD}(P, Q) = 0 \iff P = Q$이다.

### 수렴 속도

큰 수의 법칙에 따라 표본이 $N = m + n$개일 때 다음과 같다.

$$\mathbb{E}[\widehat{\text{MMD}}^2 - \text{MMD}^2(P,Q)] = O\left(\frac{1}{\min(m,n)}\right)$$

수렴이 특징 차원과 무관하므로 높은 차원의 표현에도 튼튼하다.

## 커널 고르기

### 도메인 적응에 흔한 커널

| 커널 | 식 | 좋은 점 | 맞바꿈 |
|--------|---------|-----------|-----------|
| **RBF** | $\exp(-\gamma\|\mathbf{x}-\mathbf{y}\|^2)$ | 보편적이고 해석하기 좋다 | 띠너비 고르기가 매우 중요하다 |
| **라플라스** | $\exp(-\gamma\|\mathbf{x}-\mathbf{y}\|)$ | 극단값에 튼튼하다 | 덜 매끄럽다 |
| **다항** | $(\mathbf{x}^T\mathbf{y} + c)^d$ | 서로 주고받음을 잡아낸다 | 차원의 저주 |
| **다중 커널** | $\sum_k \alpha_k k_k$ | 자유롭다 | 맞추기가 까다롭다 |

### RBF의 띠너비 고르기

RBF 커널 $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x} - \mathbf{y}\|^2)$에 대해 다음과 같다.

**중앙값 어림법**:

$$\gamma = \frac{1}{2 \text{median}\{\|\mathbf{x}_i - \mathbf{x}_j\|^2\}_{i,j}}$$

**교차 검증**: 도메인 적응 손실이 가장 작아지도록 $\gamma$을 맞춘다.

## 도메인 적응의 틀

### 학습 목표

도메인 적응은 아우른 손실을 가장 작게 한다.

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \text{MMD}^2(\mathbf{f}_s, \mathbf{f}_t)$$

여기서 각 기호는 다음과 같다.

- $\mathcal{L}_{\text{task}}$은 원천 이름표에 대한 지도 손실이다
- $\mathbf{f}_s, \mathbf{f}_t$은 배운 표현이다
- $\lambda$은 과제 성능과 분포 맞추기의 균형을 잡는다

### 구조 설계

MMD 기반 적응은 대개 다음을 쓴다.

1. **함께 쓰는 특징 뽑개**: 두 도메인에 모두 적용하는 $\mathbf{f}(\mathbf{x}) = \Phi(\mathbf{x}; \theta)$
2. **과제에 맞는 분류기**: $y = W^T \mathbf{f}(\mathbf{x}) + b$
3. **MMD 손실**: 특징 공간에서 셈한다

```
Input Domain S ──┐
                 ├──> Feature Extractor ──> Task Classifier ──> Output
Input Domain T ──┘     (Shared)
                     │
                     └──> MMD Loss (Domain Alignment)
```

## MMD의 기울기 흐름

역전파 중에 MMD의 기울기가 특징을 맞물리도록 이끈다.

$$\frac{\partial \text{MMD}^2}{\partial \Phi} = \frac{2}{m} \sum_i \phi'(\mathbf{x}_i) k'(\mathbf{x}_i, \cdot) - \frac{2}{n} \sum_j \phi'(\mathbf{y}_j) k'(\mathbf{y}_j, \cdot)$$

이것이 원천의 특징을 목표 분포 쪽으로 자연스레 몬다.

## 실용적인 고려

### 계산 복잡도

쌍마다 커널을 셈하는 데 다음이 든다.

$$O(N^2 d)$$

여기서 $N = m + n$은 전체 표본 크기이고 $d$은 특징 차원이다.

!!! note "작은 배치로 다듬기"
    작은 배치를 써서 복잡도를 줄인다. $k \ll N$일 때 $O(k^2 d)$이다.

### 키우는 방법

**무작위 푸리에 특징**: $d' \ll d$일 때 $O(N d')$ 복잡도로 RBF 커널을 어림한다.

$$k(\mathbf{x}, \mathbf{y}) \approx \frac{1}{m} \sum_{i=1}^m \cos(\mathbf{w}_i^T \mathbf{x} + b_i) \cos(\mathbf{w}_i^T \mathbf{y} + b_i)$$

**니스트룀 어림**: 표본의 일부로 커널 행렬을 어림한다.

## 넓힘과 변형

### 다중 커널 MMD

학습된 가중치로 여러 커널을 아우른다.

$$\text{MMD}^2_{\text{multi}} = \sum_{k} \beta_k \text{MMD}^2_k(P, Q)$$

여기서 $\beta_k \geq 0$이고 $\sum_k \beta_k = 1$이다.

### 결합 분포 적응 (JDA)

주변 분포와 조건부 분포를 함께 맞춘다.

$$\mathcal{L} = \text{MMD}(P_s, P_t) + \text{MMD}(P_s(y), P_t(y))$$

### 조건부 MMD

부류 이름표를 조건으로 불일치를 줄인다 (목표 이름표가 얼마간 필요하다).

$$\text{CMMD} = \sum_c P(c) \text{MMD}(P_s(\mathbf{x}|c), P_t(\mathbf{x}|c))$$

## 계량 금융에서의 쓰임

!!! warning "시장 국면 적응"
    MMD로 시장 국면을 넘나들며 모델을 맞춘다.
    
    - **상승장 → 횡보장**: 상승장 특징과 횡보장 특징 사이의 불일치를 가장 작게 한다
    - **미국 → 신흥 시장**: MMD 정렬로 시장 사이를 옮겨 간다
    - **주식 → 채권**: 특징의 맞물림을 지키는 자산 갈래 적응

## 이론적 보장

MMD는 목표 도메인 오차의 한계를 준다.

$$\mathcal{E}_t(\mathbf{f}) \leq \mathcal{E}_s(\mathbf{f}) + \text{MMD}^2(P_s, P_t) + O(\sqrt{\frac{d_A}{n}})$$

여기서 $d_A$은 도메인 적응 차원이고 $n$은 목표 표본의 크기이다.

## 관련 주제

- 도메인 적응 훑어보기 (10.2절)
- 자기 학습 방법 (10.2.3절)
- 다중 원천 도메인 적응 (10.2.2절)
- 적대적 도메인 적응

## 연습문제

**연습문제 1.**
최대 평균 불일치(MMD)를 정의하고 도메인 적응에서의 몫을 설명하라.

??? success "연습문제 1 풀이"
    MMD는 재생 커널 힐베르트 공간에서 두 분포 $P$과 $Q$의 평균 임베딩을 견주어 그 사이의 거리를 잰다: $\text{MMD}^2(P,Q) = \|\mu_P - \mu_Q\|_H^2$. 도메인 적응에서 원천과 목표의 특징 분포 사이의 MMD를 가장 작게 하면 두 도메인이 맞물려, 원천에서 학습한 분류기가 목표에서도 통하게 된다.

---

**연습문제 2.**
두 분포에서 뽑은 표본이 주어졌을 때 MMD의 불편 추정량을 이끌어 내라.

??? success "연습문제 2 풀이"
    $\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i\neq j} k(x_i, x_j) + \frac{1}{m(m-1)}\sum_{i\neq j} k(y_i, y_j) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j)$이며 여기서 $k$은 (대개 가우스 RBF인) 커널 함수이다.

---

**연습문제 3.**
도메인 적응 학습을 위한 MMD 손실을 파이토치에서 구현하라.

??? success "연습문제 3 풀이"
    ```python
    def mmd_loss(source_features, target_features, kernel='rbf', bandwidth=1.0):
        def rbf(x, y):
            return torch.exp(-torch.cdist(x, y)**2 / (2 * bandwidth**2))
        xx = rbf(source_features, source_features).mean()
        yy = rbf(target_features, target_features).mean()
        xy = rbf(source_features, target_features).mean()
        return xx + yy - 2 * xy
    ```

---

**연습문제 4.**
MMD와 적대적 도메인 적응(DANN)을 견주어라. 각각의 이점은 무엇인가?

??? success "연습문제 4 풀이"
    MMD는 거리 척도가 드러나 있고 적대적 학습의 불안정이 없으며 커널 선택이 초매개변수이다. DANN은 도메인 판별기를 써서 더 자유롭고 복잡한 분포 차이를 잡아낼 수 있다. MMD가 더 간단하고 안정적이며 DANN이 더 힘 있지만 학습이 어렵다. 도메인 이동이 어지간하면 MMD가 잘 통한다.
