# 도메인 적응을 위한 최대 평균 불일치

최대 평균 불일치(MMD)는 커널 특징 공간에서 두 분포의 평균 임베딩 차이를 잰다. 특징 분포를 맞추는 도메인 적응 손실로 사용할 수 있지만, MMD가 작다는 사실만으로 목표 도메인의 예측 품질이나 시장 간 전이가 보장되지는 않는다.

## 1. 핵심 개념

- **커널 평균 임베딩**: 분포의 특징 기댓값을 RKHS의 한 점으로 나타낸다.
- **MMD와 제곱 MMD**: 평균 임베딩 거리와 그 제곱을 구분한다.
- **경험 추정**: 표본 수·독립성·커널·대각 항 처리에 따라 추정 성질이 달라진다.
- **도메인 적응**: 소스·목표 특징을 정렬하되 예측에 필요한 정보도 유지해야 한다.

## 2. 수학적 틀

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

### 경험적 MMD 계산

$P$에서 뽑은 유한 표본 $\{\mathbf{x}_i\}_{i=1}^m$과 $Q$에서 뽑은 $\{\mathbf{y}_j\}_{j=1}^n$이 주어지면 다음과 같다.

$$\widehat{\text{MMD}}^2 = \frac{1}{m(m-1)} \sum_{i \neq i'} k(\mathbf{x}_i, \mathbf{x}_{i'}) - \frac{2}{mn} \sum_{i,j} k(\mathbf{x}_i, \mathbf{y}_j) + \frac{1}{n(n-1)} \sum_{j \neq j'} k(\mathbf{y}_j, \mathbf{y}_{j'})$$

---

## 3. 이론적 성질

### 분포를 구분하는 조건

기댓값이 존재하고 characteristic 커널을 사용하면 MMD(P,Q)=0과 P=Q가 동치다. 양의 고정 폭 RBF는 대표적인 예다. 임의의 커널에 이 성질이 성립하는 것은 아니며, 유한 차원 선형 커널은 평균 차이만 볼 수 있다. MMD 자체가 거리인 조건과 제곱 MMD가 삼각 부등식을 만족하는지는 다르다.

### 불편 추정과 표본 오차

위 대각 제외 식은 각 분포의 독립 표본과 적절한 적률 조건에서 제곱 MMD의 불편 추정량이다. 따라서 기댓값의 편향은 정확히 0이며 “평균 편향이 O(1/n)로 수렴한다”는 표현과 다르다. 유한 표본의 값은 음수가 될 수도 있다.

유계 커널에서 표본 평균 임베딩의 오차는 대표적으로 $O_p(m^{-1/2}+n^{-1/2})$ 척도로 분석할 수 있다. 차원이 경계식에 직접 없더라도 커널 폭·거리 집중·검정력과 표본 의존성 때문에 고차원 성능이 자동 보장되지는 않는다. [MMD 원 논문](https://www.jmlr.org/papers/v13/gretton12a.html)의 가정을 확인한다.

## 4. 커널 고르기

### 도메인 적응에 흔한 커널

| 커널 | 형태 | 확인할 점 |
|---|---|---|
| RBF | $\exp(-\gamma\|x-y\|^2)$ | gamma>0, 거리 스케일·폭에 민감 |
| 라플라스 계열 | $\exp(-\gamma\|x-y\|)$ 등 | 노름·도메인과 커널 조건을 명시 |
| 다항 | $(x^Ty+c)^p$ | 차수·적률 조건, 모든 분포를 구분하지는 않음 |
| 다중 커널 | 비음수 가중 커널 합 | 가중치·대역폭과 선택 절차를 검증 |

### RBF의 대역폭 고르기

RBF 커널 $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x} - \mathbf{y}\|^2)$에 대해 다음과 같다.

**중앙값 휴리스틱**: 자기 자신과의 0거리·중복을 어떻게 다룰지 정한 표본 쌍 거리의 중앙값으로 대역폭을 정할 수 있다. 아래는 gamma 정의의 한 관례이며 중앙값이 0이면 별도 처리가 필요하다.

$$\gamma = \frac{1}{2 \text{median}\{\|\mathbf{x}_i - \mathbf{x}_j\|^2\}_{i,j}}$$

**선택 기준**: MMD만 최소화해 폭을 고르면 모든 커널값을 비슷하게 만들어 겉보기 거리를 줄이는 퇴화가 가능하다. 예측 품질·검정력·도메인 적응 가정에 맞는 독립 검증 규칙을 정한다. 목표 레이블을 사용하면 비지도 적응과도 구분한다.

---

## 5. 도메인 적응의 틀

### 학습 목표

도메인 적응은 아우른 손실을 가장 작게 한다.

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \text{MMD}^2(\mathbf{f}_s, \mathbf{f}_t)$$

여기서 각 기호는 다음과 같다.

- $\mathcal{L}_{\text{task}}$은 소스 레이블에 대한 지도 손실이다
- $\mathbf{f}_s, \mathbf{f}_t$은 배운 표현이다
- $\lambda$은 과제 성능과 분포 맞추기의 균형을 잡는다

### 구조 설계

MMD 기반 적응은 대개 다음을 쓴다.

1. **공유 특징 추출기**: 두 도메인에 모두 적용하는 $\mathbf{f}(\mathbf{x}) = \Phi(\mathbf{x}; \theta)$
2. **과제에 맞는 분류기**: $y = W^T \mathbf{f}(\mathbf{x}) + b$
3. **MMD 손실**: 특징 공간에서 계산한다

```
Input Domain S ──┐
                 ├──> Feature Extractor ──> Task Classifier ──> Output
Input Domain T ──┘     (Shared)
                     │
                     └──> MMD Loss (Domain Alignment)
```

---

## 6. MMD의 기울기 흐름

미분 가능한 특징 추출기와 커널을 쓰면 표본 커널 합에서 특징으로 자동 미분할 수 있다. 아래 코드와 같은 대각 포함 추정량, RBF 폭 sigma에서 소스 특징 x_i의 기울기는 다음과 같다.

$$
\nabla_{x_i}\widehat{\mathrm{MMD}}_b^2
=-\frac{2}{m^2\sigma^2}\sum_a k(x_i,x_a)(x_i-x_a)
+\frac{2}{mn\sigma^2}\sum_j k(x_i,y_j)(x_i-y_j)
$$

특징 추출기 매개변수에는 이 경사를 연쇄 법칙으로 전달한다. 양쪽 특징을 모두 갱신할 수 있어 “소스만 목표 쪽으로 이동한다”는 해석은 부정확하다. 상수 특징으로 모두 붕괴해도 분포 차이가 작아지므로 과제 손실·구조 조건이 필요하다.

## 7. 실용적인 고려

### 계산 복잡도

쌍마다 커널을 셈하는 데 다음이 든다.

$$O(N^2 d)$$

여기서 $N = m + n$은 전체 표본 크기이고 $d$은 특징 차원이다.

!!! note "작은 배치로 다듬기"
    작은 배치를 써서 복잡도를 줄인다. $k \ll N$일 때 $O(k^2 d)$이다.

### 키우는 방법

**무작위 푸리에 특징**: r개 특징으로 RBF를 근사하면 일반적인 밀집 투영 비용은 $O(Ndr)$이고 특징 평균 집계는 $O(Nr)$다. r이 원래 d보다 작아야 한다는 필수 조건은 없으며 근사 정확도와 비용을 비교한다.

k(x,y)\approx\frac2r\sum_{i=1}^r\cos(w_i^Tx+b_i)\cos(w_i^Ty+b_i),\qquad b_i\sim U[0,2\pi]

**니스트룀 추정**: 표본의 일부로 커널 행렬을 추정한다.

---

## 8. 넓힘과 변형

### 다중 커널

비음수 가중치의 커널 합에 대한 제곱 MMD는 각 커널 제곱 MMD의 가중합이다. 학습 가중치가 가장 판별하기 어려운 커널에만 몰리지 않게 목적·제약을 함께 정해야 한다.

### 조건부 정렬

주변 특징 분포 정렬 외에 클래스별 특징을 비교할 수 있다. 예를 들어 공통 클래스에서 다음 손실을 구성한다.

$$
\mathcal L_{\mathrm{align}}=\mathrm{MMD}^2(P_s^f,P_t^f)+\lambda\sum_c w_c\mathrm{MMD}^2(P_s^{f\mid c},P_t^{f\mid c})
$$

목표 정답 대신 의사 레이블을 쓰는 구성도 있지만 오류·빈 클래스·레이블 이동에 주의해야 한다. 레이블 주변분포만 맞추는 식이 조건부 특징 분포를 맞추는 것과 같지는 않다.

## 9. 퀀트 금융에서의 쓰임

시장·자산·기간의 특징 분포 차이를 탐색하는 통계로 MMD를 사용할 수 있다. 관측 빈도·표본 의존·스케일과 선택한 커널의 영향을 함께 확인한다.

분포 거리가 작아도 수익률과 특징의 관계가 달라질 수 있다. 전체 특징을 정렬해 예측에 필요한 차이를 없애는 음의 전이도 가능하므로 시간 분할의 목표 오차를 검증해야 한다. 이 페이지에는 시장·자산 간 전이를 입증하는 실험이 없다.

## 10. 이론적 보장

목표 예측 오차를 소스 오차와 MMD만 더한 식으로 일반적으로 제한할 수는 없다. 입력 분포가 같아 MMD=0이어도 두 도메인의 정답 규칙이 반대면 소스에서 정확한 모델이 목표에서 실패할 수 있다.

MMD를 포함한 위험 경계에는 손실 함수가 해당 RKHS에서 어떻게 제한되는지, 라벨 관계·공통 가설·표본과 복잡도 등의 가정이 필요하다. 정의되지 않은 “도메인 적응 차원”을 넣은 식으로 전이 성능을 보장하지 않는다.

## 11. 관련 주제

- [미세 조정](../transfer_learning/fine_tuning.md)
- [시계열 전이](../applications/transfer_time_series.md)
- 조건부 정렬·레이블 이동·적대적 도메인 적응: 각각의 가정과 별도 검증이 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
MMD의 정의와 도메인 정렬의 해석 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    MMD는 두 커널 평균 임베딩의 RKHS 거리다. characteristic 커널 등의 조건에서 모집단 MMD가 0이면 분포가 같다. 학습 특징 분포를 맞춰도 라벨 관계가 다르거나 표현이 붕괴하면 목표 예측이 나빠질 수 있어 작은 거리만으로 좋은 전이를 입증할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
m개 소스·n개 목표 표본의 제곱 MMD 불편 추정에서 대각선을 제외하는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    같은 분포의 두 독립 표본 커널 기대값을 추정하려고 소스는 m(m−1), 목표는 n(n−1)개 서로 다른 쌍을 사용한다. 교차항은 mn개다. m,n≥2와 독립성 조건이 필요하며 이 U-통계량은 유한 표본에서 음수가 될 수 있다. 아래 코드는 대각 포함 V-통계량이라 다른 추정량이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 MMD 함수의 추정량·커널 인수·대역폭과 입력 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"
    torch import가 필요하고 kernel 인수는 사용되지 않아 항상 RBF다. bandwidth는 양의 sigma이며 0은 허용하지 않는다. cdist는 소스·목표의 같은 특징 차원·호환 장치·지원 자료형을 가정한다. 같은 도메인의 대각도 평균해 불편 추정량이 아니며 빈 표본은 mean이 유효하지 않다. 수치 오차로 미세한 음수가 날 가능성과 전체 쌍 메모리도 확인한다.

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

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
MMD와 DANN이 정렬 대상을 표현하는 방식 및 공통 한계를 비교하라.

</div>

??? success "연습문제 4 풀이"
    MMD는 선택한 커널의 평균 임베딩 차이를, DANN은 학습한 도메인 판별 신호를 이용한다. 커널·대역폭과 판별기 용량·학습 균형이 각각 중요하다. 어느 방법도 목표 라벨 관계를 자동 보존하거나 모든 분포 이동에서 안정적인 성능을 보장하지 않는다. 같은 자료·목표 평가로 비교한다.

## 정리하며

MMD와 제곱 MMD, 모집단 성질과 U·V 추정량을 구분한다. 커널·대역폭·표본 의존성과 특징 붕괴를 확인하고, 분포 정렬만으로 목표 도메인 오차를 보장하지 않는다.
