# Contrastive Divergence(CD)

에너지 기반 모델의 우도 그래디언트에는 모델 분포의 기댓값이 들어간다. 매 업데이트마다 긴 사슬을 돌리는 대신 데이터에서 시작한 짧은 사슬로 이 항을 근사하는 방법이 CD다. 계산량을 줄이는 대가로 어떤 분포의 통계를 사용하게 되는지 확인하는 것이 핵심이다.

## 1. 코드의 역할

```python
"""
맞댐 벌어짐: 효율 좋은 제한 볼츠만 기계 익히기
============================================

이 단원은 제한 볼츠만 기계를 쓸모 있게 만든 알고리즘인
견줌 갈림(CD)을 깊이 파고든다. CD-k의 변형을 살펴본다.
이어지는 맞댐 벌어짐과 모임의 성질.

걸리는 시간: 75~90분
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

# 살피기를 곁들인 두루 갖춘 맞댐 벌어짐 짜기
# ... (비슷하게 자세한 구조)

def main():
    print("Module complete - see 04_restricted_boltzmann_machines.py for RBM+CD")

if __name__ == "__main__":
    main()
```

**보존된 출력:**

```
Module complete - see 04_restricted_boltzmann_machines.py for RBM+CD
```

위 main은 [앞 페이지의 RBM 학습 코드](04_restricted_boltzmann_machines.md)를 안내하는 문구만 출력한다. CD 학습이나 데이터 로드를 이 파일에서 수행하지 않는다. 여기서는 앞 페이지의 실제 업데이트를 수식과 연결한다.

## 2. 정확한 우도 그래디언트와 CD-k

온도 1에서 모델 pθ∝exp(−Eθ)의 관측 로그 우도 그래디언트는 데이터에 대응하는 positive phase와 모델 전체의 negative phase 차이다. RBM의 결합 Wⱼᵢ에서는 다음과 같다.

$$\frac{\partial\log p_\theta(v)}{\partial W_{ji}}
=v_i\mathbb E[h_j\mid v]-\mathbb E_{p_\theta}[v_i h_j]$$

CD-k는 데이터 분포 q₀에서 시작해 모델 전이 Kθ를 k번 적용한 qₖ=q₀Kθᵏ의 통계를 두 번째 항에 사용한다. qₖ가 평형 pθ와 다르면 일반적으로 바이어스가 생긴다. 개별 샘플의 우연한 차이와 기대 그래디언트의 바이어스는 구분한다.

[Hinton의 CD 연구](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)는 짧은 사슬로 학습하는 접근을 제시한다. CD라는 양을 KL(q₀||pθ)−KL(qₖ||pθ)로 표현할 수 있지만 qₖ 자체도 θ에 의존한다. 통상 CD 업데이트는 그 의존성에 따른 추가 미분항을 생략하므로 위 KL 차이의 정확한 전체 그래디언트라고 설명하지 않는다.

## 3. 확률과 샘플을 사용하는 위치

앞 페이지 구현은 positive phase에 은닉 확률 ph₀를 사용하고 negative phase에 최종 은닉 샘플 hₖ를 사용한다. 같은 데이터 v₀에서 서로 다른 난수로 다른 업데이트가 나올 수 있다. 두 phase를 모두 샘플로 계산하는 다른 관례와 수치가 다르므로 손계산에서도 어떤 통계를 쓰는지 명시한다.

CD의 재구성 오차는 짧은 사슬 뒤 원본과의 차이다. 원본 부근에서만 잘 머무는 사슬은 낮은 오차를 보이면서 다른 중요한 모드를 탐색하지 못할 수 있다. 재구성 오차 감소를 NLL 감소나 평형 생성 품질 개선의 증거로 대체하지 않는다.

## 4. Persistent Contrastive Divergence(PCD)

PCD는 매 업데이트에 사슬을 데이터로 재초기화하지 않고 이전 negative phase의 상태를 이어 간다. 모델이 사슬의 혼합 속도보다 천천히 변하면 평형을 추적하는 데 도움이 될 수 있다. 하지만 먼 모드 사이의 높은 장벽은 persistent chain에도 문제가 되며 PCD가 자동으로 해결하지 않는다.

k를 늘리면 전이 비용이 늘고 고정 모델에서 긴 시간의 평형 수렴을 기대할 수 있다. 그렇다고 유한 k의 바이어스·분산이나 학습 결과가 항상 단조 개선되는 것은 아니다. 사슬 수, 초기화, 파라미터 변화 속도와 실효 표본 수를 함께 고려한다. 배치 크기에 따라 사슬을 재시작하는 구현은 [앞 페이지의 PCD 풀이](04_restricted_boltzmann_machines.md)처럼 별도 제한이 있다.

## 5. 자유 에너지 차이를 진단에 쓰기

같은 모델 안에서 두 관측치의 로그확률 차이는 자유 에너지 차이로 계산할 수 있다. log p(v)=−F(v)−log Z이므로 Z가 약분되기 때문이다. 그러나 모델이 바뀌는 학습 시점 간의 평균 F만 비교하면 log Z도 바뀌는 항을 놓친다.

연습문제의 monitor_training은 데이터와 독립 Bernoulli(0.5) 노이즈의 평균 자유 에너지 차이를 반환한다. 특정 노이즈와 데이터를 구분하는 상대 신호일 뿐 NLL 추정기가 아니다. 픽셀 희소성 같은 쉬운 차이만 학습해도 격차가 커질 수 있고, 전체 모델 분포가 좋아진다는 보장도 없다.

작은 RBM은 상태 열거로 Z와 테스트 우도를 직접 점검할 수 있다. 큰 모델에서는 적절한 [AIS](partition_function.md) 등 별도 정규화 상수 추정이 필요하다. 추정 바이어스·분산과 전처리 조건을 명시하고 표본·분포 평가와 함께 해석한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
앞 페이지와 같은 확률 positive phase를 사용하라. v₀=(1,0), W=(0.5,−0.3), a=b=0에서 h₀=1, v₁=(1,0), h₁=0이 뽑혔다면 CD-1 통계와 갱신 방향은 무엇인가?

</div>

??? success "연습문제 1 풀이"

    ph₀=sigmoid(0.5)≈0.62246이므로 positive W 통계는 (0.62246,0)다. h₁=0이라 negative 통계는 (0,0)이고 ΔW=lr×(0.62246,0)다. Δa=0, Δb=lr×0.62246이다. h₀ 샘플을 positive 통계로 사용하는 다른 관례의 (1,0)과 구분한다. 지정된 샘플 경로는 가능한 한 경우이며 기대 그래디언트 자체는 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CD-k와 PCD의 시작점 차이와 PCD가 실패할 수 있는 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"

    CD-k는 매번 데이터에서 시작하고 PCD는 이전 모델 사슬 상태를 유지한다. 모델이 빨리 변하거나 모드 간 혼합이 느리면 PCD 입자가 현재 평형을 따라가지 못할 수 있다. 작은 러닝레이트와 긴 유지 시간만으로 모든 모드 탐색이 보장되는 것도 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 monitor_training이 로그 우도 추정기를 구현하는가? 반환하는 세 값의 의미와 제한을 설명하라.

</div>

??? success "연습문제 3 풀이"

    데이터 평균 자유 에너지, Bernoulli(0.5) 노이즈 평균 자유 에너지, 두 값의 차이를 반환한다. Z를 추정하지 않아 NLL은 아니다. 노이즈와의 격차가 커져도 모델의 나머지 상태 질량이나 테스트 우도가 좋아졌다고 단정할 수 없다. random_v는 CPU에서 생성하므로 rbm·입력의 장치와도 맞춰야 한다. n_random과 데이터는 비어 있지 않아야 한다.

    ```python
    def monitor_training(rbm, train_data, n_random=100):
        with torch.no_grad():
            fe_data = rbm.free_energy(train_data).mean().item()
            random_v = torch.bernoulli(torch.ones(n_random, rbm.n_visible) * 0.5)
            fe_random = rbm.free_energy(random_v).mean().item()
        return fe_data, fe_random, fe_random - fe_data
    ```


## 정리하며

CD는 데이터에서 출발한 짧은 사슬로 negative phase를 근사하고 PCD는 사슬을 이어 간다. 두 방법의 혼합·바이어스 조건을 구분하며, 확률·샘플의 통계 관례와 재구성·자유 에너지 차이의 평가 범위를 명시한다.
