# 분할 정복

분할 정복(divide and conquer)은 문제를 서로 독립인 부분문제로 나누어 재귀적으로 풀고 결과를 결합한다. 이 접근법은 멀티헤드 어텐션, 재귀 신경망, 계층적 특징 추출 등 딥러닝 전반에 나타난다.

## 정의

분할 정복 알고리즘은 세 단계로 이루어진다. 문제를 크기 $n/b$인 부분문제 $a$개로 **분할** 하고, 각 부분문제를 재귀적으로 **정복** 하며, 해들을 **결합** 한다. 실행 시간은 다음 점화식을 만족한다.

$$
T(n) = a \, T\!\left(\frac{n}{b}\right) + f(n)
$$

여기서 $f(n)$은 분할하고 결합하는 비용이다.

## 설명

핵심 통찰은, 더 작은 부분문제를 독립적으로 풀고 결합하는 편이 전체 문제를 직접 푸는 것보다 빠른 경우가 많다는 것이다. 마스터 정리가 해를 알려준다.

- $f(n) = O(n^{\log_b a - \epsilon})$이면 $T(n) = \Theta(n^{\log_b a})$ (재귀가 지배)
- $f(n) = \Theta(n^{\log_b a})$이면 $T(n) = \Theta(n^{\log_b a} \log n)$ (균형)
- $f(n) = \Omega(n^{\log_b a + \epsilon})$이면 $T(n) = \Theta(f(n))$ (결합이 지배)

딥러닝에서 분할 정복은 다음과 같이 나타난다.

- **멀티헤드 어텐션**: 임베딩 차원을 $h$개의 헤드로 나누고 어텐션을 독립적으로 계산한 뒤 이어 붙인다(결합).
- **계층적 모델**: U-Net은 여러 해상도에서 특징을 처리하며 각 수준에서 나누고 합친다.
- **병렬 계산**: 배치를 여러 GPU에 나누어 경사를 독립적으로 계산하고 평균 내는(결합) 것이 분할 정복이다.

## 예제

```python
import torch

# 분할 정복 방식의 행렬 곱(Strassen 개념)
# 표준: O(n^3). Strassen: 재귀적 곱 7회로 O(n^2.81)
def recursive_sum(x: torch.Tensor) -> torch.Tensor:
    """분할 정복 합: 나누고, 재귀하고, 결합한다."""
    if x.numel() == 1:
        return x.squeeze()
    mid = x.numel() // 2
    left = recursive_sum(x[:mid])
    right = recursive_sum(x[mid:])
    return left + right

x = torch.arange(1, 9, dtype=torch.float32)
print(f"Recursive sum of {x.tolist()}: {recursive_sum(x).item()}")
print(f"Direct sum: {x.sum().item()}")

# 분할 정복으로서의 멀티헤드 어텐션
d_model, n_heads = 64, 4
d_head = d_model // n_heads
Q = torch.randn(1, 8, d_model)  # (배치, 시퀀스 길이, d_model)

# 분할: 헤드로 나눈다
heads = Q.view(1, 8, n_heads, d_head).transpose(1, 2)  # (1, n_heads, 8, d_head)
print(f"Divided into {n_heads} heads of dim {d_head}")

# 결합: 헤드를 이어 붙인다
combined = heads.transpose(1, 2).contiguous().view(1, 8, d_model)
print(f"Combined back to shape {combined.shape}")
print(f"Reconstruction matches: {torch.allclose(Q, combined)}")
```

## 연습문제

**연습문제 1.**
배열을 크기 $n/2$인 부분문제 2개로 나누고 $O(n)$ 시간에 병합하는 병합 정렬의 시간 복잡도를 마스터 정리로 구하라.

??? success "연습문제 1 풀이"
    점화식은 $T(n) = 2T(n/2) + O(n)$이다. 여기서 $a = 2$, $b = 2$, $f(n) = O(n)$이다. $\log_b a = \log_2 2 = 1$을 계산한다. $f(n) = \Theta(n^1) = \Theta(n^{\log_b a})$이므로 마스터 정리의 경우 2에 해당하며 $T(n) = \Theta(n^{\log_b a} \log n) = \Theta(n \log n)$이다.

---

**연습문제 2.**
멀티헤드 어텐션은 $d_{\text{model}} = 768$을 $h = 12$개의 헤드로 나눈다. 각 헤드는 $d_k = 64$ 차원에서 어텐션을 계산한다. 전체 계산량이 단일 헤드인 경우와 같음을 보이고, 나누는 것의 이점을 설명하라.

??? success "연습문제 2 풀이"
    $d_{\text{model}} = 768$에서의 단일 헤드 어텐션: $QK^\top$ 행렬의 비용은 $O(n^2 \cdot 768)$이다. 멀티헤드: 12개 헤드가 각각 $d_k = 64$ 차원에서 $QK^\top$을 $O(n^2 \cdot 64)$ 비용으로 계산한다. 총합 $12 \times O(n^2 \cdot 64) = O(n^2 \cdot 768)$으로 같다. 이점은 각 헤드가 독립적인 부분공간에서 작동하므로 입력의 서로 다른 측면(다른 위치, 다른 특징 상호작용)에 주목할 수 있다는 것이다. 이는 앙상블 방법과 유사하다. 여러 개의 다양한 "약한" 어텐션 패턴이 결합하여 하나의 "강한" 어텐션보다 풍부한 표현을 만든다.

---

**연습문제 3.**
U-Net은 다운샘플링과 업샘플링을 통해 여러 해상도에서 특징을 처리한다. 입력이 $256 \times 256$이고 (각각 공간 차원을 절반으로 줄이는) 다운샘플링 단계가 4개라면, 각 단계의 공간 차원을 나열하라. 이것이 분할 정복과 어떻게 관련되는가?

??? success "연습문제 3 풀이"
    0단계: $256 \times 256$, 1단계: $128 \times 128$, 2단계: $64 \times 64$, 3단계: $32 \times 32$, 4단계(병목): $16 \times 16$. 업샘플링 경로는 이 차원들을 역순으로 되돌린다. 이것은 분할 정복이다. "분할" 단계는 공간 해상도를 줄이고(더 거친 특징을 처리), "정복" 단계는 각 해상도에서 독립적으로 처리하며, "결합" 단계는 고해상도 건너뛰기 연결과 업샘플링된 특징을 합쳐 세밀한 출력을 만든다.

---

**연습문제 4.**
데이터 병렬화는 크기 $B$인 배치를 $G$개의 GPU에 나누어 경사를 독립적으로 계산하고 평균 낸다. 단일 GPU 계산 시간 $T_{\text{comp}}(B/G)$와 매개변수 $P$개를 평균 내는 통신 비용 $T_{\text{comm}}(P)$으로 시간 복잡도의 점화식을 작성하라.

??? success "연습문제 4 풀이"
    단계당 총 시간은 $T = T_{\text{comp}}(B/G) + T_{\text{comm}}(P)$이다. 계산이 분할되고(각 GPU가 $B/G$개 표본을 처리), 독립적으로 수행되며(정복), 경사가 평균된다(결합). 속도 향상은 $T_{\text{comp}}(B) / T = T_{\text{comp}}(B) / (T_{\text{comp}}(B/G) + T_{\text{comm}}(P))$이다. 선형 확장의 경우 $T_{\text{comp}}(B/G) = T_{\text{comp}}(B) / G$이므로 속도 향상은 $G / (1 + G \cdot T_{\text{comm}} / T_{\text{comp}})$가 된다. 통신 부담이 실제 속도 향상을 제한하는데, 이는 결합 단계가 분할 정복의 효율을 제한하는 것과 같은 이치이다.

---

**연습문제 5.**
Strassen 알고리즘은 $n \times n$ 행렬을 곱할 때 크기 $n/2$의 재귀적 곱셈을 8회가 아니라 7회 수행한다. 표준 점화식과 Strassen 점화식 모두에 마스터 정리를 적용하여 비교하라.

??? success "연습문제 5 풀이"
    **표준**: $T(n) = 8T(n/2) + O(n^2)$. $\log_2 8 = 3$이고 $\epsilon = 1$에 대해 $f(n) = O(n^2) = O(n^{3-\epsilon})$이다. 경우 1: $T(n) = \Theta(n^3)$. **Strassen**: $T(n) = 7T(n/2) + O(n^2)$. $\log_2 7 \approx 2.807$이고 $\epsilon \approx 0.807$에 대해 $f(n) = O(n^2) = O(n^{2.807 - \epsilon})$이다. 경우 1: $T(n) = \Theta(n^{\log_2 7}) \approx \Theta(n^{2.807})$. Strassen은 점근적으로 $n^{0.193}$ 인자를 절약하지만, 상수 인자 때문에 실제로는 $n$이 클 때만 유리하다.
