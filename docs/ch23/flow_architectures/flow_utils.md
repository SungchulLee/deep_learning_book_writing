# 고르게 하는 흐름

고르게 하는 흐름: 바탕 도구와 바탕 갈래. 가르치기 위한 것

고르게 하는 흐름은 뒤집을 수 있는 바꿈으로 정확한 가능도 셈하기를 준다. 이 짜기는 깊은 배움 개념을 보이며, 일대일 대응의 차례를 거쳐 단순한 분포가 복잡한 분포로 어떻게 바뀌는지 드러낸다.

## 코드

```python
"""
고르게 하는 흐름: 바탕 도구와 바탕 갈래

================================================================================
가르치기 위한 것
================================================================================
이 단원은 고르게 하는 흐름의 바탕 벽돌을 준다.
만들어 내는 모델과 고르게 하는 흐름을 처음 배우는
학부생을 위해 만들었다.

다루는 핵심 개념:
1. 바탕 분포(뽑기를 시작하는 곳)
2. 흐름 바꿈(분포를 바꾸는 법)
3. 야코비 행렬식(부피 바뀜 좇기)
4. 바꿈 여럿 아우르기
5. 최대 가능도 익히기

선수 지식:
- 확률 분포에 대한 이해
- 기본 신경망(PyTorch)
- 여러 변수 미적분(야코비)
- 선형 대수의 바탕

배움 길:
BaseDistribution → Flow → FlowSequence → 단순한 보기로 시작하고
이어 coupling_flows.py의 더 복잡한 바꿈으로 넘어간다

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class BaseDistribution:
    """
    바탕 분포: 고르게 하는 흐름의 출발점
    
    ============================================================================
    개념 살펴보기
    ============================================================================
    고르게 하는 흐름에는 다음이 되는 단순한 분포가 필요하다:
    1. 쉽게 뽑을 수 있다(아무 수 만들기)
    2. 확률을 쉽게 셈할 수 있다
    
    흔히 표준 정규 분포를 쓰는 까닭은 이렇다:
    - 확률 밀도 함수가 알려져 있다
    - 효율 좋게 뽑을 수 있다
    - 가운데가 0이고 흩어짐이 1이다
    
    이를 배운 바꿈으로 더 복잡한 분포로 바꿀
    "날 재료"라고 생각하라.
    
    ============================================================================
    수학의 바탕
    ============================================================================
    d차원 표준 정규 분포에서:
    
    p(z) = (1/√(2π))^d × exp(-||z||²/2)
    
    로그를 취하면:
    log p(z) = -d/2 × log(2π) - ||z||²/2
             = 모든 차원에 걸쳐 더한 -1/2 × (z² + log(2π))
    
    이 단순한 꼴 덕분에 확률을 셈하기 쉽다!
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        바탕 분포를 첫자리매김한다.
        
        인수:
            dim (int): 분포의 차원
                      그림에서는 높이 × 너비 × 채널일 수 있다
                      표 자료에서는 특징의 수
                      2차원 장난감 문제에서는 흔히 2
        
        보기:
            >>> base = BaseDistribution(dim=2)  # 2차원으로 그려 보려고
            >>> base = BaseDistribution(dim=784)  # MNIST(28×28)용
        """
        self.dim = dim
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        표준 정규 분포에서 뽑는다.
        
        주사위를 굴리는 것과 비슷하되 1~6 대신 0을 가운데로 한
        종 모양 곡선에서 수를 얻는다.
        
        인수:
            n_samples (int): 만들 표본의 수
            device (str): 'cpu'나 'cuda' - 텐서를 만들 곳
        
        반환값:
            torch.Tensor: 아무 표본, 꼴 (n_samples, dim)
                         가로줄마다 정규 분포에서 뽑은 표본 하나이다
        
        수학의 세부:
            정규 분포를 따르는 아무 수를 만들려 박스-뮐러 바꿈을 짠
            torch.randn을 쓴다.
        
        보기:
            >>> base = BaseDistribution(dim=2)
            >>> samples = base.sample(100)  # 2차원 점 100개를 얻는다
            >>> samples.shape
            torch.Size([100, 2])
            >>> samples.mean()  # 0에 가까워야 한다
            tensor(0.0234)
            >>> samples.std()   # 1에 가까워야 한다
            tensor(0.9876)
        """
        return torch.randn(n_samples, self.dim, device=device)
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        표준 정규 분포에서 표본의 로그 확률을 셈한다.
        
        왜 로그 확률인가?
        - 확률이 아주 작을 수 있어(1e-100 같은) 수치 문제가 생긴다
        - 로그를 취하면 곱셈이 덧셈이 된다(더 안정되다)
        - 로그는 한쪽으로만 간다. 곧 로그 확률이 클수록 확률도 크다
        
        인수:
            z (torch.Tensor): 따질 표본, 꼴 (batch_size, dim)
        
        반환값:
            torch.Tensor: 로그 확률, 꼴 (batch_size,)
                         표본마다 로그 확률 값 하나
        
        수학의 이끌어 내기:
            표준 정규 분포: p(z) = (2π)^(-d/2) × exp(-z²/2)
            
            양변에 로그를 취하면:
            log p(z) = log[(2π)^(-d/2)] + log[exp(-z²/2)]
                     = -d/2 × log(2π) - z²/2
                     = -0.5 × (z² + log(2π))  [차원마다]
            
            여러 차원에서는 모든 차원에 걸쳐 더한다
        
        보기:
            >>> base = BaseDistribution(dim=2)
            >>> z = torch.tensor([[0.0, 0.0],    # 평균에서
            ...                   [3.0, 3.0]])   # 평균에서 멀리
            >>> log_probs = base.log_prob(z)
            >>> log_probs
            tensor([-1.8379, -10.8379])  # 둘째 점의 확률이 더 낮다
        """
        # 원소마다: -0.5 × (z² + log(2π))
        # 상수 log(2π)은 대략 1.8379이다
        log_prob = -0.5 * (z ** 2 + np.log(2 * np.pi))
        
        # 표본마다 전체 로그 확률을 얻으려 차원에 걸쳐 더한다
        # 꼴: (묶음 크기, 차원) → (묶음 크기,)
        return log_prob.sum(dim=-1)


class Flow(nn.Module):
    """
    Flow: 뒤집을 수 있는 바꿈의 추상 바탕 갈래
    
    ============================================================================
    개념 살펴보기
    ============================================================================
    "흐름"은 다음과 같은 특별한 갈래의 바꿈이다:
    
    1. **뒤집을 수 있다**(일대일 대응)
       - z → x으로 갈 수 있다(앞으로)
       - x → z으로 갈 수 있다(거꾸로)
       - 앎을 잃지 않는다
    
    2. **다룰 수 있는 야코비**를 갖는다
       - 부피가 어떻게 바뀌는지 효율 좋게 셈할 수 있다
       - 확률을 셈하는 데 결정적이다
    
    흐름을 이렇게 생각하라:
    - 앞 방향: 단순한 분포 → 복잡한 분포
    - 뒤 방향: 복잡한 분포 → 단순한 분포
    
    ============================================================================
    왜 뒤집을 수 있음이 중요한가
    ============================================================================
    익히기와 뽑기에서:
    
    익히기(거꾸로):
        실제 자료 x → 흐름⁻¹ → 숨은 z → log p(z) 셈하기
        실제 자료의 확률을 따지려면 역이 필요하다
    
    뽑기(앞으로):
        z ~ N(0,I) 뽑기 → 흐름 → 만든 자료 x
        새 표본을 만들려면 앞 방향이 필요하다
    
    ============================================================================
    변수 바꿈 식
    ============================================================================
    이것이 고르게 하는 흐름의 수학 심장이다!
    
    z ~ p(z)이고 x = f(z)이면:
    
        p(x) = p(z) |det(∂z/∂x)|
    
    로그 공간에서(더 안정되다):
    
        log p(x) = log p(z) + log |det(∂z/∂x)|
    
    야코비 행렬식 log |det(∂z/∂x)|이 알려 주는 것:
    - 바꿈이 부피를 얼마나 늘이거나 줄이는지
    - 양의 행렬식 = 부피 늘어남
    - 음의 행렬식 = 부피 줄어듦
    
    ============================================================================
    """
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        앞 바꿈: 숨은 공간 → 자료 공간 (z → x)
        
        이것이 만들어 내는 방향이며 뽑기에 쓴다.
        
        인수:
            z (torch.Tensor): 숨은 공간의 표본, 꼴 (batch_size, dim)
                             단순한 바탕 분포에서 온다
        
        반환값:
            다음을 담은 짝:
                x (torch.Tensor): 자료 공간으로 바꾼 표본, 꼴 (batch_size, dim)
                                 실제 자료처럼 보여야 한다
                
                log_det (torch.Tensor): 야코비 행렬식 절댓값의 로그, 꼴 (batch_size,)
                                       부피가 어떻게 바뀌는지 좇는다
        
        개념의 흐름:
            단순한 정규 분포 z → [신경망 바꿈] → 복잡한 자료 x
            
        쓰는 보기:
            >>> flow = SomeFlow(dim=2)
            >>> z = torch.randn(10, 2)  # 정규 분포에서 표본 10개
            >>> x, log_det = flow.forward(z)  # 자료 공간으로 바꾼다
            >>> x.shape
            torch.Size([10, 2])
            >>> log_det.shape
            torch.Size([10])
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        역 바꿈: 자료 공간 → 숨은 공간 (x → z)
        
        이것이 추론 방향이며 확률을 셈하는 데 쓴다.
        
        인수:
            x (torch.Tensor): 자료 공간의 표본, 꼴 (batch_size, dim)
                             흔히 익히기 자료이다
        
        반환값:
            다음을 담은 짝:
                z (torch.Tensor): 숨은 공간으로 바꾼 표본, 꼴 (batch_size, dim)
                                 바탕 분포를 따라야 한다
                
                log_det (torch.Tensor): 야코비 행렬식 절댓값의 로그, 꼴 (batch_size,)
                                       역에서는 앞 방향과 부호가 반대이다
        
        개념의 흐름:
            복잡한 자료 x → [역 바꿈] → 단순한 정규 분포 z
            
        수학 메모:
            앞 방향의 야코비가 J_f이면 역의 야코비는 J_f⁻¹이다
            det(J_f⁻¹) = 1/det(J_f)
            log|det(J_f⁻¹)| = -log|det(J_f)|
            
        쓰는 보기:
            >>> flow = SomeFlow(dim=2)
            >>> x = torch.tensor([[1.0, 2.0]])  # 실제 자료 점
            >>> z, log_det = flow.inverse(x)  # 숨은 공간으로 옮긴다
            >>> # 이제 log p(x) = log p(z) + log_det을 셈할 수 있다
        """
        raise NotImplementedError("Subclasses must implement inverse()")


class FlowSequence(nn.Module):
    """
    흐름 차례: 바꿈 여럿 아우르기
    
    ============================================================================
    개념 살펴보기
    ============================================================================
    흐름 바꿈 하나로는 복잡한 분포를 나타내기에 표현력이
    모자랄 때가 많다. 대신 흐름 여럿을 아우른다:
    
        z → 흐름₁ → 흐름₂ → 흐름₃ → ... → x
    
    흐름마다 복잡함을 조금씩 더해 단순한 정규 분포를
    풍부하고 복잡한 분포로 차츰 바꾼다.
    
    이렇게 생각하라:
    - 흐름 1: 정규 분포를 늘인다
    - 흐름 2: 돌린다
    - 흐름 3: 굽음을 더한다
    - 흐름 4: 더 복잡한 결을 만든다
    - ... 그렇게 이어진다
    
    ============================================================================
    수학의 아우르기
    ============================================================================
    아우른 바꿈 f₁, f₂, ..., fₙ에서:
    
    앞으로: x = fₙ(...f₂(f₁(z)))
    
    로그 행렬식 규칙:
        log|det(∂x/∂z)| = Σᵢ log|det(∂fᵢ/∂fᵢ₋₁)|
    
    로그 행렬식을 그저 더하면 된다! 그래서 로그 공간에서 다룬다.
    
    거꾸로: z = f₁⁻¹(f₂⁻¹(...fₙ⁻¹(x)))
    
    역은 거꾸로 된 차례로 쓴다(껴입은 옷을 벗듯이).
    
    ============================================================================
    설계 원칙
    ============================================================================
    1. 번갈아 쓰는 결: 흔히 여러 갈래의 흐름을 번갈아 쓴다
       보기: [짝지음, 묶음 고르게 맞추기, 짝지음, 묶음 고르게 맞추기, ...]
    
    2. 점점 복잡하게: 앞선 흐름은 단순하게, 뒤 흐름은 복잡하게 할 수 있다
    
    3. 흐름의 수: 흔한 범위:
       - 2차원 장난감 문제: 흐름 4~8개
       - 그림 만들어 내기: 흐름 20~40개
       - 흐름이 많을수록 표현력이 좋지만 느리다
    
    ============================================================================
    """
    
    def __init__(self, flows: list, base_dist: BaseDistribution):
        """
        흐름 바꿈의 차례를 첫자리매김한다.
        
        인수:
            flows (list): 아우를 Flow 객체의 목록
                         앞먹임 동안 차례대로 쓰인다
                         보기: [CouplingLayer(), BatchNorm(), CouplingLayer()]
            
            base_dist (BaseDistribution): 출발 분포(흔히 정규 분포)
        
        설계 결:
            flows = [
                CouplingLayer(dim, hidden_dim=64),
                BatchNorm(dim),
                CouplingLayer(dim, hidden_dim=64),
                BatchNorm(dim),
            ]
            model = FlowSequence(flows, BaseDistribution(dim))
        """
        super().__init__()
        
        # PyTorch가 익힐 수 있는 매개변수임을 알도록 ModuleList을 쓴다
        self.flows = nn.ModuleList(flows)
        
        # 바탕 분포를 담는다
        self.base_dist = base_dist
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        표본을 흐름 모두를 지나 바꾼다: z → x
        
        뽑기에 쓰는 만들어 내는 방향이다.
        
        과정:
            1. 바탕 분포의 z에서 시작한다
            2. 흐름₁을 쓴다: z → h₁, log_det₁을 쌓는다
            3. 흐름₂을 쓴다: h₁ → h₂, log_det₂을 쌓는다
            4. 흐름을 모두 지날 때까지 이어 간다
            5. 마지막 내놓기: x, total_log_det
        
        인수:
            z (torch.Tensor): 바탕 분포의 숨은 표본, 꼴 (batch_size, dim)
        
        반환값:
            다음을 담은 짝:
                x (torch.Tensor): 마지막으로 바꾼 표본, 꼴 (batch_size, dim)
                log_det_sum (torch.Tensor): 로그 행렬식의 합, 꼴 (batch_size,)
        
        수학의 세부:
            log|det(∂x/∂z)| = Σᵢ log|det(∂fᵢ/∂fᵢ₋₁)|
            
        보기:
            >>> flows = [Flow1(), Flow2(), Flow3()]
            >>> model = FlowSequence(flows, BaseDistribution(2))
            >>> z = model.base_dist.sample(100)
            >>> x, log_det = model.forward(z)
            >>> # 이제 x에 만든 표본 100개가 들어 있다
        """
        # 로그 행렬식 쌓개를 첫자리매김한다
        # 묶음의 표본마다 0으로 시작한다
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        
        # z에서 시작해 차츰 바꾼다
        x = z
        
        # 흐름을 차례로 적용한다
        for flow in self.flows:
            # 바꿈: 지금 → 다음
            x, log_det = flow.forward(x)
            
            # 로그 행렬식을 쌓는다
            # 핵심 눈썰미: 로그 공간에서 로그 행렬식은 더해진다
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        표본을 흐름 모두를 거슬러 바꾼다: x → z
        
        확률을 셈하는 데 쓰는 추론 방향이다.
        
        과정:
            1. x(실제 자료)에서 시작한다
            2. 흐름ₙ⁻¹을 쓴다: x → hₙ₋₁, log_detₙ을 얻는다
            3. 흐름ₙ₋₁⁻¹을 쓴다: hₙ₋₁ → hₙ₋₂, log_detₙ₋₁을 얻는다
            4. 거꾸로 된 차례로 이어 간다
            5. 마지막 내놓기: z, total_log_det
        
        인수:
            x (torch.Tensor): 바꿀 자료 표본, 꼴 (batch_size, dim)
        
        반환값:
            다음을 담은 짝:
                z (torch.Tensor): 숨은 표본, 꼴 (batch_size, dim)
                log_det_sum (torch.Tensor): 로그 행렬식의 합, 꼴 (batch_size,)
        
        결정적인 참고:
            역은 거꾸로 된 차례로 쓴다!
            앞으로가 z → f₁ → f₂ → f₃ → x이면
            거꾸로는 x → f₃⁻¹ → f₂⁻¹ → f₁⁻¹ → z이다
            
        보기:
            >>> model = FlowSequence([Flow1(), Flow2()], BaseDistribution(2))
            >>> x = torch.randn(100, 2)  # 어떤 자료
            >>> z, log_det = model.inverse(x)
            >>> # 이제 z이 정규 표본처럼 보여야 한다
        """
        # 로그 행렬식 쌓개를 첫자리매김한다
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        
        # x에서 시작해 차츰 거꾸로 z으로 바꾼다
        z = x
        
        # 흐름마다의 역을 거꾸로 된 차례로 적용한다
        # 여기서 reversed()이 결정적이다!
        for flow in reversed(self.flows):
            # 바꿈: 지금 → 앞선 것
            z, log_det = flow.inverse(z)
            
            # 로그 행렬식을 쌓는다
            log_det_sum += log_det
        
        return z, log_det_sum
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        자료 표본의 로그 확률을 셈한다.
        
        이것이 고르게 하는 흐름의 핵심 식이다!
        
        수학 공식:
            log p(x) = log p(z) + log|det(∂z/∂x)|
            
        여기서 각 기호는 다음과 같다.
            - z = f⁻¹(x)은 숨은 나타냄이다
            - log p(z)은 바탕 분포의 로그 확률이다
            - log|det(∂z/∂x)|이 부피 바뀜을 헤아린다
        
        직관:
            1. 자료 x을 숨은 공간으로 되돌린다: x → z
            2. 바탕 분포에서 z이 얼마나 그럴듯한지 따진다
            3. 바꾸는 동안의 부피 바뀜을 헤아려 고친다
            4. 결과: 배운 분포에서 x의 가능도
        
        인수:
            x (torch.Tensor): 따질 자료 표본, 꼴 (batch_size, dim)
        
        반환값:
            torch.Tensor: 로그 확률, 꼴 (batch_size,)
                         값이 클수록 배운 분포에서 더 그럴듯하다
        
        익히기에서 쓰기:
            >>> model = FlowSequence([...], BaseDistribution(2))
            >>> real_data = load_data()
            >>> log_probs = model.log_prob(real_data)
            >>> loss = -log_probs.mean()  # 음의 로그 가능도
            >>> loss.backward()  # 가능도를 크게 하도록 가장 좋게 한다
        
        만들어 내기에서 쓰기:
            >>> # 만든 표본의 품질을 살핀다
            >>> generated = model.sample(100)
            >>> log_probs = model.log_prob(generated)
            >>> # log_probs이 클수록 품질이 좋다
        """
        # 걸음 1: 자료를 숨은 공간으로 바꾼다
        # 그러면 z과 변수 바꿈 바로잡기를 얻는다
        z, log_det = self.inverse(x)
        
        # 걸음 2: 바탕 분포에서 확률을 따진다
        log_pz = self.base_dist.log_prob(z)
        
        # 걸음 3: 변수 바꿈 식을 쓴다
        # log p(x) = log p(z) + log|det(∂z/∂x)|
        return log_pz + log_det
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        배운 분포에서 새 표본을 만든다.
        
        익힌 뒤 새 자료를 만드는 길이다!
        
        과정:
            1. 단순한 바탕 분포(정규 분포)에서 z을 뽑는다
            2. z을 흐름 모두를 지나 바꾼다
            3. 내놓는 x이 익히기 자료를 닮아야 한다
        
        인수:
            n_samples (int): 만들 표본의 수
            device (str): 만들 기기('cpu'나 'cuda')
        
        반환값:
            torch.Tensor: 만든 표본, 꼴 (n_samples, dim)
        
        익힌 뒤의 보기:
            >>> # 그림으로 모델을 익힌다
            >>> model = FlowSequence([...], BaseDistribution(784))
            >>> train(model, mnist_data)
            >>> 
            >>> # 새 그림을 만든다
            >>> new_images = model.sample(16)  # 새 그림 16개를 만든다
            >>> # new_images이 실제 같은 MNIST 숫자로 보여야 한다
        
        품질 살피기:
            익힌 뒤 표본은 다음이어야 한다:
            - 익히기 자료 분포를 닮는다
            - 다양하다(모두 같지 않다)
            - 모델에서 로그 확률이 높다
        """
        # 걸음 1: 단순한 바탕 분포에서 뽑는다
        z = self.base_dist.sample(n_samples, device=device)
        
        # 걸음 2: 흐름을 모두 지나 바꾼다
        # 뽑기에는 필요 없으므로 log_det을 버린다
        x, _ = self.forward(z)
        
        return x


class AffineTransform(Flow):
    """
    아핀 바꿈: 가장 단순한 흐름
    
    ============================================================================
    개념 살펴보기
    ============================================================================
    아핀 바꿈은 있을 수 있는 가장 단순한 흐름이다:
    
        y = 잣수 × x + 옮김
    
    이는 다음과 같다:
    - 차원마다 늘이거나 줄이기(잣수)
    - 가운데 옮기기(옮김)
    
    1차원 보기:
        잣수=2이고 옮김=3이면:
        들임: [0, 1, 2] → 내놓기: [3, 5, 7]
    
    ============================================================================
    수학의 성질
    ============================================================================
    앞으로:
        y = exp(log_scale) × x + 옮김
        
    왜 exp(log_scale)인가?
        - 잣수 대신 log_scale을 배운다
        - 그러면 잣수가 늘 양수이다
        - 가장 좋게 하기에 수치가 더 안정되다
    
    야코비:
        ∂y/∂x = diag(exp(log_scale))
        
    로그 행렬식:
        log|det(∂y/∂x)| = sum(log_scale)
        
    아주 효율이 좋다. 매개변수를 더하기만 하면 된다!
    
    ============================================================================
    쓰임새
    ============================================================================
    1. 자료 고르게 하기: 들임 자료의 가운데와 잣수 맞추기
    2. 단순한 바탕: 더 복잡한 흐름을 이것과 견준다
    3. 첫 층: 이따금 첫 바꿈으로 쓴다
    4. 벌레 잡기: 옳음을 확인하기 쉽다
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        아핀 바꿈을 첫자리매김한다.
        
        인수:
            dim (int): 자료의 차원
        
        배우는 매개변수:
            - log_scale: 차원마다 늘이기와 누르기를 다스린다
            - shift: 차원마다 옮김을 다스린다
        
        첫자리매김:
            - log_scale은 0에서 시작한다(잣수 = 1, 바뀜 없음)
            - shift은 0에서 시작한다(옮김 없음)
        """
        super().__init__()
        
        # 학습 가능한 매개변수
        # 수치 안정을 위해 log(scale)을 배운다
        # 첫 잣수 = exp(0) = 1(바뀜 없음)
        self.log_scale = nn.Parameter(torch.zeros(dim))
        
        # 첫 옮김 = 0(옮김 없음)
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        앞 바꿈: y = 잣수 * z + 옮김
        
        인수:
            z (torch.Tensor): 들임 표본, 꼴 (batch_size, dim)
        
        반환값:
            다음을 담은 짝:
                x (torch.Tensor): 바꾼 표본
                log_det (torch.Tensor): 로그 행렬식
        
        보기:
            >>> transform = AffineTransform(dim=2)
            >>> z = torch.tensor([[0., 0.], [1., 1.]])
            >>> x, log_det = transform.forward(z)
        """
        # log_scale을 실제 잣수로 바꾼다(늘 양수)
        scale = torch.exp(self.log_scale)
        
        # 아핀 바꿈을 쓴다
        # 퍼뜨리기: 잣수와 옮김을 원소마다 적용한다
        x = scale * z + self.shift
        
        # 로그 행렬식을 셈한다
        # 대각 야코비: 행렬식 = 대각의 곱
        # log(det) = 대각의 로그 합 = sum(log_scale)
        log_det = self.log_scale.sum().expand(z.shape[0])
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        역 바꿈: z = (x - 옮김) / 잣수
        
        인수:
            x (torch.Tensor): 바꾼 표본, 꼴 (batch_size, dim)
        
        반환값:
            다음을 담은 짝:
                z (torch.Tensor): 본디 표본
                log_det (torch.Tensor): 로그 행렬식(앞 방향의 음수)
        """
        # log_scale을 실제 잣수로 바꾼다
        scale = torch.exp(self.log_scale)
        
        # 역 아핀 바꿈을 쓴다
        z = (x - self.shift) / scale
        
        # 역의 로그 행렬식
        # det(J_inv) = 1/det(J_forward)
        # log(det(J_inv)) = -log(det(J_forward))
        log_det = -self.log_scale.sum().expand(x.shape[0])
        
        return z, log_det


class PlanarFlow(Flow):
    """
    평면 흐름: 바꿈에 굽음 더하기
    
    ============================================================================
    개념 살펴보기
    ============================================================================
    평면 흐름은 "평면" 바꿈으로 비선형을 더한다:
    
        f(z) = z + u × tanh(w^T z + b)
    
    이렇게 생각하라:
    - w^T z + b: 선형 쏘기(결정 가장자리 같은 것)
    - tanh: 값을 -1과 1 사이로 누른다
    - u: 바꿈의 방향
    - 결과: "평면"을 따라 공간을 굽힌다
    
    ============================================================================
    기하의 직관
    ============================================================================
    평평한 2차원 면을 떠올려 보라:
    1. 항 w^T z + b이 그 면에 선을 정한다
    2. tanh이 이 선 둘레에 매끄러운 옮아감을 만든다
    3. u이 그 선에 직각으로 얼마나 "미는지" 다스린다
    
    그래서 다음을 할 수 있는 매끄럽고 굽은 바꿈이 된다:
    - 분포에 마루나 골을 만든다
    - 정규 분포를 굽은 모양으로 휘게 한다
    
    ============================================================================
    수학의 세부
    ============================================================================
    앞으로:
        f(z) = z + u × tanh(w^T z + b)
    
    야코비:
        ∂f/∂z = I + u × ψ^T
        여기서 ψ = (1 - tanh²(w^T z + b)) × w
    
    로그 행렬식:
        log|det(∂f/∂z)| = log|1 + u^T ψ|
        
    효율을 위해 행렬식 보조 정리를 쓴다!
    
    ============================================================================
    한계
    ============================================================================
    1. 닫힌 꼴 역이 없다: 되풀이 풀개가 필요하다
       (그래서 짝지음 흐름을 흔히 더 낫게 여긴다)
    
    2. 표현력이 제한된다: 평면 흐름 하나는 꽤 단순하다
       (복잡한 분포를 나타내려면 흐름이 많이 필요하다)
    
    3. 주로 쓸모 있는 곳: 차원 낮은 문제(2차원, 3차원)
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        평면 흐름 바꿈을 첫자리매김한다.
        
        인수:
            dim (int): 공간의 차원
        
        배우는 매개변수:
            - weight (w): "평면"의 방향을 정한다, 꼴 (dim,)
            - bias (b): 평면을 옮긴다, 꼴 (1,)
            - u: 바꿈의 방향, 꼴 (dim,)
        
        첫자리매김:
            - weight: 아무 정규 분포(여러 첫 평면을 만든다)
            - bias: 0(평면이 원점을 지난다)
            - u: 아무 정규 분포(아무 바꿈 방향)
        """
        super().__init__()
        
        # 평면 바꿈의 매개변수
        # w: 들임 공간의 초평면을 정한다
        self.weight = nn.Parameter(torch.randn(dim))
        
        # b: 초평면의 치우침
        self.bias = nn.Parameter(torch.zeros(1))
        
        # u: 바꿈의 방향
        self.u = nn.Parameter(torch.randn(dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        평면 흐름을 지나는 앞 바꿈.
        
        과정:
            1. 선형 쏘기를 셈한다: w^T z + b
            2. tanh 비선형을 쓴다
            3. u으로 잣수를 맞추어 z에 더한다
            4. 야코비 행렬식을 셈한다
        
        인수:
            z (torch.Tensor): 들임 표본, 꼴 (batch_size, dim)
        
        반환값:
            다음을 담은 짝:
                x (torch.Tensor): 바꾼 표본
                log_det (torch.Tensor): 로그 행렬식
        
        수치의 안정:
            - tanh이 값을 가둔다
            - log(0)을 피하려 작은 엡실론(1e-8)을 더한다
        """
        # 걸음 1: 선형 쏘기 w^T z + b을 셈한다
        # 표본마다: 무게 벡터와의 점곱에 치우침을 더한다
        # 꼴: (묶음 크기, 차원) × (차원,) → (묶음 크기, 1)
        linear = torch.sum(self.weight * z, dim=-1, keepdim=True) + self.bias
        
        # 걸음 2: 바꿈을 쓴다
        # f(z) = z + u × tanh(w^T z + b)
        # keepdim=True이라야 퍼뜨리기가 옳게 된다
        x = z + self.u * torch.tanh(linear)
        
        # 걸음 3: 야코비 행렬식을 셈한다
        # 여기가 까다로운 곳이다!
        
        # tanh의 미분: d/dx tanh(x) = 1 - tanh²(x)
        # 이를 "tanh 미분"이나 "sech²"이라 한다
        psi = (1 - torch.tanh(linear) ** 2) * self.weight
        
        # 행렬식 보조 정리를 쓰면:
        # det(I + u × ψ^T) = 1 + u^T ψ
        det = 1 + torch.sum(psi * self.u, dim=-1)
        
        # 수치 안정을 위해 로그와 절댓값을 취한다
        # log(0)을 피하려 엡실론을 더한다
        log_det = torch.log(torch.abs(det) + 1e-8)
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        역 바꿈: x → z
        
        중요한 한계:
        평면 흐름에는 닫힌 꼴 역이 없다!
        
        뒤집으려면 다음을 풀어야 한다:
            z = x - u × tanh(w^T z + b)
            
        이는 되풀이 방법(뉴턴 방법이나 붙박이점 되풀이 같은)이 필요하다.
        
        대개의 쓰임새에서는 효율 좋은 닫힌 꼴 역이 있는 짝지음 흐름을 쓰고
        평면 흐름은 피한다.
        
        인수:
            x: 바꾼 표본
        
        일으키는 예외:
            NotImplementedError: 이 방법은 되풀이 풀개가 필요하다
        
        이것이 필요하다면:
            붙박이점 되풀이나 뉴턴-랩슨을 짜라:
            
            def inverse_iterative(x, n_iters=10):
                z = x  # 첫 짐작
                for _ in range(n_iters):
                    linear = torch.sum(weight * z, dim=-1, keepdim=True) + bias
                    z = x - u * torch.tanh(linear)
                return z
        """
        raise NotImplementedError(
            "Planar flow inverse requires iterative solver. "
            "Consider using coupling flows instead, which have analytical inverses."
        )


def visualize_2d_transformation(flow_model: nn.Module, n_points: int = 1000,
                               xlim: tuple = (-3, 3), ylim: tuple = (-3, 3),
                               filename: str = 'transformation.png'):
    """
    흐름이 2차원 분포를 어떻게 바꾸는지 그려 본다.
    
    ============================================================================
    목적
    ============================================================================
    이 함수는 고르게 하는 흐름이 무엇을 하는지 눈으로 보게 해 준다!
    
    나란한 견줌을 만든다:
    - 왼쪽: 바탕 정규 분포의 표본(숨은 공간 z)
    - 오른쪽: 흐름을 지나 바꾼 뒤(자료 공간 x)
    
    이 그림은 다음을 이해하는 데 결정적이다:
    - 흐름이 공간을 어떻게 휘게 하는지
    - 익히기가 잘 되고 있는지
    - 흐름이 어떤 결을 배웠는지
    
    ============================================================================
    풀이 길잡이
    ============================================================================
    익히기 앞:
        - 왼쪽: 말끔한 정규 분포 방울
        - 오른쪽: 조금 휜 정규 분포(아무 첫자리매김)
    
    익히는 동안:
        - 오른쪽이 차츰 자료 분포에 맞게 바뀐다
        - 복잡함과 짜임이 늘어나는지 살펴라
    
    익힌 뒤:
        - 왼쪽: 여전히 정규 분포(결코 바뀌지 않는다)
        - 오른쪽: 목표 분포와 맞아야 한다(예컨대 달 모양, 동그라미)
    
    문제 풀기:
        - 오른쪽이 그대로인가? → 모델이 익지 않고 있다
        - 오른쪽이 어지러운가? → 배움 빠르기가 너무 크다
        - 오른쪽이 뭉쳐 있는가? → 층이나 담이가 모자란다
    
    ============================================================================
    
    인수:
        flow_model (nn.Module): 그려 볼 익힌 흐름 모델
        n_points (int): 뽑아 그릴 점의 수
        xlim (tuple): 두 그림의 x축 범위
        ylim (tuple): 두 그림의 y축 범위
        filename (str): 그림을 갈무리할 곳
    
    쓰는 보기:
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> train(model, data)  # 2차원 자료로 익힌다
        >>> visualize_2d_transformation(model, n_points=2000)
        >>> # 결과를 보려면 transformation.png을 열어라!
    """
    # 모델을 값매김 방식으로 둔다(떨구기와 묶음 고르게 맞추기 새로 고침을 끈다)
    flow_model.eval()
    
    # 효율을 위해 기울기 계산 끄기
    with torch.no_grad():
        # 걸음 1: 바탕 정규 분포에서 뽑는다
        z = flow_model.base_dist.sample(n_points)
        
        # 걸음 2: 흐름을 지나 바꾼다
        x, _ = flow_model.forward(z)
        
        # 걸음 3: 그리려 CPU로 옮긴다(GPU에 있으면)
        z = z.cpu().numpy()
        x = x.cpu().numpy()
    
    # 나란한 그림을 만든다
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 왼쪽 그림: 숨은 공간(늘 정규 분포)
    axes[0].scatter(z[:, 0], z[:, 1], alpha=0.5, s=10)
    axes[0].set_title('Latent Space (z) - Base Gaussian')
    axes[0].set_xlabel('z₁')
    axes[0].set_ylabel('z₂')
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_aspect('equal')  # 가로세로 비를 같게
    axes[0].grid(True, alpha=0.3)
    
    # 오른쪽 그림: 자료 공간(바뀐 분포)
    axes[1].scatter(x[:, 0], x[:, 1], alpha=0.5, s=10, color='red')
    axes[1].set_title('Data Space (x) - Transformed Distribution')
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to {filename}")
    print(f"  Compare left (Gaussian) vs right (learned distribution)")


def visualize_2d_density(flow_model: nn.Module, xlim: tuple = (-3, 3),
                        ylim: tuple = (-3, 3), n_grid: int = 200,
                        filename: str = 'density.png'):
    """
    2차원 흐름이 배운 확률 밀도를 그려 본다.
    
    ============================================================================
    목적
    ============================================================================
    visualize_2d_transformation이 표본을 보이는 반면 이 함수는
    확률 밀도, 곧 모델에서 자리마다 얼마나 그럴듯한지를 보인다.
    
    다음과 같은 열지도를 만든다:
    - 밝은 자리 = 높은 확률(모델이 자료가 여기 있어야 한다고 본다)
    - 어두운 자리 = 낮은 확률(모델이 여기 자료가 있기 어렵다고 본다)
    
    ============================================================================
    풀이 길잡이
    ============================================================================
    잘 익힌 모델:
        - 봉우리가 자료 무리와 맞는다
        - 매끄럽고 이어진 확률 풍경
        - 봉우리 사이가 또렷이 갈린다
    
    잘못 익힌 모델:
        - 어디나 평평하다(짜임을 배우지 못했다)
        - 봉우리가 엉뚱한 곳에 있다
        - 봉우리가 지나치게 뾰족하다(지나치게 맞춰짐)
    
    자료와 견주기:
        실제 자료의 흩뿌림 그림을 겹쳐 다음을 확인하라:
        - 자료가 빽빽한 곳에서 확률이 높다
        - 자료가 성긴 곳에서 확률이 낮다
    
    ============================================================================
    셈에 대한 참고
    ============================================================================
    이는 200×200 격자의 모든 점, 곧 40,000개 점에서 확률을 셈한다!
    해상도가 높거나 차원이 높은 문제에서는 느릴 수 있다.
    
    시간 복잡도: O(n_grid² × 모델 복잡도)
    
    ============================================================================
    
    인수:
        flow_model (nn.Module): 익힌 흐름 모델
        xlim (tuple): 밀도 격자의 x축 범위
        ylim (tuple): 밀도 격자의 y축 범위
        n_grid (int): 밀도 격자의 해상도(n_grid × n_grid 점)
        filename (str): 그림을 갈무리할 곳
    
    쓰는 보기:
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> train(model, moons_data)
        >>> visualize_2d_density(model)
        >>> # 그림에 초승달 꼴의 확률 높은 자리 둘이 보여야 한다!
    """
    # 값매김 방식으로 둔다
    flow_model.eval()
    
    # 걸음 1: 공간 전체를 덮는 점 격자를 만든다
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)  # 2차원 격자를 만든다
    
    # 걸음 2: 격자를 (x, y) 점 목록으로 펼친다
    # 꼴: (n_grid*n_grid, 2)
    points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), 
        dtype=torch.float32
    )
    
    # 걸음 3: 격자 점마다 로그 확률을 셈한다
    with torch.no_grad():
        log_prob = flow_model.log_prob(points)
        
        # 로그 확률을 실제 확률로 바꾼다
        # prob = exp(log_prob)
        prob = torch.exp(log_prob).cpu().numpy()
    
    # 걸음 4: 그리려 격자 꼴로 되돌린다
    prob = prob.reshape(n_grid, n_grid)
    
    # 걸음 5: 채운 등고선 그림을 만든다
    plt.figure(figsize=(8, 7))
    
    # contourf이 매끄러운 색 자리를 만든다
    # levels=50이 잔 눈금을 준다
    plt.contourf(X, Y, prob, levels=50, cmap='viridis')
    
    # 확률 잣수를 보이려 색 막대를 더한다
    plt.colorbar(label='Probability Density')
    
    plt.title('Learned Probability Density p(x)')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved density visualization to {filename}")
    print(f"  Bright = high probability, Dark = low probability")


def train_flow(flow_model: nn.Module, dataloader, optimizer, 
              n_epochs: int = 100, device: str = 'cpu') -> list:
    """
    최대 가능도 어림으로 고르게 하는 흐름 모델을 익힌다.
    
    ============================================================================
    개념 살펴보기
    ============================================================================
    익히기 목표: 익히기 자료의 가능도를 가장 크게 하기
    
    달리 말해 배운 분포에서 실제 자료의 확률이 높아지도록
    모델 매개변수를 고친다.
    
    손실 함수:
        손실 = 익히기 자료의 x에 대해 -mean(log p(x))
        
    음의 로그 가능도를 가장 작게 한다(= 가능도를 가장 크게 한다)
    
    ============================================================================
    익히기 알고리즘
    ============================================================================
    바퀴마다:
        자료 묶음마다:
            1. 묶음의 log p(x)을 셈한다
               - 역으로 x → z을 옮긴다
               - log p(z) + log|det|을 따진다
            
            2. 손실 = -mean(log p(x))을 셈한다
               - 가장 작게 하므로 음수이다
               - 기울기가 안정되도록 묶음에 걸쳐 평균 낸다
            
            3. 뒤먹임 퍼뜨리고 매개변수를 새로 고친다
               - 기울기를 셈한다: ∂Loss/∂θ
               - 새로 고친다: θ ← θ - lr × ∇Loss
    
    ============================================================================
    익히기 지켜보기
    ============================================================================
    건강한 익히기:
        - 손실이 꾸준히 줄어든다
        - 마침내 평평해진다
        - 심하게 흔들리지 않는다
    
    살펴야 할 문제:
        - 손실이 는다 → 배움 빠르기가 너무 크다
        - 손실이 흔들린다 → 배움 빠르기나 묶음 크기를 줄여라
        - 손실이 멈춘다 → 모델이 너무 단순하거나 자료가 너무 복잡하다
        - 손실 → NaN → 수치가 불안정하다(기울기 터짐)
    
    ============================================================================
    웃매개변수 요령
    ============================================================================
    배움 빠르기:
        - 시작: 1e-3이나 1e-4
        - 익히기가 불안정하면 낮춘다
        - 배움 빠르기 차례표를 쓸 수 있다
    
    묶음 크기:
        - 클수록 안정되지만 느리다
        - 작을수록 빠르지만 잡음이 많다
        - 흔히: 작은 자료 묶음에 64~256
    
    바퀴 수:
        - 2차원 장난감 자료: 100~500바퀴
        - 그림: 50~100바퀴
        - 손실을 지켜보다 모이면 멈춘다
    
    ============================================================================
    
    인수:
        flow_model (nn.Module): 익힐 흐름 모델
        dataloader: 익히기 자료를 담은 PyTorch DataLoader
        optimizer: PyTorch 가장 좋게 하개(예컨대 Adam)
        n_epochs (int): 익히기 바퀴 수
        device (str): 'cpu'나 'cuda'
    
    반환값:
        list: 바퀴마다의 익히기 손실
    
    쓰는 보기:
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> 
        >>> # 자료를 마련한다
        >>> data = generate_toy_data('moons', n_samples=2000)
        >>> dataset = TensorDataset(data)
        >>> dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        >>> 
        >>> # 모델을 세운다
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> 
        >>> # 익힌다!
        >>> losses = train_flow(model, dataloader, optimizer, n_epochs=500)
        >>> 
        >>> # 익히기 곡선을 그린다
        >>> plot_training_loss(losses)
    """
    # 모델을 학습 모드로
    # 그러면 떨구기와 묶음 고르게 맞추기 새로 고침 같은 것이 켜진다
    flow_model.train()
    
    # 그리려 손실 값을 담는 목록
    losses = []
    
    # 학습 루프
    for epoch in range(n_epochs):
        epoch_loss = 0.0  # 이 바퀴의 쌓개
        n_batches = 0     # 묶음을 센다
        
        # 배치들을 순회한다
        for batch in dataloader:
            # 여러 자료 불러오개 꼴을 다룬다
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # 자료를 뽑는다(이름표가 있으면 무시한다)
            
            # 묶음을 기기(CPU나 GPU)로 옮긴다
            batch = batch.to(device)
            
            # 자료의 차원이 2보다 많으면 펼친다
            # 예컨대 MNIST는 (묶음 크기, 28, 28) → (묶음 크기, 784)
            if batch.dim() > 2:
                batch = batch.view(batch.shape[0], -1)
            
            # ==================== 앞먹임 ====================
            # 지금 모델에서 묶음의 로그 확률을 셈한다
            # 여기에는 다음이 든다:
            #   1. 역 바꿈: x → z
            #   2. 바탕 로그 확률 따지기: log p(z)
            #   3. 야코비 바로잡기 더하기: log|det|
            log_prob = flow_model.log_prob(batch)
            
            # ==================== 손실 셈하기 ====================
            # 음의 로그 가능도 손실
            # 로그 확률을 크게 하고 싶으므로 음의 로그 확률을 가장 작게 한다
            loss = -log_prob.mean()
            
            # ==================== 뒤먹임 ====================
            # 앞선 기울기를 지운다
            optimizer.zero_grad()
            
            # 역전파로 경사를 계산한다
            loss.backward()
            
            # 매개변수 갱신
            optimizer.step()
            
            # ==================== 기록 ====================
            epoch_loss += loss.item()
            n_batches += 1
        
        # 이 바퀴의 평균 손실을 셈한다
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # 이따금 나아감을 찍는다
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def plot_training_loss(losses: list, filename: str = 'training_loss.png'):
    """
    익히기 손실 곡선을 그린다.
    
    ============================================================================
    목적
    ============================================================================
    손실 곡선을 그려 보면 다음을 알 수 있다:
    - 익히기가 잘 되는가?(손실이 줄어야 한다)
    - 익히기가 모였는가?(손실이 평평해진다)
    - 문제가 있는가?(흔들림, 벌어짐)
    
    ============================================================================
    풀이 길잡이
    ============================================================================
    좋은 익히기 곡선:
        - 매끄럽게 줄어든다
        - 마침내 평평해진다
        - 갑작스러운 치솟음이 없다
        보기: \___  (높이 시작해 떨어지고 평평해진다)
    
    문제:
        - 평평한 선 → 모델이 배우지 않는다(배움 빠르기와 모델 담이를 살펴라)
        - 늘어남 → 벌어진다(배움 빠르기를 줄여라)
        - 흔들림 → 불안정하다(배움 빠르기나 묶음 크기를 줄여라)
        - 뚝 떨어진 뒤 평평 → 바퀴를 더 돌거나 더 나은 첫자리매김이 필요하다
    
    ============================================================================
    
    인수:
        losses (list): 익히기에서 나온 손실 값의 목록
        filename (str): 그림을 갈무리할 곳
    
    쓰는 보기:
        >>> losses = train_flow(model, dataloader, optimizer, n_epochs=500)
        >>> plot_training_loss(losses)
        >>> # 익히기가 잘 됐는지 training_loss.png으로 확인하라
    """
    plt.figure(figsize=(10, 5))
    
    # 바퀴에 대한 손실을 그린다
    plt.plot(losses, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative Log-Likelihood', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    
    # 읽기 쉽도록 격자를 더한다
    plt.grid(True, alpha=0.3)
    
    # 마지막 손실을 적어 넣는다
    final_loss = losses[-1]
    plt.annotate(f'Final: {final_loss:.4f}',
                xy=(len(losses)-1, final_loss),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.5', 
                                      facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved loss plot to {filename}")
    print(f"  Final loss: {final_loss:.4f}")


# ============================================================================
# 단원 간추리기와 다음 걸음
# ============================================================================
"""
축하한다! 바탕 벽돌을 모두 마쳤다!

배운 것:
✓ BaseDistribution: 뽑는 곳(정규 분포)
✓ Flow: 바꿈의 추상 갈래
✓ FlowSequence: 흐름 여럿 아우르기
✓ AffineTransform: 있을 수 있는 가장 단순한 흐름
✓ PlanarFlow: 비선형 더하기
✓ 2차원 흐름을 그려 보는 연장
✓ 최대 가능도 익히기 물길

다음 걸음:
1. 더 힘 있는 바꿈을 보려면 coupling_flows.py을 익혀라
   - 짝지음 층(RealNVP)
   - 묶음 고르게 맞추기 흐름
   - 그림을 위한 바둑판 결

2. 모두 도는 것을 보려면 example_2d_flows.py을 돌려라
   - 장난감 2차원 자료 묶음으로 익힌다
   - 바꿈을 그려 본다
   - 익히기의 움직임을 이해한다

3. 네 자료로 실험해 보라!
   - 2차원 장난감 문제로 시작하라
   - 여러 얼개를 시험하라
   - 배운 것을 그려 보라

읽어 볼 것:
- "Normalizing Flows for Probabilistic Modeling" (Papamakarios et al., 2019)
- "Density Estimation Using Real NVP" (Dinh et al., 2017)
- "Variational Inference with Normalizing Flows" (Rezende & Mohamed, 2015)

잘하는 요령:
- 늘 그려 보라! 준비된 그려 보기 연장을 써라
- 단순하게 시작하라: 2차원 자료, 적은 층
- 익히기 손실을 꼼꼼히 지켜보라
- 웃매개변수를 실험하라
- 뒤집을 수 있는지 살펴라: x → z → x이 x을 돌려주어야 한다

즐겁게 배우기를! 🎓
"""


if __name__ == "__main__":
    pass
```

## 논의

이 짜기는 함께 어우러져 온전한 깊은 배움 얼개를 이루는 갈래 5개(`BaseDistribution`, `Flow`, `FlowSequence`, `AffineTransform`, 그리고 하나 더)를 정한다. 갈래마다 뚜렷한 조각을 감싸 부호를 모듈답고 넓히기 쉽게 만든다. `forward` 방법이 PyTorch가 저절로 미분하는 데 쓰는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 기계 배움 일에 대한 실전 직관이 선다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 덩이의 수를 자리매김할 수 있도록 `BaseDistribution`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 층 2, 4, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이한다. (수수한 파이썬 목록이 아니라) `nn.ModuleList`을 써야 PyTorch가 모든 매개변수를 가장 좋게 하기에 올린다. 다음으로 시험하라: `for n in [2, 4, 8]: model = BaseDistribution(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
