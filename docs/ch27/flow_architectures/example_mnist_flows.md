# MNIST 정규화 흐름 예제

이 예제는 MNIST 이미지를 784차원 벡터로 펼치고 아핀 커플링과 흐름용 정규화 레이어를 쌓아 학습하는 절차를 보여 준다. 가역 흐름에서 밀도를 계산하려면 레이어의 방향·통계 처리·이미지 전처리의 확률적 의미까지 맞아야 한다. 표시된 코드의 학습 로스와 이미지 파일이 이 조건을 자동으로 충족하는 것은 아니다.

## 1. 학습·생성 코드

```python
"""
MNIST 정규화 흐름

이미지 생성을 보이려 MNIST 숫자에 정규화 흐름을 학습한다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ========================================================================
# 메인
# ========================================================================

from typing import Tuple

# --- flow_utils ---
"""
정규화 흐름: 기본 도구와 기반 클래스

================================================================================
교육 목적
================================================================================
이 모듈은 정규화 흐름의 기본 구성 요소를 제공한다.
생성 모델과 정규화 흐름을 처음 배우는
학부생을 위해 만들었다.

다루는 핵심 개념:
1. 기저 분포(샘플링을 시작하는 곳)
2. 흐름 변환(분포를 바꾸는 법)
3. 야코비안 행렬식(부피 변화 추적)
4. 여러 변환의 합성
5. 최대 가능도 학습

선수 지식:
- 확률 분포에 대한 이해
- 기본 뉴럴 네트워크(PyTorch)
- 다변수 미적분(야코비안)
- 선형대수의 기초

학습 경로:
BaseDistribution → Flow → FlowSequence → 단순한 예로 시작하고
이어 coupling_flows.py의 더 복잡한 변환으로 넘어간다

================================================================================
"""



class BaseDistribution:
    """
    기저 분포: 정규화 흐름의 출발점
    
    ============================================================================
    개념 개요
    ============================================================================
    정규화 흐름에는 다음을 만족하는 단순한 분포가 필요하다:
    1. 쉽게 샘플링할 수 있다(난수 생성)
    2. 확률을 쉽게 계산할 수 있다
    
    흔히 표준 정규 분포를 쓰는 까닭은 이렇다:
    - 확률 밀도 함수가 알려져 있다
    - 효율적으로 뽑을 수 있다
    - 평균이 0이고 분산이 1이다
    
    이를 학습한 변환으로 더 복잡한 분포로 바꿀
    "원재료"라고 생각하라.
    
    ============================================================================
    수학적 배경
    ============================================================================
    d차원 표준 정규 분포에서:
    
    p(z) = (1/√(2π))^d × exp(-||z||²/2)
    
    로그를 취하면:
    log p(z) = -d/2 × log(2π) - ||z||²/2
             = 모든 차원에 걸쳐 더한 -1/2 × (z² + log(2π))
    
    이 단순한 형태 덕분에 확률을 계산하기 쉽다!
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        기저 분포를 초기화한다.
        
        인수:
            dim (int): 분포의 차원
                      이미지에서는 높이 × 너비 × 채널일 수 있다
                      표 데이터에서는 피처의 수
                      2차원 장난감 문제에서는 흔히 2
        
        보기:
            >>> base = BaseDistribution(dim=2)  # 2차원으로 그려 보려고
            >>> base = BaseDistribution(dim=784)  # MNIST(28×28)용
        """
        self.dim = dim
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        표준 정규 분포에서 뽑는다.
        
        주사위를 굴리는 것과 비슷하되 1~6 대신 0을 중심으로 한
        종 모양 곡선에서 수를 얻는다.
        
        인수:
            n_samples (int): 만들 표본의 수
            device (str): 'cpu'나 'cuda' - 텐서를 만들 곳
        
        반환값:
            torch.Tensor: 무작위 표본, 형태 (n_samples, dim)
                         행마다 정규 분포에서 뽑은 표본 하나이다
        
        수학적 세부:
            정규 분포를 따르는 난수를 만들려고 박스-뮐러 변환을 구현한
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
        표준 정규 분포에서 표본의 로그 확률을 계산한다.
        
        왜 로그 확률인가?
        - 확률이 아주 작을 수 있어(1e-100 같은) 수치 문제가 생긴다
        - 로그를 취하면 곱셈이 덧셈이 된다(더 안정적이다)
        - 로그는 단조 증가 함수다. 곧 로그 확률이 클수록 확률도 크다
        
        인수:
            z (torch.Tensor): 평가할 표본, 형태 (batch_size, dim)
        
        반환값:
            torch.Tensor: 로그 확률, 형태 (batch_size,)
                         표본마다 로그 확률 값 하나
        
        수식 유도:
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
        # 상수 log(2π)는 대략 1.8379이다
        log_prob = -0.5 * (z ** 2 + np.log(2 * np.pi))
        
        # 표본마다 전체 로그 확률을 얻으려 차원에 걸쳐 더한다
        # 형태: (배치 크기, 차원) → (배치 크기,)
        return log_prob.sum(dim=-1)


class Flow(nn.Module):
    """
    Flow: 가역 변환의 추상 기반 클래스
    
    ============================================================================
    개념 개요
    ============================================================================
    "흐름"은 다음 성질을 갖춘 특별한 종류의 변환이다:
    
    1. **가역이다**(일대일 대응)
       - z → x로 갈 수 있다(순방향)
       - x → z로 갈 수 있다(역방향)
       - 정보를 잃지 않는다
    
    2. **계산 가능한(tractable) 야코비안**을 갖는다
       - 부피가 어떻게 바뀌는지 효율적으로 계산할 수 있다
       - 확률을 계산하는 데 결정적이다
    
    흐름을 이렇게 생각하라:
    - 순방향: 단순한 분포 → 복잡한 분포
    - 역방향: 복잡한 분포 → 단순한 분포
    
    ============================================================================
    왜 가역성이 중요한가
    ============================================================================
    학습과 샘플링에서:
    
    학습(역방향):
        실제 데이터 x → 흐름⁻¹ → 잠재 z → log p(z) 계산
        실제 데이터의 확률을 평가하려면 역변환이 필요하다
    
    샘플링(순방향):
        z ~ N(0,I) 샘플링 → 흐름 → 생성한 데이터 x
        새 표본을 만들려면 순방향이 필요하다
    
    ============================================================================
    변수 변환 공식
    ============================================================================
    이것이 정규화 흐름의 수학적 핵심이다!
    
    z ~ p(z)이고 x = f(z)이면:
    
        p(x) = p(z) |det(∂z/∂x)|
    
    로그 공간에서(더 안정적이다):
    
        log p(x) = log p(z) + log |det(∂z/∂x)|
    
    야코비안 행렬식 log |det(∂z/∂x)|이 알려 주는 것:
    - 변환이 부피를 얼마나 늘이거나 줄이는지
    - 양의 행렬식 = 부피 늘어남
    - 음의 행렬식 = 부피 줄어듦
    
    ============================================================================
    """
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순방향 변환: 잠재 공간 → 데이터 공간 (z → x)
        
        이것이 생성 방향이며 샘플링에 쓴다.
        
        인수:
            z (torch.Tensor): 잠재 공간의 표본, 형태 (batch_size, dim)
                             단순한 기저 분포에서 온다
        
        반환값:
            다음을 담은 튜플:
                x (torch.Tensor): 데이터 공간으로 바꾼 표본, 형태 (batch_size, dim)
                                 실제 데이터처럼 보여야 한다
                
                log_det (torch.Tensor): 야코비안 행렬식 절댓값의 로그, 형태 (batch_size,)
                                       부피가 어떻게 바뀌는지 추적한다
        
        개념의 흐름:
            단순한 정규 분포 z → [뉴럴 네트워크 변환] → 복잡한 데이터 x
            
        사용 예:
            >>> flow = SomeFlow(dim=2)
            >>> z = torch.randn(10, 2)  # 정규 분포에서 표본 10개
            >>> x, log_det = flow.forward(z)  # 데이터 공간으로 바꾼다
            >>> x.shape
            torch.Size([10, 2])
            >>> log_det.shape
            torch.Size([10])
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        역변환: 데이터 공간 → 잠재 공간 (x → z)
        
        이것이 추론 방향이며 확률을 계산하는 데 쓴다.
        
        인수:
            x (torch.Tensor): 데이터 공간의 표본, 형태 (batch_size, dim)
                             흔히 학습 데이터이다
        
        반환값:
            다음을 담은 튜플:
                z (torch.Tensor): 잠재 공간으로 바꾼 표본, 형태 (batch_size, dim)
                                 기저 분포를 따라야 한다
                
                log_det (torch.Tensor): 야코비안 행렬식 절댓값의 로그, 형태 (batch_size,)
                                       역에서는 순방향과 부호가 반대이다
        
        개념의 흐름:
            복잡한 데이터 x → [역변환] → 단순한 정규 분포 z
            
        수학 메모:
            순방향의 야코비안이 J_f이면 역변환의 야코비안은 J_f⁻¹이다
            det(J_f⁻¹) = 1/det(J_f)
            log|det(J_f⁻¹)| = -log|det(J_f)|
            
        사용 예:
            >>> flow = SomeFlow(dim=2)
            >>> x = torch.tensor([[1.0, 2.0]])  # 실제 데이터 점
            >>> z, log_det = flow.inverse(x)  # 잠재 공간으로 옮긴다
            >>> # 이제 log p(x) = log p(z) + log_det을 계산할 수 있다
        """
        raise NotImplementedError("Subclasses must implement inverse()")


class FlowSequence(nn.Module):
    """
    흐름 시퀀스: 여러 변환의 합성
    
    ============================================================================
    개념 개요
    ============================================================================
    흐름 변환 하나로는 복잡한 분포를 나타내기에 표현력이
    모자랄 때가 많다. 대신 흐름 여러 개를 합성한다:
    
        z → 흐름₁ → 흐름₂ → 흐름₃ → ... → x
    
    흐름마다 복잡도를 조금씩 더해 단순한 정규 분포를
    풍부하고 복잡한 분포로 차츰 바꾼다.
    
    이렇게 생각하라:
    - 흐름 1: 정규 분포를 늘인다
    - 흐름 2: 돌린다
    - 흐름 3: 비선형성을 더한다
    - 흐름 4: 더 복잡한 패턴을 만든다
    - ... 그렇게 이어진다
    
    ============================================================================
    합성의 수학
    ============================================================================
    합성한 변환 f₁, f₂, ..., fₙ에서:
    
    순방향: x = fₙ(...f₂(f₁(z)))
    
    로그 행렬식 규칙:
        log|det(∂x/∂z)| = Σᵢ log|det(∂fᵢ/∂fᵢ₋₁)|
    
    로그 행렬식을 그저 더하면 된다! 그래서 로그 공간에서 다룬다.
    
    역방향: z = f₁⁻¹(f₂⁻¹(...fₙ⁻¹(x)))
    
    역변환은 역순으로 적용한다(껴입은 옷을 벗듯이).
    
    ============================================================================
    설계 원칙
    ============================================================================
    1. 교대 패턴: 흔히 여러 종류의 흐름을 번갈아 쓴다
       보기: [커플링, 배치 정규화, 커플링, 배치 정규화, ...]
    
    2. 점점 복잡하게: 앞선 흐름은 단순하게, 뒤 흐름은 복잡하게 할 수 있다
    
    3. 흐름의 수: 흔한 범위:
       - 2차원 장난감 문제: 흐름 4~8개
       - 이미지 생성: 흐름 20~40개
       - 흐름이 많을수록 표현력이 좋지만 느리다
    
    ============================================================================
    """
    
    def __init__(self, flows: list, base_dist: BaseDistribution):
        """
        흐름 변환의 시퀀스를 초기화한다.
        
        인수:
            flows (list): 합성할 Flow 객체의 목록
                         순전파 동안 차례대로 적용된다
                         보기: [CouplingLayer(), BatchNorm(), CouplingLayer()]
            
            base_dist (BaseDistribution): 출발 분포(흔히 정규 분포)
        
        설계 패턴:
            flows = [
                CouplingLayer(dim, hidden_dim=64),
                BatchNorm(dim),
                CouplingLayer(dim, hidden_dim=64),
                BatchNorm(dim),
            ]
            model = FlowSequence(flows, BaseDistribution(dim))
        """
        super().__init__()
        
        # PyTorch가 학습 가능한 파라메터임을 알도록 ModuleList를 쓴다
        self.flows = nn.ModuleList(flows)
        
        # 기저 분포를 담는다
        self.base_dist = base_dist
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        표본을 모든 흐름에 통과시켜 변환한다: z → x
        
        샘플링에 쓰는 생성 방향이다.
        
        과정:
            1. 기저 분포의 z에서 시작한다
            2. 흐름₁을 쓴다: z → h₁, log_det₁을 누적한다
            3. 흐름₂을 쓴다: h₁ → h₂, log_det₂을 누적한다
            4. 흐름을 모두 지날 때까지 이어 간다
            5. 최종 출력: x, total_log_det
        
        인수:
            z (torch.Tensor): 기저 분포의 잠재 표본, 형태 (batch_size, dim)
        
        반환값:
            다음을 담은 튜플:
                x (torch.Tensor): 최종 변환된 표본, 형태 (batch_size, dim)
                log_det_sum (torch.Tensor): 로그 행렬식의 합, 형태 (batch_size,)
        
        수학적 세부:
            log|det(∂x/∂z)| = Σᵢ log|det(∂fᵢ/∂fᵢ₋₁)|
            
        보기:
            >>> flows = [Flow1(), Flow2(), Flow3()]
            >>> model = FlowSequence(flows, BaseDistribution(2))
            >>> z = model.base_dist.sample(100)
            >>> x, log_det = model.forward(z)
            >>> # 이제 x에 생성한 표본 100개가 들어 있다
        """
        # 로그 행렬식 누적 변수를 초기화한다
        # 배치의 표본마다 0으로 시작한다
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        
        # z에서 시작해 차츰 바꾼다
        x = z
        
        # 흐름을 차례로 적용한다
        for flow in self.flows:
            # 변환: 지금 → 다음
            x, log_det = flow.forward(x)
            
            # 로그 행렬식을 누적한다
            # 핵심 통찰: 로그 공간에서 로그 행렬식은 더해진다
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        표본을 모든 흐름에 역방향으로 통과시켜 변환한다: x → z
        
        확률을 계산하는 데 쓰는 추론 방향이다.
        
        과정:
            1. x(실제 데이터)에서 시작한다
            2. 흐름ₙ⁻¹을 쓴다: x → hₙ₋₁, log_detₙ을 얻는다
            3. 흐름ₙ₋₁⁻¹을 쓴다: hₙ₋₁ → hₙ₋₂, log_detₙ₋₁을 얻는다
            4. 역순으로 이어 간다
            5. 최종 출력: z, total_log_det
        
        인수:
            x (torch.Tensor): 바꿀 데이터 표본, 형태 (batch_size, dim)
        
        반환값:
            다음을 담은 튜플:
                z (torch.Tensor): 잠재 표본, 형태 (batch_size, dim)
                log_det_sum (torch.Tensor): 로그 행렬식의 합, 형태 (batch_size,)
        
        중요한 참고:
            역변환은 역순으로 적용한다!
            순방향이 z → f₁ → f₂ → f₃ → x이면
            역방향은 x → f₃⁻¹ → f₂⁻¹ → f₁⁻¹ → z이다
            
        보기:
            >>> model = FlowSequence([Flow1(), Flow2()], BaseDistribution(2))
            >>> x = torch.randn(100, 2)  # 어떤 데이터
            >>> z, log_det = model.inverse(x)
            >>> # 이제 z가 정규 표본처럼 보여야 한다
        """
        # 로그 행렬식 누적 변수를 초기화한다
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        
        # x에서 시작해 차츰 거꾸로 z로 바꾼다
        z = x
        
        # 흐름마다의 역변환을 역순으로 적용한다
        # 여기서 reversed()가 결정적이다!
        for flow in reversed(self.flows):
            # 변환: 지금 → 앞선 것
            z, log_det = flow.inverse(z)
            
            # 로그 행렬식을 누적한다
            log_det_sum += log_det
        
        return z, log_det_sum
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        데이터 표본의 로그 확률을 계산한다.
        
        이것이 정규화 흐름의 핵심 식이다!
        
        수학 공식:
            log p(x) = log p(z) + log|det(∂z/∂x)|
            
        여기서 각 기호는 다음과 같다.
            - z = f⁻¹(x)는 잠재 표현이다
            - log p(z)는 기저 분포의 로그 확률이다
            - log|det(∂z/∂x)|가 부피 변화를 반영한다
        
        직관:
            1. 데이터 x를 잠재 공간으로 되돌린다: x → z
            2. 기저 분포에서 z가 얼마나 그럴듯한지 평가한다
            3. 변환하는 동안의 부피 변화를 반영해 보정한다
            4. 결과: 학습된 분포에서 x의 가능도
        
        인수:
            x (torch.Tensor): 평가할 데이터 표본, 형태 (batch_size, dim)
        
        반환값:
            torch.Tensor: 로그 확률, 형태 (batch_size,)
                         값이 클수록 학습된 분포에서 더 그럴듯하다
        
        학습에서 쓰기:
            >>> model = FlowSequence([...], BaseDistribution(2))
            >>> real_data = load_data()
            >>> log_probs = model.log_prob(real_data)
            >>> loss = -log_probs.mean()  # 음의 로그 가능도
            >>> loss.backward()  # 가능도를 키우도록 최적화한다
        
        생성에서 쓰기:
            >>> # 생성한 표본의 품질을 확인한다
            >>> generated = model.sample(100)
            >>> log_probs = model.log_prob(generated)
            >>> # log_probs가 클수록 품질이 좋다
        """
        # 단계 1: 데이터를 잠재 공간으로 바꾼다
        # 그러면 z와 변수 변환 보정항을 얻는다
        z, log_det = self.inverse(x)
        
        # 단계 2: 기저 분포에서 확률을 평가한다
        log_pz = self.base_dist.log_prob(z)
        
        # 단계 3: 변수 변환 공식을 쓴다
        # log p(x) = log p(z) + log|det(∂z/∂x)|
        return log_pz + log_det
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        학습된 분포에서 새 표본을 만든다.
        
        학습한 뒤 새 데이터를 생성하는 방법이다!
        
        과정:
            1. 단순한 기저 분포(정규 분포)에서 z를 뽑는다
            2. z를 모든 흐름에 통과시켜 변환한다
            3. 출력 x가 학습 데이터를 닮아야 한다
        
        인수:
            n_samples (int): 만들 표본의 수
            device (str): 만들 기기('cpu'나 'cuda')
        
        반환값:
            torch.Tensor: 생성한 표본, 형태 (n_samples, dim)
        
        학습한 뒤의 예:
            >>> # 이미지로 모델을 학습한다
            >>> model = FlowSequence([...], BaseDistribution(784))
            >>> train(model, mnist_data)
            >>> 
            >>> # 새 이미지를 만든다
            >>> new_images = model.sample(16)  # 새 이미지 16개를 만든다
            >>> # new_images가 실제 같은 MNIST 숫자로 보여야 한다
        
        품질 확인:
            학습한 뒤 표본은 다음이어야 한다:
            - 학습 데이터 분포를 닮는다
            - 다양하다(모두 같지 않다)
            - 모델에서 로그 확률이 높다
        """
        # 단계 1: 단순한 기저 분포에서 뽑는다
        z = self.base_dist.sample(n_samples, device=device)
        
        # 단계 2: 모든 흐름에 통과시켜 변환한다
        # 샘플링에는 필요 없으므로 log_det을 버린다
        x, _ = self.forward(z)
        
        return x


class AffineTransform(Flow):
    """
    아핀 변환: 가장 단순한 흐름
    
    ============================================================================
    개념 개요
    ============================================================================
    아핀 변환은 가장 단순한 흐름이다:
    
        y = 스케일 × x + 이동
    
    이는 다음과 같다:
    - 차원마다 늘이거나 줄이기(스케일)
    - 중심 이동(이동)
    
    1차원 보기:
        스케일=2이고 이동=3이면:
        입력: [0, 1, 2] → 출력: [3, 5, 7]
    
    ============================================================================
    수학의 성질
    ============================================================================
    순방향:
        y = exp(log_scale) × x + 이동
        
    왜 exp(log_scale)인가?
        - 스케일 대신 log_scale을 배운다
        - 그러면 스케일이 늘 양수이다
        - 최적화에서 수치적으로 더 안정적이다
    
    야코비안:
        ∂y/∂x = diag(exp(log_scale))
        
    로그 행렬식:
        log|det(∂y/∂x)| = sum(log_scale)
        
    아주 효율이 좋다. 파라메터를 더하기만 하면 된다!
    
    ============================================================================
    쓰임새
    ============================================================================
    1. 데이터 정규화: 입력 데이터의 중심과 스케일 맞추기
    2. 단순한 베이스라인: 더 복잡한 흐름을 이것과 견준다
    3. 첫 레이어: 이따금 첫 변환으로 쓴다
    4. 디버깅: 올바른지 확인하기 쉽다
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        아핀 변환을 초기화한다.
        
        인수:
            dim (int): 데이터의 차원
        
        학습되는 파라메터:
            - log_scale: 차원마다 늘이기와 줄이기를 조절한다
            - shift: 차원마다 이동을 조절한다
        
        초기화:
            - log_scale은 0에서 시작한다(스케일 = 1, 바뀜 없음)
            - shift는 0에서 시작한다(이동 없음)
        """
        super().__init__()
        
        # 학습 가능한 파라메터
        # 수치 안정을 위해 log(scale)을 배운다
        # 초기 스케일 = exp(0) = 1(바뀜 없음)
        self.log_scale = nn.Parameter(torch.zeros(dim))
        
        # 초기 이동 = 0(이동 없음)
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순방향 변환: y = 스케일 * z + 이동
        
        인수:
            z (torch.Tensor): 입력 표본, 형태 (batch_size, dim)
        
        반환값:
            다음을 담은 튜플:
                x (torch.Tensor): 바꾼 표본
                log_det (torch.Tensor): 로그 행렬식
        
        보기:
            >>> transform = AffineTransform(dim=2)
            >>> z = torch.tensor([[0., 0.], [1., 1.]])
            >>> x, log_det = transform.forward(z)
        """
        # log_scale을 실제 스케일로 바꾼다(늘 양수)
        scale = torch.exp(self.log_scale)
        
        # 아핀 변환을 쓴다
        # 브로드캐스팅: 스케일과 이동을 원소마다 적용한다
        x = scale * z + self.shift
        
        # 로그 행렬식을 계산한다
        # 대각 야코비안: 행렬식 = 대각의 곱
        # log(det) = 대각의 로그 합 = sum(log_scale)
        log_det = self.log_scale.sum().expand(z.shape[0])
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        역변환: z = (x - 이동) / 스케일
        
        인수:
            x (torch.Tensor): 바꾼 표본, 형태 (batch_size, dim)
        
        반환값:
            다음을 담은 튜플:
                z (torch.Tensor): 원래 표본
                log_det (torch.Tensor): 로그 행렬식(순방향의 음수)
        """
        # log_scale을 실제 스케일로 바꾼다
        scale = torch.exp(self.log_scale)
        
        # 아핀 역변환을 쓴다
        z = (x - self.shift) / scale
        
        # 역의 로그 행렬식
        # det(J_inv) = 1/det(J_forward)
        # log(det(J_inv)) = -log(det(J_forward))
        log_det = -self.log_scale.sum().expand(x.shape[0])
        
        return z, log_det


class PlanarFlow(Flow):
    """
    평면 흐름: 변환에 비선형성 더하기
    
    ============================================================================
    개념 개요
    ============================================================================
    평면 흐름은 "평면" 변환으로 비선형성을 더한다:
    
        f(z) = z + u × tanh(w^T z + b)
    
    이렇게 생각하라:
    - w^T z + b: 선형 투영(결정 경계 같은 것)
    - tanh: 값을 -1과 1 사이로 압축한다
    - u: 변환의 방향
    - 결과: "평면"을 따라 공간을 굽힌다
    
    ============================================================================
    기하학적 직관
    ============================================================================
    평평한 2차원 면을 떠올려 보라:
    1. 항 w^T z + b가 그 면에 선을 정한다
    2. tanh가 이 선 둘레에 매끄러운 전이를 만든다
    3. u가 그 선에 수직으로 얼마나 "미는지" 조절한다
    
    그래서 다음을 할 수 있는 매끄럽고 굽은 변환이 된다:
    - 분포에 마루나 골을 만든다
    - 정규 분포를 굽은 모양으로 휘게 한다
    
    ============================================================================
    수학적 세부
    ============================================================================
    순방향:
        f(z) = z + u × tanh(w^T z + b)
    
    야코비안:
        ∂f/∂z = I + u × ψ^T
        여기서 ψ = (1 - tanh²(w^T z + b)) × w
    
    로그 행렬식:
        log|det(∂f/∂z)| = log|1 + u^T ψ|
        
    효율을 위해 행렬식 보조 정리를 쓴다!
    
    ============================================================================
    한계
    ============================================================================
    1. 닫힌 형태의 역변환이 없다: 반복 솔버가 필요하다
       (그래서 커플링 흐름을 흔히 더 낫게 여긴다)
    
    2. 표현력이 제한된다: 평면 흐름 하나는 꽤 단순하다
       (복잡한 분포를 나타내려면 흐름이 많이 필요하다)
    
    3. 주로 쓸모 있는 곳: 저차원 문제(2차원, 3차원)
    
    ============================================================================
    """
    
    def __init__(self, dim: int):
        """
        평면 흐름 변환을 초기화한다.
        
        인수:
            dim (int): 공간의 차원
        
        학습되는 파라메터:
            - weight (w): "평면"의 방향을 정한다, 형태 (dim,)
            - bias (b): 평면을 옮긴다, 형태 (1,)
            - u: 변환의 방향, 형태 (dim,)
        
        초기화:
            - weight: 정규 분포로 무작위 초기화(다양한 초기 평면을 만든다)
            - bias: 0(평면이 원점을 지난다)
            - u: 정규 분포로 무작위 초기화(무작위 변환 방향)
        """
        super().__init__()
        
        # 평면 변환의 파라메터
        # w: 입력 공간의 초평면을 정한다
        self.weight = nn.Parameter(torch.randn(dim))
        
        # b: 초평면의 바이어스
        self.bias = nn.Parameter(torch.zeros(1))
        
        # u: 변환의 방향
        self.u = nn.Parameter(torch.randn(dim))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        평면 흐름을 지나는 순방향 변환.
        
        과정:
            1. 선형 투영을 계산한다: w^T z + b
            2. tanh 비선형성을 적용한다
            3. u로 스케일을 맞추어 z에 더한다
            4. 야코비안 행렬식을 계산한다
        
        인수:
            z (torch.Tensor): 입력 표본, 형태 (batch_size, dim)
        
        반환값:
            다음을 담은 튜플:
                x (torch.Tensor): 바꾼 표본
                log_det (torch.Tensor): 로그 행렬식
        
        수치 안정성:
            - tanh가 값을 제한한다
            - log(0)을 피하려 작은 엡실론(1e-8)을 더한다
        """
        # 단계 1: 선형 투영 w^T z + b를 계산한다
        # 표본마다: 웨이트 벡터와의 내적에 바이어스를 더한다
        # 형태: (배치 크기, 차원) × (차원,) → (배치 크기, 1)
        linear = torch.sum(self.weight * z, dim=-1, keepdim=True) + self.bias
        
        # 단계 2: 변환을 쓴다
        # f(z) = z + u × tanh(w^T z + b)
        # keepdim=True여야 브로드캐스팅이 올바르게 된다
        x = z + self.u * torch.tanh(linear)
        
        # 단계 3: 야코비안 행렬식을 계산한다
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
        역변환: x → z
        
        중요한 한계:
        평면 흐름에는 닫힌 형태의 역변환이 없다!
        
        뒤집으려면 다음을 풀어야 한다:
            z = x - u × tanh(w^T z + b)
            
        이는 반복법(뉴턴 방법이나 고정점 반복 같은)이 필요하다.
        
        대개의 쓰임새에서는 효율적인 닫힌 형태의 역변환이 있는 커플링 흐름을 쓰고
        평면 흐름은 피한다.
        
        인수:
            x: 바꾼 표본
        
        발생시키는 예외:
            NotImplementedError: 이 메서드는 반복 솔버가 필요하다
        
        이것이 필요하다면:
            고정점 반복이나 뉴턴-랩슨을 구현하라:
            
            def inverse_iterative(x, n_iters=10):
                z = x  # 초기 추정값
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
    이 함수는 정규화 흐름이 무엇을 하는지 눈으로 보게 해 준다!
    
    나란히 놓은 비교 그림을 만든다:
    - 왼쪽: 기저 정규 분포의 표본(잠재 공간 z)
    - 오른쪽: 흐름을 지나 바꾼 뒤(데이터 공간 x)
    
    이 그림은 다음을 이해하는 데 결정적이다:
    - 흐름이 공간을 어떻게 휘게 하는지
    - 학습이 잘 되고 있는지
    - 흐름이 어떤 패턴을 학습했는지
    
    ============================================================================
    해석 가이드
    ============================================================================
    학습 전:
        - 왼쪽: 깔끔한 정규 분포 덩어리
        - 오른쪽: 조금 휜 정규 분포(무작위 초기화)
    
    학습하는 동안:
        - 오른쪽이 차츰 데이터 분포에 맞게 바뀐다
        - 복잡도와 구조가 늘어나는지 확인하라
    
    학습한 뒤:
        - 왼쪽: 여전히 정규 분포(결코 바뀌지 않는다)
        - 오른쪽: 목표 분포와 맞아야 한다(예컨대 달 모양, 동그라미)
    
    문제 해결:
        - 오른쪽이 그대로인가? → 모델이 학습되지 않고 있다
        - 오른쪽이 어지러운가? → 러닝레이트가 너무 크다
        - 오른쪽이 뭉쳐 있는가? → 레이어나 용량이 모자란다
    
    ============================================================================
    
    인수:
        flow_model (nn.Module): 시각화할 학습된 흐름 모델
        n_points (int): 뽑아 그릴 점의 수
        xlim (tuple): 두 그림의 x축 범위
        ylim (tuple): 두 그림의 y축 범위
        filename (str): 그림을 갈무리할 곳
    
    사용 예:
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> train(model, data)  # 2차원 데이터로 학습한다
        >>> visualize_2d_transformation(model, n_points=2000)
        >>> # 결과를 보려면 transformation.png를 열어라!
    """
    # 모델을 평가 모드로 둔다(드롭아웃과 배치 정규화 갱신을 끈다)
    flow_model.eval()
    
    # 효율을 위해 그래디언트 계산 끄기
    with torch.no_grad():
        # 단계 1: 기저 정규 분포에서 뽑는다
        z = flow_model.base_dist.sample(n_points)
        
        # 단계 2: 흐름을 지나 바꾼다
        x, _ = flow_model.forward(z)
        
        # 단계 3: 그리려 CPU로 옮긴다(GPU에 있으면)
        z = z.cpu().numpy()
        x = x.cpu().numpy()
    
    # 나란한 그림을 만든다
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 왼쪽 그림: 잠재 공간(늘 정규 분포)
    axes[0].scatter(z[:, 0], z[:, 1], alpha=0.5, s=10)
    axes[0].set_title('Latent Space (z) - Base Gaussian')
    axes[0].set_xlabel('z₁')
    axes[0].set_ylabel('z₂')
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_aspect('equal')  # 가로세로 비를 같게
    axes[0].grid(True, alpha=0.3)
    
    # 오른쪽 그림: 데이터 공간(바뀐 분포)
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
    2차원 흐름이 학습한 확률 밀도를 그려 본다.
    
    ============================================================================
    목적
    ============================================================================
    visualize_2d_transformation이 표본을 보이는 반면 이 함수는
    확률 밀도, 곧 모델에서 위치마다 얼마나 그럴듯한지를 보인다.
    
    다음과 같은 히트맵을 만든다:
    - 밝은 영역 = 높은 확률(모델이 데이터가 여기 있어야 한다고 본다)
    - 어두운 영역 = 낮은 확률(모델이 여기 데이터가 있기 어렵다고 본다)
    
    ============================================================================
    해석 가이드
    ============================================================================
    잘 학습된 모델:
        - 봉우리가 데이터 군집과 맞는다
        - 매끄럽고 이어진 확률 풍경
        - 봉우리 사이가 또렷이 갈린다
    
    잘못 학습된 모델:
        - 어디나 평평하다(구조를 학습하지 못했다)
        - 봉우리가 엉뚱한 곳에 있다
        - 봉우리가 지나치게 뾰족하다(과적합)
    
    데이터와 견주기:
        실제 데이터의 산점도를 겹쳐 다음을 확인하라:
        - 데이터가 빽빽한 곳에서 확률이 높다
        - 데이터가 성긴 곳에서 확률이 낮다
    
    ============================================================================
    계산에 대한 참고
    ============================================================================
    이는 200×200 격자의 모든 점, 곧 40,000개 점에서 확률을 계산한다!
    해상도가 높거나 차원이 높은 문제에서는 느릴 수 있다.
    
    시간 복잡도: O(n_grid² × 모델 복잡도)
    
    ============================================================================
    
    인수:
        flow_model (nn.Module): 학습된 흐름 모델
        xlim (tuple): 밀도 격자의 x축 범위
        ylim (tuple): 밀도 격자의 y축 범위
        n_grid (int): 밀도 격자의 해상도(n_grid × n_grid 점)
        filename (str): 그림을 갈무리할 곳
    
    사용 예:
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> train(model, moons_data)
        >>> visualize_2d_density(model)
        >>> # 그림에 초승달 형태의 확률 높은 영역 둘이 보여야 한다!
    """
    # 평가 모드로 둔다
    flow_model.eval()
    
    # 단계 1: 공간 전체를 덮는 점 격자를 만든다
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)  # 2차원 격자를 만든다
    
    # 단계 2: 격자를 (x, y) 점 목록으로 펼친다
    # 형태: (n_grid*n_grid, 2)
    points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), 
        dtype=torch.float32
    )
    
    # 단계 3: 격자 점마다 로그 확률을 계산한다
    with torch.no_grad():
        log_prob = flow_model.log_prob(points)
        
        # 로그 확률을 실제 확률로 바꾼다
        # prob = exp(log_prob)
        prob = torch.exp(log_prob).cpu().numpy()
    
    # 단계 4: 그리려 격자 형태로 되돌린다
    prob = prob.reshape(n_grid, n_grid)
    
    # 단계 5: 채운 등고선 그림을 만든다
    plt.figure(figsize=(8, 7))
    
    # contourf가 매끄러운 색 영역을 만든다
    # levels=50이 촘촘한 등고 단계를 준다
    plt.contourf(X, Y, prob, levels=50, cmap='viridis')
    
    # 확률 스케일을 보이려 컬러바를 더한다
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
    최대 가능도 추정으로 정규화 흐름 모델을 학습한다.
    
    ============================================================================
    개념 개요
    ============================================================================
    학습 목표: 학습 데이터의 가능도 최대화
    
    달리 말해 학습된 분포에서 실제 데이터의 확률이 높아지도록
    모델 파라메터를 조정한다.
    
    로스 함수:
        로스 = 학습 데이터의 x에 대해 -mean(log p(x))
        
    음의 로그 가능도를 최소화한다(= 가능도를 최대화한다)
    
    ============================================================================
    학습 알고리즘
    ============================================================================
    에포크마다:
        배치마다:
            1. 배치의 log p(x)를 계산한다
               - 역변환으로 x → z를 구한다
               - log p(z) + log|det|을 평가한다
            
            2. 로스 = -mean(log p(x))를 계산한다
               - 최소화하므로 음수를 붙인다
               - 그래디언트가 안정되도록 배치에 걸쳐 평균 낸다
            
            3. 역전파하고 파라메터를 갱신한다
               - 그래디언트를 계산한다: ∂Loss/∂θ
               - 갱신한다: θ ← θ - lr × ∇Loss
    
    ============================================================================
    학습 모니터링
    ============================================================================
    정상적인 학습:
        - 로스가 꾸준히 줄어든다
        - 마침내 평평해진다
        - 심하게 흔들리지 않는다
    
    살펴야 할 문제:
        - 로스가 는다 → 러닝레이트가 너무 크다
        - 로스가 흔들린다 → 러닝레이트나 배치 크기를 줄여라
        - 로스가 멈춘다 → 모델이 너무 단순하거나 데이터가 너무 복잡하다
        - 로스 → NaN → 수치가 불안정하다(그래디언트 폭발)
    
    ============================================================================
    하이퍼파라메터 요령
    ============================================================================
    러닝레이트:
        - 시작: 1e-3이나 1e-4
        - 학습이 불안정하면 낮춘다
        - 러닝레이트 스케줄을 쓸 수 있다
    
    배치 크기:
        - 클수록 안정되지만 느리다
        - 작을수록 빠르지만 잡음이 많다
        - 흔히: 작은 데이터셋에 64~256
    
    에포크 수:
        - 2차원 장난감 데이터: 100~500에포크
        - 이미지: 50~100에포크
        - 로스를 지켜보다 수렴하면 멈춘다
    
    ============================================================================
    
    인수:
        flow_model (nn.Module): 학습할 흐름 모델
        dataloader: 학습 데이터를 담은 PyTorch DataLoader
        optimizer: PyTorch 옵티마이저(예컨대 Adam)
        n_epochs (int): 학습 에포크 수
        device (str): 'cpu'나 'cuda'
    
    반환값:
        list: 에포크마다의 학습 로스
    
    사용 예:
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> 
        >>> # 데이터를 준비한다
        >>> data = generate_toy_data('moons', n_samples=2000)
        >>> dataset = TensorDataset(data)
        >>> dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        >>> 
        >>> # 모델을 만든다
        >>> model = build_realnvp_model(dim=2, n_layers=6)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> 
        >>> # 학습한다!
        >>> losses = train_flow(model, dataloader, optimizer, n_epochs=500)
        >>> 
        >>> # 학습 곡선을 그린다
        >>> plot_training_loss(losses)
    """
    # 모델을 학습 모드로
    # 그러면 드롭아웃과 배치 정규화 갱신 같은 것이 켜진다
    flow_model.train()
    
    # 그리려 로스 값을 담는 목록
    losses = []
    
    # 학습 루프
    for epoch in range(n_epochs):
        epoch_loss = 0.0  # 이 에포크의 누적값
        n_batches = 0     # 배치를 센다
        
        # 배치들을 순회한다
        for batch in dataloader:
            # 여러 데이터 로더 형식을 처리한다
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # 데이터를 꺼낸다(라벨이 있으면 무시한다)
            
            # 배치를 기기(CPU나 GPU)로 옮긴다
            batch = batch.to(device)
            
            # 데이터의 차원이 2보다 많으면 펼친다
            # 예컨대 MNIST는 (배치 크기, 28, 28) → (배치 크기, 784)
            if batch.dim() > 2:
                batch = batch.view(batch.shape[0], -1)
            
            # ==================== 순전파 ====================
            # 지금 모델에서 배치의 로그 확률을 계산한다
            # 여기에는 다음이 든다:
            #   1. 역변환: x → z
            #   2. 기저 로그 확률 평가: log p(z)
            #   3. 야코비안 보정항 더하기: log|det|
            log_prob = flow_model.log_prob(batch)
            
            # ==================== 로스 계산 ====================
            # 음의 로그 가능도 로스
            # 로그 확률을 크게 하고 싶으므로 음의 로그 확률을 최소화한다
            loss = -log_prob.mean()
            
            # ==================== 역전파 ====================
            # 이전 그래디언트를 지운다
            optimizer.zero_grad()
            
            # 역전파로 그래디언트를 계산한다
            loss.backward()
            
            # 파라메터 갱신
            optimizer.step()
            
            # ==================== 기록 ====================
            epoch_loss += loss.item()
            n_batches += 1
        
        # 이 에포크의 평균 로스를 계산한다
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # 주기적으로 진행 상황을 출력한다
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def plot_training_loss(losses: list, filename: str = 'training_loss.png'):
    """
    학습 로스 곡선을 그린다.
    
    ============================================================================
    목적
    ============================================================================
    로스 곡선을 그려 보면 다음을 알 수 있다:
    - 학습이 잘 되는가?(로스가 줄어야 한다)
    - 학습이 수렴했는가?(로스가 평평해진다)
    - 문제가 있는가?(진동, 발산)
    
    ============================================================================
    해석 가이드
    ============================================================================
    좋은 학습 곡선:
        - 매끄럽게 줄어든다
        - 마침내 평평해진다
        - 갑자기 치솟지 않는다
        보기: \___  (높이 시작해 떨어지고 평평해진다)
    
    문제:
        - 평평한 선 → 모델이 학습하지 못한다(러닝레이트와 모델 용량을 확인하라)
        - 증가 → 발산한다(러닝레이트를 줄여라)
        - 진동 → 불안정하다(러닝레이트나 배치 크기를 줄여라)
        - 뚝 떨어진 뒤 평평 → 에포크를 더 돌리거나 더 나은 초기화가 필요하다
    
    ============================================================================
    
    인수:
        losses (list): 학습에서 나온 로스 값의 목록
        filename (str): 그림을 갈무리할 곳
    
    사용 예:
        >>> losses = train_flow(model, dataloader, optimizer, n_epochs=500)
        >>> plot_training_loss(losses)
        >>> # 학습이 잘 됐는지 training_loss.png로 확인하라
    """
    plt.figure(figsize=(10, 5))
    
    # 에포크에 대한 로스를 그린다
    plt.plot(losses, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative Log-Likelihood', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    
    # 읽기 쉽도록 격자를 더한다
    plt.grid(True, alpha=0.3)
    
    # 마지막 로스를 적어 넣는다
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
# 모듈 요약과 다음 단계
# ============================================================================
"""
축하한다! 기본 구성 요소를 모두 마쳤다!

배운 것:
✓ BaseDistribution: 샘플링의 출발점(정규 분포)
✓ Flow: 변환의 추상 클래스
✓ FlowSequence: 여러 흐름의 합성
✓ AffineTransform: 가장 단순한 흐름
✓ PlanarFlow: 비선형성 더하기
✓ 2차원 흐름 시각화 도구
✓ 최대 가능도 학습 파이프라인

다음 단계:
1. 더 강력한 변환을 보려면 coupling_flows.py를 공부하라
   - 커플링 레이어(RealNVP)
   - 배치 정규화 흐름
   - 이미지를 위한 바둑판 패턴

2. 모두 도는 것을 보려면 example_2d_flows.py를 돌려라
   - 장난감 2차원 데이터셋으로 학습한다
   - 변환을 그려 본다
   - 학습 동역학을 이해한다

3. 네 데이터로 실험해 보라!
   - 2차원 장난감 문제로 시작하라
   - 여러 구조를 시험하라
   - 배운 것을 그려 보라

읽어 볼 것:
- "Normalizing Flows for Probabilistic Modeling" (Papamakarios et al., 2019)
- "Density Estimation Using Real NVP" (Dinh et al., 2017)
- "Variational Inference with Normalizing Flows" (Rezende & Mohamed, 2015)

잘하는 요령:
- 늘 그려 보라! 준비된 시각화 도구를 써라
- 단순하게 시작하라: 2차원 데이터, 적은 레이어
- 학습 로스를 꼼꼼히 지켜보라
- 하이퍼파라메터를 실험하라
- 가역성을 확인하라: x → z → x가 x를 돌려주어야 한다

즐겁게 배우기를! 🎓
"""
from coupling_flows import CouplingLayer, BatchNorm


class MNISTFlow:
    """MNIST 흐름 학습의 래퍼 클래스."""
    
    def __init__(self, n_layers: int = 8, hidden_dim: int = 256,
                 batch_size: int = 128, lr: float = 1e-4, device: str = None):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lr = lr
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 모델을 만든다
        dim = 28 * 28  # MNIST 이미지 크기
        flows = []
        
        for i in range(n_layers):
            # 마스크를 번갈아 쓴다
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[dim // 2:] = 1
            else:
                mask[:dim // 2] = 1
            
            flows.append(CouplingLayer(dim, hidden_dim, mask))
            flows.append(BatchNorm(dim))
        
        base_dist = BaseDistribution(dim)
        self.model = FlowSequence(flows, base_dist).to(self.device)
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_dataloader(self, train: bool = True):
        """MNIST 데이터 로더를 만든다."""
        # 연속화(dequantization)하고 [0, 1] 범위로 맞춘다
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x + torch.rand_like(x) / 256.))  # 연속화(dequantization)
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )
    
    def train_epoch(self, dataloader):
        """한 에포크를 학습한다."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, _ in pbar:
            images = images.to(self.device)
            images = images.view(images.shape[0], -1)  # 편다
            
            # 음의 로그가능도를 계산한다
            log_prob = self.model.log_prob(images)
            loss = -log_prob.mean()
            
            # 차원당 비트(bits/dim) 지표를 계산한다
            bpd = loss / (28 * 28 * np.log(2))
            
            # 최적화
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'bpd': f'{bpd.item():.4f}'})
        
        return total_loss / num_batches
    
    def sample_images(self, n_samples: int = 64):
        """표본 이미지를 만든다."""
        self.model.eval()
        
        with torch.no_grad():
            samples = self.model.sample(n_samples, device=self.device)
            samples = samples.view(n_samples, 1, 28, 28)
            samples = torch.clamp(samples, 0, 1)
        
        return samples
    
    def train(self, n_epochs: int = 50, save_interval: int = 10):
        """흐름 모델을 학습한다."""
        dataloader = self.get_dataloader(train=True)
        
        os.makedirs('samples', exist_ok=True)
        losses = []
        
        print(f"\nTraining for {n_epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            avg_loss = self.train_epoch(dataloader)
            losses.append(avg_loss)
            
            print(f"Average loss: {avg_loss:.4f}")
            
            # 표본 만들기
            if epoch % save_interval == 0 or epoch == 1:
                print("Generating samples...")
                samples = self.sample_images(64)
                self.visualize_samples(samples, f'samples/epoch_{epoch:04d}.png')
            
            # 체크포인트 저장
            if epoch % 25 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # 학습 로스 그리기
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('MNIST Flow Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=150)
        plt.close()
        
        print("\nTraining complete!")
        
        return losses
    
    def visualize_samples(self, samples, filename='samples.png'):
        """생성한 표본을 그려 본다."""
        from torchvision.utils import make_grid
        
        grid = make_grid(samples, nrow=8, padding=2)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid[0].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved samples to {filename}")
    
    def save_checkpoint(self, filename: str):
        """모델 체크포인트를 저장한다."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Saved checkpoint to {filename}")
    
    def load_checkpoint(self, filename: str):
        """모델 체크포인트를 불러온다."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")


import numpy as np

def main():
    """메인 학습 스크립트."""
    print("=" * 50)
    print("MNIST Normalizing Flow Training")
    print("=" * 50)
    
    # 초기화한다
    mnist_flow = MNISTFlow(
        n_layers=8,
        hidden_dim=256,
        batch_size=128,
        lr=1e-4
    )
    
    # 학습
    mnist_flow.train(n_epochs=50, save_interval=10)
    
    # 마지막 표본을 만든다
    print("\nGenerating final samples...")
    samples = mnist_flow.sample_images(64)
    mnist_flow.visualize_samples(samples, 'final_samples.png')
    
    # 모델을 저장한다
    mnist_flow.save_checkpoint('mnist_flow_final.pt')
    
    print("\nDone! Check the 'samples' folder for generated images.")


if __name__ == "__main__":
    main()
```

기본은 커플링 8개와 BatchNorm 8개, 은닉 256, 배치 128, Adam 러닝레이트 0.0001이며 main은 50에포크를 요청한다. flow_utils의 내용은 위 코드에 그대로 들어 있지만 CouplingLayer·BatchNorm은 로컬 coupling_flows에서 import하므로 실행 환경에서 이 모듈이 검색되어야 한다. numpy는 클래스 정의 아래에서 import하지만 main 실행 전에 수행되므로 그 위치 자체가 np 참조 오류를 만들지는 않는다.

입력은 `(B,1,28,28)`에서 `(B,784)`로 펼친다. 로컬 CouplingLayer는 전체 784차원을 받는 스케일·이동 MLP 두 개를 사용하고 한쪽 좌표를 마스크로 0 처리한다. 앞 절의 절반 차원 MLP 구현과 다르다. 기본 전체 학습 파라메터는 7,504,392개이며 마스크·running 통계 버퍼는 제외한 값이다.

## 2. Dequantization과 bits/dim

8비트 정수 픽셀 k에 대한 흔한 연속화는 u~Uniform(0,1)을 뽑아 y=(k+u)/256으로 만드는 것이다. 정수 관측의 가능도(likelihood, 우도라고도 한다)는 각 픽셀 구간에 연속 밀도를 적분한 질량이고, 균일 잡음을 사용한 평균 연속 로그밀도는 이에 대한 하한을 제공한다. 이때 D개 픽셀의 음의 로그밀도 NLL에서 통상적인 8비트 bits/dim 상한을 계산하면 8+NLL/(D ln 2)가 된다.

본문 코드는 ToTensor의 k/255에 u/256을 더한다. 따라서 위 (k+u)/256과 다른 위치의 구간을 사용하고 흰 픽셀은 1을 넘을 수 있다. 연속 흐름 입력으로 1 초과가 금지되는 것은 아니지만 표준 256개 픽셀 구간의 가능도로 바로 해석할 수는 없다. 또한 출력 변수 bpd는 NLL/(784 ln 2)만 계산해 위 8비트 보정 8이 없다. 이 값을 정수 MNIST의 통상적인 bits/dim 결과와 직접 비교해서는 안 된다.

[RealNVP 논문](https://arxiv.org/abs/1605.08803)과 [정규화 흐름 리뷰](https://jmlr.org/papers/v22/19-1028.html)는 연속 밀도와 이미지 가능도 평가의 관계를 다룬다. 로그밀도는 좌표 스케일에 의존하므로 전처리와 보정 항을 함께 보고한다.

## 3. 학습 모드 정규화의 한계

로컬 BatchNorm.inverse는 학습 모드에서 현재 입력 배치의 평균·분산을 계산해 정규화한다. 이 통계를 상수로 취급한 로그 스케일을 반환하지만 실제 통계가 배치에 의존하는 연산 전체의 표본별 야코비안을 나타내지는 않는다. 배치 평균을 제거하는 연산은 배치 전체의 공통 이동 정보를 잃기도 한다. 따라서 이 학습 로스를 엄밀한 독립 표본의 정확한 로그가능도로 단정할 수 없다.

평가 모드에서는 고정 running 통계를 사용하므로 양의 분산과 유한 스케일 아래 가역 선형 변환으로 해석할 수 있다. 그러나 학습 모드의 통계를 양방향에서 새로 계산하면 같은 통계가 보장되지 않아 왕복 복원도 일반적으로 성립하지 않는다. BatchNorm이라는 이름이 같다고 흐름용 정규화의 가역성이 보장되는 것은 아니다.

CouplingLayer의 로그 스케일은 tanh에 학습 가능한 exp(scale_cap)을 곱한다. 초기에는 제한되지만 이 배율 자체가 커질 수 있으므로 전체 학습에서 고정된 스케일 상한이 있는 것은 아니다. 그래디언트 클리핑도 이 문제나 위 밀도 계약을 자동 해결하지 않는다.

## 4. 기록·생성과 체크포인트

에포크 로스는 배치 평균들의 단순 평균이므로 마지막 배치 크기가 다르면 전체 표본당 평균과 다르다. 코드에는 검증·독립 시험 평가·조기 종료가 없다. 샘플은 매 에포크가 아니라 첫 에포크와 지정 간격에 저장하며 25에포크 간격으로 체크포인트를 저장한다.

sample_images는 eval·no_grad로 생성한 연속 출력을 0~1로 clipping한다. 이는 표시용 후처리이며 clipping 전 연속 흐름의 표본과 다르다. clipping한 픽셀은 경계에 질량을 만들 수 있어 원래 log_prob로 그 후처리 분포의 밀도를 평가할 수 없다. 다음 train_epoch는 다시 train 모드를 설정한다.

체크포인트는 모델·옵티마이저 상태만 저장하고 구조·전처리·에포크·난수 상태는 저장하지 않는다. 복원 전에 같은 구조를 만들어야 하며 재학습 재현에 필요한 설정은 별도 기록해야 한다. 이 절은 새 성능 수치·그림을 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
매 배치 zero_grad가 필요한 이유와 누적 학습의 차이를 설명하라.

</div>

??? success "풀이"

    현재 루프는 매 배치 step하므로 기존 그래디언트를 초기화하지 않으면 이전 배치 그래디언트가 계속 더해져 원래와 다른 업데이트가 된다. 단순한 러닝레이트 배수나 반드시 발산하는 과정이라고 설명할 수는 없다. 의도한 누적은 여러 배치 로스의 스케일을 맞춰 backward한 뒤 한 번 step하고 초기화한다. 정규화 레이어의 배치 통계도 바뀌므로 누적만으로 큰 배치와 완전히 같은 학습이 되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 옵티마이저와 비교 실험의 우선 조건을 설명하라.

</div>

??? success "풀이"

    이미 Adam을 사용하며 기본 러닝레이트는 0.0001이다. Adam 0.001로 바꾸면 종류가 아닌 러닝레이트 비교다. SGD 등과의 비교에는 동일한 검증 목적·구조·전처리와 같은 튜닝 예산이 필요하다. 먼저 정규화·전처리·로그밀도 계약을 확인해야 최적화 방법에 따른 차이를 올바르게 해석할 수 있다. 한 번의 로스 곡선으로 일반적 우열을 정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 예제의 밀도 평가에서 확인할 두 가지 핵심 문제를 설명하라.

</div>

??? success "풀이"

    첫째, 학습 배치 통계를 새로 계산하는 BatchNorm의 반환값을 독립 표본의 정확한 야코비안 로그값으로 읽을 수 없다. 둘째, k/255+u/256 전처리와 보정 없는 bpd는 표준 정수 MNIST bits/dim과 다르다. 평가 모드의 고정 통계에서 모형 밀도를 정의하고, 정수 픽셀 구간·연속화·야코비안 보정을 일관되게 정해야 한다. 밝은 샘플 격자나 감소한 학습 로스만으로 두 문제가 사라지지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 골격에 필요한 보완을 설명하라.

</div>

??? success "풀이"

    본문에는 검증 로더·evaluate가 없으므로 같은 전처리·평가 모드에서 비교 가능한 검증 목적을 먼저 정의해야 한다. state_dict().copy는 얕은 복사라 최적 웨이트의 독립 보존에 충분하지 않다. 깊은 복제나 파일 저장이 필요하고 정상 종료에서도 최적 상태를 복원해야 한다. 검증 잡음과 모델 선택 기준을 기록하며 시험 집합은 선택 과정과 분리한다.

    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```


## 정리하며

MNIST 흐름 예제는 학습·생성의 연결을 보여 주지만 전처리·bits/dim·배치 정규화의 밀도 해석에 주의가 필요하다. 고정 통계의 평가 모형과 학습 중 연산을 구분하고 출력 clipping·체크포인트 설정도 명시해야 결과를 비교할 수 있다.
