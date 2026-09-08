# PGD 치기

62.2 묶음: 되비춘 기울기 내림(PGD) 치기 - 가운데 걸음. 이 묶음은 더 센 치기인 되비춘 기울기 내림(PGD)을 짜 놓았다

맞섬에 든든하기는 안전이 걸린 자리에 신경 그물을 내놓을 때 종요로운 걱정거리다. 이 짜보기는 작은 흔듦이 어떻게 모형을 속이는지, 막이는 어떻게 지을 수 있는지 보이며 맞섬에 든든하기의 깨침을 드러낸다.

## 1. 코드

```python
"""
62.2 묶음: 되비춘 기울기 내림(PGD) 치기 - 가운데 걸음

이 묶음은 FGSM의 더 센 되돌이 갈래인 되비춘 기울기 내림(PGD) 치기를
짜 놓았다. PGD은 가장 센 일차 맞서는 치기의 하나로 여겨지며
모형의 든든함을 따지는 데 흔히 쓰인다.

수학 밑그림:
=======================

PGD은 작은 기울기 걸음을 여러 번 밟고 걸음마다 받아 주는 흔듦 모임으로
되비추어 FGSM을 넓힌다. 그래서 한 걸음짜리 FGSM보다
훨씬 세다.

매개변수가 θ인 모형 f, 들임 x, 이름표 y, 흔듦 예산 ε이 있을 때:

PGD은 다음을 푼다:
    maximize_{||δ||_∞ ≤ ε} L(θ, x + δ, y)

알고리즘(과녁 없음):
-----------------------
1. 첫자리: x^(0) = x + uniform_noise[-ε, ε]  (아무 비롯 자리)
2. t = 0에서 T-1까지:
       x^(t+1) = Π_{x+S}(x^(t) + α · sign(∇_x L(θ, x^(t), y)))
3. x^(T)을 돌려준다

여기서:
- Π_{x+S}은 ε 공으로의 되비춤: Π(z) = clip(z, x-ε, x+ε)
- α은 걸음 크기(흔히 α = ε/num_iter 또는 α = 2.5·ε/num_iter)
- S = {δ : ||δ||_∞ ≤ ε}은 받아 주는 흔듦 모임
- T은 되돌이 횟수

FGSM과 다른 고갱이:
===========================
1. **여러 걸음**: PGD은 작은 걸음을 여러 번 밟는다(흔히 10~100)
2. **아무 첫자리**: ε 공 안의 아무 점에서 비롯한다
3. **되비추기**: 걸음마다 ε 공으로 되비춘다
4. **더 셈**: 맞서는 보기를 훨씬 잘 찾는다

아무 첫자리:
=====================
아무 첫자리는 PGD의 세기에 종요롭다:
- 그 자리의 나쁜 봉우리를 벗어나게 돕는다
- ε 공의 여러 자리를 둘러본다
- 치기가 첫 조건에 덜 예민해진다

흔한 꾀:
- 고르게: x^(0) ~ Uniform[x-ε, x+ε]
- 가우스: x^(0) ~ N(x, σ²I) 뒤 ε 공으로 되비춤

되비추는 셈:
===================
되비춤 Π은 흔듦이 ε 공 안에 머물게 한다:

L∞ 노름에서는:
    Π(z)_i = clip(z_i, x_i - ε, x_i + ε)

이는 [x-ε, x+ε] 상자로 낱낱이 잘라 내는 것이다.

L2 노름에서는:
    ||z - x||_2 ≤ ε이면: Π(z) = z
    아니면: Π(z) = x + ε · (z - x) / ||z - x||_2

걸음 크기 고르기:
===================
걸음 크기 α은 다음의 맞바꿈을 다룬다:
- α이 크면: 빨리 모이나 지나칠 수 있다
- α이 작으면: 더 꼭 집으나 되돌이가 더 든다

흔히 고르는 것:
- α = ε / T (걸음마다 온 예산의 1/T)
- α = 2.5 · ε / T (조금 더 세게)
- α = 2 · ε / T (매드리 등, 2018)

지은이: 가르침 감
날짜: 2025년 11월
어려움: 가운데 걸음
먼저 알 것: FGSM(62.1 묶음), 되돌이 다듬기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Literal
from tqdm import tqdm
import copy

# ========================================================================
# 메인
# ========================================================================


class PGD:
    """
    되비춘 기울기 내림(PGD) 치기
    
    PGD은 기울기 걸음을 여러 번 밟고 받아 주는 흔듦 모임으로 되비추는
    되돌이 맞서는 치기다. FGSM보다 훨씬 세며
    든든함을 따지는 사실상의 잣대다.
    
    수학 꼴:
    -------------------------
    PGD 치기는 옭아맨 다듬기 문제를 푼다:
    
        maximize L(θ, x + δ, y)  subject to ||δ||_p ≤ ε
    
    되비춘 기울기 내림을 써서:
    
        x^(t+1) = Π_{x+S}(x^(t) + α · sign(∇_x L(θ, x^(t), y)))
    
    여기서 Π은 ε 공으로 되비추는 셈이다.
    
    속성:
    -----------
    model : nn.Module
        칠 신경 그물
    epsilon : float
        가장 큰 흔듦의 크기
    alpha : float
        되돌 때마다의 걸음 크기
    num_iter : int
        돌릴 되돌이 횟수
    norm : str
        쓸 노름('linf' 또는 'l2')
    random_init : bool
        아무 첫자리를 쓸지
    loss_fn : nn.Module
        가장 크게 할 잃음 함수
    early_stop : bool
        치기가 먹히면 일찍 멈출지
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 40,
        norm: Literal['linf', 'l2'] = 'linf',
        random_init: bool = True,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        early_stop: bool = False
    ):
        """
        PGD 치기의 첫자리를 잡는다.
        
        매개변수:
        -----------
        model : nn.Module
            칠 신경 그물
        epsilon : float, 기본값=0.03
            가장 큰 흔듦의 크기(L∞ 또는 L2 노름)
            CIFAR-10에서는 ε=8/255≈0.031이 여느 값이다
        alpha : float, 기본값=0.01
            되돌 때마다의 걸음 크기
            흔히 고르는 값: α = 2.5 * ε / num_iter
            밝히지 않으면 기본값은 ε / 4이다
        num_iter : int, 기본값=40
            PGD 되돌이 횟수
            되돌이가 많을수록 치기는 세지고 느려진다
            흔한 값: 10(빠름), 40(여느 것), 100(셈)
        norm : str, 기본값='linf'
            쓸 노름: 'linf'(L∞) 또는 'l2'(L2)
            L∞: 낱그림점 모두가 ε으로 마디 지어진다
            L2: 온 흔듦이 ε으로 마디 지어진다
        random_init : bool, 기본값=True
            흔듦의 첫자리를 아무렇게나 잡을지
            True: 더 든든한 치기(즐겨 씀)
            False: 본디 그림에서 비롯한다(I-FGSM처럼)
        loss_fn : nn.Module, 골라 씀
            쓸 잃음 함수
        device : torch.device, 골라 씀
            셈할 장치
        clip_min : float, 기본값=0.0
            옳은 낱그림점의 가장 작은 값
        clip_max : float, 기본값=1.0
            옳은 낱그림점의 가장 큰 값
        early_stop : bool, 기본값=False
            치기가 먹히면 되돌이를 멈춘다
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon / 4.0
        self.num_iter = num_iter
        self.norm = norm
        self.random_init = random_init
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else next(model.parameters()).device
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.early_stop = early_stop
        
        # 모형을 따짐 모드로 둔다
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # 매개변수를 따진다
        if self.norm not in ['linf', 'l2']:
            raise ValueError(f"norm must be 'linf' or 'l2', got {self.norm}")
        
        # 차림을 찍는다
        print(f"PGD 치기 차림:")
        print(f"  엡실론 (ε): {self.epsilon}")
        print(f"  알파 (α): {self.alpha}")
        print(f"  되돌이: {self.num_iter}")
        print(f"  노름: L{self.norm}")
        print(f"  아무 첫자리: {self.random_init}")
        print(f"  일찍 멈추기: {self.early_stop}")
    
    def _initialize_perturbation(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        흔듦의 첫자리를 잡는다.
        
        두 가지 꾀가 있다:
        1. 아무 첫자리(즐겨 씀):
           - ε 공에서 고르게 뽑는다
           - 그 자리 봉우리를 벗어나게 돕는다
           - 더 든든한 치기
        
        2. 0에서 비롯하기:
           - 본디 그림에서 비롯한다
           - 되돌이 FGSM(I-FGSM)과 같다
           - 빠르나 더 여릴 수 있다
        
        수학의 자세한 것(L∞):
        -------------------------
        아무 첫자리: 차수마다 δ^(0) ~ Uniform[-ε, ε]
        그다음 되비춘다: x^(0) = clip(x + δ^(0), [clip_min, clip_max])
        
        매개변수:
        -----------
        images : torch.Tensor
            맑은 그림
        
        돌려주는 것:
        --------
        perturbed_images : torch.Tensor
            첫자리를 잡은 맞서는 그림
        """
        if self.random_init:
            # ε 공 안의 아무 첫자리
            if self.norm == 'linf':
                # [-ε, +ε]의 고른 아무 잡음
                # 이는 L∞ 공 전체를 고르게 둘러본다
                delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            else:  # l2
                # 아무 방향을 잡고 반지름 ≤ ε으로 아무렇게나 잣대를 잡는다
                delta = torch.randn_like(images)
                # 낱 공으로 잣대를 맞춘다
                delta_norm = delta.view(len(delta), -1).norm(p=2, dim=1)
                delta = delta / delta_norm.view(-1, 1, 1, 1)
                # [0, ε]의 아무 반지름으로 잣대를 잡는다
                random_radius = torch.rand(len(delta), device=images.device)
                random_radius = random_radius * self.epsilon
                delta = delta * random_radius.view(-1, 1, 1, 1)
            
            # 흔듦을 걸고 옳은 자리로 잘라 낸다
            perturbed_images = images + delta
            perturbed_images = torch.clamp(perturbed_images, self.clip_min, self.clip_max)
        else:
            # 맑은 그림에서 비롯한다(흔듦 없음)
            perturbed_images = images.clone()
        
        return perturbed_images
    
    def _project(
        self,
        perturbed_images: torch.Tensor,
        original_images: torch.Tensor
    ) -> torch.Tensor:
        """
        흔든 그림을 본디 그림 둘레 ε 공으로 되비춘다.
        
        이것이 흔듦을 마디 안에 붙잡아 두는 고갱이 셈이다.
        기울기 한 걸음을 밟으면 ||δ|| > ε이 될 수 있으므로
        될 수 있는 모임으로 되비춰야 한다.
        
        L∞ 되비춤:
        --------------
        낱그림점 i마다:
            δ_i = clip(δ_i, -ε, +ε)
        
        이는 [-ε, +ε] 상자로 낱낱이 잘라 내는 것이다.
        
        L2 되비춤:
        --------------
        ||δ||_2 ≤ ε이면 되비출 것이 없다
        아니면: δ = ε · δ / ||δ||_2
        
        이는 방향은 지키면서 δ의 노름을 꼭 ε으로 맞춘다.
        
        매개변수:
        -----------
        perturbed_images : torch.Tensor
            이제의 맞서는 그림
        original_images : torch.Tensor
            본디 맑은 그림
        
        돌려주는 것:
        --------
        projected_images : torch.Tensor
            ε 공으로 되비춘 그림
        """
        # 이제의 흔듦을 셈한다
        delta = perturbed_images - original_images
        
        if self.norm == 'linf':
            # L∞ 되비춤: 낱낱을 [-ε, +ε]으로 잘라 낸다
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        else:  # l2
            # L2 되비춤: 노름 ≤ ε이 되도록 잣대를 잡는다
            batch_size = len(delta)
            # 보기마다 L2 노름을 셈한다
            delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1)
            # ε을 넘는 보기의 가리개를 만든다
            exceed_mask = delta_norm > self.epsilon
            # ε을 넘는 흔듦을 줄인다
            if exceed_mask.any():
                scale = self.epsilon / delta_norm[exceed_mask]
                delta[exceed_mask] = delta[exceed_mask] * scale.view(-1, 1, 1, 1)
        
        # 되비춘 흔듦을 건다
        projected_images = original_images + delta
        
        # 옳은 낱그림점 자리 [clip_min, clip_max]으로도 잘라 낸다
        # 이로써 그림이 옳게 남는다([0, 1] 따위)
        projected_images = torch.clamp(projected_images, self.clip_min, self.clip_max)
        
        return projected_images
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        PGD으로 맞서는 보기를 만든다.
        
        이것이 PGD의 으뜸 알고리즘이다:
        
        알고리즘:
        ----------
        1. x^(0)의 첫자리를 잡는다(아무렇게나 또는 x에서)
        2. t = 0에서 T-1까지:
             a. 잃음 L(θ, x^(t), y)을 셈한다
             b. 기울기 g = ∇_x L을 셈한다
             c. 고친다: x^(t+1) = x^(t) + α · sign(g)  [L∞에서]
             d. 되비춘다: x^(t+1) = Π(x^(t+1))
             e. 옳은 자리로 잘라 낸다
        3. x^(T)을 돌려준다
        
        되비추는 걸음 (d)이 종요롭다. 흔듦을 ε 공 안에 붙잡아 둔다.
        
        매개변수:
        -----------
        images : torch.Tensor
            맑은 그림
        labels : torch.Tensor
            참 이름표(과녁 있는 치기에서는 과녁 이름표)
        targeted : bool, 기본값=False
            과녁 있는 치기를 할지
        target_labels : torch.Tensor, 골라 씀
            과녁 있는 치기의 과녁 이름표
        verbose : bool, 기본값=False
            되돌이가 나아가는 것을 찍는다
        
        돌려주는 것:
        --------
        adv_images : torch.Tensor
            PGD 되돌이를 마친 맞서는 그림
        """
        # 장치로 옮긴다
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # 맞서는 그림의 첫자리를 잡는다
        adv_images = self._initialize_perturbation(images)
        
        # 과녁 있는 치기에서
        if targeted and target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        # 되돌이 치기
        iterator = range(self.num_iter)
        if verbose:
            iterator = tqdm(iterator, desc="PGD iterations")
        
        for i in iterator:
            # 맞서는 그림에 기울기 좇기를 켠다
            adv_images = adv_images.detach().clone()
            adv_images.requires_grad = True
            
            # 앞으로 걸음
            outputs = self.model(adv_images)
            
            # 잃음을 셈한다
            if targeted:
                # 과녁 있는 치기: 과녁에 대한 잃음을 가장 작게
                loss = -self.loss_fn(outputs, target_labels)
            else:
                # 과녁 없는 치기: 참 이름표에 대한 잃음을 가장 크게
                loss = self.loss_fn(outputs, labels)
            
            # 되돌아 걸음
            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()
            loss.backward()
            
            # 기울기를 얻는다
            grad = adv_images.grad.data
            
            # 기울기 오름 걸음
            if self.norm == 'linf':
                # L∞: 기울기 부호 방향으로 걷는다
                # x^(t+1) = x^(t) + α · sign(∇L)
                adv_images = adv_images + self.alpha * torch.sign(grad)
            else:  # l2
                # L2: 잣대 맞춘 기울기 방향으로 걷는다
                # x^(t+1) = x^(t) + α · ∇L / ||∇L||_2
                grad_norm = grad.view(len(grad), -1).norm(p=2, dim=1)
                # 0으로 나누기를 비껴간다
                grad_norm = torch.clamp(grad_norm, min=1e-12)
                normalized_grad = grad / grad_norm.view(-1, 1, 1, 1)
                adv_images = adv_images + self.alpha * normalized_grad
            
            # 본디 그림 둘레 ε 공으로 되비춘다
            adv_images = self._project(adv_images, images)
            
            # 일찍 멈추기: 치기가 먹히면 더 갈 것 없다
            if self.early_stop:
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs, 1)
                    if targeted:
                        # 과녁 있음: 보기가 모두 과녁에 닿았는지 살핀다
                        if (predicted == target_labels).all():
                            if verbose:
                                print(f"\n{i+1}번째 되돌이에서 일찍 멈춘다: 치기가 모두 먹혔다")
                            break
                    else:
                        # 과녁 없음: 보기가 모두 틀리게 갈렸는지 살핀다
                        if (predicted != labels).all():
                            if verbose:
                                print(f"\n{i+1}번째 되돌이에서 일찍 멈춘다: 치기가 모두 먹혔다")
                            break
        
        return adv_images.detach()
    
    def generate_with_restarts(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        num_restarts: int = 5,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        아무렇게나 여러 번 다시 비롯하는 PGD.
        
        여러 번 다시 비롯하면 PGD이 더 세진다:
        1. ε 공 안의 여러 비롯 자리를 해 본다
        2. 돌린 것 가운데 가장 좋은 맞서는 보기를 고른다
        3. 첫자리에 덜 예민해진다
        
        알고리즘:
        ----------
        r = 1에서 num_restarts까지:
            아무 첫자리로 x_adv^(r) = PGD(x, y)
        잃음이 가장 큰(가장 잘 속인) x_adv을 돌려준다
        
        이것이 PGD의 가장 센 갈래이며 든든함을 따질 때
        즐겨 쓴다.
        
        매개변수:
        -----------
        images : torch.Tensor
            맑은 그림
        labels : torch.Tensor
            참 이름표
        num_restarts : int, 기본값=5
            아무렇게나 다시 비롯하는 횟수
            많을수록 세지고 느려진다
            흔한 값: 1(다시 비롯 없음), 5(여느 것), 10(셈)
        targeted : bool, 기본값=False
            과녁 있는 치기 표시
        target_labels : torch.Tensor, 골라 씀
            과녁 있는 치기의 과녁 이름표
        verbose : bool, 기본값=False
            나아가는 것을 찍는다
        
        돌려주는 것:
        --------
        best_adv_images : torch.Tensor
            다시 비롯한 것 가운데 잃음이 가장 큰 맞서는 보기
        """
        # 아무 첫자리가 켜져 있는지 확인한다
        original_random_init = self.random_init
        self.random_init = True
        
        best_adv_images = None
        best_loss = None
        
        for restart in range(num_restarts):
            if verbose:
                print(f"\n다시 비롯 {restart + 1}/{num_restarts}")
            
            # 이번 다시 비롯으로 맞서는 보기를 만든다
            adv_images = self.generate(
                images, labels, targeted, target_labels, verbose=False
            )
            
            # 이 맞서는 보기의 잃음을 셈한다
            with torch.no_grad():
                outputs = self.model(adv_images)
                if targeted:
                    loss = -self.loss_fn(outputs, target_labels).item()
                else:
                    loss = self.loss_fn(outputs, labels).item()
            
            # 가장 좋은 맞서는 보기를 남긴다(잃음이 가장 큰 것)
            if best_loss is None or loss > best_loss:
                best_loss = loss
                best_adv_images = adv_images.clone()
            
            if verbose:
                print(f"잃음: {loss:.4f} (가장 좋음: {best_loss:.4f})")
        
        # 본디 차림을 되돌린다
        self.random_init = original_random_init
        
        return best_adv_images
    
    def evaluate(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        adv_images: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        PGD 치기를 두루 따진다.
        
        치기가 잘 먹히는지 재는 여러 자를 셈한다:
        - 맑은 맞음
        - 맞섬 맞음
        - 치기 먹힘 비율
        - 흔듦의 자(L∞, L2, L1)
        
        매개변수:
        -----------
        clean_images : torch.Tensor
            본디 맑은 그림
        labels : torch.Tensor
            참 이름표
        adv_images : torch.Tensor
            맞서는 그림
        verbose : bool, 기본값=True
            결과를 찍는다
        
        돌려주는 것:
        --------
        metrics : Dict[str, float]
            따지는 자를 담은 사전
        """
        with torch.no_grad():
            # 맑은 맞음
            clean_outputs = self.model(clean_images.to(self.device))
            _, clean_pred = torch.max(clean_outputs, 1)
            clean_correct = (clean_pred == labels.to(self.device)).sum().item()
            clean_accuracy = clean_correct / len(labels)
            
            # 맞섬 맞음
            adv_outputs = self.model(adv_images.to(self.device))
            _, adv_pred = torch.max(adv_outputs, 1)
            adv_correct = (adv_pred == labels.to(self.device)).sum().item()
            adv_accuracy = adv_correct / len(labels)
            
            # 치기 먹힘 비율
            success_rate = 1.0 - adv_accuracy
            
            # 흔듦의 자
            perturbation = (adv_images - clean_images).cpu()
            
            # L∞ 노름(가장 큰 바뀜의 크기)
            linf_norm = torch.max(torch.abs(perturbation)).item()
            
            # L2 노름(유클리드 거리)
            l2_norms = torch.norm(perturbation.view(len(perturbation), -1), p=2, dim=1)
            l2_norm = l2_norms.mean().item()
            
            # L1 노름(크기의 합)
            l1_norms = torch.norm(perturbation.view(len(perturbation), -1), p=1, dim=1)
            l1_norm = l1_norms.mean().item()
        
        metrics = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': success_rate,
            'avg_linf_perturbation': linf_norm,
            'avg_l2_perturbation': l2_norm,
            'avg_l1_perturbation': l1_norm,
        }
        
        if verbose:
            print("=" * 60)
            print("PGD 치기 따짐 결과")
            print("=" * 60)
            print(f"차림:")
            print(f"  엡실론: {self.epsilon}")
            print(f"  알파: {self.alpha}")
            print(f"  되돌이: {self.num_iter}")
            print(f"  노름: L{self.norm}")
            print(f"\n결과:")
            print(f"  맑은 맞음: {clean_accuracy:.2%}")
            print(f"  맞섬 맞음: {adv_accuracy:.2%}")
            print(f"  치기 먹힘 비율: {success_rate:.2%}")
            print(f"\n흔듦의 자:")
            print(f"  가장 큰 L∞: {linf_norm:.6f}")
            print(f"  평균 L2: {l2_norm:.6f}")
            print(f"  평균 L1: {l1_norm:.6f}")
            print("=" * 60)
        
        return metrics


def compare_fgsm_pgd(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    num_iter_list: List[int] = [1, 10, 40, 100],
    device: Optional[torch.device] = None
) -> Dict[int, Dict[str, float]]:
    """
    되돌이 횟수를 달리한 PGD을 견준다(FGSM을 아울러).
    
    이 함수는 되돌이가 늘수록 PGD이 세지는 것을 보여 준다:
    - 되돌이 1 = FGSM(밑금)
    - 되돌이 10 = 빠른 PGD
    - 되돌이 40 = 여느 PGD
    - 되돌이 100 = 센 PGD
    
    매개변수:
    -----------
    model : nn.Module
        칠 모형
    images : torch.Tensor
        맑은 그림
    labels : torch.Tensor
        참 이름표
    epsilon : float, 기본값=0.03
        흔듦 예산
    num_iter_list : List[int], 기본값=[1, 10, 40, 100]
        해 볼 되돌이 횟수의 목록
    device : torch.device, 골라 씀
        셈할 장치
    
    돌려주는 것:
    --------
    results : Dict[int, Dict[str, float]]
        되돌이 횟수마다의 결과
    """
    if device is None:
        device = next(model.parameters()).device
    
    results = {}
    
    for num_iter in num_iter_list:
        print(f"\n{'='*60}")
        print(f"되돌이 {num_iter}번으로 PGD을 해 본다")
        print(f"{'='*60}")
        
        # PGD 치기를 만든다
        alpha = 2.5 * epsilon / num_iter if num_iter > 1 else epsilon
        attack = PGD(
            model=model,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            random_init=(num_iter > 1),  # FGSM doesn't use random init
            device=device
        )
        
        # 맞서는 보기를 만든다
        adv_images = attack.generate(images, labels)
        
        # 따진다
        metrics = attack.evaluate(clean_images=images, labels=labels, adv_images=adv_images)
        results[num_iter] = metrics
    
    # 견줌을 찍는다
    print(f"\n{'='*60}")
    print("PGD 되돌이 견주기")
    print(f"{'='*60}")
    print(f"{'되돌이':<12} {'먹힘 비율':<15} {'맞섬 맞음':<15}")
    print(f"{'-'*60}")
    for num_iter in num_iter_list:
        success = results[num_iter]['attack_success_rate']
        adv_acc = results[num_iter]['adversarial_accuracy']
        print(f"{num_iter:<12} {success:< 15.2%} {adv_acc:<15.2%}")
    print(f"{'='*60}")
    
    return results


# 쓰는 보기
if __name__ == "__main__":
    """
    PGD 치기를 보여 준다.
    
    이 보기는 다음을 보인다:
    1. 밑바탕 PGD 치기
    2. 여러 번 다시 비롯하는 PGD
    3. 되돌이 횟수를 달리한 견줌
    """
    print("=" * 70)
    print("PGD 치기 보여 주기")
    print("=" * 70)
    print("\n이 글은 센 되돌이 맞서는 치기인 되비춘 기울기 내림(PGD)")
    print("치기를 보여 준다.")
    print("\n붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.")
    print("=" * 70)```

## 2. 논의

잃음 셈하기는 모형의 날임을 다듬는 목표에 이어 준다. 알맞은 잃음 함수를 고르는 일이 종요로운 까닭은, 그것이 모형이 무엇을 다듬도록 배울지를 정하여 배운 드러냄과 판단의 금을 곧바로 빚기 때문이다.

그려 보기는 모형의 결을 알아보고 익힘의 탈을 짚어내는 데 큰 몫을 한다. 그리는 코드는 배운 드러냄, 모여 가는 결, 따지는 자를 들여다보게 하여 어림잡기 어려운 셈을 손에 잡히게 한다.

여기서 보인 결은 더 얽힌 자리로도 자연스레 넓어진다. 하이퍼파라미터, 얼개의 갈래, 다른 자료 꾸러미로 해 보면 앎이 깊어지고 모형 지킴 일에 손에 잡히는 느낌이 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 맞섬에 든든하기에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
PGD 치기 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_pgd():
        model = PGD(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — PGD 치기

잃음 셈하기는 모형의 날임을 다듬는 목표에 이어 준다.

고갱이 갈래는 `PGD`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
