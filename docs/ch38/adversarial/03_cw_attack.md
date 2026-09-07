# C&W 치기

62.3 묶음: 칼리니 & 와그너(C&W) 치기 - 가운데 걸음. 이 묶음은 가장 센 치기의 하나인 칼리니 & 와그너(C&W) 치기를 짜 놓았다

맞섬에 든든하기는 안전이 걸린 자리에 신경 그물을 내놓을 때 종요로운 걱정거리다. 이 짜보기는 작은 흔듦이 어떻게 모형을 속이는지, 막이는 어떻게 지을 수 있는지 보이며 맞섬에 든든하기의 깨침을 드러낸다.

## 코드

```python
"""
62.3 묶음: 칼리니 & 와그너(C&W) 치기 - 가운데 걸음

이 묶음은 가장 센 다듬기 바탕의 맞서는 치기 가운데 하나인 칼리니 & 와그너(C&W)
치기를 짜 놓았다. C&W은 맞서는 보기 만들기를 옭아맴 없는 다듬기 문제로
다시 적으며, FGSM이나 PGD 같은 기울기 바탕 방법보다
훨씬 세다.

수학 밑그림:
=======================

흔듦의 크기를 곧바로 옭아매는 FGSM/PGD과 달리, C&W은 공들여 꾸민
목표 함수를 지닌 옭아맴 없는 다듬기를 쓴다.

문제 꼴:
-------------------
다음을 푸는 δ을 찾는다:
    minimize ||δ||_p + c · f(x + δ)

여기서:
- ||δ||_p은 흔듦의 크기(p = 2, ∞, 0)
- c > 0은 맞바꿈 붙박이
- f(·)은 틀리게 가르게 이끄는 목표 함수

고갱이 새로움은 목표 함수 f(·)이다.

로짓에 기댄 목표:
=====================
C&W은 로짓(소프트맥스 앞의 날임)에 기댄 꾀바른 목표를 쓴다:

과녁 없는 치기에서:
    f(x') = max(max_{i≠t} Z(x')_i - Z(x')_t, -κ)

과녁 있는 치기(과녁 갈래 t')에서:
    f(x') = max(Z(x')_t - Z(x')_{t'}, -κ)

여기서:
- Z(x')은 로짓(소프트맥스 앞의 날임)
- t은 참 갈래
- t'은 과녁 갈래
- κ ≥ 0은 자신함 매개변수(흔히 0)

느낌으로 알기:
----------
1. 과녁 없음: max_{i≠t} Z_i > Z_t이길 바란다(아무 틀린 갈래가 참 갈래를 이긴다)
2. 과녁 있음: Z_{t'} > Z_t이길 바란다(과녁 갈래가 참 갈래를 이긴다)
3. max(..., -κ)은 다음을 지킨다:
   - 이미 틀리게 갈렸으면: f(x') ≤ 0(치기가 먹혔다)
   - 옳게 갈렸으면: f(x') > 0(더 다듬는다)
4. κ > 0은 자신함의 여유를 더한다. 과녁이 참 갈래를 κ만큼 이겨야 한다

변수 바꾸기:
===================
상자 옭아맴 x' ∈ [0, 1]을 다루려고 C&W은 변수를 바꾼다:

    x' = 0.5 * (tanh(w) + 1)

여기서 w은 다듬기 변수다. 이러면
- tanh(w) ∈ (-1, 1)
- x' ∈ (0, 1)이 절로 채워진다
- 드러난 옭아맴 없이 w을 다듬을 수 있다

그러면 흔듦은
    δ = x' - x = 0.5 * (tanh(w) + 1) - x

다듬기:
============
C&W은 Adam으로 다음을 푼다:
    minimize_{w} ||0.5*(tanh(w)+1) - x||_p + c · f(0.5*(tanh(w)+1))

c을 두 쪽 갈라 찾기:
-------------------
가장 좋은 c을 미리 알 수 없으므로 C&W은 두 쪽 갈라 찾는다:

1. 첫자리: c_low = 0, c_high = 큰 수
2. c마다:
   - 다듬기를 돌린다
   - 치기가 먹히면: c_high = c(더 작은 c을 해 본다)
   - 치기가 안 먹히면: c_low = c(더 큰 c을 해 본다)
3. 먹힌 것 가운데 가장 작은 c을 돌려준다

이러면 틀리게 가르게 하는 가장 작은 흔듦을 찾는다.

PGD과 다른 고갱이:
=========================
1. **다듬기 바탕**: 기울기 오름 대신 Adam을 쓴다
2. **옭아맴 없음**: 드러난 되비춤 없이 변수 바꾸기를 쓴다
3. **맞추어 감**: 두 쪽 갈라 찾기로 가장 좋은 맞바꿈 매개변수를 찾는다
4. **더 셈**: 흔히 PGD보다 작은 흔듦을 찾는다
5. **더 느림**: 두 쪽 갈라 찾으며 다듬기를 여러 번 돌려야 한다

노름:
======
C&W은 여러 노름을 가장 작게 할 수 있다:

L2(가장 흔함):
    ||δ||_2 = sqrt(Σ δ_i²)
    온 흔듦의 힘을 잰다

L∞:
    ||δ||_∞ = max_i |δ_i|
    낱그림점마다의 가장 큰 바뀜을 잰다

L0(띄엄함):
    ||δ||_0 = 바뀐 낱그림점의 수
    흔듦의 성김을 잰다

지은이: 가르침 감
날짜: 2025년 11월
어려움: 가운데에서 한발 더
먼저 알 것: PGD(62.2 묶음), 다듬기 이론, PyTorch 가장 좋게 하는 개
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, List, Dict, Literal
from tqdm import tqdm

# ========================================================================
# 메인
# ========================================================================


class CarliniWagnerL2:
    """
    칼리니 & 와그너(C&W) L2 치기
    
    이 갈래는 틀리게 가르게 하면서 흔듦의 L2 노름을 가장 작게 하는
    C&W 치기를 짜 놓았다. 알려진 가장 센 치기의 하나이며
    PGD보다 작은 흔듦을 찾는 일이 잦다.
    
    수학 꼴:
    -------------------------
    C&W L2 치기는 다음을 푼다:
    
        minimize ||δ||_2² + c · f(x + δ)
    
    여기서 f은 로짓에 기댄 목표다:
        f(x') = max(max_{i≠t} Z(x')_i - Z(x')_t, -κ)
    
    다듬기는 바꾼 변수 w에 대해 한다:
        x' = 0.5 * (tanh(w) + 1)
    
    가장 좋은 붙박이 c은 두 쪽 갈라 찾는다.
    
    속성:
    -----------
    model : nn.Module
        칠 신경 그물
    c : float
        맞바꿈 붙박이(두 쪽 갈라 찾는다)
    kappa : float
        자신함 매개변수
    learning_rate : float
        Adam의 배움 비율
    max_iter : int
        가장 많은 다듬기 되돌이
    binary_search_steps : int
        c을 두 쪽 갈라 찾는 걸음 수
    """
    
    def __init__(
        self,
        model: nn.Module,
        c: float = 1.0,
        kappa: float = 0.0,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        binary_search_steps: int = 9,
        initial_const: float = 1e-3,
        device: Optional[torch.device] = None,
        abort_early: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        """
        C&W L2 치기의 첫자리를 잡는다.
        
        매개변수:
        -----------
        model : nn.Module
            칠 신경 그물
        c : float, 기본값=1.0
            처음 맞바꿈 붙박이
            두 쪽 갈라 찾으며 손본다
        kappa : float, 기본값=0.0
            자신함 매개변수
            κ = 0: 틀리게 가르기만 한다
            κ > 0: 자신함의 여유 κ을 두고 틀리게 가른다
            κ이 클수록 치기는 세지고 흔듦은 커진다
        learning_rate : float, 기본값=0.01
            Adam의 배움 비율
            흔한 값: 0.01(기본), 0.001(조심), 0.1(세게)
        max_iter : int, 기본값=1000
            다듬기마다의 가장 많은 되돌이
            되돌이가 많을수록 치기는 세지고 느려진다
            흔한 값: 100(빠름), 1000(여느 것), 10000(꼼꼼함)
        binary_search_steps : int, 기본값=9
            c을 두 쪽 갈라 찾는 되돌이 횟수
            걸음이 많을수록 c은 좋아지고 느려진다
            걸음마다 온 다듬기 때가 곱절이 된다
        initial_const : float, 기본값=1e-3
            두 쪽 갈라 찾기에서 c의 첫 값
            가장 작은 흔듦을 찾으려 작게 비롯한다
        device : torch.device, 골라 씀
            셈할 장치
        abort_early : bool, 기본값=True
            치기가 먹히면 다듬기를 일찍 멈춘다
            때를 아끼나 더 작은 흔듦을 놓칠 수 있다
        clip_min : float, 기본값=0.0
            옳은 낱그림점의 가장 작은 값
        clip_max : float, 기본값=1.0
            옳은 낱그림점의 가장 큰 값
        """
        self.model = model
        self.c = c
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.device = device if device is not None else next(model.parameters()).device
        self.abort_early = abort_early
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # 모형을 따짐 모드로 둔다
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print(f"C&W L2 치기 차림:")
        print(f"  카파 (κ): {self.kappa}")
        print(f"  배움 비율: {self.learning_rate}")
        print(f"  가장 많은 되돌이: {self.max_iter}")
        print(f"  두 쪽 갈라 찾는 걸음: {self.binary_search_steps}")
        print(f"  일찍 그만두기: {self.abort_early}")
    
    def _arctanh(self, x: torch.Tensor) -> torch.Tensor:
        """
        arctanh(쌍곡 탄젠트의 거꿀)을 셈한다.
        
        이는 변수 바꾸기의 거꿀이다.
        x ∈ (0, 1)이 있을 때 다음을 채우는 w을 찾는다:
            x = 0.5 * (tanh(w) + 1)
        
        w에 대해 풀면:
            2x - 1 = tanh(w)
            w = arctanh(2x - 1)
        
        수학 붙임말:
            arctanh(z) = 0.5 * log((1+z)/(1-z))
        
        금 언저리에서 셈이 탈나지 않도록 작은 엡실론을 더한다.
        
        매개변수:
        -----------
        x : torch.Tensor
            (0, 1)의 값
        
        돌려주는 것:
        --------
        w : torch.Tensor
            거꿀로 바꾼 값
        """
        # log(0)을 비껴가려 (epsilon, 1-epsilon)으로 자른다
        epsilon = 1e-6
        x = torch.clamp(x, epsilon, 1.0 - epsilon)
        
        # 바꾼다: x ∈ (0,1) → z ∈ (-1,1)
        z = 2 * x - 1
        z = torch.clamp(z, -1 + epsilon, 1 - epsilon)
        
        # arctanh(z)을 셈한다
        w = 0.5 * torch.log((1 + z) / (1 - z))
        
        return w
    
    def _f_objective(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        C&W 목표 함수 f(x')을 셈한다.
        
        이것이 로짓(소프트맥스 앞의 점수)으로 틀리게 가르게 이끄는
        C&W의 꾀바른 대목이다.
        
        과녁 없는 치기에서:
        ----------------------
        가장 크게 하려는 것: Z_target - max_{i≠target} Z_i
        이는 다음을 가장 작게 하는 것과 같다:
            f = max(max_{i≠target} Z_i - Z_target, -κ)
        
        f ≤ 0이면 치기가 먹힌 것이다(자신함 κ으로).
        
        과녁 있는 치기에서:
        -------------------
        가장 크게 하려는 것: Z_target - Z_true
        이는 다음을 가장 작게 하는 것이다:
            f = max(Z_true - Z_target, -κ)
        
        매개변수:
        -----------
        outputs : torch.Tensor
            모형의 날임(낌새가 아니라 로짓)
        labels : torch.Tensor
            참 갈래 이름표
        targeted : bool
            과녁 있는 치기인지
        target_labels : torch.Tensor, 골라 씀
            과녁 갈래 이름표(과녁 있는 치기에)
        
        돌려주는 것:
        --------
        f_value : torch.Tensor
            목표 값(꼴: (batch_size,))
        """
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        
        if targeted:
            # 과녁 있는 치기: Z_true - Z_target을 가장 작게
            if target_labels is None:
                raise ValueError("target_labels required for targeted attack")
            
            # 참 갈래의 로짓을 얻는다
            true_logits = outputs[torch.arange(batch_size), labels]
            # 과녁 갈래의 로짓을 얻는다
            target_logits = outputs[torch.arange(batch_size), target_labels]
            
            # 목표: max(Z_true - Z_target, -κ)
            # Z_target > Z_true + κ이길 바란다
            f_value = torch.clamp(true_logits - target_logits, min=-self.kappa)
        else:
            # 과녁 없는 치기: max_{i≠t} Z_i - Z_t을 가장 작게
            
            # 참 갈래의 로짓을 얻는다
            true_logits = outputs[torch.arange(batch_size), labels]
            
            # 다른 갈래 모두의 로짓을 얻는다
            # 가리개를 만든다: 참 갈래는 1, 나머지는 0
            one_hot = F.one_hot(labels, num_classes).bool()
            
            # max에 뽑히지 않도록 참 갈래의 로짓을 -무한으로 둔다
            other_logits = outputs.clone()
            other_logits[one_hot] = float('-inf')
            
            # 틀린 갈래 가운데 가장 큰 로짓을 얻는다
            max_other_logits, _ = torch.max(other_logits, dim=1)
            
            # 목표: max(max_{i≠t} Z_i - Z_t, -κ)
            # Z_t < max_{i≠t} Z_i + κ이길 바란다
            f_value = torch.clamp(max_other_logits - true_logits, min=-self.kappa)
        
        return f_value
    
    def _optimize(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        c_value: float,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        주어진 c 값으로 다듬기를 한다.
        
        이것이 맞서는 보기를 찾는 고갱이 다듬기 돌기다.
        
        알고리즘:
        ----------
        1. 첫자리: w = arctanh(2x - 1)
        2. iter = 1에서 max_iter까지:
             a. x' = 0.5 * (tanh(w) + 1)  [바꿈을 건다]
             b. x'에 대한 모형 날임을 얻는다
             c. 잃음 = ||δ||_2² + c · f(x')을 셈한다
             d. 되돌아가며 Adam으로 w을 고친다
             e. 이제까지 찾은 가장 좋은 맞서는 보기를 좇는다
        3. 가장 좋은 맞서는 보기를 돌려준다
        
        매개변수:
        -----------
        images : torch.Tensor
            맑은 그림
        labels : torch.Tensor
            참 이름표
        c_value : float
            이번 다듬기의 이제 c 값
        targeted : bool
            과녁 있는 치기 표시
        target_labels : torch.Tensor, 골라 씀
            과녁 이름표
        verbose : bool
            나아가는 것을 찍는다
        
        돌려주는 것:
        --------
        best_adv : torch.Tensor
            찾은 가장 좋은 맞서는 보기
        best_l2 : torch.Tensor
            가장 좋은 흔듦의 L2 거리
        success : bool
            보기 모두에서 치기가 먹혔는지
        """
        batch_size = images.size(0)
        
        # 거꿀 바꿈으로 w의 첫자리를 잡는다
        # x = 0.5*(tanh(w) + 1)이 되도록 w = arctanh(2x - 1)
        w = self._arctanh(images)
        w = w.to(self.device).detach()
        w.requires_grad = True
        
        # w의 가장 좋게 하는 개를 마련한다
        optimizer = optim.Adam([w], lr=self.learning_rate)
        
        # 그림마다 가장 좋은 맞서는 보기를 좇는다
        best_adv = images.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        best_attack_success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # 다듬기 돌기
        iterator = range(self.max_iter)
        if verbose:
            iterator = tqdm(iterator, desc=f"C&W optimization (c={c_value:.2e})")
        
        for iteration in iterator:
            # 바꿈을 건다: x' = 0.5 * (tanh(w) + 1)
            # 이러면 x' ∈ (0, 1)이 절로 지켜진다
            adv_images = 0.5 * (torch.tanh(w) + 1)
            
            # 옳은 자리로 잘라 낸다(tanh 덕에 거의 군더더기다)
            adv_images = torch.clamp(adv_images, self.clip_min, self.clip_max)
            
            # 모형 날임을 얻는다
            outputs = self.model(adv_images)
            
            # 흔듦을 셈한다
            delta = adv_images - images
            
            # L2 잃음(L2 노름의 제곱)
            # ||δ||_2² = Σ δ²
            l2_loss = torch.sum(delta.view(batch_size, -1) ** 2, dim=1)
            
            # C&W 목표 f(x')
            f_loss = self._f_objective(outputs, labels, targeted, target_labels)
            
            # 온 잃음: ||δ||_2² + c · f(x')
            # c은 흔듦 크기와 치기 먹힘의 맞바꿈을 다룬다
            loss = torch.sum(l2_loss + c_value * f_loss)
            
            # 다듬기 한 걸음
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 어느 맞서는 보기가 먹혔는지 살핀다
            # 먹힘: f(x') ≤ 0(자신함 κ으로 틀리게 갈림)
            current_success = (f_loss <= 0)
            
            # 가장 좋은 맞서는 보기를 고친다
            # 먹힌 치기 가운데 L2 거리가 가장 작은 것을 남긴다
            improved_mask = current_success & (l2_loss < best_l2)
            
            if improved_mask.any():
                best_adv[improved_mask] = adv_images[improved_mask].detach()
                best_l2[improved_mask] = l2_loss[improved_mask].detach()
                best_attack_success[improved_mask] = True
            
            # 치기가 모두 먹혔고 더 작은 흔듦을 찾지 않는다면 일찍 그만둔다
            if self.abort_early and current_success.all():
                if verbose:
                    print(f"\n{iteration+1}번째 되돌이에서 일찍 그만둔다: 치기가 모두 먹혔다")
                break
        
        # 통틀어 먹혔는지 살핀다(보기가 모두 먹혔는지)
        success = best_attack_success.all().item()
        
        return best_adv, best_l2, success
    
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        두 쪽 갈라 찾기를 곁들인 C&W 치기로 맞서는 보기를 만든다.
        
        이 방법은 c을 두 쪽 갈라 찾는 온전한 C&W 치기를 짜 놓았다:
        
        알고리즘:
        ----------
        1. 첫자리: c_low = 0, c_high = 1e10
        2. b = 1에서 binary_search_steps까지:
             a. c = (c_low + c_high) / 2
             b. 이제의 c으로 다듬기를 돌린다
             c. 치기가 먹히면: c_high = c(더 작은 c을 해 본다)
             d. 치기가 안 먹히면: c_low = c(더 큰 c을 해 본다)
        3. 찾은 것 가운데 가장 좋은 맞서는 보기를 돌려준다
        
        두 쪽 갈라 찾기는 모형을 속이는 가장 작은 c(따라서 가장 작은 흔듦)을
        찾는다.
        
        매개변수:
        -----------
        images : torch.Tensor
            맑은 그림
        labels : torch.Tensor
            참 이름표
        targeted : bool, 기본값=False
            과녁 있는 치기 표시
        target_labels : torch.Tensor, 골라 씀
            과녁 있는 치기의 과녁 이름표
        verbose : bool, 기본값=False
            자세히 나아가는 것을 찍는다
        
        돌려주는 것:
        --------
        adv_images : torch.Tensor
            맞서는 그림
        """
        # 장치로 옮긴다
        images = images.to(self.device)
        labels = labels.to(self.device)
        if targeted and target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        batch_size = images.size(0)
        
        # c을 두 쪽 갈라 찾을 테두리를 잡는다
        # c_low: 치기가 안 먹히면 c이 모자랐다는 뜻
        # c_high: 치기가 먹히면 c이 너무 컸을 수 있다
        c_low = torch.zeros(batch_size, device=self.device)
        c_high = torch.full((batch_size,), 1e10, device=self.device)
        
        # 두 쪽 갈라 찾는 걸음에 걸쳐 가장 좋은 맞서는 보기를 좇는다
        best_adv = images.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        if verbose:
            print(f"\n걸음 {self.binary_search_steps}번으로 두 쪽 갈라 찾기를 비롯한다")
        
        # c을 두 쪽 갈라 찾는다
        for search_step in range(self.binary_search_steps):
            # 이제의 c 값: [c_low, c_high]의 가운데
            # 첫 되돌이에는 initial_const을 쓴다
            if search_step == 0:
                c_current = torch.full((batch_size,), self.initial_const, device=self.device)
            else:
                c_current = (c_low + c_high) / 2
            
            if verbose:
                c_min, c_max, c_mean = c_current.min().item(), c_current.max().item(), c_current.mean().item()
                print(f"\n[Step {search_step+1}/{self.binary_search_steps}] "
                      f"c range: [{c_min:.2e}, {c_max:.2e}], mean: {c_mean:.2e}")
            
            # 이제의 c으로 다듬기를 돌린다
            # 단순하게 c의 평균을 쓴다(보기마다 따로 다듬을 수도 있다)
            c_value = c_current.mean().item()
            adv_images, l2_dist, success = self._optimize(
                images, labels, c_value, targeted, target_labels, verbose=False
            )
            
            # 가장 좋은 맞서는 보기를 고친다
            improved_mask = l2_dist < best_l2
            if improved_mask.any():
                best_adv[improved_mask] = adv_images[improved_mask]
                best_l2[improved_mask] = l2_dist[improved_mask]
            
            # 두 쪽 갈라 찾기의 테두리를 고친다
            # 보기마다 치기가 먹혔는지 살핀다
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs, 1)
                
                if targeted:
                    # 과녁 있음: 미루어 봄 == 과녁이면 먹힌 것
                    success_mask = (predicted == target_labels)
                else:
                    # 과녁 없음: 미루어 봄 != 참 이름표이면 먹힌 것
                    success_mask = (predicted != labels)
            
            # 먹힘에 따라 테두리를 고친다
            # 먹혔으면: 더 작은 c을 해 본다(c_high = c)
            # 안 먹혔으면: 더 큰 c을 해 본다(c_low = c)
            c_high[success_mask] = c_current[success_mask]
            c_low[~success_mask] = c_current[~success_mask]
            
            if verbose:
                success_rate = success_mask.float().mean().item()
                avg_l2 = l2_dist[success_mask].mean().item() if success_mask.any() else float('inf')
                print(f"먹힘 비율: {success_rate:.2%}, 평균 L2: {avg_l2:.4f}")
        
        if verbose:
            print(f"\n두 쪽 갈라 찾기를 마쳤다.")
            final_success = (best_l2 < float('inf')).float().mean().item()
            final_l2 = best_l2[best_l2 < float('inf')].mean().item() if (best_l2 < float('inf')).any() else float('inf')
            print(f"마지막 먹힘 비율: {final_success:.2%}")
            print(f"마지막 평균 L2: {final_l2:.4f}")
        
        return best_adv
    
    def evaluate(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        adv_images: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        C&W 치기가 잘 먹히는지 따진다.
        
        매개변수:
        -----------
        clean_images : torch.Tensor
            본디 그림
        labels : torch.Tensor
            참 이름표
        adv_images : torch.Tensor
            맞서는 그림
        verbose : bool
            결과를 찍는다
        
        돌려주는 것:
        --------
        metrics : Dict[str, float]
            따지는 자
        """
        with torch.no_grad():
            # 맑은 맞음
            clean_outputs = self.model(clean_images.to(self.device))
            _, clean_pred = torch.max(clean_outputs, 1)
            clean_accuracy = (clean_pred == labels.to(self.device)).float().mean().item()
            
            # 맞섬 맞음
            adv_outputs = self.model(adv_images.to(self.device))
            _, adv_pred = torch.max(adv_outputs, 1)
            adv_accuracy = (adv_pred == labels.to(self.device)).float().mean().item()
            
            # 흔듦의 자
            perturbation = (adv_images - clean_images).cpu()
            
            # L2 노름(보기마다 셈한 뒤 고르게 함)
            l2_norms = torch.norm(perturbation.view(len(perturbation), -1), p=2, dim=1)
            l2_mean = l2_norms.mean().item()
            l2_median = l2_norms.median().item()
            
            # L∞ 노름
            linf_norm = torch.max(torch.abs(perturbation)).item()
        
        metrics = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': 1.0 - adv_accuracy,
            'avg_l2_perturbation': l2_mean,
            'median_l2_perturbation': l2_median,
            'max_linf_perturbation': linf_norm,
        }
        
        if verbose:
            print("=" * 60)
            print("C&W L2 치기 따짐")
            print("=" * 60)
            print(f"맑은 맞음: {clean_accuracy:.2%}")
            print(f"맞섬 맞음: {adv_accuracy:.2%}")
            print(f"치기 먹힘 비율: {metrics['attack_success_rate']:.2%}")
            print(f"\n흔듦의 자:")
            print(f"  평균 L2: {l2_mean:.4f}")
            print(f"  가운뎃값 L2: {l2_median:.4f}")
            print(f"  가장 큰 L∞: {linf_norm:.6f}")
            print("=" * 60)
        
        return metrics


# 쓰는 보기
if __name__ == "__main__":
    """
    C&W 치기를 보여 준다.
    """
    print("=" * 70)
    print("칼리니 & 와그너 L2 치기 보여 주기")
    print("=" * 70)
    print("\n이 글은 센 다듬기 바탕의 맞서는 치기인")
    print("C&W 치기를 보여 준다.")
    print("\n붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.")
    print("=" * 70)```

## 논의

익힘 돌기는 여느 PyTorch 결을 따른다. 앞으로 걸어 미루어 봄을 셈하고, 잃음을 재고, 되돌아 걸어 기울기를 셈하고, 가장 좋게 하는 개로 매개변수를 고친다. 판에 걸쳐 자를 좇으면 모여 가는 결이 드러나고 덜 맞추기나 지나치게 맞추기 같은 탈을 짚어내는 데 도움이 된다.

여기서 보인 결은 더 얽힌 자리로도 자연스레 넓어진다. 하이퍼파라미터, 얼개의 갈래, 다른 자료 꾸러미로 해 보면 앎이 깊어지고 모형 지킴 일에 손에 잡히는 느낌이 붙는다.

## 익힘 문제

**익힘 1.**
익힘 돌기에서 `optimizer.zero_grad()` 부름을 없애면 어떻게 되는지 밝혀라. 고친 코드를 돌려 익힘 잃음이 모이는 데 어떤 일이 생기는지 적어라.

??? success "익힘 1 풀이"
    `optimizer.zero_grad()`이 없으면 PyTorch이 새 기울기를 이미 있는 `.grad` 텐서에 갈음하지 않고 더하므로 기울기가 되돌이마다 쌓인다. 이는 사실상 배움 비율에 쌓인 걸음 수를 곱하는 셈이라 다듬기가 점점 크고 들쭉날쭉한 걸음을 밟게 된다. 익힘 잃음은 매끄럽게 모이는 대신 크게 출렁이거나 터진다. 고치기는 쉽다. `loss.backward()`을 부르기 앞서 늘 기울기를 0으로 만든다.

---

**익힘 2.**
가장 좋게 하는 개를 Adam(`torch.optim.Adam`, `lr=0.001`)으로 갈음하고 본디 것과 익힘이 모이는 결을 견주어라. 두 잃음 굽이를 한 그림에 그려라.

??? success "익힘 2 풀이"
    가장 좋게 하는 개를 짓는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 갈음한다. Adam은 매개변수마다 맞추어 가는 배움 비율과 밀어 나감 어림을 지니므로 이른 판에서 더 빨리 모이는 것이 보통이다. Adam의 잃음 굽이는 첫 몇 판에서 더 가파르게 떨어지지만, 가장 좋은 자리 언저리에서는 밀어 나감을 곁들인 SGD보다 조금 더 출렁일 수 있다. 고르게 견주려면 아무렇게나 하는 씨앗과 판 수를 똑같이 두고 돌린다.

---

**익힘 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "익힘 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**익힘 4.**
일찍 멈추기를 짜 넣어라. 판마다 따짐 잃음을 좇다가 열 판 잇달아 나아지지 않으면 익힘을 멈춘다. 가장 좋은 모형 짐을 담아 두었다가 되돌려라.

??? success "익힘 4 풀이"
    참을 수 세개와 가장 좋은 잃음 좇개를 더한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 익힘 걸음 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'{epoch}판에서 일찍 멈춘다')
            model.load_state_dict(best_state)
            break
    ```
    이러면 남겨 둔 자료에서 모형이 더 나아지지 않을 때 멈추므로 지나치게 맞추기를 막는다.
