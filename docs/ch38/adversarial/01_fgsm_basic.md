# FGSM 밑바탕

62.1 묶음: 빠른 기울기 부호 방법(FGSM) - 첫걸음. 이 묶음은 가장 단순한 치기 가운데 하나인 빠른 기울기 부호 방법(FGSM)을 짜 놓았다

맞섬에 든든하기는 안전이 걸린 자리에 신경 그물을 내놓을 때 종요로운 걱정거리다. 이 짜보기는 작은 흔듦이 어떻게 모형을 속이는지, 막이는 어떻게 지을 수 있는지 보이며 맞섬에 든든하기의 깨침을 드러낸다.

## 코드

```python
"""
62.1 묶음: 빠른 기울기 부호 방법(FGSM) - 첫걸음

이 묶음은 가장 단순하고 널리 힘을 미친 맞서는 치기 방법 가운데 하나인
빠른 기울기 부호 방법(FGSM)을 짜 놓았다. FGSM은 신경 그물이 작고 공들여
지은 흔듦에 무르다는 것을 보여 준다.

수학 밑그림:
=======================

FGSM 치기는 이제의 들임 언저리에서 잃음 함수를 곧게 편 데 기댄다.
매개변수가 θ인 신경 그물 f, 들임 x, 참 이름표 y이 있을 때,
잃음을 가장 크게 하는 흔듦 δ을 찾으려 한다:

    maximize L(θ, x + δ, y)  subject to ||δ||_∞ ≤ ε

FGSM은 잃음의 일차 테일러 어림을 쓴다:
    
    L(θ, x + δ, y) ≈ L(θ, x, y) + δ^T ∇_x L(θ, x, y)

이를 ||δ||_∞ ≤ ε 아래 가장 크게 하려면 이렇게 한다:
    
    δ = ε · sign(∇_x L(θ, x, y))

그러면 맞서는 보기가 나온다:
    
    x_adv = x + ε · sign(∇_x L(θ, x, y))

느낌으로 알기:
==========
- 기울기 ∇_x L은 잃음을 키우려면 x을 어떻게 바꿀지 알려 준다
- 부호를 취하면 방향(+ 또는 -)만 남는다
- ε을 곱하면 차수마다 받아 주는 가장 큰 흔듦이 된다
- 이는 "가장 가파른 오름" 방향으로 한 걸음 밟는 치기다

고갱이 결:
===============
1. 셈이 잘 든다: O(1) - 기울기 셈 한 번
2. 한 걸음에 맞서는 보기를 만든다
3. 많은 모형을 속이기에 흔히 넉넉하다
4. 더 센 치기의 비롯 자리가 된다

지은이: 가르침 감
날짜: 2025년 11월
어려움: 첫걸음
먼저 알 것: PyTorch 밑바탕, 되돌아가기, CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm

# ========================================================================
# 메인
# ========================================================================


class FGSM:
    """
    빠른 기울기 부호 방법(FGSM) 치기
    
    이 갈래는 잃음을 가장 크게 하는 방향으로 기울기 한 걸음을 밟아
    맞서는 보기를 만드는 FGSM 치기를 짜 놓았다.
    
    수학 꼴:
    -------------------------
    주어진 것:
        - 매개변수가 θ인 모형 f(x; θ)
        - 참 이름표가 y인 들임 x
        - 잃음 함수 L(θ, x, y)
        - 흔듦 예산 ε
    
    FGSM 치기는 다음을 셈한다:
        x_adv = x + ε · sign(∇_x L(θ, x, y))
    
    여기서:
        - ∇_x L은 들임에 대한 잃음의 기울기
        - sign(·)은 낱낱에 +1, 0, -1을 돌려준다
        - ε은 흔듦의 크기를 다스린다
    
    속성:
    -----------
    model : nn.Module
        칠 신경 그물
    epsilon : float
        가장 큰 흔듦의 크기(L∞ 노름)
    loss_fn : nn.Module
        가장 크게 할 잃음 함수(기본값: CrossEntropyLoss)
    device : torch.device
        셈할 장치(CPU 또는 GPU)
    clip_min : float
        옳은 낱그림점의 가장 작은 값(기본값: 0.0)
    clip_max : float
        옳은 낱그림점의 가장 큰 값(기본값: 1.0)
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.3,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        """
        FGSM 치기의 첫자리를 잡는다.
        
        매개변수:
        -----------
        model : nn.Module
            칠 신경 그물. 따짐 모드여야 한다.
        epsilon : float, 기본값=0.3
            가장 큰 L∞ 흔듦. [0,1]으로 잣대 맞춘 그림에서
            ε=0.3이면 낱그림점마다 많아야 0.3만큼 바뀐다.
            흔한 값: 0.03(은근함), 0.1(가운데), 0.3(셈)
        loss_fn : nn.Module, 골라 씀
            쓸 잃음 함수. None이면 CrossEntropyLoss을 쓴다.
        device : torch.device, 골라 씀
            셈할 장치. None이면 모형의 장치를 쓴다.
        clip_min : float, 기본값=0.0
            날임의 가장 작은 값(그림에서는 흔히 0)
        clip_max : float, 기본값=1.0
            날임의 가장 큰 값(잣대 맞춘 그림에서는 흔히 1)
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else next(model.parameters()).device
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # 모형을 따짐 모드로 둔다(묶음 잣대 잡기와 드롭아웃에 종요롭다)
        self.model.eval()
        
        # 있어야 하면 모형을 장치로 옮긴다
        self.model = self.model.to(self.device)
        
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        targeted: bool = False,
        target_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FGSM으로 맞서는 보기를 만든다.
        
        이 방법은 FGSM의 고갱이 알고리즘을 짜 놓았다:
        1. 본디 들임의 잃음을 셈한다
        2. 들임에 대한 잃음의 기울기를 셈한다
        3. 흔듦을 만든다: δ = ε · sign(∇_x L)
        4. 맞서는 보기를 만든다: x_adv = x + δ
        5. 옳은 자리 [clip_min, clip_max]으로 잘라 낸다
        
        매개변수:
        -----------
        images : torch.Tensor
            꼴이 (batch_size, channels, height, width)인 맑은 그림
            [clip_min, clip_max] 자리에 있어야 한다
        labels : torch.Tensor
            꼴이 (batch_size,)인 참 이름표
        targeted : bool, 기본값=False
            True이면 target_labels으로 갈리는 보기를 만든다
            False이면 틀리게 갈리는 보기를 만든다(아무 틀린 갈래)
        target_labels : torch.Tensor, 골라 씀
            과녁 있는 치기의 과녁 이름표. targeted=True이면 있어야 한다.
        
        돌려주는 것:
        --------
        adv_images : torch.Tensor
            들임과 꼴이 같은 맞서는 그림
            ||x_adv - x||_∞ ≤ ε을 반드시 채운다
        
        수학의 자세한 것:
        ---------------------
        과녁 없는 치기(targeted=False)에서는
            잃음을 가장 크게 한다: δ = ε · sign(∇_x L(f(x), y))
            이는 미루어 봄을 참 이름표에서 밀어낸다
        
        과녁 있는 치기(targeted=True)에서는
            잃음을 가장 작게 한다: δ = -ε · sign(∇_x L(f(x), y_target))
            이는 미루어 봄을 과녁 이름표 쪽으로 끌어당긴다
        """
        # 들임을 장치로 옮긴다
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # 흔들 베낌을 만든다(셈 그림에서 떼어 낸다)
        # 모형 매개변수가 아니라 그림에 대한 기울기를 좇아야 한다
        images_adv = images.clone().detach()
        
        # 그림에 기울기 셈을 켠다
        # 이것이 종요롭다: ∇_θ L이 아니라 ∇_x L이 있어야 한다
        images_adv.requires_grad = True
        
        # 앞으로 걸음: 모형의 미루어 봄을 셈한다
        outputs = self.model(images_adv)
        
        # 잃음을 셈한다
        # 과녁 있는 치기에는 target_labels을, 없는 치기에는 참 이름표를 쓴다
        if targeted:
            if target_labels is None:
                raise ValueError("target_labels must be provided for targeted attack")
            target_labels = target_labels.to(self.device)
            loss = self.loss_fn(outputs, target_labels)
        else:
            loss = self.loss_fn(outputs, labels)
        
        # 되돌아 걸음: 들임 그림에 대한 잃음의 기울기를 셈한다
        # 이로써 ∇_x L(θ, x, y)을 얻는다
        self.model.zero_grad()  # Clear any existing gradients
        if images_adv.grad is not None:
            images_adv.grad.zero_()
        loss.backward()
        
        # 기울기를 뽑아낸다
        # images_adv.grad의 꼴은 (batch_size, channels, height, width)이다
        grad = images_adv.grad.data
        
        # 기울기의 부호로 흔듦을 만든다
        # 과녁 없는 치기: δ = ε · sign(∇_x L)
        # 과녁 있는 치기: δ = -ε · sign(∇_x L)(과녁 쪽으로 잃음을 가장 작게)
        if targeted:
            perturbation = -self.epsilon * torch.sign(grad)
        else:
            perturbation = self.epsilon * torch.sign(grad)
        
        # 맞서는 보기를 만든다
        # x_adv = x + δ
        images_adv = images + perturbation
        
        # 맞서는 보기가 옳은 그림이 되도록 옳은 자리로 잘라 낸다
        # 이것이 종요롭다: x_adv ∈ [clip_min, clip_max]이어야 한다
        images_adv = torch.clamp(images_adv, self.clip_min, self.clip_max)
        
        # 셈 그림에서 떼어 낸다(이제 기울기가 있어야 하지 않다)
        images_adv = images_adv.detach()
        
        return images_adv
    
    def generate_with_budget_search(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon_values: List[float],
        target_success_rate: float = 0.9
    ) -> Tuple[torch.Tensor, float]:
        """
        과녁으로 삼은 먹힘 비율을 이루는 가장 작은 ε을 찾는다.
        
        이 방법은 엡실론 값을 여럿 뒤져 바라는 먹힘 비율에 드는
        가장 작은 흔듦을 찾는다.
        
        알고리즘:
        ----------
        1. 엡실론 값을 작은 것부터 차례로 해 본다
        2. 엡실론마다 맞서는 보기를 만든다
        3. 치기가 먹힌 비율을 셈한다
        4. target_success_rate을 이루는 첫 엡실론을 돌려준다
        
        모형이 얼마나 든든한지 알아보는 데 쓸모 있다. 엡실론이 작을수록
        모형이 더 무르다는 뜻이다.
        
        매개변수:
        -----------
        images : torch.Tensor
            맑은 그림
        labels : torch.Tensor
            참 이름표
        epsilon_values : List[float]
            해 볼 엡실론 값의 목록(작은 것부터 줄 세워야 한다)
        target_success_rate : float, 기본값=0.9
            바라는 치기 먹힘 비율(0에서 1)
        
        돌려주는 것:
        --------
        best_adv_images : torch.Tensor
            먹힌 것 가운데 엡실론이 가장 작은 맞서는 그림
        best_epsilon : float
            과녁 먹힘 비율을 이루는 가장 작은 엡실론
        """
        best_epsilon = None
        best_adv_images = None
        
        for eps in epsilon_values:
            # 엡실론을 잠깐 바꾼다
            original_epsilon = self.epsilon
            self.epsilon = eps
            
            # 맞서는 보기를 만든다
            adv_images = self.generate(images, labels)
            
            # 먹힘 비율을 따진다
            success_rate = self.compute_success_rate(images, labels, adv_images)
            
            # 본디 엡실론을 되돌린다
            self.epsilon = original_epsilon
            
            # 과녁 먹힘 비율에 들었는지 살핀다
            if success_rate >= target_success_rate:
                best_epsilon = eps
                best_adv_images = adv_images
                break
        
        if best_epsilon is None:
            print(f"알림: 먹힘 비율 {target_success_rate}에 이르지 못했다")
            print(f"가장 큰 엡실론을 쓴다: {epsilon_values[-1]}")
            self.epsilon = epsilon_values[-1]
            best_adv_images = self.generate(images, labels)
            best_epsilon = epsilon_values[-1]
            self.epsilon = original_epsilon
        
        return best_adv_images, best_epsilon
    
    def compute_success_rate(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        adv_images: torch.Tensor
    ) -> float:
        """
        치기가 먹힌 비율을 셈한다.
        
        먹힘 비율은 맞서는 보기 가운데 틀리게 갈린(과녁 없는 치기) 또는
        과녁으로 갈린(과녁 있는 치기) 것의
        몫이다.
        
        수학의 뜻매김:
        ------------------------
        먹힘 비율 = (1/n) * Σ I[f(x_adv) ≠ y]
        
        여기서:
        - n은 보기의 수
        - I[·]은 알림 함수(참이면 1, 거짓이면 0)
        - f(x_adv)은 맞서는 보기에 대한 모형의 미루어 봄
        - y은 참 이름표
        
        매개변수:
        -----------
        clean_images : torch.Tensor
            본디 맑은 그림(쓰지 않으나 API을 한결같게 하려고 둔다)
        labels : torch.Tensor
            참 이름표
        adv_images : torch.Tensor
            맞서는 그림
        
        돌려주는 것:
        --------
        success_rate : float
            먹힌 치기의 몫(0에서 1)
        """
        with torch.no_grad():  # No gradients needed for evaluation
            # 맞서는 보기의 미루어 봄을 얻는다
            outputs = self.model(adv_images.to(self.device))
            _, predicted = torch.max(outputs, 1)
            
            # 틀리게 갈린 것을 센다
            # 미루어 봄 != 참 이름표이면 먹힌 것이다
            successful_attacks = (predicted != labels.to(self.device)).sum().item()
            
            # 먹힘 비율을 셈한다
            success_rate = successful_attacks / len(labels)
        
        return success_rate
    
    def evaluate(
        self,
        clean_images: torch.Tensor,
        labels: torch.Tensor,
        adv_images: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        치기가 잘 먹히는지 두루 따진다.
        
        이 방법은 치기를 재는 여러 자를 셈한다:
        1. 맑은 맞음: 본디 그림의 맞음
        2. 맞섬 맞음: 흔든 그림의 맞음
        3. 치기 먹힘 비율: 먹힌 치기의 몫
        4. 평균 L∞ 흔듦: 가장 큰 바뀜의 크기
        5. 평균 L2 흔듦: 유클리드 거리
        
        매개변수:
        -----------
        clean_images : torch.Tensor
            본디 맑은 그림
        labels : torch.Tensor
            참 이름표
        adv_images : torch.Tensor
            맞서는 그림
        verbose : bool, 기본값=True
            True이면 결과를 찍는다
        
        돌려주는 것:
        --------
        metrics : Dict[str, float]
            셈한 자를 모두 담은 사전
        """
        with torch.no_grad():
            # 맑은 그림으로 따진다
            clean_outputs = self.model(clean_images.to(self.device))
            _, clean_pred = torch.max(clean_outputs, 1)
            clean_correct = (clean_pred == labels.to(self.device)).sum().item()
            clean_accuracy = clean_correct / len(labels)
            
            # 맞서는 그림으로 따진다
            adv_outputs = self.model(adv_images.to(self.device))
            _, adv_pred = torch.max(adv_outputs, 1)
            adv_correct = (adv_pred == labels.to(self.device)).sum().item()
            adv_accuracy = adv_correct / len(labels)
            
            # 치기 먹힘 비율을 셈한다
            success_rate = 1.0 - adv_accuracy
            
            # 흔듦의 자를 셈한다
            perturbation = (adv_images - clean_images).cpu()
            
            # L∞ 노름: 낱그림점 모두에 걸친 가장 큰 바뀜의 크기
            linf_norm = torch.max(torch.abs(perturbation)).item()
            
            # L2 노름: 유클리드 거리
            # 보기마다 L2 노름을 셈한 뒤 고르게 한다
            l2_norms = torch.norm(perturbation.view(len(perturbation), -1), p=2, dim=1)
            l2_norm = l2_norms.mean().item()
            
            # L0 노름: 바뀐 낱그림점의 수(엄밀하지는 않다)
            # |바뀜| > 문턱인 낱그림점을 센다
            threshold = 1e-5
            l0_norm = (torch.abs(perturbation) > threshold).sum().item() / len(labels)
        
        # 자를 모은다
        metrics = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': success_rate,
            'avg_linf_perturbation': linf_norm,
            'avg_l2_perturbation': l2_norm,
            'avg_l0_perturbation': l0_norm,
        }
        
        if verbose:
            print("=" * 60)
            print("FGSM 치기 따짐 결과")
            print("=" * 60)
            print(f"엡실론 (ε): {self.epsilon}")
            print(f"맑은 맞음: {clean_accuracy:.2%}")
            print(f"맞섬 맞음: {adv_accuracy:.2%}")
            print(f"치기 먹힘 비율: {success_rate:.2%}")
            print(f"평균 L∞ 흔듦: {linf_norm:.6f}")
            print(f"평균 L2 흔듦: {l2_norm:.6f}")
            print(f"평균 L0 흔듦: 낱그림점 {l0_norm:.2f}개")
            print("=" * 60)
        
        return metrics


def visualize_attack(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: Optional[List[str]] = None,
    num_examples: int = 5,
    epsilon: float = 0.3,
    save_path: Optional[str] = None
):
    """
    맞서는 보기와 흔듦을 그린다.
    
    이 함수는 다음을 보이는 그림을 만든다:
    - 참 이름표를 붙인 본디 맑은 그림
    - 미루어 본 이름표를 붙인 맞서는 그림
    - 키운 흔듦(눈에 띄게 부풀림)
    
    이 그림은 FGSM이 그림에 무엇을 하는지 알아보는 데 도움이 된다.
    
    매개변수:
    -----------
    clean_images : torch.Tensor
        꼴이 (batch_size, C, H, W)인 본디 그림
    adv_images : torch.Tensor
        맞서는 그림
    labels : torch.Tensor
        참 이름표
    predictions : torch.Tensor
        맞서는 그림에 대한 모형의 미루어 봄
    class_names : List[str], 골라 씀
        이름표에 쓸 갈래 이름
    num_examples : int, 기본값=5
        그릴 보기의 수
    epsilon : float, 기본값=0.3
        쓴 엡실론 값(그림 이름에 쓴다)
    save_path : str, 골라 씀
        주어지면 이 자리에 그림을 담는다
    """
    # matplotlib에 쓰려고 텐서를 넘파이로 옮긴다
    clean_np = clean_images[:num_examples].cpu().numpy()
    adv_np = adv_images[:num_examples].cpu().numpy()
    labels_np = labels[:num_examples].cpu().numpy()
    pred_np = predictions[:num_examples].cpu().numpy()
    
    # 흔듦을 셈한다
    perturbations = adv_np - clean_np
    
    # 3줄짜리 그림을 만든다: 맑음, 맞섬, 흔듦(키움)
    fig, axes = plt.subplots(3, num_examples, figsize=(3*num_examples, 9))
    
    if num_examples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_examples):
        # matplotlib에 맞게 (C, H, W)을 (H, W, C)으로 옮긴다
        # 잿빛과 RGB을 함께 다룬다
        if clean_np.shape[1] == 1:  # Grayscale
            clean_img = clean_np[i, 0]
            adv_img = adv_np[i, 0]
            pert_img = perturbations[i, 0]
            cmap = 'gray'
        else:  # RGB
            clean_img = np.transpose(clean_np[i], (1, 2, 0))
            adv_img = np.transpose(adv_np[i], (1, 2, 0))
            pert_img = np.transpose(perturbations[i], (1, 2, 0))
            cmap = None
        
        # 1줄: 맑은 그림
        axes[0, i].imshow(clean_img, cmap=cmap)
        true_label = class_names[labels_np[i]] if class_names else labels_np[i]
        axes[0, i].set_title(f'Clean\nTrue: {true_label}', fontsize=10)
        axes[0, i].axis('off')
        
        # 2줄: 맞서는 그림
        axes[1, i].imshow(adv_img, cmap=cmap)
        pred_label = class_names[pred_np[i]] if class_names else pred_np[i]
        color = 'red' if pred_np[i] != labels_np[i] else 'green'
        axes[1, i].set_title(f'Adversarial\nPred: {pred_label}', 
                            fontsize=10, color=color)
        axes[1, i].axis('off')
        
        # 3줄: 흔듦(눈에 띄게 10배로 키움)
        # 더 잘 보이도록 흔듦의 잣대를 맞춘다
        pert_magnified = pert_img * 10
        pert_magnified = np.clip(pert_magnified + 0.5, 0, 1)  # Center around 0.5
        
        axes[2, i].imshow(pert_magnified, cmap='RdBu_r')  # Red-Blue colormap
        axes[2, i].set_title(f'Perturbation\n(10× magnified)', fontsize=10)
        axes[2, i].axis('off')
    
    plt.suptitle(f'FGSM Attack Visualization (ε = {epsilon})', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그림을 {save_path}에 담았다")
    
    plt.show()


# 쓰는 보기와 보여 주기
if __name__ == "__main__":
    """
    CIFAR-10 자료 꾸러미에서 FGSM 치기를 보여 준다.
    
    이 보기는 다음을 보인다:
    1. 미리 익힌 모형 얹기
    2. FGSM 치기 만들기
    3. 맞서는 보기 만들기
    4. 치기가 잘 먹히는지 따지기
    5. 결과 그리기
    """
    print("=" * 70)
    print("FGSM 치기 보여 주기")
    print("=" * 70)
    print("\n이 글은 미리 익힌 ResNet-18 모형으로 CIFAR-10 자료 꾸러미에서")
    print("빠른 기울기 부호 방법(FGSM) 치기를 보여 준다.")
    print("\n붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.")
    print("=" * 70)```

## 논의

잃음 셈하기는 모형의 날임을 다듬는 목표에 이어 준다. 알맞은 잃음 함수를 고르는 일이 종요로운 까닭은, 그것이 모형이 무엇을 다듬도록 배울지를 정하여 배운 드러냄과 판단의 금을 곧바로 빚기 때문이다.

그려 보기는 모형의 결을 알아보고 익힘의 탈을 짚어내는 데 큰 몫을 한다. 그리는 코드는 배운 드러냄, 모여 가는 결, 따지는 자를 들여다보게 하여 어림잡기 어려운 셈을 손에 잡히게 한다.

여기서 보인 결은 더 얽힌 자리로도 자연스레 넓어진다. 하이퍼파라미터, 얼개의 갈래, 다른 자료 꾸러미로 해 보면 앎이 깊어지고 모형 지킴 일에 손에 잡히는 느낌이 붙는다.

## 익힘 문제

**익힘 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 맞섬에 든든하기에 알맞은지 밝혀라.

??? success "익힘 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**익힘 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "익힘 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**익힘 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "익힘 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**익힘 4.**
FGSM 밑바탕 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "익힘 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_fgsm():
        model = FGSM(...)
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
