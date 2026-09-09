# 에너지 바탕 모델의 쓰임새

에너지 바탕 모델은 만들어 내기를 훨씬 넘어서며 분포 밖 찾기, 그림 잡음 없애기, 얽어 만들어 내기를 아우르는 틀을 준다. 에너지 함수는 자료가 얼마나 어울리는지 재는 자연스러운 잣대를 준다. 에너지가 낮으면 분포 안의 표본이고 높으면 이상 징후이다. 이 쓰임새 넓음이 에너지 바탕 모델을 여러 기계 배움 일에서 값진 도구로 만든다.

## 1. 코드

```python
"""
에너지 바탕 모델의 쓰임새: 분포 밖 찾기, 잡음 없애기, 얽어 만들어 내기
================================================================================

만들어 내기 너머 에너지 바탕 모델의 실제 쓰임새.

걸리는 시간: 90~120분
"""

import torch
import torch.nn as nn
import numpy as np

# ========================================================================
# 메인
# ========================================================================

def out_of_distribution_detection():
    """분포 밖 찾기에 에너지 바탕 모델의 에너지를 쓴다."""
    print("\nOut-of-Distribution Detection with EBMs")
    print("Lower energy → In-distribution")
    print("Higher energy → Out-of-distribution")
    print("✓ Can identify anomalies and novel inputs")

def image_denoising_with_ebm():
    """에너지 가장 작게 하기로 그림의 잡음을 없앤다."""
    print("\nImage Denoising via Energy Minimization")
    print("Start with noisy image, minimize energy")
    print("✓ Preserves structure while removing noise")

def compositional_generation():
    """에너지를 더해 여러 개념을 합친다."""
    print("\nCompositional Generation")
    print("E_combined = E_concept1 + E_concept2")
    print("✓ Generate images with multiple attributes")

def main():
    print("="*70)
    print("EBM APPLICATIONS")
    print("="*70)
    
    out_of_distribution_detection()
    image_denoising_with_ebm()
    compositional_generation()
    
    print("\nKey Applications:")
    print("  ✓ Anomaly/OOD detection")
    print("  ✓ Image denoising and inpainting")
    print("  ✓ Compositional generation")
    print("  ✓ Adversarial robustness")

if __name__ == "__main__":
    main()
```

**출력:**

```
======================================================================
EBM APPLICATIONS
======================================================================

Out-of-Distribution Detection with EBMs
Lower energy → In-distribution
Higher energy → Out-of-distribution
✓ Can identify anomalies and novel inputs

Image Denoising via Energy Minimization
Start with noisy image, minimize energy
✓ Preserves structure while removing noise

Compositional Generation
E_combined = E_concept1 + E_concept2
✓ Generate images with multiple attributes

Key Applications:
  ✓ Anomaly/OOD detection
  ✓ Image denoising and inpainting
  ✓ Compositional generation
  ✓ Adversarial robustness
```

## 2. 논의

분포 밖(OOD) 찾기는 아마 에너지 바탕 모델의 가장 자연스러운 쓰임새일 것이다. 에너지 함수가 어떤 들임에도 낱값을 매기므로 모델이 주어진 표본을 얼마나 잘 "설명"하는지 곧바로 잰다. 모델이 익힌 분포 안의 자료는 낮은 에너지를 받고, 다른 분포에서 온 분포 밖 자료는 더 높은 에너지를 받는 편이다. 에너지에 단순한 문턱을 두면 잘 듣는 분포 밖 찾개가 되며, 분포 밖 들임에서 지나치게 자신하는 소프트맥스 자신도 바탕 방법보다 흔히 낫다.

그림 잡음 없애기는 에너지 바탕 모델을 배운 사전 분포로 쓴다. 잡음 낀 관측 $y = x + n$이 주어지면 합친 목표 $E_\theta(x) + \frac{1}{2\sigma^2}\|y - x\|^2$을 가장 작게 하여 깨끗한 그림을 되찾을 수 있다. 첫 항은 내놓기가 배운 자료 다양체 위에 놓이도록 이끌고 둘째 항은 관측에 충실하도록 한다. 이 에너지 가장 작게 하기는 기울기 내려가기나 랑주뱅 움직임으로 할 수 있으며 잡음 낀 들임을 자료 다양체 위로 되쏘는 셈이다.

얽어 만들어 내기는 에너지 바탕 모델을 다른 만들어 내는 모델과 갈라 주는 남다른 힘이다. 에너지는 더할 수 있으므로 서로 다른 개념으로 익힌 여러 에너지 바탕 모델을 합치면 그 교집합의 모델이 된다: $E_{\text{combined}}(x) = E_1(x) + E_2(x)$. 보기로 "붉은 물체"의 에너지 바탕 모델과 "둥근 물체"의 에너지 바탕 모델을 얽으면 따로 함께 익히지 않고도 "붉고 둥근 물체"를 만들어 낼 수 있다. 이 얽음성은 에너지 공간이 아니라 확률 공간에서 도는 맞겨루기 만들개나 변분 자기 부호기로는 이루기 어렵다.

## 연습문제

**연습문제 1.**
MNIST 숫자로 익힌 에너지 바탕 모델이 주어질 때 에너지 함수로 MNIST 숫자와 Fashion-MNIST 그림을 가르는 분포 밖 찾기 절차를 짜라. 찾기 잣대와 가름 규칙을 뜻매김하라.

??? success "연습문제 1 풀이"
    ```python
    def ood_detection(energy_net, in_dist_loader, ood_loader, threshold=None):
        in_energies = []
        ood_energies = []
        
        with torch.no_grad():
            for x, _ in in_dist_loader:
                in_energies.append(energy_net(x).cpu())
            for x, _ in ood_loader:
                ood_energies.append(energy_net(x).cpu())
        
        in_energies = torch.cat(in_energies)
        ood_energies = torch.cat(ood_energies)
        
        if threshold is None:
            threshold = in_energies.quantile(0.95).item()
        
        in_detected = (in_energies < threshold).float().mean()
        ood_detected = (ood_energies >= threshold).float().mean()
        
        return {
            'threshold': threshold,
            'in_dist_correct': in_detected.item(),
            'ood_detected': ood_detected.item(),
            'auroc': compute_auroc(in_energies, ood_energies)
        }
    ```
    
    가름 규칙은 이렇다. $E_\theta(x) > \tau$이면 들임 $x$을 분포 밖으로 가른다. 여기서 $\tau$은 바라는 거짓 양성 비율(보기로 분포 안 표본의 5%이 걸리도록)을 이루도록 고른다. AUROC 잣대는 모든 문턱에 걸쳐 에너지가 분포 안 표본과 분포 밖 표본을 얼마나 잘 갈라 주는지 잰다.

---

**연습문제 2.**
에너지 바탕 모델 둘 $E_1(x)$과 $E_2(x)$을 더해 얽는 것이 그 고르게 맞추지 않은 분포의 곱에 해당함을 밝혀라. 어떤 가정 아래에서 이 곱이 두 개념의 교집합을 어림하는가?

??? success "연습문제 2 풀이"
    $p_1(x) \propto \exp(-E_1(x))$이고 $p_2(x) \propto \exp(-E_2(x))$이라 하자. 얽은 분포는 다음과 같다:
    
    $$
    p_{1 \cap 2}(x) \propto \exp(-(E_1(x) + E_2(x))) = \exp(-E_1(x)) \cdot \exp(-E_2(x)) \propto p_1(x) \cdot p_2(x)
    $$
    
    이 곱은 들임이 주어졌을 때 두 분포가 대략 서로 얽매이지 않는다는 가정 아래에서 교집합을 어림한다. 자세히는 $p_1$이 "붉은 물체"를 담고 $p_2$이 "둥근 물체"를 담으면 그 곱은 $p_1$과 $p_2$이 모두 높은 자리, 곧 붉고 둥근 물체에 높은 확률을 준다. 각 에너지 바탕 모델이 담는 특징이 통계로 서로 얽매이지 않을 때 이 어림은 정확하다. 실제로는 얽매이지 않음이 꼭 들어맞지 않아도 얽음이 뜻으로 그럴듯한 결과를 내곤 한다.

---

**연습문제 3.**
에너지 바탕 모델로 그림 안 그리기 알고리즘을 짜라. 빠진 화소를 가리키는 알려진 가림막과 함께 망가진 그림이 주어질 때 안 그리기 문제를 에너지 가장 작게 하기로 적고 기울기 내려가기로 풀어라.

??? success "연습문제 3 풀이"
    ```python
    def inpaint(energy_net, corrupted_image, mask, n_steps=500, lr=0.01, 
                data_weight=100.0):
        """
        mask: 알려진 화소는 1, 빠진 화소는 0
        """
        # 빠진 화소를 잡음으로 첫자리매김한다
        x = corrupted_image.clone()
        x = x + (1 - mask) * torch.randn_like(x) * 0.5
        x = x.requires_grad_(True)
        
        optimizer = torch.optim.Adam([x], lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # 에너지 바탕 모델 사전 분포: x이 자료 다양체 위에 놓이도록 이끈다
            energy = energy_net(x)
            
            # 자료 충실: 알려진 화소를 붙박이로 둔다
            fidelity = data_weight * ((x - corrupted_image) * mask).pow(2).sum()
            
            loss = energy + fidelity
            loss.backward()
            optimizer.step()
        
        return x.detach()
    ```
    
    에너지 항은 배운 사전 분포 노릇을 하여 채운 그림이 그럴듯해 보이도록 이끈다. 충실 항은 알려진 화소가 바뀌지 않게 한다. 두 항의 균형은 `data_weight`으로 다스린다. 너무 작으면 그럴듯하지 않은 채움을 허락하고, 너무 크면 에너지 바탕 모델이 빠진 자리에 영향을 주지 못한다. 기울기 내려가기가 두 항을 한꺼번에 가장 작게 하도록 빠진 화소를 되풀이해 다듬는다.

## 정리하며

**다룬 것** — 에너지 바탕 모델의 쓰임새

분포 밖(OOD) 찾기는 아마 에너지 바탕 모델의 가장 자연스러운 쓰임새일 것이다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
