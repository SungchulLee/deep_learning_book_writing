# 밝혀 낸 막이

62.5 묶음: 밝혀 낸 막이 - 한발 더. 이 묶음은 증명할 수 있는 다짐을 주는 밝혀 낸 든든함 막이를 짜 놓았다

맞섬에 든든하기는 안전이 걸린 자리에 신경 그물을 내놓을 때 종요로운 걱정거리다. 이 짜보기는 작은 흔듦이 어떻게 모형을 속이는지, 막이는 어떻게 지을 수 있는지 보이며 맞섬에 든든하기의 깨침을 드러낸다.

## 1. 코드

```python
"""
62.5 묶음: 밝혀 낸 막이 - 한발 더

이 묶음은 모형의 미루어 봄에 증명할 수 있는 다짐을 주는 밝혀 낸 든든함 막이를
짜 놓았다. 남다른 치기로 시험해 보는 겪은 막이(맞서며 익히기 따위)와 달리,
밝혀 낸 막이는 주어진 반지름 안의 모든 흔듦에 대해
수학의 다짐을 준다.

수학 밑그림:
=======================

겪은 든든함과 밝혀 낸 든든함:
-----------------------------------
**겪은 든든함**:
- 아는 치기로 시험한다(FGSM, PGD, C&W)
- 모르는 치기에는 다짐이 없다
- 기울기 가리기에 걸릴 수 있다

**밝혀 낸 든든함**:
- 미루어 봄이 든든하다는 수학의 증명
- ε 공 안의 모든 흔듦에 다짐이 선다
- 밝혀 낸 반지름 안에서는 어떤 치기에도 속지 않는다

아무렇게나 매끄럽게 하기:
====================
아무렇게나 매끄럽게 하기는 본디 가름개를 가우스 잡음으로 매끄럽게 하여
증명할 수 있게 든든한 가름개를 만든다.

짓기:
-------------
밑 가름개 f: R^d → {1,...,k}으로 매끄럽게 한 가름개 g을 짓는다:

    g(x) = argmax_c P(f(x + ε) = c)  where ε ~ N(0, σ²I)

느낌으로 말하면:
- 들임에 가우스 잡음을 더한다
- 잡음 섞인 미루어 봄에서 많은 쪽을 고른다
- 이것이 가름개를 "매끄럽게" 한다

밝히기 정리:
=====================
코언 등(2019)이 증명했다:

어떤 들임 x에서
    P(f(x + ε) = c_A) ≥ p_A  [으뜸 갈래의 낌새]
    P(f(x + ε) = c_B) ≤ p_B  [다음 갈래의 낌새]

이면 g(x) = c_A은 다음 L2 반지름 안에서 밝혀 낸 든든함을 지닌다:

    R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))

여기서 Φ^(-1)은 잣대 정규 분포의 쌓인 분포 함수의 거꿀이다.

느낌으로 알기:
----------
- p_A은 갈래 c_A이 많은 쪽이 될 낌새
- p_B은 다른 갈래가 이길 낌새
- p_A >> p_B이면(크게 자신하면) R이 크다
- R은 밝혀 낸 든든함의 반지름이다

짜기:
===============

두 도막:
------------------
1. **고르기**: 미루어 볼 갈래 c_A을 찾는다
   - 잡음 섞인 미루어 봄을 n0번 뽑는다
   - 많은 쪽을 고른다

2. **밝히기**: p_A을 어림하고 반지름을 밝힌다
   - 잡음 섞인 미루어 봄을 n번 더 뽑는다
   - p_A, p_B의 믿음 구간을 셈한다
   - 밝혀 낸 반지름 R을 셈한다

몬테카를로 뽑기:
--------------------
낌새는 몬테카를로로 어림한다:
    
    P(f(x + ε) = c) ≈ (# times f(x + ε_i) = c) / N

여기서 ε_1, ..., ε_N ~ N(0, σ²I)

통계의 다짐:
----------------------
클로퍼-피어슨 믿음 구간을 쓰면 낌새의 다짐을 얻는다:
    
    낌새 ≥ 1-α으로: p_A ≥ p̂_A이고 p_B ≤ p̂_B

이로써 낌새 ≥ 1-α으로 밝혀 낸 반지름 R을 얻는다.

고갱이 매개변수:
===============
1. **σ(시그마)**: 잡음의 잣대 어긋남
   - σ이 크면: 밝혀 낸 반지름은 크고 맞음은 낮다
   - 흔함: σ ∈ [0.12, 1.0]

2. **n0**: 고르는 데 쓸 표본 수
   - 흔함: n0 = 100

3. **n**: 밝히는 데 쓸 표본 수
   - 표본이 많을수록 믿음 구간이 촘촘하다
   - 흔함: n = 10,000 이상

4. **α(알파)**: 믿음 켜
   - 흔함: α = 0.001(믿음 99.9%)

맞바꿈:
==========
1. **맞음과 밝혀 낸 반지름**:
   - σ이 크면: R은 크고 맑은 맞음은 낮다
   - σ이 작으면: 맞음은 높고 R은 작다

2. **셈 값**:
   - 밝히려면 앞으로 걸음이 많이 든다(표본 N개)
   - 여느 미루어 봄보다 훨씬 느리다
   - 나란히 할 수 있다

3. **L2과 L∞**:
   - 아무렇게나 매끄럽게 하기는 L2을 밝힌다
   - L∞을 밝히기는 더 어렵다

지은이: 가르침 감
날짜: 2025년 11월
어려움: 한발 더
먼저 알 것: 낌새 이론, 통계, 믿음 구간
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from scipy.stats import norm, binom
from tqdm import tqdm
import math

# ========================================================================
# 메인
# ========================================================================


class RandomizedSmoothing:
    """
    밝혀 낸 든든함을 위한 아무렇게나 매끄럽게 하기
    
    이 갈래는 어떤 가름개에도 증명할 수 있는 L2 든든함 다짐을 주는
    아무렇게나 매끄럽게 하기를 짜 놓았다.
    
    수학 꼴:
    -------------------------
    밑 가름개 f으로 매끄럽게 한 가름개를 짓는다:
    
        g(x) = argmax_c E_{ε~N(0,σ²I)}[1{f(x + ε) = c}]
    
    밝히기: P(f(x + ε) = c_A) ≥ p_A이면 g(x)은 다음 L2 반지름 안에서
    밝혀 낸 든든함을 지닌다:
    
        R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))
    
    여기서 c_B은 다음 갈래이고 Φ^(-1)은 잣대 정규 분포의 쌓인 분포 함수의 거꿀이다.
    
    속성:
    -----------
    base_classifier : nn.Module
        매끄럽게 할 밑 가름개
    sigma : float
        가우스 잡음의 잣대 어긋남
    device : torch.device
        셈할 장치
    """
    
    def __init__(
        self,
        base_classifier: nn.Module,
        sigma: float = 0.25,
        device: Optional[torch.device] = None
    ):
        """
        아무렇게나 매끄럽게 하기의 첫자리를 잡는다.
        
        매개변수:
        -----------
        base_classifier : nn.Module
            매끄럽게 할 밑 가름개
            로짓(소프트맥스 앞의 날임)을 내야 한다
        sigma : float, 기본값=0.25
            가우스 잡음의 잣대 어긋남
            σ이 크면: 더 매끄럽고 밝혀 낸 반지름이 크다
            σ이 작으면: 덜 매끄럽고 맞음이 높다
            흔한 값: σ ∈ [0.12, 0.25, 0.50, 1.0]
        device : torch.device, 골라 씀
            셈할 장치
        """
        self.base_classifier = base_classifier
        self.sigma = sigma
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 밑 가름개를 따짐 모드로 둔다
        self.base_classifier.eval()
        self.base_classifier = self.base_classifier.to(self.device)
        
        print(f"아무렇게나 매끄럽게 하기 차림:")
        print(f"  시그마 (σ): {self.sigma}")
        print(f"  장치: {self.device}")
    
    def _sample_noise(
        self,
        x: torch.Tensor,
        num_samples: int,
        batch_size: int = 1000
    ) -> torch.Tensor:
        """
        잡음 섞인 표본의 미루어 봄을 만든다.
        
        이것이 몬테카를로 어림 걸음이다. 다음을 한다:
        1. 가우스 잡음을 더한다: x + ε, ε ~ N(0, σ²I)
        2. 미루어 봄을 얻는다: f(x + ε)
        3. N번 되풀이한다
        
        기억을 아끼려 묶음으로 다룬다.
        
        매개변수:
        -----------
        x : torch.Tensor
            들임 그림(그림 하나)
        num_samples : int
            만들 잡음 섞인 표본의 수
        batch_size : int, 기본값=1000
            표본을 다룰 묶음 크기
        
        돌려주는 것:
        --------
        counts : torch.Tensor
            갈래마다의 미루어 봄 셈
            꼴: (num_classes,)
        """
        with torch.no_grad():
            # 앞으로 걸음으로 갈래의 수를 얻는다
            if not hasattr(self, 'num_classes'):
                test_output = self.base_classifier(x.unsqueeze(0))
                self.num_classes = test_output.size(1)
            
            # 갈래마다의 셈을 마련한다
            counts = torch.zeros(self.num_classes, device=self.device)
            
            # 기억을 아끼려 묶음으로 다룬다
            num_batches = math.ceil(num_samples / batch_size)
            
            for _ in range(num_batches):
                # 참 묶음 크기를 정한다(마지막 묶음은 작을 수 있다)
                current_batch_size = min(batch_size, num_samples - len(counts.nonzero()))
                
                # 들임을 되풀이해 묶음을 만든다
                batch = x.repeat(current_batch_size, 1, 1, 1)
                
                # 가우스 잡음을 더한다: ε ~ N(0, σ²I)
                noise = torch.randn_like(batch) * self.sigma
                noisy_batch = batch + noise
                
                # 잡음 섞인 표본의 미루어 봄을 얻는다
                outputs = self.base_classifier(noisy_batch)
                predictions = outputs.argmax(dim=1)
                
                # 갈래마다 미루어 봄을 센다
                for pred in predictions:
                    counts[pred] += 1
        
        return counts
    
    def predict(
        self,
        x: torch.Tensor,
        n: int = 1000,
        alpha: float = 0.001,
        batch_size: int = 1000
    ) -> Tuple[int, float]:
        """
        들임 하나의 갈래를 미루어 보고 든든함을 밝힌다.
        
        두 도막으로 이루어진다:
        ------------------
        1. 고르기: 표본 n0개로 미루어 볼 갈래를 찾는다
        2. 밝히기: 낌새를 어림하고 반지름을 셈한다
        
        매개변수:
        -----------
        x : torch.Tensor
            들임 그림(그림 하나, 꼴: (C, H, W))
        n : int, 기본값=1000
            밝히는 데 쓸 표본 수
            표본이 많을수록 믿음 구간이 촘촘하다
            흔함: 좋은 다짐에는 n ≥ 10,000
        alpha : float, 기본값=0.001
            믿음 켜(어긋날 낌새)
            흔함: α = 0.001(믿음 99.9%)
        batch_size : int, 기본값=1000
            표본을 다룰 묶음 크기
        
        돌려주는 것:
        --------
        prediction : int
            미루어 본 갈래 이름표
        radius : float
            밝혀 낸 L2 반지름
            반지름 = 0이면 밝히기가 어그러진 것이다
        """
        # 1도막: 고르기(으뜸 갈래 찾기)
        # 잘 들도록 표본 n/10개를 쓴다
        n_selection = max(100, n // 10)
        counts_selection = self._sample_noise(x, n_selection, batch_size)
        top_class = counts_selection.argmax().item()
        
        # 2도막: 밝히기(낌새 어림)
        counts_cert = self._sample_noise(x, n, batch_size)
        
        # 낌새의 믿음 구간을 셈한다
        # 클로퍼-피어슨(정확한) 두 값 믿음 구간을 쓴다
        
        # 으뜸 갈래의 셈
        count_top = counts_cert[top_class].item()
        
        # p_A(으뜸 갈래의 낌새)의 믿음 아래끝
        p_A_lower = self._lower_confidence_bound(count_top, n, alpha)
        
        # 다음 갈래를 찾는다(으뜸 갈래는 뺀다)
        counts_cert[top_class] = -1  # Temporarily remove top class
        runner_up_class = counts_cert.argmax().item()
        count_runner_up = counts_cert[runner_up_class].item()
        
        # p_B(다음 갈래의 낌새)의 믿음 위끝
        p_B_upper = self._upper_confidence_bound(count_runner_up, n, alpha)
        
        # 밝혀 낸 반지름을 셈한다
        # R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))
        if p_A_lower > p_B_upper:
            # 밝히기가 되었다
            radius = self._compute_radius(p_A_lower, p_B_upper)
        else:
            # 밝히기가 어그러졌다(믿음이 모자라다)
            radius = 0.0
        
        return top_class, radius
    
    def _lower_confidence_bound(
        self,
        count: int,
        n: int,
        alpha: float
    ) -> float:
        """
        두 값 몫의 믿음 아래끝을 셈한다.
        
        클로퍼-피어슨 길(정확한 두 값 믿음 구간)을 쓴다.
        
        수학 꼴:
        -------------------------
        n번 가운데 count번 들어맞았다고 하자.
        참 낌새 p은 다음을 채운다:
            P(count | p) ≥ α/2
        
        이로써 다음을 채우는 아래끝 p_lower을 얻는다:
            P(p ≥ p_lower) ≥ 1 - α/2
        
        매개변수:
        -----------
        count : int
            들어맞은 횟수
        n : int
            해 본 횟수
        alpha : float
            뜻있음 켜
        
        돌려주는 것:
        --------
        p_lower : float
            믿음 아래끝
        """
        return binom.ppf(alpha/2, n, count/n) / n if count > 0 else 0.0
    
    def _upper_confidence_bound(
        self,
        count: int,
        n: int,
        alpha: float
    ) -> float:
        """
        두 값 몫의 믿음 위끝을 셈한다.
        
        매개변수:
        -----------
        count : int
            들어맞은 횟수
        n : int
            해 본 횟수
        alpha : float
            뜻있음 켜
        
        돌려주는 것:
        --------
        p_upper : float
            믿음 위끝
        """
        return binom.ppf(1 - alpha/2, n, count/n) / n if count < n else 1.0
    
    def _compute_radius(
        self,
        p_A: float,
        p_B: float
    ) -> float:
        """
        낌새로 밝혀 낸 반지름을 셈한다.
        
        식:
        --------
        R = σ/2 * (Φ^(-1)(p_A) - Φ^(-1)(p_B))
        
        여기서:
        - Φ^(-1)은 잣대 정규 분포의 쌓인 분포 함수의 거꿀
        - p_A은 으뜸 갈래 낌새의 아래끝
        - p_B은 다음 갈래 낌새의 위끝
        
        느낌으로 알기:
        ----------
        - p_A이 1에 가깝고 p_B이 0에 가까우면: R이 크다(크게 자신함)
        - p_A ≈ p_B이면: R이 작다(머뭇거림)
        - σ이 반지름의 잣대를 잡는다. σ이 크면 R도 크다
        
        매개변수:
        -----------
        p_A : float
            으뜸 갈래 낌새의 아래끝
        p_B : float
            다음 갈래 낌새의 위끝
        
        돌려주는 것:
        --------
        radius : float
            밝혀 낸 L2 반지름
        """
        # 쌓인 분포 함수의 거꿀 값을 셈한다
        # norm.ppf은 잣대 정규 쌓인 분포 함수의 거꿀이다
        if p_A >= 1.0:
            p_A = 0.999999  # Avoid infinity
        if p_B <= 0.0:
            p_B = 0.000001
        
        phi_inv_pA = norm.ppf(p_A)
        phi_inv_pB = norm.ppf(p_B)
        
        # 반지름을 셈한다
        radius = (self.sigma / 2.0) * (phi_inv_pA - phi_inv_pB)
        
        return max(0.0, radius)  # Ensure non-negative
    
    def certify_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        n: int = 10000,
        alpha: float = 0.001,
        batch_size: int = 1000,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        그림 묶음을 밝히고 자를 셈한다.
        
        그림마다:
        1. 갈래와 밝혀 낸 반지름을 미루어 본다
        2. 미루어 봄이 맞는지 살핀다
        3. 치기에 대해 밝혀졌는지 살핀다
        
        자:
        --------
        - 맑은 맞음: 옳게 미루어 본 몫
        - 반지름 r의 밝혀 낸 맞음: 옳으면서 밝혀 낸 반지름 ≥ r인 몫
        
        매개변수:
        -----------
        images : torch.Tensor
            그림 묶음
        labels : torch.Tensor
            참 이름표
        n : int, 기본값=10000
            그림마다의 표본 수
        alpha : float, 기본값=0.001
            믿음 켜
        batch_size : int, 기본값=1000
            잡음을 뽑을 묶음 크기
        verbose : bool, 기본값=True
            나아가는 것을 찍는다
        
        돌려주는 것:
        --------
        results : Dict[str, float]
            밝히기 자
        """
        num_images = len(images)
        predictions = []
        radii = []
        
        if verbose:
            pbar = tqdm(range(num_images), desc="Certifying")
        else:
            pbar = range(num_images)
        
        for i in pbar:
            pred, radius = self.predict(images[i], n, alpha, batch_size)
            predictions.append(pred)
            radii.append(radius)
        
        # 텐서로 옮긴다
        predictions = torch.tensor(predictions, device=labels.device)
        radii = torch.tensor(radii)
        
        # 자를 셈한다
        correct = (predictions == labels)
        
        # 맑은 맞음
        clean_accuracy = correct.float().mean().item()
        
        # 반지름마다 밝혀 낸 맞음
        radius_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        certified_accuracies = {}
        
        for r in radius_levels:
            # 밝혀짐: 미루어 봄이 옳고 밝혀 낸 반지름 ≥ r
            certified = correct & (radii >= r)
            certified_accuracies[f'certified_acc_r={r}'] = certified.float().mean().item()
        
        # 밝혀 낸 반지름의 평균(옳게 가른 보기에 대해)
        avg_radius = radii[correct].mean().item() if correct.any() else 0.0
        
        results = {
            'clean_accuracy': clean_accuracy,
            'avg_certified_radius': avg_radius,
            **certified_accuracies
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("밝히기 결과")
            print("=" * 60)
            print(f"맑은 맞음: {clean_accuracy:.2%}")
            print(f"밝혀 낸 반지름 평균: {avg_radius:.4f}")
            print("\n반지름마다 밝혀 낸 맞음:")
            for r in radius_levels:
                key = f'certified_acc_r={r}'
                print(f"  r = {r}: {results[key]:.2%}")
            print("=" * 60)
        
        return results


# 쓰는 보기
if __name__ == "__main__":
    """
    아무렇게나 매끄럽게 하여 밝혀 낸 든든함을 보여 준다.
    """
    print("=" * 70)
    print("아무렇게나 매끄럽게 하여 밝혀 낸 든든함")
    print("=" * 70)
    print("\n이 글은 아무렇게나 매끄럽게 하기로 증명할 수 있는")
    print("든든함 다짐을 보여 준다.")
    print("\n붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.")
    print("\n알림: 밝히기는 셈이 아주 비싸다!")
    print("=" * 70)
```

## 2. 논의

이 짜보기는 맑고 읽기 쉬운 PyTorch 코드로 맞섬에 든든하기의 고갱이 깨침을 드러낸다. 묶음으로 나눈 얼개 덕에 몫 하나하나를 살피고 다른 일이나 자료 꾸러미에 맞춰 고치기 쉽다.

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
밝혀 낸 막이 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_randomizedsmoothing():
        model = RandomizedSmoothing(...)
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

**다룬 것** — 밝혀 낸 막이

이 짜보기는 맑고 읽기 쉬운 PyTorch 코드로 맞섬에 든든하기의 고갱이 깨침을 드러낸다.

고갱이 갈래는 `RandomizedSmoothing`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
