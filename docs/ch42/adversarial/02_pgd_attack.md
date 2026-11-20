# PGD 공격

중급 단계의 구현이다. FGSM보다 센 반복 공격인 투영 경사 하강법(PGD)을 클래스로 구현한다. 무작위 시작, 매 스텝의 $\varepsilon$ 공 투영, 여러 번의 재시작(restart), 반복 횟수에 따른 FGSM과의 비교 함수를 담았다.

적대적 강건성은 안전이 걸린 곳에 뉴럴 네트워크를 배포할 때 중요한 문제다. 다만 `__main__` 블록은 안내 메시지만 출력하고, 모델을 불러와 공격을 실제로 돌리지는 않는다(아래 출력 참고).

## 1. 코드

??? note "코드 (735줄)"

    ```python
    """
    62.2 배치: 투영 경사 하강법(PGD) 치기 - 가운데 스텝

    이 배치는 FGSM의 더 센 회귀 클래스인 투영 경사 하강법(PGD) 치기를
    구현했다. PGD은 가장 센 일차 맞서는 치기의 하나로 여겨지며
    모형의 강건성을 평가하는 데 흔히 쓰인다.

    수학 밑그림:
    =======================

    PGD은 작은 그래디언트 스텝을 여러 번 밟고 스텝마다 받아 주는 섭동 모임으로
    되비추어 FGSM을 넓힌다. 그래서 한 스텝짜리 FGSM보다
    훨씬 세다.

    파라메터가 θ인 모형 f, 들임 x, 이름표 y, 섭동 예산 ε이 있을 때:

    PGD은 다음을 푼다:
        maximize_{||δ||_∞ ≤ ε} L(θ, x + δ, y)

    알고리즘(타깃 없음):
    -----------------------
    1. 첫자리: x^(0) = x + uniform_noise[-ε, ε]  (아무 비롯 자리)
    2. t = 0에서 T-1까지:
           x^(t+1) = Π_{x+S}(x^(t) + α · sign(∇_x L(θ, x^(t), y)))
    3. x^(T)을 돌려준다

    여기서:
    - Π_{x+S}은 ε 공으로의 되비춤: Π(z) = clip(z, x-ε, x+ε)
    - α은 스텝 크기(흔히 α = ε/num_iter 또는 α = 2.5·ε/num_iter)
    - S = {δ : ||δ||_∞ ≤ ε}은 받아 주는 섭동 모임
    - T은 회귀 횟수

    FGSM과 다른 핵심:
    ===========================
    1. **여러 스텝**: PGD은 작은 스텝을 여러 번 밟는다(흔히 10~100)
    2. **아무 첫자리**: ε 공 안의 아무 점에서 비롯한다
    3. **되비추기**: 스텝마다 ε 공으로 되비춘다
    4. **더 셈**: 맞서는 보기를 훨씬 잘 찾는다

    아무 첫자리:
    =====================
    아무 첫자리는 PGD의 세기에 종요롭다:
    - 그 자리의 나쁜 봉우리를 벗어나게 돕는다
    - ε 공의 여러 자리를 둘러본다
    - 치기가 첫 조건에 덜 예민해진다

    흔한 전략:
    - 고르게: x^(0) ~ Uniform[x-ε, x+ε]
    - 가우스: x^(0) ~ N(x, σ²I) 뒤 ε 공으로 되비춤

    되비추는 셈:
    ===================
    되비춤 Π은 섭동이 ε 공 안에 머물게 한다:

    L∞ 노름에서는:
        Π(z)_i = clip(z_i, x_i - ε, x_i + ε)

    이는 [x-ε, x+ε] 상자로 낱낱이 잘라 내는 것이다.

    L2 노름에서는:
        ||z - x||_2 ≤ ε이면: Π(z) = z
        아니면: Π(z) = x + ε · (z - x) / ||z - x||_2

    스텝 크기 고르기:
    ===================
    스텝 크기 α은 다음의 맞바꿈을 다룬다:
    - α이 크면: 빨리 모이나 지나칠 수 있다
    - α이 작으면: 더 꼭 집으나 회귀가 더 든다

    흔히 고르는 것:
    - α = ε / T (스텝마다 온 예산의 1/T)
    - α = 2.5 · ε / T (조금 더 세게)
    - α = 2 · ε / T (매드리 등, 2018)

    지은이: 가르침 감
    날짜: 2025년 11월
    어려움: 가운데 스텝
    먼저 알 것: FGSM(62.1 배치), 회귀 다듬기
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
        투영 경사 하강법(PGD) 치기

        PGD은 그래디언트 스텝을 여러 번 밟고 받아 주는 섭동 모임으로 되비추는
        회귀 맞서는 치기다. FGSM보다 훨씬 세며
        강건성을 평가하는 사실상의 기준이다.

        수학 형태:
        -------------------------
        PGD 치기는 옭아맨 다듬기 문제를 푼다:

            maximize L(θ, x + δ, y)  subject to ||δ||_p ≤ ε

        투영 경사 하강법을 써서:

            x^(t+1) = Π_{x+S}(x^(t) + α · sign(∇_x L(θ, x^(t), y)))

        여기서 Π은 ε 공으로 되비추는 셈이다.

        속성:
        -----------
        model : nn.Module
            칠 뉴럴 네트워크
        epsilon : float
            가장 큰 섭동의 크기
        alpha : float
            되돌 때마다의 스텝 크기
        num_iter : int
            돌릴 회귀 횟수
        norm : str
            쓸 노름('linf' 또는 'l2')
        random_init : bool
            아무 첫자리를 쓸지
        loss_fn : nn.Module
            가장 크게 할 로스 함수
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

            파라메터:
            -----------
            model : nn.Module
                칠 뉴럴 네트워크
            epsilon : float, 기본값=0.03
                가장 큰 섭동의 크기(L∞ 또는 L2 노름)
                CIFAR-10에서는 ε=8/255≈0.031이 여느 값이다
            alpha : float, 기본값=0.01
                되돌 때마다의 스텝 크기
                흔히 고르는 값: α = 2.5 * ε / num_iter
                밝히지 않으면 기본값은 ε / 4이다
            num_iter : int, 기본값=40
                PGD 회귀 횟수
                회귀가 많을수록 치기는 세지고 느려진다
                흔한 값: 10(빠름), 40(여느 것), 100(셈)
            norm : str, 기본값='linf'
                쓸 노름: 'linf'(L∞) 또는 'l2'(L2)
                L∞: 픽셀 모두가 ε으로 마디 지어진다
                L2: 온 섭동이 ε으로 마디 지어진다
            random_init : bool, 기본값=True
                섭동의 첫자리를 아무렇게나 잡을지
                True: 더 강건한 치기(즐겨 씀)
                False: 본디 그림에서 비롯한다(I-FGSM처럼)
            loss_fn : nn.Module, 골라 씀
                쓸 로스 함수
            device : torch.device, 골라 씀
                셈할 장치
            clip_min : float, 기본값=0.0
                옳은 픽셀의 가장 작은 값
            clip_max : float, 기본값=1.0
                옳은 픽셀의 가장 큰 값
            early_stop : bool, 기본값=False
                치기가 먹히면 회귀를 멈춘다
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

            # 모형을 평가 모드로 둔다
            self.model.eval()
            self.model = self.model.to(self.device)

            # 파라메터를 따진다
            if self.norm not in ['linf', 'l2']:
                raise ValueError(f"norm must be 'linf' or 'l2', got {self.norm}")

            # 차림을 찍는다
            print(f"PGD 치기 차림:")
            print(f"  엡실론 (ε): {self.epsilon}")
            print(f"  알파 (α): {self.alpha}")
            print(f"  회귀: {self.num_iter}")
            print(f"  노름: L{self.norm}")
            print(f"  아무 첫자리: {self.random_init}")
            print(f"  일찍 멈추기: {self.early_stop}")

        def _initialize_perturbation(
            self,
            images: torch.Tensor
        ) -> torch.Tensor:
            """
            섭동의 첫자리를 잡는다.

            두 가지 전략이 있다:
            1. 아무 첫자리(즐겨 씀):
               - ε 공에서 고르게 뽑는다
               - 그 자리 봉우리를 벗어나게 돕는다
               - 더 강건한 치기

            2. 0에서 비롯하기:
               - 본디 그림에서 비롯한다
               - 회귀 FGSM(I-FGSM)과 같다
               - 빠르나 더 여릴 수 있다

            수학의 자세한 것(L∞):
            -------------------------
            아무 첫자리: 차수마다 δ^(0) ~ Uniform[-ε, ε]
            그다음 되비춘다: x^(0) = clip(x + δ^(0), [clip_min, clip_max])

            파라메터:
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
                    # 아무 방향을 잡고 반지름 ≤ ε으로 아무렇게나 스케일을 맞춘다
                    delta = torch.randn_like(images)
                    # 낱 공으로 기준을 맞춘다
                    delta_norm = delta.view(len(delta), -1).norm(p=2, dim=1)
                    delta = delta / delta_norm.view(-1, 1, 1, 1)
                    # [0, ε]의 아무 반지름으로 스케일을 맞춘다
                    random_radius = torch.rand(len(delta), device=images.device)
                    random_radius = random_radius * self.epsilon
                    delta = delta * random_radius.view(-1, 1, 1, 1)

                # 섭동을 걸고 옳은 자리로 잘라 낸다
                perturbed_images = images + delta
                perturbed_images = torch.clamp(perturbed_images, self.clip_min, self.clip_max)
            else:
                # 맑은 그림에서 비롯한다(섭동 없음)
                perturbed_images = images.clone()

            return perturbed_images

        def _project(
            self,
            perturbed_images: torch.Tensor,
            original_images: torch.Tensor
        ) -> torch.Tensor:
            """
            흔든 그림을 본디 그림 둘레 ε 공으로 되비춘다.

            이것이 섭동을 마디 안에 붙잡아 두는 핵심 셈이다.
            그래디언트 한 스텝을 밟으면 ||δ|| > ε이 될 수 있으므로
            될 수 있는 모임으로 되비춰야 한다.

            L∞ 되비춤:
            --------------
            픽셀 i마다:
                δ_i = clip(δ_i, -ε, +ε)

            이는 [-ε, +ε] 상자로 낱낱이 잘라 내는 것이다.

            L2 되비춤:
            --------------
            ||δ||_2 ≤ ε이면 되비출 것이 없다
            아니면: δ = ε · δ / ||δ||_2

            이는 방향은 지키면서 δ의 노름을 꼭 ε으로 맞춘다.

            파라메터:
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
            # 이제의 섭동을 셈한다
            delta = perturbed_images - original_images

            if self.norm == 'linf':
                # L∞ 되비춤: 낱낱을 [-ε, +ε]으로 잘라 낸다
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:  # l2
                # L2 되비춤: 노름 ≤ ε이 되도록 스케일을 맞춘다
                batch_size = len(delta)
                # 보기마다 L2 노름을 셈한다
                delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1)
                # ε을 넘는 보기의 마스크를 만든다
                exceed_mask = delta_norm > self.epsilon
                # ε을 넘는 섭동을 줄인다
                if exceed_mask.any():
                    scale = self.epsilon / delta_norm[exceed_mask]
                    delta[exceed_mask] = delta[exceed_mask] * scale.view(-1, 1, 1, 1)

            # 되비춘 섭동을 건다
            projected_images = original_images + delta

            # 옳은 픽셀 자리 [clip_min, clip_max]으로도 잘라 낸다
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
                 a. 로스 L(θ, x^(t), y)을 셈한다
                 b. 그래디언트 g = ∇_x L을 셈한다
                 c. 고친다: x^(t+1) = x^(t) + α · sign(g)  [L∞에서]
                 d. 되비춘다: x^(t+1) = Π(x^(t+1))
                 e. 옳은 자리로 잘라 낸다
            3. x^(T)을 돌려준다

            되비추는 스텝 (d)이 종요롭다. 섭동을 ε 공 안에 붙잡아 둔다.

            파라메터:
            -----------
            images : torch.Tensor
                맑은 그림
            labels : torch.Tensor
                참 이름표(타깃 있는 치기에서는 타깃 이름표)
            targeted : bool, 기본값=False
                타깃 있는 치기를 할지
            target_labels : torch.Tensor, 골라 씀
                타깃 있는 치기의 타깃 이름표
            verbose : bool, 기본값=False
                회귀가 나아가는 것을 찍는다

            돌려주는 것:
            --------
            adv_images : torch.Tensor
                PGD 회귀를 마친 맞서는 그림
            """
            # 장치로 옮긴다
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 맞서는 그림의 첫자리를 잡는다
            adv_images = self._initialize_perturbation(images)

            # 타깃 있는 치기에서
            if targeted and target_labels is not None:
                target_labels = target_labels.to(self.device)

            # 회귀 치기
            iterator = range(self.num_iter)
            if verbose:
                iterator = tqdm(iterator, desc="PGD iterations")

            for i in iterator:
                # 맞서는 그림에 그래디언트 좇기를 켠다
                adv_images = adv_images.detach().clone()
                adv_images.requires_grad = True

                # 앞으로 스텝
                outputs = self.model(adv_images)

                # 로스를 셈한다
                if targeted:
                    # 타깃 있는 치기: 타깃에 대한 로스를 가장 작게
                    loss = -self.loss_fn(outputs, target_labels)
                else:
                    # 타깃 없는 치기: 참 이름표에 대한 로스를 가장 크게
                    loss = self.loss_fn(outputs, labels)

                # 되돌아 스텝
                self.model.zero_grad()
                if adv_images.grad is not None:
                    adv_images.grad.zero_()
                loss.backward()

                # 그래디언트를 얻는다
                grad = adv_images.grad.data

                # 그래디언트 오름 스텝
                if self.norm == 'linf':
                    # L∞: 그래디언트 부호 방향으로 걷는다
                    # x^(t+1) = x^(t) + α · sign(∇L)
                    adv_images = adv_images + self.alpha * torch.sign(grad)
                else:  # l2
                    # L2: 기준 맞춘 그래디언트 방향으로 걷는다
                    # x^(t+1) = x^(t) + α · ∇L / ||∇L||_2
                    grad_norm = grad.view(len(grad), -1).norm(p=2, dim=1)
                    # 0으로 나누기를 회피한다
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
                            # 타깃 있음: 보기가 모두 타깃에 닿았는지 살핀다
                            if (predicted == target_labels).all():
                                if verbose:
                                    print(f"\n{i+1}번째 회귀에서 일찍 멈춘다: 치기가 모두 먹혔다")
                                break
                        else:
                            # 타깃 없음: 보기가 모두 틀리게 갈렸는지 살핀다
                            if (predicted != labels).all():
                                if verbose:
                                    print(f"\n{i+1}번째 회귀에서 일찍 멈춘다: 치기가 모두 먹혔다")
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
            로스가 가장 큰(가장 잘 속인) x_adv을 돌려준다

            이것이 PGD의 가장 센 클래스이며 강건성을 따질 때
            즐겨 쓴다.

            파라메터:
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
                타깃 있는 치기 표시
            target_labels : torch.Tensor, 골라 씀
                타깃 있는 치기의 타깃 이름표
            verbose : bool, 기본값=False
                나아가는 것을 찍는다

            돌려주는 것:
            --------
            best_adv_images : torch.Tensor
                다시 비롯한 것 가운데 로스가 가장 큰 맞서는 보기
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

                # 이 맞서는 보기의 로스를 셈한다
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    if targeted:
                        loss = -self.loss_fn(outputs, target_labels).item()
                    else:
                        loss = self.loss_fn(outputs, labels).item()

                # 가장 좋은 맞서는 보기를 남긴다(로스가 가장 큰 것)
                if best_loss is None or loss > best_loss:
                    best_loss = loss
                    best_adv_images = adv_images.clone()

                if verbose:
                    print(f"로스: {loss:.4f} (가장 좋음: {best_loss:.4f})")

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
            - 맑은 정확도
            - 적대적 정확도
            - 치기 공격 성공률
            - 섭동의 자(L∞, L2, L1)

            파라메터:
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
                평가하는 자를 담은 사전
            """
            with torch.no_grad():
                # 맑은 정확도
                clean_outputs = self.model(clean_images.to(self.device))
                _, clean_pred = torch.max(clean_outputs, 1)
                clean_correct = (clean_pred == labels.to(self.device)).sum().item()
                clean_accuracy = clean_correct / len(labels)

                # 적대적 정확도
                adv_outputs = self.model(adv_images.to(self.device))
                _, adv_pred = torch.max(adv_outputs, 1)
                adv_correct = (adv_pred == labels.to(self.device)).sum().item()
                adv_accuracy = adv_correct / len(labels)

                # 치기 공격 성공률
                success_rate = 1.0 - adv_accuracy

                # 섭동의 자
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
                print("PGD 치기 평가 결과")
                print("=" * 60)
                print(f"차림:")
                print(f"  엡실론: {self.epsilon}")
                print(f"  알파: {self.alpha}")
                print(f"  회귀: {self.num_iter}")
                print(f"  노름: L{self.norm}")
                print(f"\n결과:")
                print(f"  맑은 정확도: {clean_accuracy:.2%}")
                print(f"  적대적 정확도: {adv_accuracy:.2%}")
                print(f"  치기 공격 성공률: {success_rate:.2%}")
                print(f"\n섭동의 자:")
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
        회귀 횟수를 달리한 PGD을 견준다(FGSM을 아울러).

        이 함수는 회귀가 늘수록 PGD이 세지는 것을 보여 준다:
        - 회귀 1 = FGSM(베이스라인)
        - 회귀 10 = 빠른 PGD
        - 회귀 40 = 여느 PGD
        - 회귀 100 = 센 PGD

        파라메터:
        -----------
        model : nn.Module
            칠 모형
        images : torch.Tensor
            맑은 그림
        labels : torch.Tensor
            참 이름표
        epsilon : float, 기본값=0.03
            섭동 예산
        num_iter_list : List[int], 기본값=[1, 10, 40, 100]
            해 볼 회귀 횟수의 목록
        device : torch.device, 골라 씀
            셈할 장치

        돌려주는 것:
        --------
        results : Dict[int, Dict[str, float]]
            회귀 횟수마다의 결과
        """
        if device is None:
            device = next(model.parameters()).device

        results = {}

        for num_iter in num_iter_list:
            print(f"\n{'='*60}")
            print(f"회귀 {num_iter}번으로 PGD을 해 본다")
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
        print("PGD 회귀 견주기")
        print(f"{'='*60}")
        print(f"{'회귀':<12} {'공격 성공률':<15} {'적대적 정확도':<15}")
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
        3. 회귀 횟수를 달리한 견줌
        """
        print("=" * 70)
        print("PGD 치기 보여 주기")
        print("=" * 70)
        print("\n이 글은 센 회귀 맞서는 치기인 투영 경사 하강법(PGD)")
        print("치기를 보여 준다.")
        print("=" * 70)
    ```


**출력:**

```
======================================================================
PGD 치기 보여 주기
======================================================================

이 글은 센 회귀 맞서는 치기인 투영 경사 하강법(PGD)
치기를 보여 준다.
======================================================================
```

## 2. 논의

PGD의 핵심은 매 스텝 뒤의 투영 `_project` 다. 그래디언트 스텝을 밟으면 섭동이 $\varepsilon$ 공 밖으로 나갈 수 있으므로, $\ell_\infty$에서는 성분별로 잘라 내고 $\ell_2$에서는 노름이 $\varepsilon$을 넘는 샘플만 크기를 줄여 되돌린다. 그 뒤 픽셀 범위 $[0, 1]$로 한 번 더 잘라 낸다.

`generate_with_restarts` 는 무작위 시작점을 바꿔 가며 PGD를 여러 번 돌리고, 로스가 가장 큰 결과를 고른다. 다만 이 코드는 배치 **전체의 평균 로스**로 고르므로 샘플마다 가장 좋은 재시작을 고르는 구현보다 약하다. `compare_fgsm_pgd` 는 반복 1회(사실상 FGSM)부터 100회까지 공격 성공률을 나란히 찍어, 반복이 늘수록 공격이 세지는 것을 보여 준다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 구현에서 선택한 것 셋을 들고, 각각 왜 적대적 강건성에 알맞은지 설명하여라.

</div>

??? success "풀이"
    이 구현에서 눈여겨볼 선택은 셋이다. (1) **무작위 시작(`random_init=True`)** -- $\varepsilon$ 공 안의 임의의 점에서 출발하면 클린 입력 바로 근처의 평평하거나 날카로운 로스 지형을 피해, 같은 반복 횟수로 더 높은 로스에 이르기 쉽다. (2) **매 스텝 뒤의 투영 `_project`** -- 그래디언트 스텝이 섭동을 $\varepsilon$ 공 밖으로 밀어내도 다음 스텝 전에 되돌려 놓으므로, 몇 번을 반복하든 위협 모델의 예산을 지킨다. 픽셀 범위 $[0, 1]$ 클리핑도 같은 자리에서 한다. (3) **`generate_with_restarts`의 여러 번 재시작** -- 로스 지형이 볼록하지 않아 한 번의 PGD는 국소 최댓값에 머물 수 있으므로, 시작점을 바꿔 여러 번 돌리고 로스가 가장 큰 것을 고른다. 다만 위 논의대로 배치 평균 로스로 고르므로 샘플마다 고르는 것보다 약하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 자료형이 기대한 대로인지 확인하는 입력 검증을 주 함수나 클래스에 추가하여라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `generate` 메서드의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 추가한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기에 강건하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 둘을 들고, 각각 어떻게 탐지하고 고칠지 설명하여라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **스텝 크기가 맞지 않는다** -- $\alpha$가 너무 작으면 $T$번을 다 밟아도 $\varepsilon$ 공의 경계에 닿지 못하고, 너무 크면 매 스텝 투영에 잘려 경계 위에서 진동한다. 반복마다 로스와 $\|x_{\text{adv}} - x\|_\infty$를 찍어 진단하고, $\alpha \approx 2.5\varepsilon/T$처럼 $T$번이면 경계를 넉넉히 지날 크기로 고른다. (2) **그래디언트 마스킹으로 공격이 약해 보인다** -- 반복 횟수나 재시작을 늘려도 강건 정확도가 그대로이거나, 블랙박스 공격이 PGD보다 더 잘 통하면 그래디언트가 공격에 쓸모없게 가려진 것이다. `compare_fgsm_pgd`로 반복 횟수에 따른 곡선을 보고, 재시작 수를 늘리거나 로짓 마진 로스(C&W)·APGD·전이 공격으로 교차 확인해 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
PGD 공격 구현을 검사하는 포괄적인 테스트 함수를 작성하여라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 조건을 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 시험 함수를 작성한다.
    ```python
    def test_pgd():
        model = PGD(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — PGD 공격

PGD는 작은 그래디언트 스텝과 $\varepsilon$ 공 투영을 번갈아 되풀이해 FGSM보다 센 적대적 샘플을 찾는다.

핵심 클래스는 `PGD`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
