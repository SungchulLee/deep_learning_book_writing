# FGSM 기초

입문 단계의 구현이다. 가장 단순한 적대적 공격 가운데 하나인 빠른 그래디언트 부호 방법(FGSM)을 클래스로 구현한다.

적대적 강건성은 안전이 걸린 곳에 뉴럴 네트워크를 배포할 때 중요한 문제다. 이 구현은 그래디언트의 부호만으로 만든 작은 섭동이 어떻게 모델을 속이는지 보여 준다. 다만 `__main__` 블록은 안내 메시지만 출력하고, 모델을 불러와 공격을 실제로 돌리지는 않는다(아래 출력 참고).

## 1. 코드

??? note "코드 (582줄)"

    ```python
    """
    62.1 배치: 빠른 그래디언트 부호 방법(FGSM) - 첫스텝

    이 배치는 가장 단순하고 널리 힘을 미친 맞서는 치기 방법 가운데 하나인
    빠른 그래디언트 부호 방법(FGSM)을 구현했다. FGSM은 뉴럴 네트워크가 작고 공들여
    지은 섭동에 취약하다는 것을 보여 준다.

    수학 밑그림:
    =======================

    FGSM 치기는 이제의 들임 언저리에서 로스 함수를 곧게 편 데 기댄다.
    파라메터가 θ인 뉴럴 네트워크 f, 들임 x, 참 이름표 y이 있을 때,
    로스를 가장 크게 하는 섭동 δ을 찾으려 한다:

        maximize L(θ, x + δ, y)  subject to ||δ||_∞ ≤ ε

    FGSM은 로스의 일차 테일러 어림을 쓴다:

        L(θ, x + δ, y) ≈ L(θ, x, y) + δ^T ∇_x L(θ, x, y)

    이를 ||δ||_∞ ≤ ε 아래 가장 크게 하려면 이렇게 한다:

        δ = ε · sign(∇_x L(θ, x, y))

    그러면 맞서는 보기가 나온다:

        x_adv = x + ε · sign(∇_x L(θ, x, y))

    느낌으로 알기:
    ==========
    - 그래디언트 ∇_x L은 로스를 키우려면 x을 어떻게 바꿀지 알려 준다
    - 부호를 취하면 방향(+ 또는 -)만 남는다
    - ε을 곱하면 차수마다 받아 주는 가장 큰 섭동이 된다
    - 이는 "가장 가파른 오름" 방향으로 한 스텝 밟는 치기다

    핵심 결:
    ===============
    1. 셈이 잘 든다: O(1) - 그래디언트 셈 한 번
    2. 한 스텝에 맞서는 보기를 만든다
    3. 많은 모형을 속이기에 흔히 넉넉하다
    4. 더 센 치기의 비롯 자리가 된다

    지은이: 가르침 감
    날짜: 2025년 11월
    어려움: 첫스텝
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
        빠른 그래디언트 부호 방법(FGSM) 치기

        이 클래스는 로스를 가장 크게 하는 방향으로 그래디언트 한 스텝을 밟아
        맞서는 보기를 만드는 FGSM 치기를 구현했다.

        수학 형태:
        -------------------------
        주어진 것:
            - 파라메터가 θ인 모형 f(x; θ)
            - 참 이름표가 y인 들임 x
            - 로스 함수 L(θ, x, y)
            - 섭동 예산 ε

        FGSM 치기는 다음을 셈한다:
            x_adv = x + ε · sign(∇_x L(θ, x, y))

        여기서:
            - ∇_x L은 들임에 대한 로스의 그래디언트
            - sign(·)은 낱낱에 +1, 0, -1을 돌려준다
            - ε은 섭동의 크기를 다스린다

        속성:
        -----------
        model : nn.Module
            칠 뉴럴 네트워크
        epsilon : float
            가장 큰 섭동의 크기(L∞ 노름)
        loss_fn : nn.Module
            가장 크게 할 로스 함수(기본값: CrossEntropyLoss)
        device : torch.device
            셈할 장치(CPU 또는 GPU)
        clip_min : float
            옳은 픽셀의 가장 작은 값(기본값: 0.0)
        clip_max : float
            옳은 픽셀의 가장 큰 값(기본값: 1.0)
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

            파라메터:
            -----------
            model : nn.Module
                칠 뉴럴 네트워크. 평가 모드여야 한다.
            epsilon : float, 기본값=0.3
                가장 큰 L∞ 섭동. [0,1]으로 기준 맞춘 그림에서
                ε=0.3이면 픽셀마다 많아야 0.3만큼 바뀐다.
                흔한 값: 0.03(은근함), 0.1(가운데), 0.3(셈)
            loss_fn : nn.Module, 골라 씀
                쓸 로스 함수. None이면 CrossEntropyLoss을 쓴다.
            device : torch.device, 골라 씀
                셈할 장치. None이면 모형의 장치를 쓴다.
            clip_min : float, 기본값=0.0
                날임의 가장 작은 값(그림에서는 흔히 0)
            clip_max : float, 기본값=1.0
                날임의 가장 큰 값(기준 맞춘 그림에서는 흔히 1)
            """
            self.model = model
            self.epsilon = epsilon
            self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
            self.device = device if device is not None else next(model.parameters()).device
            self.clip_min = clip_min
            self.clip_max = clip_max

            # 모형을 평가 모드로 둔다(배치 표준화와 드롭아웃에 종요롭다)
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

            이 방법은 FGSM의 핵심 알고리즘을 구현했다:
            1. 본디 들임의 로스를 셈한다
            2. 들임에 대한 로스의 그래디언트를 셈한다
            3. 섭동을 만든다: δ = ε · sign(∇_x L)
            4. 맞서는 보기를 만든다: x_adv = x + δ
            5. 옳은 자리 [clip_min, clip_max]으로 잘라 낸다

            파라메터:
            -----------
            images : torch.Tensor
                형태가 (batch_size, channels, height, width)인 맑은 그림
                [clip_min, clip_max] 자리에 있어야 한다
            labels : torch.Tensor
                형태가 (batch_size,)인 참 이름표
            targeted : bool, 기본값=False
                True이면 target_labels으로 갈리는 보기를 만든다
                False이면 틀리게 갈리는 보기를 만든다(아무 틀린 클래스)
            target_labels : torch.Tensor, 골라 씀
                타깃 있는 치기의 타깃 이름표. targeted=True이면 있어야 한다.

            돌려주는 것:
            --------
            adv_images : torch.Tensor
                들임과 형태가 같은 맞서는 그림
                ||x_adv - x||_∞ ≤ ε을 반드시 채운다

            수학의 자세한 것:
            ---------------------
            타깃 없는 치기(targeted=False)에서는
                로스를 가장 크게 한다: δ = ε · sign(∇_x L(f(x), y))
                이는 미루어 봄을 참 이름표에서 밀어낸다

            타깃 있는 치기(targeted=True)에서는
                로스를 가장 작게 한다: δ = -ε · sign(∇_x L(f(x), y_target))
                이는 미루어 봄을 타깃 이름표 쪽으로 끌어당긴다
            """
            # 들임을 장치로 옮긴다
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 흔들 베낌을 만든다(셈 그림에서 떼어 낸다)
            # 모형 파라메터가 아니라 그림에 대한 그래디언트를 좇아야 한다
            images_adv = images.clone().detach()

            # 그림에 그래디언트 셈을 켠다
            # 이것이 종요롭다: ∇_θ L이 아니라 ∇_x L이 있어야 한다
            images_adv.requires_grad = True

            # 앞으로 스텝: 모형의 미루어 봄을 셈한다
            outputs = self.model(images_adv)

            # 로스를 셈한다
            # 타깃 있는 치기에는 target_labels을, 없는 치기에는 참 이름표를 쓴다
            if targeted:
                if target_labels is None:
                    raise ValueError("target_labels must be provided for targeted attack")
                target_labels = target_labels.to(self.device)
                loss = self.loss_fn(outputs, target_labels)
            else:
                loss = self.loss_fn(outputs, labels)

            # 되돌아 스텝: 들임 그림에 대한 로스의 그래디언트를 셈한다
            # 이로써 ∇_x L(θ, x, y)을 얻는다
            self.model.zero_grad()  # Clear any existing gradients
            if images_adv.grad is not None:
                images_adv.grad.zero_()
            loss.backward()

            # 그래디언트를 뽑아낸다
            # images_adv.grad의 형태는 (batch_size, channels, height, width)이다
            grad = images_adv.grad.data

            # 그래디언트의 부호로 섭동을 만든다
            # 타깃 없는 치기: δ = ε · sign(∇_x L)
            # 타깃 있는 치기: δ = -ε · sign(∇_x L)(타깃 쪽으로 로스를 가장 작게)
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

            # 셈 그림에서 떼어 낸다(이제 그래디언트가 있어야 하지 않다)
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
            타깃으로 삼은 공격 성공률을 이루는 가장 작은 ε을 찾는다.

            이 방법은 엡실론 값을 여럿 뒤져 바라는 공격 성공률에 드는
            가장 작은 섭동을 찾는다.

            알고리즘:
            ----------
            1. 엡실론 값을 작은 것부터 차례로 해 본다
            2. 엡실론마다 맞서는 보기를 만든다
            3. 치기가 먹힌 비율을 셈한다
            4. target_success_rate을 이루는 첫 엡실론을 돌려준다

            모형이 얼마나 강건한지 알아보는 데 쓸모 있다. 엡실론이 작을수록
            모형이 더 취약하다는 뜻이다.

            파라메터:
            -----------
            images : torch.Tensor
                맑은 그림
            labels : torch.Tensor
                참 이름표
            epsilon_values : List[float]
                해 볼 엡실론 값의 목록(작은 것부터 줄 세워야 한다)
            target_success_rate : float, 기본값=0.9
                바라는 치기 공격 성공률(0에서 1)

            돌려주는 것:
            --------
            best_adv_images : torch.Tensor
                먹힌 것 가운데 엡실론이 가장 작은 맞서는 그림
            best_epsilon : float
                타깃 공격 성공률을 이루는 가장 작은 엡실론
            """
            best_epsilon = None
            best_adv_images = None

            for eps in epsilon_values:
                # 엡실론을 잠깐 바꾼다
                original_epsilon = self.epsilon
                self.epsilon = eps

                # 맞서는 보기를 만든다
                adv_images = self.generate(images, labels)

                # 공격 성공률을 따진다
                success_rate = self.compute_success_rate(images, labels, adv_images)

                # 본디 엡실론을 되돌린다
                self.epsilon = original_epsilon

                # 타깃 공격 성공률에 들었는지 살핀다
                if success_rate >= target_success_rate:
                    best_epsilon = eps
                    best_adv_images = adv_images
                    break

            if best_epsilon is None:
                print(f"알림: 공격 성공률 {target_success_rate}에 이르지 못했다")
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

            공격 성공률은 맞서는 보기 가운데 틀리게 갈린(타깃 없는 치기) 또는
            타깃으로 갈린(타깃 있는 치기) 것의
            기여도다.

            수학의 뜻매김:
            ------------------------
            공격 성공률 = (1/n) * Σ I[f(x_adv) ≠ y]

            여기서:
            - n은 보기의 수
            - I[·]은 알림 함수(참이면 1, 거짓이면 0)
            - f(x_adv)은 맞서는 보기에 대한 모형의 미루어 봄
            - y은 참 이름표

            파라메터:
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
                먹힌 치기의 기여도(0에서 1)
            """
            with torch.no_grad():  # No gradients needed for evaluation
                # 맞서는 보기의 미루어 봄을 얻는다
                outputs = self.model(adv_images.to(self.device))
                _, predicted = torch.max(outputs, 1)

                # 틀리게 갈린 것을 센다
                # 미루어 봄 != 참 이름표이면 먹힌 것이다
                successful_attacks = (predicted != labels.to(self.device)).sum().item()

                # 공격 성공률을 셈한다
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
            1. 맑은 정확도: 본디 그림의 정확도
            2. 적대적 정확도: 흔든 그림의 정확도
            3. 치기 공격 성공률: 먹힌 치기의 기여도
            4. 평균 L∞ 섭동: 가장 큰 바뀜의 크기
            5. 평균 L2 섭동: 유클리드 거리

            파라메터:
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

                # 치기 공격 성공률을 셈한다
                success_rate = 1.0 - adv_accuracy

                # 섭동의 자를 셈한다
                perturbation = (adv_images - clean_images).cpu()

                # L∞ 노름: 픽셀 모두에 걸친 가장 큰 바뀜의 크기
                linf_norm = torch.max(torch.abs(perturbation)).item()

                # L2 노름: 유클리드 거리
                # 보기마다 L2 노름을 셈한 뒤 고르게 한다
                l2_norms = torch.norm(perturbation.view(len(perturbation), -1), p=2, dim=1)
                l2_norm = l2_norms.mean().item()

                # L0 노름: 바뀐 픽셀의 수(엄밀하지는 않다)
                # |바뀜| > 임계인 픽셀을 센다
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
                print("FGSM 치기 평가 결과")
                print("=" * 60)
                print(f"엡실론 (ε): {self.epsilon}")
                print(f"맑은 정확도: {clean_accuracy:.2%}")
                print(f"적대적 정확도: {adv_accuracy:.2%}")
                print(f"치기 공격 성공률: {success_rate:.2%}")
                print(f"평균 L∞ 섭동: {linf_norm:.6f}")
                print(f"평균 L2 섭동: {l2_norm:.6f}")
                print(f"평균 L0 섭동: 픽셀 {l0_norm:.2f}개")
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
        맞서는 보기와 섭동을 그린다.

        이 함수는 다음을 보이는 그림을 만든다:
        - 참 이름표를 붙인 본디 맑은 그림
        - 미루어 본 이름표를 붙인 맞서는 그림
        - 키운 섭동(눈에 띄게 부풀림)

        이 그림은 FGSM이 그림에 무엇을 하는지 알아보는 데 도움이 된다.

        파라메터:
        -----------
        clean_images : torch.Tensor
            형태가 (batch_size, C, H, W)인 본디 그림
        adv_images : torch.Tensor
            맞서는 그림
        labels : torch.Tensor
            참 이름표
        predictions : torch.Tensor
            맞서는 그림에 대한 모형의 미루어 봄
        class_names : List[str], 골라 씀
            이름표에 쓸 클래스 이름
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

        # 섭동을 셈한다
        perturbations = adv_np - clean_np

        # 3줄짜리 그림을 만든다: 원본, 적대적 샘플, 섭동(키움)
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

            # 3줄: 섭동(눈에 띄게 10배로 키움)
            # 더 잘 보이도록 섭동의 기준을 맞춘다
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
        4. 치기가 잘 먹히는지 평가
        5. 결과 그리기
        """
        print("=" * 70)
        print("FGSM 치기 보여 주기")
        print("=" * 70)
        print("\n이 글은 미리 익힌 ResNet-18 모형으로 CIFAR-10 자료 꾸러미에서")
        print("빠른 그래디언트 부호 방법(FGSM) 치기를 보여 준다.")
        print("\n붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.")
        print("=" * 70)
    ```


**출력:**

```
======================================================================
FGSM 치기 보여 주기
======================================================================

이 글은 미리 익힌 ResNet-18 모형으로 CIFAR-10 자료 꾸러미에서
빠른 그래디언트 부호 방법(FGSM) 치기를 보여 준다.

붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.
======================================================================
```

## 2. 논의

FGSM에서 로스는 학습할 때와 반대로 쓰인다. 웨이트는 고정하고, 입력에 대한 로스의 그래디언트 $\nabla_x \mathcal{L}$을 구해 로스를 **키우는** 쪽으로 입력을 한 번 옮긴다. 그래서 `generate` 는 `images_adv.requires_grad = True` 로 입력에 그래디언트를 켜고, 모델을 `eval()` 로 두어 배치 정규화와 드롭아웃이 공격 도중에 흔들리지 않게 한다.

`visualize_attack` 은 원본, 적대적 샘플, 10배로 키운 섭동을 세 줄로 나란히 그린다. 섭동이 눈으로는 거의 보이지 않는데 예측이 바뀌는 것을 한눈에 보여 주는 것이 이 그림의 목적이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 구현에서 선택한 것 셋을 들고, 각각 왜 적대적 강건성에 알맞은지 설명하여라.

</div>

??? success "풀이"
    이 구현에서 눈여겨볼 선택은 셋이다. (1) **생성자에서 `model.eval()`을 부른다** -- 학습 모드로 두면 배치 정규화가 공격 배치의 통계를 쓰고 드롭아웃이 뉴런을 무작위로 끄므로, 공격하는 모델과 평가하는 모델이 서로 달라진다. (2) **`images.clone().detach()`에 `requires_grad = True`를 켠다** -- 필요한 것은 웨이트가 아니라 입력에 대한 그래디언트 $\nabla_x \mathcal{L}$이고, 원본 텐서와 계산 그래프를 끊어 두어야 원본이 오염되지 않는다. (3) **섭동을 더한 뒤 `torch.clamp`로 `[clip_min, clip_max]`에 잘라 낸다** -- 잘라 내지 않으면 픽셀값 범위를 벗어난, 실제로는 존재할 수 없는 이미지가 나온다. 잘라 내도 $\|x_{\text{adv}} - x\|_\infty \leq \varepsilon$은 그대로 지켜진다.

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
    흔한 실패 양상은 이렇다. (1) **그래디언트가 0에 가깝다(그래디언트 마스킹)** -- 소프트맥스가 포화했거나 미분할 수 없는 전처리가 끼어 있으면 `grad`의 대부분이 0이 되고, `torch.sign(0) = 0`이라 그 픽셀은 전혀 움직이지 않는다. `grad.abs().mean()`과 `(grad == 0).float().mean()`을 찍어 진단하고, 무작위 시작을 더하거나 로스를 로짓 마진으로 바꾸거나 반복 공격(PGD)으로 넘어가 고친다. (2) **입력 범위가 어긋난다** -- 모델이 평균·표준편차로 정규화한 입력을 받는데 `clip_min=0`, `clip_max=1`로 잘라 내면 섭동이 엉뚱한 범위에서 잘린다. `images.min()`, `images.max()`를 확인해 진단하고, 정규화를 모델 안으로 옮겨 공격은 늘 $[0, 1]$ 픽셀 공간에서 하도록 고친다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
FGSM 기초 구현을 검사하는 포괄적인 테스트 함수를 작성하여라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 조건을 시험하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 시험 함수를 작성한다.
    ```python
    def test_fgsm():
        model = FGSM(...)
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

**다룬 것** — FGSM 기초

FGSM은 입력에 대한 로스의 그래디언트 부호로 한 스텝을 밟아 적대적 샘플을 만든다.

핵심 클래스는 `FGSM`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
