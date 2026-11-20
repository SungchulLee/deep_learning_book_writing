# 적대적 학습

심화 단계의 구현이다. 가장 잘 통하는 방어로 알려진 적대적 학습을 클래스로 구현한다. 미니배치마다 PGD로 적대적 샘플을 만들어 그것으로 학습하는 `AdversarialTrainer`(PGD-AT)와, 클린 로스에 KL 발산 항을 더하는 `TRADESTrainer`를 담았다.

적대적 강건성은 안전이 걸린 곳에 뉴럴 네트워크를 배포할 때 중요한 문제다. 다만 `__main__` 블록은 안내 메시지만 출력하고, 모델을 실제로 학습하지는 않는다(아래 출력 참고).

## 1. 코드

??? note "코드 (755줄)"

    ```python
    """
    62.4 배치: 맞서며 익히기 - 심화

    이 배치는 맞서는 치기를 막는 가장 잘 듣는 길인 맞서며 익히기를 구현했다.
    맞서며 익히기는 익힘 꾸러미에 맞서는 보기를 불려 넣어
    모형이 섭동에 강건해지게 한다.

    수학 밑그림:
    =======================

    여느 익힘:
    -----------------
    여느 겪은 무릅씀 가장 작게 하기(ERM):

        min_θ E_{(x,y)~D}[L(θ, x, y)]

    이는 익힘 분포의 정확도를 다듬지만 맞서는 섭동은
    헤아리지 않는다.

    맞서며 익히기(강건하게 다듬기):
    ------------------------------------------
    맞서며 익히기는 가장 작게-가장 크게 하는 다듬기 문제를 푼다:

        min_θ E_{(x,y)~D}[ max_{||δ||≤ε} L(θ, x + δ, y) ]

    안쪽 가장 크게 하기: 가장 나쁜 섭동을 찾는다(치기)
    바깥 가장 작게 하기: 그에 강건하도록 모형을 익힌다

    이는 익힘 보기마다 ε 공 안에서 가장 나쁜 로스를 가장 작게 하려는
    강건하게 다듬기 문제다.

    풀이:
    ==============
    1. **안쪽 가장 크게**: 익힘 보기마다 로스를 가장 크게 하는
       맞서는 섭동을 찾는다(예산 ε 안의 가장 센 치기)
    2. **바깥 가장 작게**: 가장 나쁜 로스가 작아지도록 파라메터를 고친다
    3. **결과**: 모형이 익힘 자료의 ε 공 안에서 강건해진다

    참으로 짜기:
    =========================
    안쪽 가장 크게 하기를 다룰 수 없으므로 PGD으로 어림한다:

    알고리즘(PGD 맞서며 익히기):
    -------------------------------------
    판마다:
        잔 배치 (x, y)마다:
            1. 맞서는 보기를 만든다:
               x_adv = PGD(x, y, ε, α, K)  [K스텝 PGD]

            2. 맞서는 보기의 로스를 셈한다:
               L_adv = L(θ, x_adv, y)

            3. 파라메터를 고친다:
               θ ← θ - η∇_θ L_adv

    이를 "PGD 바탕 맞서며 익히기" 또는 "PGD-AT"이라 한다.

    핵심 하이퍼파라미터:
    ===================
    1. **ε(엡실론)**: 익힘의 섭동 예산
       - 강건성의 레이어를 정한다
       - 흔함: CIFAR-10에서 ε = 8/255 ≈ 0.031

    2. **K(PGD 스텝)**: PGD 회귀 횟수
       - 스텝이 많을수록 익힘 때의 치기가 세다
       - 흔함: K = 10(익힘), K = 20(평가)

    3. **α(스텝 크기)**: PGD 스텝 크기
       - 흔히 α = 2ε/K 또는 α = 2.5ε/K

    맞서며 익히기의 클래스:
    =================================

    1. **여느 PGD-AT**(매드리 등, 2018):
       - PGD으로 만든 맞서는 보기를 쓴다
       - 가장 세지만 셈이 비싸다

    2. **TRADES**(장 등, 2019):
       - 강건성과 정확도 사이의 이론에 닿는 맞바꿈
       - 맑은 정확도와 강건한 정확도를 저울질한다
       - 로스: L_nat + β·KL(f(x)||f(x_adv))

    3. **MART**(왕 등, 2020):
       - 잘못 가름을 아는 맞서며 익히기
       - 잘못 가른 보기에 힘을 모은다
       - 잘못 가른 것에는 북돋운 엇갈린 엔트로피, 옳게 가른 것에는 KL 발산

    4. **빠른 맞서며 익히기**(웡 등, 2020):
       - 잘 들도록 한 스텝 FGSM을 쓴다
       - 훨씬 빠르나 조금 덜 강건하다

    5. **값싼 맞서며 익히기**(샤파히 등, 2019):
       - 되돌아가기의 그래디언트를 되쓴다
       - 여느 익힘과 값이 같다

    정확도와 강건성의 맞바꿈:
    ==============================
    맞서며 익히면 흔히 맑은 정확도가 떨어진다:
    - 여느 익힘: 맑은 정확도 약 95%, 강건한 정확도 약 0%
    - 맞서며 익히기: 맑은 정확도 약 85%, 강건한 정확도 약 50%

    이 맞바꿈은 밑바탕부터 있는 것이며 잘 알려져 있다. TRADES은 두 목표를
    드러내 놓고 저울질하여 이를 완화하려 한다.

    참으로 헤아릴 것:
    ========================
    1. **셈 값**: PGD 탓에 여느 익힘보다 7~10배 느리다
    2. **지나치게 맞추기**: 조심스러운 다독임과 자료 불리기가 있어야 한다
    3. **무너지듯 지나친 맞춤**: 로스가 갑자기 치솟을 수 있다. 일찍 멈추기를 쓴다
    4. **평가**: 센 치기로 따져야 한다(PGD, AutoAttack)

    지은이: 가르침 감
    날짜: 2025년 11월
    어려움: 심화
    먼저 알 것: PGD(62.2 배치), 뉴럴 네트워크 익히기
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from typing import Optional, Dict, List, Tuple, Callable
    from tqdm import tqdm
    import copy

    # ========================================================================
    # 메인
    # ========================================================================


    class AdversarialTrainer:
        """
        강건한 모형을 위한 맞서며 익히기

        이 클래스는 PGD으로 만든 맞서는 보기로 맞서며 익히기를 구현했다.
        모형은 익힘 보기마다 ε 공 안에서 가장 나쁜 로스를 가장 작게 하도록
        익힌다.

        수학 형태:
        -------------------------
        강건하게 다듬는 문제를 푼다:

            min_θ E_{(x,y)}[ max_{||δ||≤ε} L(θ, x + δ, y) ]

        안쪽 가장 크게 하기를 K스텝 PGD으로 어림한다:

            x_adv = PGD(x, y, ε, α, K)
            θ ← θ - η∇_θ L(θ, x_adv, y)

        속성:
        -----------
        model : nn.Module
            익힐 모형
        epsilon : float
            섭동 예산
        alpha : float
            PGD 스텝 크기
        num_iter : int
            PGD 회귀 횟수
        """

        def __init__(
            self,
            model: nn.Module,
            epsilon: float = 0.031,
            alpha: float = 0.007,
            num_iter: int = 10,
            device: Optional[torch.device] = None,
            loss_fn: Optional[nn.Module] = None,
            norm: str = 'linf'
        ):
            """
            맞서며 익히는 개의 첫자리를 잡는다.

            파라메터:
            -----------
            model : nn.Module
                강건하게 익힐 모형
            epsilon : float, 기본값=0.031
                맞서며 익히기의 섭동 예산
                여느 값: CIFAR-10에서 ε = 8/255 ≈ 0.031
            alpha : float, 기본값=0.007
                익히는 동안의 PGD 스텝 크기
                흔히 α = 2ε/K이고 K은 num_iter이다
            num_iter : int, 기본값=10
                익히는 동안의 PGD 회귀 횟수
                빠르려면 적게, 세려면 많게
                익힘: 7~10, 평가: 20~100
            device : torch.device, 골라 씀
                셈할 장치
            loss_fn : nn.Module, 골라 씀
                로스 함수(기본값: CrossEntropyLoss)
            norm : str, 기본값='linf'
                섭동의 노름('linf' 또는 'l2')
            """
            self.model = model
            self.epsilon = epsilon
            self.alpha = alpha
            self.num_iter = num_iter
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
            self.norm = norm

            # 모형을 장치로 옮긴다
            self.model = self.model.to(self.device)

            print(f"맞서며 익히기 차림:")
            print(f"  엡실론 (ε): {self.epsilon}")
            print(f"  알파 (α): {self.alpha}")
            print(f"  PGD 회귀: {self.num_iter}")
            print(f"  노름: L{self.norm}")
            print(f"  장치: {self.device}")

        def _generate_adversarial_examples(
            self,
            images: torch.Tensor,
            labels: torch.Tensor,
            random_init: bool = True
        ) -> torch.Tensor:
            """
            PGD으로 맞서는 보기를 만든다.

            이것이 맞서며 익히기의 안쪽 가장 크게 하기다.
            가장 나쁜 섭동을 PGD으로 어림해 찾는다.

            알고리즘:
            ----------
            1. 첫자리: δ^(0) ~ Uniform[-ε, ε](아무렇게나) 또는 δ^(0) = 0
            2. t = 1에서 K까지:
                 δ^(t) = Π_ε(δ^(t-1) + α·sign(∇_δ L(θ, x+δ^(t-1), y)))
            3. x + δ^(K)을 돌려준다

            여기서 Π_ε은 ε 공으로의 되비춤이다.

            파라메터:
            -----------
            images : torch.Tensor
                맑은 그림
            labels : torch.Tensor
                참 이름표
            random_init : bool, 기본값=True
                PGD의 첫자리를 아무렇게나 잡을지

            돌려주는 것:
            --------
            adv_images : torch.Tensor
                맞서는 보기
            """
            # 치기를 만들 때는 모형을 평가 모드로 둔다
            # (치는 동안 배치 표준화와 드롭아웃은 평가 모드여야 한다)
            self.model.eval()

            # 섭동의 첫자리를 잡는다
            if random_init:
                # ε 공 안의 아무 첫자리
                delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
                delta = delta.to(self.device)
            else:
                delta = torch.zeros_like(images).to(self.device)

            # delta에 그래디언트를 켠다
            delta.requires_grad = True

            # PGD 회귀
            for _ in range(self.num_iter):
                # 맞서는 그림을 셈한다
                adv_images = images + delta

                # 앞으로 스텝
                outputs = self.model(adv_images)

                # 로스를 셈한다
                loss = self.loss_fn(outputs, labels)

                # 되돌아 스텝
                self.model.zero_grad()
                if delta.grad is not None:
                    delta.grad.zero_()
                loss.backward()

                # 그래디언트 오름 스텝
                if self.norm == 'linf':
                    delta_grad = delta.grad.detach()
                    delta = delta + self.alpha * torch.sign(delta_grad)
                    # ε 공으로 되비춘다
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                else:  # l2
                    delta_grad = delta.grad.detach()
                    # 그래디언트의 기준을 맞춘다
                    grad_norm = delta_grad.view(len(delta_grad), -1).norm(p=2, dim=1)
                    grad_norm = torch.clamp(grad_norm, min=1e-12)
                    normalized_grad = delta_grad / grad_norm.view(-1, 1, 1, 1)
                    delta = delta + self.alpha * normalized_grad
                    # ε 공으로 되비춘다
                    delta_norm = delta.view(len(delta), -1).norm(p=2, dim=1)
                    scale = torch.clamp(delta_norm / self.epsilon, min=1.0)
                    delta = delta / scale.view(-1, 1, 1, 1)

                # 옳은 그림 자리 [0, 1]으로 잘라 낸다
                delta = torch.clamp(images + delta, 0, 1) - images
                delta = delta.detach()
                delta.requires_grad = True

            # 맞서는 보기를 돌려준다
            adv_images = images + delta.detach()
            adv_images = torch.clamp(adv_images, 0, 1)

            # 모형을 익힘 모드로 되돌린다
            self.model.train()

            return adv_images

        def train_epoch(
            self,
            train_loader: DataLoader,
            optimizer: optim.Optimizer,
            epoch: int,
            verbose: bool = True
        ) -> Dict[str, float]:
            """
            맞서며 익히기로 한 판 익힌다.

            이것이 맞서며 익히기의 바깥 가장 작게 하기다.
            배치마다:
            1. 맞서는 보기를 만든다(안쪽 가장 크게)
            2. 맞서는 보기의 로스를 셈한다
            3. 모형 파라메터를 고친다(바깥 가장 작게)

            알고리즘:
            ----------
            배치 (x, y)마다:
                # 안쪽 가장 크게 하기(PGD으로 어림)
                x_adv = PGD(x, y, ε, α, K)

                # 적대적 로스를 셈한다
                L_adv = L(θ, x_adv, y)

                # 바깥 가장 작게 하기
                θ ← θ - η∇_θ L_adv

            파라메터:
            -----------
            train_loader : DataLoader
                익힘 자료 실개
            optimizer : optim.Optimizer
                모형 파라메터의 가장 좋게 하는 개
            epoch : int
                이제의 판 수
            verbose : bool, 기본값=True
                나아가는 것을 찍는다

            돌려주는 것:
            --------
            metrics : Dict[str, float]
                익힘 자(로스, 정확도)
            """
            self.model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            # 나아감 막대
            if verbose:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            else:
                pbar = train_loader

            for batch_idx, (images, labels) in enumerate(pbar):
                # 장치로 옮긴다
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 맞서는 보기를 만든다
                adv_images = self._generate_adversarial_examples(images, labels)

                # 맞서는 보기로 앞으로 스텝
                outputs = self.model(adv_images)
                loss = self.loss_fn(outputs, labels)

                # 되돌아 스텝과 다듬기
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 자를 좇는다
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 나아감 막대를 고친다
                if verbose:
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100. * correct / total
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.3f}',
                        'acc': f'{accuracy:.2f}%'
                    })

            # 판의 자를 셈한다
            metrics = {
                'train_loss': total_loss / len(train_loader),
                'train_accuracy': correct / total
            }

            return metrics

        def evaluate(
            self,
            test_loader: DataLoader,
            attack_epsilon: Optional[float] = None,
            attack_iterations: int = 20,
            verbose: bool = True
        ) -> Dict[str, float]:
            """
            맑은 보기와 맞서는 보기로 모형을 따진다.

            이는 적대적 강건하기의 두 핵심 자인 맑은 정확도와
            강건한 정확도를 함께 준다.

            파라메터:
            -----------
            test_loader : DataLoader
                시험 자료 실개
            attack_epsilon : float, 골라 씀
                따질 때 쓸 치기의 엡실론(기본값: 익힘과 같음)
            attack_iterations : int, 기본값=20
                따질 때의 PGD 회귀 횟수
                익힘보다 많아야 한다(익힘 7~10, 평가 20~100)
            verbose : bool, 기본값=True
                결과를 찍는다

            돌려주는 것:
            --------
            metrics : Dict[str, float]
                맑은 정확도, 강건한 정확도 따위
            """
            if attack_epsilon is None:
                attack_epsilon = self.epsilon

            self.model.eval()

            clean_correct = 0
            robust_correct = 0
            total = 0

            if verbose:
                pbar = tqdm(test_loader, desc="Evaluating")
            else:
                pbar = test_loader

            with torch.no_grad():
                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # 맑은 정확도
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    clean_correct += predicted.eq(labels).sum().item()

                    total += labels.size(0)

            # 강건성을 평가하려 맞서는 보기를 만든다
            # 치기 파라메터를 잠깐 바꾼다
            original_epsilon = self.epsilon
            original_num_iter = self.num_iter
            self.epsilon = attack_epsilon
            self.num_iter = attack_iterations

            if verbose:
                pbar = tqdm(test_loader, desc="Robust evaluation")
            else:
                pbar = test_loader

            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 맞서는 보기를 만든다
                adv_images = self._generate_adversarial_examples(images, labels)

                # 강건한 정확도
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = outputs.max(1)
                    robust_correct += predicted.eq(labels).sum().item()

            # 본디 파라메터를 되돌린다
            self.epsilon = original_epsilon
            self.num_iter = original_num_iter

            # 자를 셈한다
            clean_accuracy = clean_correct / total
            robust_accuracy = robust_correct / total

            metrics = {
                'clean_accuracy': clean_accuracy,
                'robust_accuracy': robust_accuracy,
                'accuracy_drop': clean_accuracy - robust_accuracy
            }

            if verbose:
                print("\n" + "=" * 60)
                print("평가 결과")
                print("=" * 60)
                print(f"맑은 정확도: {clean_accuracy:.2%}")
                print(f"강건한 정확도 (ε={attack_epsilon}): {robust_accuracy:.2%}")
                print(f"정확도 떨어짐: {metrics['accuracy_drop']:.2%}")
                print("=" * 60)

            return metrics

        def train(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int,
            optimizer: Optional[optim.Optimizer] = None,
            scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
            save_path: Optional[str] = None,
            eval_frequency: int = 1
        ) -> Dict[str, List[float]]:
            """
            온전한 맞서며 익히기 돌기.

            이것이 다음을 아우르는 으뜸 익힘 함수다:
            1. 여러 판 익히기
            2. 때때로 평가
            3. 모형 찰칵 담기
            4. 러닝 비율 짜임

            파라메터:
            -----------
            train_loader : DataLoader
                익힘 자료
            test_loader : DataLoader
                시험 자료
            epochs : int
                익힘 판 수
            optimizer : optim.Optimizer, 골라 씀
                가장 좋게 하는 개(기본값: 밀어 나감을 곁들인 SGD)
            scheduler : lr_scheduler, 골라 씀
                러닝 비율 짜임개
            save_path : str, 골라 씀
                가장 좋은 모형을 담을 자리
            eval_frequency : int, 기본값=1
                N판마다 따진다

            돌려주는 것:
            --------
            history : Dict[str, List[float]]
                익힘 이력(로스, 정확도)
            """
            # 기본 가장 좋게 하는 개: 밀어 나감을 곁들인 SGD
            if optimizer is None:
                optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=5e-4
                )

            # 기본 짜임개: 여러 스텝 러닝 비율 줄이기
            if scheduler is None:
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[int(0.5*epochs), int(0.75*epochs)],
                    gamma=0.1
                )

            # 익힘 이력을 좇는다
            history = {
                'train_loss': [],
                'train_accuracy': [],
                'clean_accuracy': [],
                'robust_accuracy': []
            }

            best_robust_acc = 0.0

            print(f"\n{epochs}판 동안 맞서며 익히기를 비롯한다")
            print(f"ε={self.epsilon}, PGD 스텝={self.num_iter}으로 익힌다")
            print("=" * 60)

            for epoch in range(1, epochs + 1):
                # 한 판 익힌다
                train_metrics = self.train_epoch(train_loader, optimizer, epoch)
                history['train_loss'].append(train_metrics['train_loss'])
                history['train_accuracy'].append(train_metrics['train_accuracy'])

                # 따진다
                if epoch % eval_frequency == 0:
                    eval_metrics = self.evaluate(test_loader, verbose=False)
                    history['clean_accuracy'].append(eval_metrics['clean_accuracy'])
                    history['robust_accuracy'].append(eval_metrics['robust_accuracy'])

                    print(f"{epoch}/{epochs}판:")
                    print(f"  익힘 로스: {train_metrics['train_loss']:.4f}")
                    print(f"  익힘 정확도: {train_metrics['train_accuracy']:.2%}")
                    print(f"  맑은 정확도: {eval_metrics['clean_accuracy']:.2%}")
                    print(f"  강건한 정확도: {eval_metrics['robust_accuracy']:.2%}")

                    # 가장 좋은 모형을 담는다
                    if save_path and eval_metrics['robust_accuracy'] > best_robust_acc:
                        best_robust_acc = eval_metrics['robust_accuracy']
                        torch.save(self.model.state_dict(), save_path)
                        print(f"  → 가장 좋은 모형을 담았다(강건한 정확도: {best_robust_acc:.2%})")

                # 러닝 비율을 고친다
                scheduler.step()

                print("-" * 60)

            print(f"\n익힘을 마쳤다!")
            print(f"가장 좋은 강건한 정확도: {best_robust_acc:.2%}")

            return history


    class TRADESTrainer(AdversarialTrainer):
        """
        TRADES: 강건성과 정확도의 맞바꿈

        TRADES은 여느 로스와 KL 발산을 아울러 맑은 정확도와 강건한 정확도를
        드러내 놓고 저울질한다.

        로스 함수:
        --------------
        L_TRADES = L_CE(f(x), y) + β·KL(f(x) || f(x_adv))

        where:
        - L_CE은 맑은 보기의 여느 엇갈린 엔트로피
        - KL은 맑은 미루어 봄과 맞서는 미루어 봄 사이의 KL 발산
        - β은 맞바꿈을 다룬다

        KL 항은 맑은 보기와 맞서는 보기에서 미루어 봄이 비슷해지도록 이끌어
        그 자리의 매끄러움을 북돋운다.

        살펴볼 거리: Zhang et al., "Theoretically Principled Trade-off between
                   Robustness and Accuracy" (ICML 2019)
        """

        def __init__(
            self,
            model: nn.Module,
            epsilon: float = 0.031,
            alpha: float = 0.007,
            num_iter: int = 10,
            beta: float = 6.0,
            device: Optional[torch.device] = None,
            norm: str = 'linf'
        ):
            """
            TRADES 익힘개의 첫자리를 잡는다.

            파라메터:
            -----------
            beta : float, 기본값=6.0
                여느 로스와 강건성 로스 사이의 맞바꿈 파라메터
                β이 클수록 강건성을 앞세운다
                β이 작을수록 맑은 정확도를 앞세운다
            """
            super().__init__(model, epsilon, alpha, num_iter, device, None, norm)
            self.beta = beta
            print(f"  TRADES β: {self.beta}")

        def train_epoch(
            self,
            train_loader: DataLoader,
            optimizer: optim.Optimizer,
            epoch: int,
            verbose: bool = True
        ) -> Dict[str, float]:
            """
            TRADES으로 한 판 익힌다.

            Loss: L_natural + β·KL(f(x) || f(x_adv))
            """
            self.model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            if verbose:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch} (TRADES)")
            else:
                pbar = train_loader

            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 맑은 보기로 앞으로 스텝
                logits_clean = self.model(images)
                loss_natural = F.cross_entropy(logits_clean, labels)

                # 맞서는 보기를 만든다
                adv_images = self._generate_adversarial_examples(images, labels)

                # 맞서는 보기로 앞으로 스텝
                logits_adv = self.model(adv_images)

                # 맑은 미루어 봄과 맞서는 미루어 봄의 KL 발산
                # KL(P||Q), 여기서 P=f(x), Q=f(x_adv)
                loss_robust = F.kl_div(
                    F.log_softmax(logits_adv, dim=1),
                    F.softmax(logits_clean, dim=1),
                    reduction='batchmean'
                )

                # 온 TRADES 로스
                loss = loss_natural + self.beta * loss_robust

                # 다듬기 한 스텝
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 자를 좇는다
                total_loss += loss.item()
                _, predicted = logits_clean.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if verbose:
                    avg_loss = total_loss / (batch_idx + 1)
                    accuracy = 100. * correct / total
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.3f}',
                        'acc': f'{accuracy:.2f}%'
                    })

            metrics = {
                'train_loss': total_loss / len(train_loader),
                'train_accuracy': correct / total
            }

            return metrics


    # 쓰는 보기
    if __name__ == "__main__":
        """
        맞서며 익히기를 보여 준다.
        """
        print("=" * 70)
        print("맞서며 익히기 보여 주기")
        print("=" * 70)
        print("\n이 글은 강건한 모형을 위한 맞서며 익히기를 보여 준다.")
        print("=" * 70)
    ```


**출력:**

```
======================================================================
맞서며 익히기 보여 주기
======================================================================

이 글은 강건한 모형을 위한 맞서며 익히기를 보여 준다.
======================================================================
```

## 2. 논의

`train_epoch` 의 바깥 루프는 표준 학습 루프와 같지만, 순전파에 넣는 것이 클린 이미지가 아니라 `_generate_adversarial_examples` 가 PGD로 만든 적대적 샘플이다. 이것이 min-max 문제의 두 층이다. 안쪽 PGD는 섭동 `delta` 에 대해 로스를 **키우고**, 바깥 옵티마이저는 그 적대적 샘플의 로스를 웨이트에 대해 **줄인다**. 미니배치마다 PGD를 `num_iter` 번 돌리므로 한 에폭이 일반 학습보다 대략 `num_iter + 1` 배 비싸다.

`_generate_adversarial_examples` 는 섭동을 만드는 동안 모델을 `eval()` 로 두고, 끝에서 **늘** `train()` 으로 되돌린다. 학습 중에는 이것이 맞지만, `evaluate` 도 같은 함수를 부르므로 강건 정확도를 재는 순전파가 학습 모드에서 이루어진다. 배치 정규화가 있는 모델이면 시험 배치의 통계를 쓰고 running 통계까지 바뀌므로, 실제로 쓸 때는 그 순전파 앞에서 `model.eval()` 을 다시 불러야 한다.

`TRADESTrainer` 는 로스를 $\mathcal{L}_{\text{CE}}(f(x), y) + \beta \cdot \mathrm{KL}(f(x) \,\|\, f(x_{\text{adv}}))$로 바꾼다. 다만 적대적 샘플은 부모 클래스의 함수로, 곧 KL이 아니라 크로스 엔트로피를 키우는 PGD로 만든다. 원래 TRADES는 안쪽 최대화에서도 KL 항을 키우므로, 이 구현은 그 단순화판이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떻게 되는지 설명하여라. 고친 코드를 돌려 학습 로스의 수렴에 어떤 일이 생기는지 적어라.

</div>

??? success "풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 그래디언트를 이미 있는 `.grad` 텐서에 덮어쓰지 않고 더하므로 그래디언트가 반복마다 쌓인다. 이는 사실상 러닝레이트에 쌓인 스텝 수를 곱하는 셈이라 최적화가 점점 크고 들쭉날쭉한 스텝을 밟게 된다. 학습 로스는 매끄럽게 수렴하는 대신 크게 출렁이거나 발산한다. 고치기는 쉽다. `loss.backward()`를 호출하기 전에 늘 그래디언트를 0으로 만든다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam(`torch.optim.Adam`, `lr=0.001`)으로 바꾸고 원래 것과 수렴 양상을 비교하여라. 두 로스 곡선을 한 그림에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 파라메터마다 적응적인 러닝레이트와 모멘텀 추정을 지니므로 초기 에폭에서 더 빨리 수렴하는 것이 보통이다. Adam의 로스 곡선은 첫 몇 에폭에서 더 가파르게 떨어지지만, 최적점 부근에서는 모멘텀을 곁들인 SGD보다 조금 더 출렁일 수 있다. 공정하게 비교하려면 무작위 시드와 에폭 수를 똑같이 두고 돌린다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 둘을 들고, 각각 어떻게 탐지하고 고칠지 설명하여라.

</div>

??? success "풀이"
    흔한 실패 양상은 이렇다. (1) **강건 과적합(robust overfitting)** -- 러닝레이트를 줄인 직후 테스트 강건 정확도가 정점을 찍은 뒤 학습 강건 정확도는 계속 오르는데 테스트 쪽은 떨어진다. 에폭마다 `evaluate`로 강건 정확도를 기록해 진단하고, 강건 정확도가 가장 높았던 체크포인트를 쓰거나(이 코드의 `save_path`가 그 일을 한다) 데이터 어그멘테이션과 웨이트 감쇠를 늘려 고친다. (2) **그래디언트 마스킹으로 강건성이 부풀려진다** -- PGD 스텝이 너무 적거나 한 스텝 공격으로 학습하면, 모델이 실제로 강건해지는 대신 학습에 쓴 공격의 그래디언트만 쓸모없게 만드는 길을 배운다. 평가 때 PGD 반복과 재시작을 늘렸을 때 강건 정확도가 크게 떨어지거나 블랙박스 공격이 더 잘 통하면 의심하고, 학습에 쓰는 `num_iter`를 늘리거나 AutoAttack으로 다시 평가해 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
조기 종료(early stopping)를 구현하여라. 에폭마다 평가 로스를 좇다가 열 에폭 연속으로 나아지지 않으면 학습을 멈춘다. 가장 좋은 모델 웨이트를 저장해 두었다가 되돌려라.

</div>

??? success "풀이"
    인내 카운터와 최고 로스 추적 변수를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 익힘 스텝 ...
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
    이러면 검증 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막는다.

## 정리하며

**다룬 것** — 적대적 학습

적대적 학습은 미니배치마다 PGD로 최악의 섭동을 찾고(안쪽 최대화), 그 적대적 샘플의 로스로 웨이트를 갱신한다(바깥 최소화).

핵심 클래스는 `AdversarialTrainer`, `TRADESTrainer`이며 앞의 연습문제 4개로 스스로 점검해 볼 수 있다.
