# 맞서며 익히기

62.4 묶음: 맞서며 익히기 - 한발 더. 이 묶음은 가장 잘 듣는 막이인 맞서며 익히기를 짜 놓았다

맞섬에 든든하기는 안전이 걸린 자리에 신경 그물을 내놓을 때 종요로운 걱정거리다. 이 짜보기는 작은 흔듦이 어떻게 모형을 속이는지, 막이는 어떻게 지을 수 있는지 보이며 맞섬에 든든하기의 깨침을 드러낸다.

## 코드

```python
"""
62.4 묶음: 맞서며 익히기 - 한발 더

이 묶음은 맞서는 치기를 막는 가장 잘 듣는 길인 맞서며 익히기를 짜 놓았다.
맞서며 익히기는 익힘 꾸러미에 맞서는 보기를 불려 넣어
모형이 흔듦에 든든해지게 한다.

수학 밑그림:
=======================

여느 익힘:
-----------------
여느 겪은 무릅씀 가장 작게 하기(ERM):
    
    min_θ E_{(x,y)~D}[L(θ, x, y)]

이는 익힘 분포의 맞음을 다듬지만 맞서는 흔듦은
헤아리지 않는다.

맞서며 익히기(든든하게 다듬기):
------------------------------------------
맞서며 익히기는 가장 작게-가장 크게 하는 다듬기 문제를 푼다:

    min_θ E_{(x,y)~D}[ max_{||δ||≤ε} L(θ, x + δ, y) ]

안쪽 가장 크게 하기: 가장 나쁜 흔듦을 찾는다(치기)
바깥 가장 작게 하기: 그에 든든하도록 모형을 익힌다

이는 익힘 보기마다 ε 공 안에서 가장 나쁜 잃음을 가장 작게 하려는
든든하게 다듬기 문제다.

풀이:
==============
1. **안쪽 가장 크게**: 익힘 보기마다 잃음을 가장 크게 하는
   맞서는 흔듦을 찾는다(예산 ε 안의 가장 센 치기)
2. **바깥 가장 작게**: 가장 나쁜 잃음이 작아지도록 매개변수를 고친다
3. **결과**: 모형이 익힘 자료의 ε 공 안에서 든든해진다

참으로 짜기:
=========================
안쪽 가장 크게 하기를 다룰 수 없으므로 PGD으로 어림한다:

알고리즘(PGD 맞서며 익히기):
-------------------------------------
판마다:
    잔 묶음 (x, y)마다:
        1. 맞서는 보기를 만든다:
           x_adv = PGD(x, y, ε, α, K)  [K걸음 PGD]
        
        2. 맞서는 보기의 잃음을 셈한다:
           L_adv = L(θ, x_adv, y)
        
        3. 매개변수를 고친다:
           θ ← θ - η∇_θ L_adv

이를 "PGD 바탕 맞서며 익히기" 또는 "PGD-AT"이라 한다.

고갱이 하이퍼파라미터:
===================
1. **ε(엡실론)**: 익힘의 흔듦 예산
   - 든든함의 켜를 정한다
   - 흔함: CIFAR-10에서 ε = 8/255 ≈ 0.031
   
2. **K(PGD 걸음)**: PGD 되돌이 횟수
   - 걸음이 많을수록 익힘 때의 치기가 세다
   - 흔함: K = 10(익힘), K = 20(따짐)
   
3. **α(걸음 크기)**: PGD 걸음 크기
   - 흔히 α = 2ε/K 또는 α = 2.5ε/K

맞서며 익히기의 갈래:
=================================

1. **여느 PGD-AT**(매드리 등, 2018):
   - PGD으로 만든 맞서는 보기를 쓴다
   - 가장 세지만 셈이 비싸다

2. **TRADES**(장 등, 2019):
   - 든든함과 맞음 사이의 이론에 닿는 맞바꿈
   - 맑은 맞음과 든든한 맞음을 저울질한다
   - 잃음: L_nat + β·KL(f(x)||f(x_adv))
   
3. **MART**(왕 등, 2020):
   - 잘못 가름을 아는 맞서며 익히기
   - 잘못 가른 보기에 힘을 모은다
   - 잘못 가른 것에는 북돋운 엇갈린 엔트로피, 옳게 가른 것에는 KL 갈림

4. **빠른 맞서며 익히기**(웡 등, 2020):
   - 잘 들도록 한 걸음 FGSM을 쓴다
   - 훨씬 빠르나 조금 덜 든든하다
   
5. **값싼 맞서며 익히기**(샤파히 등, 2019):
   - 되돌아가기의 기울기를 되쓴다
   - 여느 익힘과 값이 같다

맞음과 든든함의 맞바꿈:
==============================
맞서며 익히면 흔히 맑은 맞음이 떨어진다:
- 여느 익힘: 맑은 맞음 약 95%, 든든한 맞음 약 0%
- 맞서며 익히기: 맑은 맞음 약 85%, 든든한 맞음 약 50%

이 맞바꿈은 밑바탕부터 있는 것이며 잘 알려져 있다. TRADES은 두 목표를
드러내 놓고 저울질하여 이를 눅이려 한다.

참으로 헤아릴 것:
========================
1. **셈 값**: PGD 탓에 여느 익힘보다 7~10배 느리다
2. **지나치게 맞추기**: 조심스러운 다독임과 자료 불리기가 있어야 한다
3. **무너지듯 지나친 맞춤**: 잃음이 갑자기 치솟을 수 있다. 일찍 멈추기를 쓴다
4. **따지기**: 센 치기로 따져야 한다(PGD, 오토어택)

지은이: 가르침 감
날짜: 2025년 11월
어려움: 한발 더
먼저 알 것: PGD(62.2 묶음), 신경 그물 익히기
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
    든든한 모형을 위한 맞서며 익히기
    
    이 갈래는 PGD으로 만든 맞서는 보기로 맞서며 익히기를 짜 놓았다.
    모형은 익힘 보기마다 ε 공 안에서 가장 나쁜 잃음을 가장 작게 하도록
    익힌다.
    
    수학 꼴:
    -------------------------
    든든하게 다듬는 문제를 푼다:
    
        min_θ E_{(x,y)}[ max_{||δ||≤ε} L(θ, x + δ, y) ]
    
    안쪽 가장 크게 하기를 K걸음 PGD으로 어림한다:
    
        x_adv = PGD(x, y, ε, α, K)
        θ ← θ - η∇_θ L(θ, x_adv, y)
    
    속성:
    -----------
    model : nn.Module
        익힐 모형
    epsilon : float
        흔듦 예산
    alpha : float
        PGD 걸음 크기
    num_iter : int
        PGD 되돌이 횟수
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
        
        매개변수:
        -----------
        model : nn.Module
            든든하게 익힐 모형
        epsilon : float, 기본값=0.031
            맞서며 익히기의 흔듦 예산
            여느 값: CIFAR-10에서 ε = 8/255 ≈ 0.031
        alpha : float, 기본값=0.007
            익히는 동안의 PGD 걸음 크기
            흔히 α = 2ε/K이고 K은 num_iter이다
        num_iter : int, 기본값=10
            익히는 동안의 PGD 되돌이 횟수
            빠르려면 적게, 세려면 많게
            익힘: 7~10, 따짐: 20~100
        device : torch.device, 골라 씀
            셈할 장치
        loss_fn : nn.Module, 골라 씀
            잃음 함수(기본값: CrossEntropyLoss)
        norm : str, 기본값='linf'
            흔듦의 노름('linf' 또는 'l2')
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
        print(f"  PGD 되돌이: {self.num_iter}")
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
        가장 나쁜 흔듦을 PGD으로 어림해 찾는다.
        
        알고리즘:
        ----------
        1. 첫자리: δ^(0) ~ Uniform[-ε, ε](아무렇게나) 또는 δ^(0) = 0
        2. t = 1에서 K까지:
             δ^(t) = Π_ε(δ^(t-1) + α·sign(∇_δ L(θ, x+δ^(t-1), y)))
        3. x + δ^(K)을 돌려준다
        
        여기서 Π_ε은 ε 공으로의 되비춤이다.
        
        매개변수:
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
        # 치기를 만들 때는 모형을 따짐 모드로 둔다
        # (치는 동안 묶음 잣대 잡기와 드롭아웃은 따짐 모드여야 한다)
        self.model.eval()
        
        # 흔듦의 첫자리를 잡는다
        if random_init:
            # ε 공 안의 아무 첫자리
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            delta = delta.to(self.device)
        else:
            delta = torch.zeros_like(images).to(self.device)
        
        # delta에 기울기를 켠다
        delta.requires_grad = True
        
        # PGD 되돌이
        for _ in range(self.num_iter):
            # 맞서는 그림을 셈한다
            adv_images = images + delta
            
            # 앞으로 걸음
            outputs = self.model(adv_images)
            
            # 잃음을 셈한다
            loss = self.loss_fn(outputs, labels)
            
            # 되돌아 걸음
            self.model.zero_grad()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            
            # 기울기 오름 걸음
            if self.norm == 'linf':
                delta_grad = delta.grad.detach()
                delta = delta + self.alpha * torch.sign(delta_grad)
                # ε 공으로 되비춘다
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:  # l2
                delta_grad = delta.grad.detach()
                # 기울기의 잣대를 맞춘다
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
        묶음마다:
        1. 맞서는 보기를 만든다(안쪽 가장 크게)
        2. 맞서는 보기의 잃음을 셈한다
        3. 모형 매개변수를 고친다(바깥 가장 작게)
        
        알고리즘:
        ----------
        묶음 (x, y)마다:
            # 안쪽 가장 크게 하기(PGD으로 어림)
            x_adv = PGD(x, y, ε, α, K)
            
            # 맞섬 잃음을 셈한다
            L_adv = L(θ, x_adv, y)
            
            # 바깥 가장 작게 하기
            θ ← θ - η∇_θ L_adv
        
        매개변수:
        -----------
        train_loader : DataLoader
            익힘 자료 실개
        optimizer : optim.Optimizer
            모형 매개변수의 가장 좋게 하는 개
        epoch : int
            이제의 판 수
        verbose : bool, 기본값=True
            나아가는 것을 찍는다
        
        돌려주는 것:
        --------
        metrics : Dict[str, float]
            익힘 자(잃음, 맞음)
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
            
            # 맞서는 보기로 앞으로 걸음
            outputs = self.model(adv_images)
            loss = self.loss_fn(outputs, labels)
            
            # 되돌아 걸음과 다듬기
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
        
        이는 맞섬에 든든하기의 두 고갱이 자인 맑은 맞음과
        든든한 맞음을 함께 준다.
        
        매개변수:
        -----------
        test_loader : DataLoader
            시험 자료 실개
        attack_epsilon : float, 골라 씀
            따질 때 쓸 치기의 엡실론(기본값: 익힘과 같음)
        attack_iterations : int, 기본값=20
            따질 때의 PGD 되돌이 횟수
            익힘보다 많아야 한다(익힘 7~10, 따짐 20~100)
        verbose : bool, 기본값=True
            결과를 찍는다
        
        돌려주는 것:
        --------
        metrics : Dict[str, float]
            맑은 맞음, 든든한 맞음 따위
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
                
                # 맑은 맞음
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                clean_correct += predicted.eq(labels).sum().item()
                
                total += labels.size(0)
        
        # 든든함을 따지려 맞서는 보기를 만든다
        # 치기 매개변수를 잠깐 바꾼다
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
            
            # 든든한 맞음
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = outputs.max(1)
                robust_correct += predicted.eq(labels).sum().item()
        
        # 본디 매개변수를 되돌린다
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
            print("따짐 결과")
            print("=" * 60)
            print(f"맑은 맞음: {clean_accuracy:.2%}")
            print(f"든든한 맞음 (ε={attack_epsilon}): {robust_accuracy:.2%}")
            print(f"맞음 떨어짐: {metrics['accuracy_drop']:.2%}")
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
        2. 때때로 따지기
        3. 모형 찰칵 담기
        4. 배움 비율 짜임
        
        매개변수:
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
            배움 비율 짜임개
        save_path : str, 골라 씀
            가장 좋은 모형을 담을 자리
        eval_frequency : int, 기본값=1
            N판마다 따진다
        
        돌려주는 것:
        --------
        history : Dict[str, List[float]]
            익힘 자취(잃음, 맞음)
        """
        # 기본 가장 좋게 하는 개: 밀어 나감을 곁들인 SGD
        if optimizer is None:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=5e-4
            )
        
        # 기본 짜임개: 여러 걸음 배움 비율 줄이기
        if scheduler is None:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(0.5*epochs), int(0.75*epochs)],
                gamma=0.1
            )
        
        # 익힘 자취를 좇는다
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'clean_accuracy': [],
            'robust_accuracy': []
        }
        
        best_robust_acc = 0.0
        
        print(f"\n{epochs}판 동안 맞서며 익히기를 비롯한다")
        print(f"ε={self.epsilon}, PGD 걸음={self.num_iter}으로 익힌다")
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
                print(f"  익힘 잃음: {train_metrics['train_loss']:.4f}")
                print(f"  익힘 맞음: {train_metrics['train_accuracy']:.2%}")
                print(f"  맑은 맞음: {eval_metrics['clean_accuracy']:.2%}")
                print(f"  든든한 맞음: {eval_metrics['robust_accuracy']:.2%}")
                
                # 가장 좋은 모형을 담는다
                if save_path and eval_metrics['robust_accuracy'] > best_robust_acc:
                    best_robust_acc = eval_metrics['robust_accuracy']
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  → 가장 좋은 모형을 담았다(든든한 맞음: {best_robust_acc:.2%})")
            
            # 배움 비율을 고친다
            scheduler.step()
            
            print("-" * 60)
        
        print(f"\n익힘을 마쳤다!")
        print(f"가장 좋은 든든한 맞음: {best_robust_acc:.2%}")
        
        return history


class TRADESTrainer(AdversarialTrainer):
    """
    TRADES: 든든함과 맞음의 맞바꿈
    
    TRADES은 여느 잃음과 KL 갈림을 아울러 맑은 맞음과 든든한 맞음을
    드러내 놓고 저울질한다.
    
    잃음 함수:
    --------------
    L_TRADES = L_CE(f(x), y) + β·KL(f(x) || f(x_adv))
    
    where:
    - L_CE은 맑은 보기의 여느 엇갈린 엔트로피
    - KL은 맑은 미루어 봄과 맞서는 미루어 봄 사이의 KL 갈림
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
        
        매개변수:
        -----------
        beta : float, 기본값=6.0
            여느 잃음과 든든함 잃음 사이의 맞바꿈 매개변수
            β이 클수록 든든함을 앞세운다
            β이 작을수록 맑은 맞음을 앞세운다
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
            
            # 맑은 보기로 앞으로 걸음
            logits_clean = self.model(images)
            loss_natural = F.cross_entropy(logits_clean, labels)
            
            # 맞서는 보기를 만든다
            adv_images = self._generate_adversarial_examples(images, labels)
            
            # 맞서는 보기로 앞으로 걸음
            logits_adv = self.model(adv_images)
            
            # 맑은 미루어 봄과 맞서는 미루어 봄의 KL 갈림
            # KL(P||Q), 여기서 P=f(x), Q=f(x_adv)
            loss_robust = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean, dim=1),
                reduction='batchmean'
            )
            
            # 온 TRADES 잃음
            loss = loss_natural + self.beta * loss_robust
            
            # 다듬기 한 걸음
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
    print("\n이 글은 든든한 모형을 위한 맞서며 익히기를 보여 준다.")
    print("\n붙임말: 자료 얹기와 모형 잔손질에 utils.py이 있어야 한다.")
    print("=" * 70)```

## 논의

익힘 돌기는 여느 PyTorch 결을 따른다. 앞으로 걸어 미루어 봄을 셈하고, 잃음을 재고, 되돌아 걸어 기울기를 셈하고, 가장 좋게 하는 개로 매개변수를 고친다. 판에 걸쳐 자를 좇으면 모여 가는 결이 드러나고 덜 맞추기나 지나치게 맞추기 같은 탈을 짚어내는 데 도움이 된다.

여기서 보인 결은 더 얽힌 자리로도 자연스레 넓어진다. 하이퍼파라미터, 얼개의 갈래, 다른 자료 꾸러미로 해 보면 앎이 깊어지고 모형 지킴 일에 손에 잡히는 느낌이 붙는다.

## 익힘 문제

**익힘 1.**
익힘 돌기에서 `optimizer.zero_grad()` 부름을 없애면 어떻게 되는지 밝혀라. 고친 코드를 돌려 익힘 잃음이 모이는 데 어떤 일이 생기는지 적어라.

??? success "익힘 1 풀이"
    `optimizer.zero_grad()`이 없으면 PyTorch가 새 기울기를 이미 있는 `.grad` 텐서에 갈음하지 않고 더하므로 기울기가 되돌이마다 쌓인다. 이는 사실상 배움 비율에 쌓인 걸음 수를 곱하는 셈이라 다듬기가 점점 크고 들쭉날쭉한 걸음을 밟게 된다. 익힘 잃음은 매끄럽게 모이는 대신 크게 출렁이거나 터진다. 고치기는 쉽다. `loss.backward()`을 부르기 앞서 늘 기울기를 0으로 만든다.

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
