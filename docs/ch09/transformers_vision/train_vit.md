# 비전 트랜스포머 학습

비전 트랜스포머 학습 스크립트. 비전 트랜스포머를 학습시키고 합성곱 신경망 방식과 견주는 법을 보인다

트랜스포머에 바탕을 둔 구조는 자연어 처리를 뒤바꾸어 놓았다. 이 구현은 트랜스포머의 개념을 살피며, 갖가지 과제에서 최고 수준의 성능을 내게 하는 주의 얼개와 구조의 본을 보여 준다.

## 1. 코드

```python
"""
비전 트랜스포머 학습 스크립트
비전 트랜스포머를 학습시키고 합성곱 신경망 방식과 견주는 법을 보인다
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
from typing import Tuple, Dict
from vit_model import create_vit_tiny, create_vit_small, create_vit_base

# ========================================================================
# 메인
# ========================================================================


class Trainer:
    """비전 트랜스포머 모형을 위한 학습 도구"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   criterion: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """한 에포크 동안 학습한다"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 순전파
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 통계
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 진행 막대를 고친다
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, 
                val_loader: DataLoader,
                criterion: nn.Module) -> Dict[str, float]:
        """모형을 검증한다"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="[Validation]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             n_epochs: int = 10,
             learning_rate: float = 3e-4,
             weight_decay: float = 0.1) -> Dict[str, list]:
        """온전한 학습 고리"""
        
        # 준비
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 코사인 담금질 일정 조정기
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"\n{'='*60}")
        print(f"Training Vision Transformer")
        print(f"Device: {self.device}")
        print(f"Epochs: {n_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, n_epochs + 1):
            start_time = time.time()
            
            # 학습
            train_metrics = self.train_epoch(
                train_loader, criterion, optimizer, epoch
            )
            
            # 검증
            val_metrics = self.validate(val_loader, criterion)
            
            # 스케줄러 갱신
            scheduler.step()
            
            # 이력 기록
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # 요약 출력
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{n_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"{'-'*60}\n")
        
        return history


def get_data_loaders(data_dir: str = "./data",
                    batch_size: int = 128,
                    num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    학습을 위한 데이터 로더를 마련한다.
    표준 데이터 불리기 기법을 쓴다.
    """
    
    # 데이터를 불리는 학습 변환
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 검증 변환
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 보기로 CIFAR-10을 싣는다
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """주 학습 함수"""
    
    # 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    n_epochs = 50
    learning_rate = 3e-4
    
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size,
        num_workers=4
    )
    
    print("Creating model...")
    # 더 빠른 학습을 위해 ViT-Tiny를 쓴다
    model = create_vit_tiny(n_classes=10)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 학습
    trainer = Trainer(model, device=device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )
    
    # 모델을 저장한다
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, 'vit_checkpoint.pth')
    
    print("\nTraining complete! Model saved to 'vit_checkpoint.pth'")


if __name__ == "__main__":
    main()
```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
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
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.

## 정리하며

**다룬 것** — 비전 트랜스포머 학습

학습 루프는 표준적인 PyTorch 패턴을 따른다.

핵심 클래스는 `Trainer`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
