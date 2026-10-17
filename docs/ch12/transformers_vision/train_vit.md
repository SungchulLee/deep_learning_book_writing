# 비전 트랜스포머 학습

ViT를 CIFAR-10 분류에 학습하는 Trainer와 데이터 로더 예제다. 이 페이지는 학습 절차를 설명하며 CNN을 실제로 학습·비교하거나 사전 학습 ViT를 불러오지는 않는다.

## 1. 코드

```python
"""
비전 트랜스포머 학습 스크립트
비전 트랜스포머를 학습시키고 컨볼루션 뉴럴 넷트웍 방식과 견주는 법을 보인다
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

train_epoch는 학습 모드에서 교차 엔트로피·역전파·AdamW 갱신을 수행하고 validate는 eval·no_grad로 평가한다. 정확도는 표본 수로 집계하지만 로스는 배치 평균의 단순 평균이라 마지막 배치 크기가 다르면 전체 표본 평균과 다르다. 빈 로더는 처리하지 않는다.

train은 에포크마다 CosineAnnealingLR을 진행하며 별도의 워밍업은 없다. 모든 계수에 같은 weight_decay를 적용한다. 반환 history에는 관측 지표만 있고 가장 좋은 모델의 저장·복원은 없다. main의 체크포인트도 마지막 웨이트와 history만 저장해 옵티마이저·스케줄러·난수·전처리 설정을 보존하지 않는다.

CIFAR-10의 32×32 이미지를 224×224 크기로 변환한다. 학습의 RandomResizedCrop은 확대·자르기를, 평가는 Resize(256)·CenterCrop(224)를 사용해 서로 다른 입력 분포가 된다. 224 패치 모델을 사용하는 선택이지 원래 해상도 학습과 같은 실험은 아니다.

val_dataset은 CIFAR-10의 공식 테스트 분할이다. 에포크별 지표를 보며 모델·설정을 선택하면 테스트가 검증 역할을 하므로 독립 최종 평가로 사용할 수 없다. 학습 분할에서 검증을 따로 나누는 설계가 필요하다. 기본 Trainer 장치는 CUDA를 가정하지만 main은 사용 가능 여부에 따라 선택한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
zero_grad를 생략할 때의 경사 누적과 의도적인 누적 학습을 구분하라.

</div>

??? success "연습문제 1 풀이"
    역전파는 기존 grad에 경사를 더하므로 이전 갱신 시점의 경사가 계속 남을 수 있다. 이는 일반적으로 러닝레이트에 단계 수를 곱하는 것과 같지 않다. 의도적 경사 누적은 로스 웨이트·갱신 간격·초기화 시점을 정한 방법이다. 로스가 반드시 발산한다고 단정할 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이미 AdamW인 기본 설정을 Adam과 비교하려면 무엇을 통제해야 하는가?

</div>

??? success "연습문제 2 풀이"
    두 옵티마이저의 감쇠 방식·기본값이 다르므로 lr만 바꾼 비교의 의미를 먼저 정한다. 같은 초기 모델·데이터 분할·학습 예산에서 각각 검증 러닝레이트를 선택하고 어그멘테이션·정밀도·스케줄을 기록한다. 테스트 분할을 그 선택에 사용하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
검증 결과를 해석할 때 이 데이터 로더의 두 가지 제약을 설명하라.

</div>

??? success "연습문제 3 풀이"
    공식 테스트를 에포크별 검증으로 써 최종 평가의 독립성이 사라질 수 있다. 또한 32→224 변환과 학습·평가 자르기 차이가 지표에 영향을 준다. 별도 검증 분할과 목적에 맞는 전처리를 정하고 학습·평가 모드가 다른 정확도 격차를 곧바로 과적합의 크기로 해석하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 조각이 실제 최고 웨이트를 보존하는지 검토하고 Trainer에 연결하기 위한 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래의 state_dict().copy()는 사전만 얕게 복사해 텐서 저장소를 공유할 수 있으므로 학습 중 최고 상태가 덮여 바뀔 수 있다. 독립적인 텐서 복사나 파일 저장이 필요하다. 조각은 외부 evaluate·num_epochs를 가정하고 학습 갱신도 보여 주지 않는다.
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
    인내 횟수를 채울 때만 복원하므로 전체 에포크가 끝날 때도 최고 웨이트를 선택할지 정해야 한다. 모델 모드·장치와 검증 분할을 맞추고 학습 재시작용 상태는 별도로 저장한다.

## 정리하며

Trainer의 갱신·평가 흐름과 데이터 분할을 연결한다. 이 예제는 새 ViT를 학습하며 테스트 분할을 검증으로 사용한다는 제약이 있다. 마지막 체크포인트와 최고 모델·학습 재시작 상태를 구분하고 조기 종료의 독립 웨이트 복사를 확인한다.
