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
import torch.nn.functional as F
from typing import Tuple

# --- vit_model ---
"""
비전 트랜스포머(ViT) 구현
이미지 분류에서 합성곱 신경망과 트랜스포머를 잇는다
"""


# ========================================================================
# 메인
# ========================================================================


class PatchEmbedding(nn.Module):
    """
    그림을 조각으로 바꾸고 임베딩으로 사영한다.
    합성곱 신경망 방식의 입력과 트랜스포머 처리를 잇는 다리이다.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 편 조각의 선형 사영 (합성곱 신경망의 합성곱과 비슷하다)
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, channels, height, width)
        반환값:
            (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    트랜스포머에서 온 다중 머리 자기 주의 얼개.
    모형이 그림의 여러 부분에 한꺼번에 주의하게 해 준다.
    """
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, seq_len, embed_dim)
        반환값:
            (batch_size, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        # Q, K, V를 만든다
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 주의 점수
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 값에 어텐션 적용
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    트랜스포머 블록에서 쓰는 순전파 신경망.
    비선형성과 특징 변환을 준다.
    """
    def __init__(self, embed_dim: int = 768, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    자기 주의와 다층 퍼셉트론을 갖춘 표준 트랜스포머 인코더 블록.
    """
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, 
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 잔차 연결을 곁들인 주의
        x = x + self.attn(self.norm1(x))
        # 잔차 연결을 곁들인 다층 퍼셉트론
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    비전 트랜스포머(ViT) 모형.
    
    핵심 혁신:
    1. 그림을 조각의 수열로 다룬다
    2. (본디 자연어 처리에서 온) 트랜스포머 인코더를 이미지 분류에 쓴다
    3. 합성곱 신경망 방식의 입력 처리와 트랜스포머 구조를 잇는다
    """
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 n_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # 조각 임베딩 층 (그림에서 토큰으로 잇는 다리)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # 분류 토큰 (학습되며 수열 앞에 붙인다)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 위치 임베딩 (학습되는 것)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 트랜스포머 인코더 블록
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # 분류 머리
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # 가중치 초기화
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        인수:
            x: (batch_size, channels, height, width)
        반환값:
            (batch_size, n_classes)
        """
        B = x.shape[0]
        
        # 그림을 조각 임베딩으로 바꾼다
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # 분류 토큰을 앞에 붙인다
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # 위치 임베딩을 더한다
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 트랜스포머 블록을 적용한다
        for block in self.blocks:
            x = block(x)
            
        # 분류 토큰으로 분류한다
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 분류 토큰만 쓴다
        x = self.head(cls_token_final)
        
        return x


def create_vit_tiny(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Tiny: 매개변수 500만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, 
        depth=12, n_heads=3, n_classes=n_classes
    )


def create_vit_small(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Small: 매개변수 2200만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384,
        depth=12, n_heads=6, n_classes=n_classes
    )


def create_vit_base(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Base: 매개변수 8600만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768,
        depth=12, n_heads=12, n_classes=n_classes
    )


def create_vit_large(n_classes: int = 1000) -> VisionTransformer:
    """ViT-Large: 매개변수 3억 700만"""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024,
        depth=24, n_heads=16, n_classes=n_classes
    )


if __name__ == "__main__":
    # 사용 예
    model = create_vit_base(n_classes=10)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

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

??? success "풀이"
    역전파는 기존 grad에 경사를 더하므로 이전 갱신 시점의 경사가 계속 남을 수 있다. 이는 일반적으로 러닝레이트에 단계 수를 곱하는 것과 같지 않다. 의도적 경사 누적은 로스 웨이트·갱신 간격·초기화 시점을 정한 방법이다. 로스가 반드시 발산한다고 단정할 수는 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이미 AdamW인 기본 설정을 Adam과 비교하려면 무엇을 통제해야 하는가?

</div>

??? success "풀이"
    두 옵티마이저의 감쇠 방식·기본값이 다르므로 lr만 바꾼 비교의 의미를 먼저 정한다. 같은 초기 모델·데이터 분할·학습 예산에서 각각 검증 러닝레이트를 선택하고 어그멘테이션·정밀도·스케줄을 기록한다. 테스트 분할을 그 선택에 사용하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
검증 결과를 해석할 때 이 데이터 로더의 두 가지 제약을 설명하라.

</div>

??? success "풀이"
    공식 테스트를 에포크별 검증으로 써 최종 평가의 독립성이 사라질 수 있다. 또한 32→224 변환과 학습·평가 자르기 차이가 지표에 영향을 준다. 별도 검증 분할과 목적에 맞는 전처리를 정하고 학습·평가 모드가 다른 정확도 격차를 곧바로 과적합의 크기로 해석하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 조각이 실제 최고 웨이트를 보존하는지 검토하고 Trainer에 연결하기 위한 조건을 설명하라.

</div>

??? success "풀이"
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
