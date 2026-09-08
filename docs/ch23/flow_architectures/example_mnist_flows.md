# MNIST 흐름 보기

MNIST 고르게 하는 흐름. 그림 만들어 내기를 보이려 MNIST 숫자에 고르게 하는 흐름을 익힌다.

고르게 하는 흐름은 뒤집을 수 있는 바꿈으로 정확한 가능도 셈하기를 준다. 이 짜기는 깊은 배움 개념을 보이며, 일대일 대응의 차례를 거쳐 단순한 분포가 복잡한 분포로 어떻게 바뀌는지 드러낸다.

## 1. 코드

```python
"""
MNIST 고르게 하는 흐름

그림 만들어 내기를 보이려 MNIST 숫자에 고르게 하는 흐름을 익힌다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ========================================================================
# 메인
# ========================================================================

from flow_utils import BaseDistribution, FlowSequence
from coupling_flows import CouplingLayer, BatchNorm


class MNISTFlow:
    """MNIST 흐름 익히기의 감개 갈래."""
    
    def __init__(self, n_layers: int = 8, hidden_dim: int = 256,
                 batch_size: int = 128, lr: float = 1e-4, device: str = None):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lr = lr
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 모형을 세운다
        dim = 28 * 28  # MNIST 그림 크기
        flows = []
        
        for i in range(n_layers):
            # 가림막을 번갈아 쓴다
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[dim // 2:] = 1
            else:
                mask[:dim // 2] = 1
            
            flows.append(CouplingLayer(dim, hidden_dim, mask))
            flows.append(BatchNorm(dim))
        
        base_dist = BaseDistribution(dim)
        self.model = FlowSequence(flows, base_dist).to(self.device)
        
        # 최적화기
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_dataloader(self, train: bool = True):
        """MNIST 자료 불러오개를 만든다."""
        # 양자화를 되돌리고 [0, 1]으로 고른다
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x + torch.rand_like(x) / 256.))  # 양자화 되돌리기
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )
    
    def train_epoch(self, dataloader):
        """한 에폭을 학습한다."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, _ in pbar:
            images = images.to(self.device)
            images = images.view(images.shape[0], -1)  # 편다
            
            # 음의 로그가능도를 계산한다
            log_prob = self.model.log_prob(images)
            loss = -log_prob.mean()
            
            # 차원마다의 비트 잣대를 더한다
            bpd = loss / (28 * 28 * np.log(2))
            
            # 최적화
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'bpd': f'{bpd.item():.4f}'})
        
        return total_loss / num_batches
    
    def sample_images(self, n_samples: int = 64):
        """표본 그림을 만든다."""
        self.model.eval()
        
        with torch.no_grad():
            samples = self.model.sample(n_samples, device=self.device)
            samples = samples.view(n_samples, 1, 28, 28)
            samples = torch.clamp(samples, 0, 1)
        
        return samples
    
    def train(self, n_epochs: int = 50, save_interval: int = 10):
        """흐름 모델을 익힌다."""
        dataloader = self.get_dataloader(train=True)
        
        os.makedirs('samples', exist_ok=True)
        losses = []
        
        print(f"\nTraining for {n_epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            avg_loss = self.train_epoch(dataloader)
            losses.append(avg_loss)
            
            print(f"Average loss: {avg_loss:.4f}")
            
            # 표본 만들기
            if epoch % save_interval == 0 or epoch == 1:
                print("Generating samples...")
                samples = self.sample_images(64)
                self.visualize_samples(samples, f'samples/epoch_{epoch:04d}.png')
            
            # 검사점 저장
            if epoch % 25 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # 학습 손실 그리기
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('MNIST Flow Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=150)
        plt.close()
        
        print("\nTraining complete!")
        
        return losses
    
    def visualize_samples(self, samples, filename='samples.png'):
        """만든 표본을 그려 본다."""
        from torchvision.utils import make_grid
        
        grid = make_grid(samples, nrow=8, padding=2)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid[0].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved samples to {filename}")
    
    def save_checkpoint(self, filename: str):
        """모델 되짚기 지점을 갈무리한다."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Saved checkpoint to {filename}")
    
    def load_checkpoint(self, filename: str):
        """모델 되짚기 지점을 불러온다."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")


import numpy as np

def main():
    """으뜸 익히기 각본."""
    print("=" * 50)
    print("MNIST Normalizing Flow Training")
    print("=" * 50)
    
    # 초기화한다
    mnist_flow = MNISTFlow(
        n_layers=8,
        hidden_dim=256,
        batch_size=128,
        lr=1e-4
    )
    
    # 학습
    mnist_flow.train(n_epochs=50, save_interval=10)
    
    # 마지막 표본을 만든다
    print("\nGenerating final samples...")
    samples = mnist_flow.sample_images(64)
    mnist_flow.visualize_samples(samples, 'final_samples.png')
    
    # 모델을 저장한다
    mnist_flow.save_checkpoint('mnist_flow_final.pt')
    
    print("\nDone! Check the 'samples' folder for generated images.")


if __name__ == "__main__":
    main()```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 기계 배움 일에 대한 실전 직관이 선다.

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

**다룬 것** — MNIST 흐름 보기

학습 루프는 표준적인 PyTorch 패턴을 따른다.

고갱이 갈래는 `MNISTFlow`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
