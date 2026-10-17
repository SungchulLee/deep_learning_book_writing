# MNIST 정규화 흐름 예제

이 예제는 MNIST 이미지를 784차원 벡터로 펼치고 아핀 커플링과 흐름용 정규화 레이어를 쌓아 학습하는 절차를 보여 준다. 가역 흐름에서 밀도를 계산하려면 레이어의 방향·통계 처리·이미지 전처리의 확률적 의미까지 맞아야 한다. 표시된 코드의 학습 로스와 이미지 파일이 이 조건을 자동으로 충족하는 것은 아니다.

## 1. 학습·생성 코드

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
    """MNIST 흐름 익히기의 감개 클래스."""
    
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
        
        # 학습 로스 그리기
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
    main()
```

기본은 커플링 8개와 BatchNorm 8개, 은닉 256, 배치 128, Adam 러닝레이트 0.0001이며 main은 50에포크를 요청한다. 로컬 flow_utils와 coupling_flows를 import하므로 실행 환경에서 이 모듈들이 검색되어야 한다. numpy는 클래스 정의 아래에서 import하지만 main 실행 전에 수행되므로 그 위치 자체가 np 참조 오류를 만들지는 않는다.

입력은 `(B,1,28,28)`에서 `(B,784)`로 펼친다. 로컬 CouplingLayer는 전체 784차원을 받는 스케일·이동 MLP 두 개를 사용하고 한쪽 좌표를 마스크로 0 처리한다. 앞 절의 절반 차원 MLP 구현과 다르다. 기본 전체 학습 파라미터는 7,504,392개이며 마스크·running 통계 버퍼는 제외한 값이다.

## 2. Dequantization과 bits/dim

8비트 정수 픽셀 k에 대한 흔한 연속화는 u~Uniform(0,1)을 뽑아 y=(k+u)/256으로 만드는 것이다. 정수 관측의 우도는 각 픽셀 구간에 연속 밀도를 적분한 질량이고, 균일 잡음을 사용한 평균 연속 로그밀도는 이에 대한 하한을 제공한다. 이때 D개 픽셀의 음의 로그밀도 NLL에서 통상적인 8비트 bits/dim 상한을 계산하면 8+NLL/(D ln 2)가 된다.

본문 코드는 ToTensor의 k/255에 u/256을 더한다. 따라서 위 (k+u)/256과 다른 위치의 구간을 사용하고 흰 픽셀은 1을 넘을 수 있다. 연속 흐름 입력으로 1 초과가 금지되는 것은 아니지만 표준 256개 픽셀 구간의 우도로 바로 해석할 수는 없다. 또한 출력 변수 bpd는 NLL/(784 ln 2)만 계산해 위 8비트 보정 8이 없다. 이 값을 정수 MNIST의 통상적인 bits/dim 결과와 직접 비교해서는 안 된다.

[RealNVP 논문](https://arxiv.org/abs/1605.08803)과 [정규화 흐름 리뷰](https://jmlr.org/papers/v22/19-1028.html)는 연속 밀도와 이미지 우도 평가의 관계를 다룬다. 로그밀도는 좌표 스케일에 의존하므로 전처리와 보정 항을 함께 보고한다.

## 3. 학습 모드 정규화의 한계

로컬 BatchNorm.inverse는 학습 모드에서 현재 입력 배치의 평균·분산을 계산해 정규화한다. 이 통계를 상수로 취급한 로그 스케일을 반환하지만 실제 통계가 배치에 의존하는 연산 전체의 표본별 야코비안을 나타내지는 않는다. 배치 평균을 제거하는 연산은 배치 전체의 공통 이동 정보를 잃기도 한다. 따라서 이 학습 로스를 엄밀한 독립 표본의 정확한 로그우도로 단정할 수 없다.

평가 모드에서는 고정 running 통계를 사용하므로 양의 분산과 유한 스케일 아래 가역 선형 변환으로 해석할 수 있다. 그러나 학습 모드의 통계를 양방향에서 새로 계산하면 같은 통계가 보장되지 않아 왕복 복원도 일반적으로 성립하지 않는다. 흐름용 정규화가 보통 BatchNorm과 같다는 이름만으로 가역성을 보장하지는 않는다.

CouplingLayer의 로그 스케일은 tanh에 학습 가능한 exp(scale_cap)을 곱한다. 초기에는 제한되지만 이 배율 자체가 커질 수 있으므로 전체 학습에서 고정된 스케일 상한이 있는 것은 아니다. 그래디언트 클리핑도 이 문제나 위 밀도 계약을 자동 해결하지 않는다.

## 4. 기록·생성과 체크포인트

에포크 로스는 배치 평균들의 단순 평균이므로 마지막 배치 크기가 다르면 전체 표본당 평균과 다르다. 코드에는 검증·독립 테스트 평가·조기 종료가 없다. 샘플은 매 에포크가 아니라 첫 에포크와 지정 간격에 저장하며 25에포크 간격으로 체크포인트를 저장한다.

sample_images는 eval·no_grad로 생성한 연속 출력을 0~1로 clipping한다. 이는 표시 후처리이며 원래 연속 흐름의 그대로인 표본과 다르다. clipping한 픽셀은 경계에 질량을 만들 수 있어 원래 log_prob로 그 후처리 분포의 밀도를 평가할 수 없다. 다음 train_epoch는 다시 train 모드를 설정한다.

체크포인트는 모델·옵티마이저 상태만 저장하고 구조·전처리·에포크·난수 상태는 저장하지 않는다. 복원 전에 같은 구조를 만들어야 하며 재학습 재현에 필요한 설정은 별도 기록해야 한다. 이 절에서는 코드나 학습을 실행하지 않았으며 새 성능 수치·그림을 제시하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
매 배치 zero_grad가 필요한 이유와 누적 학습의 차이를 설명하라.

</div>

??? success "풀이"

    현재 루프는 매 배치 step하므로 기존 그래디언트를 초기화하지 않으면 이전 배치 그래디언트가 계속 더해져 원래와 다른 업데이트가 된다. 단순한 러닝레이트 배수나 반드시 발산하는 과정이라고 설명할 수는 없다. 의도한 누적은 여러 배치 로스의 스케일을 맞춰 backward한 뒤 한 번 step하고 초기화한다. 정규화 레이어의 배치 통계도 바뀌므로 누적만으로 큰 배치와 완전히 같은 학습이 되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 옵티마이저와 비교 실험의 우선 조건을 설명하라.

</div>

??? success "풀이"

    이미 Adam을 사용하며 기본 러닝레이트는 0.0001이다. Adam 0.001로 바꾸면 종류가 아닌 러닝레이트 비교다. SGD 등과의 비교에는 동일한 검증 목적·구조·전처리와 같은 튜닝 예산이 필요하다. 먼저 정규화·전처리·로그밀도 계약을 확인해야 최적화 방법에 따른 차이를 올바르게 해석할 수 있다. 한 번의 로스 곡선으로 일반적 우열을 정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 예제의 밀도 평가에서 확인할 두 가지 핵심 문제를 설명하라.

</div>

??? success "풀이"

    첫째, 학습 배치 통계를 새로 계산하는 BatchNorm의 반환값을 독립 표본의 정확한 야코비안 로그값으로 읽을 수 없다. 둘째, k/255+u/256 전처리와 보정 없는 bpd는 표준 정수 MNIST bits/dim과 다르다. 평가 모드의 고정 통계에서 모형 밀도를 정의하고, 정수 픽셀 구간·연속화·야코비안 보정을 일관되게 정해야 한다. 밝은 샘플 격자나 감소한 훈련 로스만으로 두 문제가 사라지지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 골격에 필요한 보완을 설명하라.

</div>

??? success "풀이"

    본문에는 검증 로더·evaluate가 없으므로 같은 전처리·평가 모드에서 비교 가능한 검증 목적을 먼저 정의해야 한다. state_dict().copy는 얕은 복사라 최적 웨이트의 독립 보존에 충분하지 않다. 깊은 복제나 파일 저장이 필요하고 정상 종료에서도 최적 상태를 복원해야 한다. 검증 잡음과 모델 선택 기준을 기록하며 테스트셋은 선택 과정과 분리한다.

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


## 정리하며

MNIST 흐름 예제는 학습·생성의 연결을 보여 주지만 전처리·bits/dim·배치 정규화의 밀도 해석에 주의가 필요하다. 고정 통계의 평가 모형과 학습 중 연산을 구분하고 출력 clipping·체크포인트 설정도 명시해야 결과를 비교할 수 있다.
