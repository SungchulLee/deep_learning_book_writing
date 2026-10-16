# MNIST 확산 모델 학습 파이프라인

이 페이지는 앞에서 정의한 SimpleUNet과 확산 유틸리티를 연결한다. MNIST 이미지에 노이즈를 섞어 예측기를 학습하고, EMA 모델로 이미지를 생성하며, 웨이트와 optimizer 상태를 저장하는 흐름이다. 코드에 적힌 학습 설정과 실제로 검증된 생성 품질은 구분해야 한다.

## 1. 코드

```python
"""
MNIST 확산 모델

MNIST 숫자 만들어 내기를 위한 확산 모델의 온전한 짜기.
학부생을 위한 그럴듯한 그림 만들어 내기 보기를 준다.
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

from diffusion_utils import (
    cosine_beta_schedule,
    get_diffusion_parameters,
    get_loss,
    sample,
    visualize_samples
)
from unet_architecture import SimpleUNet


class MNISTDiffusion:
    """
    MNIST 확산 모델의 익히기와 뽑기를 감싸는 클래스.
    """
    
    def __init__(self, 
                 timesteps: int = 1000,
                 batch_size: int = 128,
                 learning_rate: float = 2e-4,
                 device: str = None):
        """
        인수:
            timesteps: 확산 때 걸음 수
            batch_size: 익히기 배치 크기
            learning_rate: 최적화기의 러닝레이트
            device: 익힐 장치('cuda' 또는 'cpu')
        """
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # 확산 파라메터를 채비한다
        betas = cosine_beta_schedule(timesteps)
        self.diffusion_params = get_diffusion_parameters(betas)
        
        # 파라메터를 기기로 옮긴다
        for key in self.diffusion_params:
            self.diffusion_params[key] = self.diffusion_params[key].to(self.device)
        
        # 모형을 시작한다
        self.model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_emb_dim=256
        ).to(self.device)
        
        # 최적화기
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # 더 나은 표본을 위한 지수 이동 평균 모델
        self.ema_model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=64,
            time_emb_dim=256
        ).to(self.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_decay = 0.9999
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def update_ema(self):
        """모델 파라메터의 지수 이동 평균을 고친다."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), 
                                        self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def get_dataloader(self, train: bool = True):
        """
        MNIST 자료 불러오개를 만든다.
        
        인수:
            train: 익히기 배치를 불러올지 시험 배치를 불러올지
        
        반환값:
            MNIST용 DataLoader
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1, 1]로 정규화
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
            
            # 아무 때 걸음을 뽑는다
            t = torch.randint(
                0, self.timesteps,
                (images.shape[0],),
                device=self.device
            )
            
            # 로스를 계산한다
            loss = get_loss(self.model, images, t, self.diffusion_params)
            
            # 최적화
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 지수 이동 평균을 고친다
            self.update_ema()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def sample_images(self, n_samples: int = 64, use_ema: bool = True):
        """
        표본 그림을 만든다.
        
        인수:
            n_samples: 만들 표본의 개수
            use_ema: 지수 이동 평균 모델을 쓸지 여부
        
        반환값:
            만든 그림 텐서
        """
        model = self.ema_model if use_ema else self.model
        
        samples = sample(
            model,
            shape=(n_samples, 1, 28, 28),
            timesteps=self.timesteps,
            diffusion_params=self.diffusion_params,
            device=self.device
        )
        
        return samples
    
    def train(self, epochs: int, save_interval: int = 10):
        """
        확산 모델을 익힌다.
        
        인수:
            epochs: 학습 에포크 수
            save_interval: N바퀴마다 표본을 갈무리한다
        """
        dataloader = self.get_dataloader(train=True)
        
        # 표본을 담을 자리를 만든다
        os.makedirs('samples', exist_ok=True)
        
        losses = []
        
        print(f"\nTraining for {epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # 학습
            avg_loss = self.train_epoch(dataloader)
            losses.append(avg_loss)
            
            print(f"Average loss: {avg_loss:.4f}")
            
            # 표본을 만들어 갈무리한다
            if epoch % save_interval == 0 or epoch == 1:
                print("Generating samples...")
                samples = self.sample_images(n_samples=64)
                visualize_samples(
                    samples,
                    nrow=8,
                    filename=f'samples/epoch_{epoch:04d}.png'
                )
            
            # 검사점 저장
            if epoch % 50 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # 학습 로스 그리기
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MNIST Diffusion Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nSaved training loss plot to training_loss.png")
        
        print("\n" + "=" * 50)
        print("Training complete!")
        print("=" * 50)
    
    def save_checkpoint(self, filename: str):
        """모델 되짚기 지점을 갈무리한다."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'diffusion_params': self.diffusion_params,
        }, filename)
        print(f"Saved checkpoint to {filename}")
    
    def load_checkpoint(self, filename: str):
        """모델 되짚기 지점을 불러온다."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")


def main():
    """
    MNIST 확산의 으뜸 익히기 대본.
    """
    print("=" * 50)
    print("MNIST Diffusion Model Training")
    print("=" * 50)
    
    # 확산 모델을 첫자리매김한다
    mnist_diffusion = MNISTDiffusion(
        timesteps=1000,
        batch_size=128,
        learning_rate=2e-4
    )
    
    # 학습
    mnist_diffusion.train(epochs=100, save_interval=10)
    
    # 마지막 표본을 만든다
    print("\nGenerating final samples...")
    samples = mnist_diffusion.sample_images(n_samples=64)
    visualize_samples(samples, nrow=8, filename='final_samples.png')
    
    # 마지막 모델을 저장한다
    mnist_diffusion.save_checkpoint('mnist_diffusion_final.pt')
    
    print("\nAll done! Check the 'samples' folder for generated images.")


if __name__ == "__main__":
    main()
```

## 2. 데이터와 노이즈 예측

ToTensor 뒤 Normalize(0.5,0.5)를 적용하므로 원본 픽셀의 [0,1] 범위는 [−1,1]로 바뀐다. 입력 배치는 B×1×28×28이며 숫자 레이블은 학습에 사용하지 않는다. 클래스 조건이 없는 생성 모델이다.

매 배치에서 각 이미지의 시간 인덱스를 0부터 T−1까지 균등하게 뽑는다. get_loss는 독립 표준정규 노이즈 ε를 생성하고 다음 입력으로 모델을 호출한다.

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

모델 출력과 ε 사이의 MSE는 배치·채널·픽셀 전체의 평균이다. 시간별 웨이트가 모두 1인 단순 노이즈 예측 목적이며, 전체 변분 하한의 모든 항을 그대로 계산하지 않는다. 코드의 인덱스 0은 첫 노이즈 단계이므로 노이즈가 전혀 없는 원본과 같지 않다.

코사인 스케줄과 확산 계수는 CPU에서 만든 다음 사전의 각 텐서를 선택 장치로 옮긴다. 러닝레이트 0.0002의 Adam을 사용하며, 역전파 뒤 모델 파라미터의 전체 그래디언트 노름을 1로 제한한다. 이 제한은 optimizer가 만드는 파라미터 변화량의 노름을 직접 제한하는 연산이 아니다.

## 3. SimpleUNet과 텐서 크기

여기서 가져오는 클래스는 일반 UNet이 아니라 SimpleUNet이다. base_channels=64에서 잔차 블록은 채널을 64→128→256→256→128→64로 바꾸고 공간 크기는 28→14→7→14→28을 따른다. 확대 경로에서 같은 해상도의 피처를 연결한 뒤 채널 수를 줄인다. 최종 출력은 입력과 같은 B×1×28×28이다.

시간은 사인·코사인 임베딩과 MLP로 256차원 조건을 만든 뒤 각 잔차 블록에 더한다. GroupNorm과 시간 조건 잔차 블록을 사용하지만 이 SimpleUNet에는 공간 어텐션이 없다. 출력 활성화로 픽셀 범위를 제한하지 않으며 학습 목표도 깨끗한 이미지 자체가 아니라 노이즈다.

## 4. 에포크 평균과 EMA

훈련 이미지 60,000개와 배치 크기 128에서 한 에포크는 469번의 업데이트다. 마지막 96개 배치를 버리지 않으므로 100에포크를 완료하면 46,900번 업데이트한다. 반환하는 에포크 로스는 각 배치 평균을 같은 비중으로 평균한 값이다. 따라서 마지막 작은 배치까지 샘플 수로 가중한 정확한 전체 평균과는 다르다.

EMA 모델은 시작할 때 학습 모델의 웨이트를 복사하고 첫 optimizer 업데이트부터 다음 식으로 갱신한다.

$$\theta_{\mathrm{EMA}}\leftarrow0.9999\theta_{\mathrm{EMA}}+0.0001\theta$$

별도 대기 기간은 없다. 파라메터를 완만하게 평균해 생성에 활용하는 방식이며, 특정 품질 향상이나 최적 시점을 보장하지는 않는다. 현재 함수는 파라미터만 갱신한다. 다른 구조에 BatchNorm 같은 상태 버퍼를 추가한다면 해당 상태의 처리도 검토해야 한다.

## 5. 생성과 시각화

sample_images는 기본적으로 EMA 모델을 사용한다. 표준정규 B×1×28×28에서 출발해 1,000단계 역과정을 수행하며 유틸리티의 샘플러는 후방분산을 사용한다. 최종 단계에는 추가 노이즈를 넣지 않는다. 생성은 no_grad와 평가 모드에서 수행하지만 출력 텐서가 반드시 [−1,1]에 들어오는 것은 아니다.

기본 실행은 에포크 1과 이후 10에포크마다 64개를 생성하고, 학습이 끝나면 최종 샘플을 별도로 만든다. visualize_samples는 배치 전체 최솟값·최댓값으로 화면 표시 범위를 조정한다. 이는 학습 정규화의 고정 역변환 (x+1)/2와 다르므로 밝기·범위의 차이를 가릴 수 있다. 모든 값이 같으면 분모가 0이 되는 점도 고려한다.

샘플 이미지와 학습 로스는 점검 자료지만 분포의 다양성이나 일반화를 수치로 입증하지 않는다. 이 코드는 검증 로스, FID나 IS를 계산하지 않으며 train=False 로더를 만들 수 있다는 사실만으로 테스트 평가를 수행한 것은 아니다.

## 6. 체크포인트가 복원하는 범위

50에포크마다 저장하고 마지막에도 체크포인트를 남긴다. 저장 항목은 학습 모델, EMA 모델, optimizer 상태와 확산 계수다. 그러나 load_checkpoint는 **저장된 diffusion_params를 복원하지 않는다**. 새 객체를 만들 때 지정한 시간 단계 수와 스케줄이 계속 사용된다.

따라서 웨이트가 로드되어도 학습 당시와 다른 T나 스케줄로 생성할 수 있다. 같은 모델 구조와 확산 설정을 별도로 확인해야 한다. 에포크 번호, 난수 상태, 전체 설정, 로스 이력과 EMA 감쇠율도 복원하지 않으므로 중단 지점과 완전히 같은 학습 궤적을 재개하는 기능으로 설명할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 설정의 입력·출력 모양과 한 에포크의 업데이트 수를 구하라.

</div>

??? success "연습문제 1 풀이"

    입력과 노이즈 예측 출력은 B×1×28×28이다. 60,000=468×128+96이므로 마지막 배치를 포함해 469번 업데이트한다. 100에포크면 46,900번이다. 로스 반환값은 469개 배치 평균의 산술평균이어서 샘플 수에 따른 가중 평균과는 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
MSE 로스의 평균 축과 EMA 갱신 시점을 설명하라.

</div>

??? success "연습문제 2 풀이"

    로스는 모든 배치·채널·공간 원소의 제곱 오차를 평균한다. 전체 원소 수가 M이면 그래디언트는 (2/M)Σ(εθ−ε)∇θεθ다. EMA는 초기 웨이트를 복사하고 첫 optimizer 업데이트부터 0.9999의 감쇠율로 파라미터를 평균한다. 단순 MSE가 전체 ELBO와 항별로 같은 목적이라는 뜻은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
다른 노이즈 스케줄로 객체를 만든 뒤 기존 체크포인트를 불러오면 어떤 문제가 생기는가?

</div>

??? success "연습문제 3 풀이"

    load_checkpoint는 저장된 확산 계수를 읽어 객체에 대입하지 않으므로 새 생성자의 스케줄이 남는다. 웨이트의 학습 조건과 샘플링 조건이 달라진다. 시간 단계 수와 스케줄, 모델 구조를 함께 일치시켜야 하며, 스케줄의 품질 비교에는 같은 데이터·예산·평가 조건과 별도의 실험이 필요하다.



## 정리하며

MNIST 파이프라인은 노이즈 MSE 학습, 그래디언트 clipping, 첫 단계부터 적용하는 EMA, 후방분산을 이용한 샘플링을 연결한다. 배치 평균의 의미와 표시용 정규화를 구분하고, 체크포인트 로더가 확산 계수를 복원하지 않는 제한을 확인한다.
