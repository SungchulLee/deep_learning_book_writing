# 단원 34: 영상 이해

단원 34: 영상 이해 — 첫걸음 수준. 파일 03: 단순 영상 갈래 매개 — 온전한 영상 가르기 체계 세우기

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 영상 이해를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

```python
"""
단원 34: 영상 이해 — 첫걸음 수준
파일 03: 단순 영상 갈래 매개 — 온전한 영상 가르기 체계 세우기

이 파일이 다루는 것:
- 온전한 3차원 누비기 신경망 갈래 매개 세우기
- 인공 영상 자료 뭉치 만들기
- 익히기 되풀이 짜기
- 영상 가르기의 값매김 잣대
- 모델 중간 저장과 미룸

수학적 바탕:
갈래 매기기 목표:
    영상-이름표 짝에 걸쳐 엇갈린 엔트로피 손실을 가장 작게 한다
    
    L = -Σ y_i * log(p_i)
    
    여기서 각 기호는 다음과 같다.
    - y_i = 참 이름표(원핫 부호)
    - p_i = 갈래 i의 어림 확률
    - p = softmax(f(V)), 여기서 f는 3차원 누비기 신경망
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings

# ========================================================================
# 메인
# ========================================================================
warnings.filterwarnings('ignore')


#=============================================================================
# 1부: 단순 3차원 누비기 신경망 갈래 매개
#=============================================================================

class Simple3DCNN(nn.Module):
    """
    영상 가르기를 위한 가벼운 3차원 누비기 신경망.
    
    구조:
        - 채널이 늘어나는 누비기 덩이 3개
        - 전체 평균 모으기
        - 온전히 이은 갈래 매개
        
    C3D보다 단순해 배우기와 작은 자료 뭉치에 좋다.
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 input_channels: int = 3,
                 dropout: float = 0.5):
        """
        단순 3차원 누비기 신경망을 첫자리매김한다.
        
        인수:
            num_classes: 몸짓 갈래의 개수
            input_channels: 들임 채널의 개수(RGB는 3)
            dropout: 드롭아웃 확률
        """
        super().__init__()
        
        # 누비기 덩이 1: 3 → 32 채널
        # 들임: (B, 3, T, H, W)
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        # 내놓음: (B, 32, T/2, H/2, W/2)
        
        # 누비기 덩이 2: 32 → 64 채널
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        # 내놓음: (B, 64, T/4, H/4, W/4)
        
        # 누비기 덩이 3: 64 → 128 채널
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        # 내놓음: (B, 128, T/8, H/8, W/8)
        
        # 전체 평균 모으기
        # 자리와 때 차원을 1x1x1로 줄인다
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        신경망을 통과하는 순전파.
        
        인수:
            x: 들임 영상 (B, C, T, H, W)
            
        반환값:
            갈래 로짓 (B, num_classes)
        """
        # 누비기 덩이를 거쳐 특징 뽑기
        x = self.conv_block1(x)  # (B, 32, T/2, H/2, W/2)
        x = self.conv_block2(x)  # (B, 64, T/4, H/4, W/4)
        x = self.conv_block3(x)  # (B, 128, T/8, H/8, W/8)
        
        # 전체 모으기: (B, 128, T/8, H/8, W/8) → (B, 128, 1, 1, 1)
        x = self.global_avg_pool(x)
        
        # 분류
        x = self.classifier(x)
        
        return x


#=============================================================================
# 2부: 인공 영상 자료 뭉치
#=============================================================================

class SyntheticVideoDataset(Dataset):
    """
    보여 주기 위한 인공 영상 자료 뭉치.
    
    갈래 매기기를 위해 움직임 무늬가 다른 영상을 만든다:
        - 갈래 0: 가로 움직임(왼쪽에서 오른쪽)
        - 갈래 1: 세로 움직임(위에서 아래)
        - 갈래 2: 대각선 움직임
        - 갈래 3: 도는 움직임
        - 갈래 4: 멈춤(움직임 없음)
    """
    
    def __init__(self,
                 num_samples: int = 1000,
                 num_frames: int = 16,
                 height: int = 64,
                 width: int = 64,
                 num_classes: int = 5):
        """
        인공 자료 뭉치를 첫자리매김한다.
        
        인수:
            num_samples: 영상 표본의 개수
            num_frames: 영상마다의 틀 개수
            height: 틀 높이
            width: 틀 너비
            num_classes: 갈래의 개수
        """
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_classes = num_classes
        
        # 영상과 이름표를 모두 만들기
        self.videos, self.labels = self._generate_dataset()
    
    def _create_moving_square(self, 
                            motion_type: int,
                            t: int) -> np.ndarray:
        """
        움직임 갈래에 따라 움직이는 정사각형이 있는 틀을 만든다.
        
        인수:
            motion_type: 움직임의 갈래(0~4)
            t: 때 걸음(틀 번호)
            
        반환값:
            numpy 배열로 된 틀 (H, W, 3)
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # 정사각형 크기
        square_size = 8
        
        # 움직임 갈래에 따라 자리 셈하기
        if motion_type == 0:  # 가로
            x = int((t / self.num_frames) * (self.width - square_size))
            y = self.height // 2 - square_size // 2
        
        elif motion_type == 1:  # 세로
            x = self.width // 2 - square_size // 2
            y = int((t / self.num_frames) * (self.height - square_size))
        
        elif motion_type == 2:  # 대각선
            progress = t / self.num_frames
            x = int(progress * (self.width - square_size))
            y = int(progress * (self.height - square_size))
        
        elif motion_type == 3:  # 돌기
            angle = 2 * np.pi * (t / self.num_frames)
            radius = min(self.height, self.width) // 4
            center_x, center_y = self.width // 2, self.height // 2
            x = int(center_x + radius * np.cos(angle) - square_size // 2)
            y = int(center_y + radius * np.sin(angle) - square_size // 2)
        
        else:  # 멈춰 있음
            x = self.width // 2 - square_size // 2
            y = self.height // 2 - square_size // 2
        
        # 테두리 안에 있는지 확인
        x = max(0, min(x, self.width - square_size))
        y = max(0, min(y, self.height - square_size))
        
        # 정사각형 그리기(갈래마다 다른 빛깔)
        color = np.array([
            [1.0, 0.0, 0.0],  # 빨강
            [0.0, 1.0, 0.0],  # 초록
            [0.0, 0.0, 1.0],  # 파랑
            [1.0, 1.0, 0.0],  # 노랑
            [1.0, 0.0, 1.0],  # 자홍
        ])[motion_type]
        
        frame[y:y+square_size, x:x+square_size] = color
        
        return frame
    
    def _generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        온전한 자료 뭉치를 만든다.
        
        반환값:
            videos: 꼴이 (N, T, H, W, 3)인 텐서
            labels: 꼴이 (N,)인 텐서
        """
        videos = []
        labels = []
        
        for i in range(self.num_samples):
            # 마구잡이 갈래
            label = np.random.randint(0, self.num_classes)
            
            # 영상 만들기
            video_frames = []
            for t in range(self.num_frames):
                frame = self._create_moving_square(label, t)
                video_frames.append(frame)
            
            video = np.array(video_frames)  # (T, H, W, 3)
            
            # 튼튼함을 위해 잡음을 살짝 더하기
            video = video + np.random.randn(*video.shape) * 0.02
            video = np.clip(video, 0, 1)
            
            videos.append(video)
            labels.append(label)
        
        # 텐서로 바꾼다
        # (N, T, H, W, 3) → (N, T, 3, H, W) → (N, 3, T, H, W)
        videos = torch.FloatTensor(np.array(videos))
        videos = videos.permute(0, 1, 4, 2, 3)  # (N, T, 3, H, W)
        videos = videos.permute(0, 2, 1, 3, 4)  # (N, 3, T, H, W)
        
        labels = torch.LongTensor(labels)
        
        return videos, labels
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        영상-이름표 짝 하나를 얻는다.
        
        인수:
            idx: 번호
            
        반환값:
            video: (3, T, H, W)
            label: 갈래 번호
        """
        return self.videos[idx], self.labels[idx]


#=============================================================================
# 3부: 익히기 물길
#=============================================================================

class VideoClassifier:
    """
    영상 가르기의 온전한 익히기·값매김 물길.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu'):
        """
        갈래 매개를 첫자리매김한다.
        
        인수:
            model: 3차원 누비기 신경망 모델
            device: 익히기에 쓸 기기('cpu' 또는 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        한 세대를 학습한다.
        
        인수:
            train_loader: 학습 데이터 로더
            optimizer: 최적화기
            criterion: 손실 함수
            
        반환값:
            그 세대의 평균 손실과 정확도
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in train_loader:
            # 장치로 옮긴다
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            # 순전파
            optimizer.zero_grad()
            outputs = self.model(videos)
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 통계
            total_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self,
                val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float]:
        """
        검증 뭉치로 값매김한다.
        
        인수:
            val_loader: 검증 데이터 로더
            criterion: 손실 함수
            
        반환값:
            평균 손실과 정확도
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for videos, labels in val_loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(videos)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int = 20,
             learning_rate: float = 0.001) -> Dict:
        """
        온전한 익히기 되풀이.
        
        인수:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            num_epochs: 학습 에포크 수
            learning_rate: 배움 비율
            
        반환값:
            학습 이력 사전
        """
        # 학습 준비
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        print(f"\nTraining on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*80)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 검증
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 학습률 스케줄링
            scheduler.step(val_loss)
            
            # 진행 상황 출력
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
        
        print("="*80)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc
        }


#=============================================================================
# 4부: 그려 보기
#=============================================================================

def plot_training_history(history: Dict):
    """
    익히기와 검증 잣대를 그린다.
    
    인수:
        history: 익히기 발자취 사전
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 손실 그래프
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 정확도 그림
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/34_video_understanding/03_training_curves.png',
                dpi=150, bbox_inches='tight')
    print(f"Training curves saved to 03_training_curves.png")
    plt.close()


#=============================================================================
# 5부: 주된 보임
#=============================================================================

def main():
    """
    단순 영상 갈래 매개의 주된 보임.
    """
    print(__doc__)
    
    # 난수 씨앗 고정
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*80)
    print("SIMPLE VIDEO CLASSIFIER DEMONSTRATION")
    print("="*80)
    
    # 설정
    num_classes = 5
    num_frames = 16
    height, width = 64, 64
    batch_size = 16
    num_epochs = 15
    
    # 데이터셋 생성
    print("\n1. Creating synthetic video dataset...")
    train_dataset = SyntheticVideoDataset(
        num_samples=800,
        num_frames=num_frames,
        height=height,
        width=width,
        num_classes=num_classes
    )
    
    val_dataset = SyntheticVideoDataset(
        num_samples=200,
        num_frames=num_frames,
        height=height,
        width=width,
        num_classes=num_classes
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Video shape: (3, {num_frames}, {height}, {width})")
    print(f"   Number of classes: {num_classes}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 벌레잡기 때는 0으로
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 모델 생성
    print("\n2. Creating 3D CNN model...")
    model = Simple3DCNN(num_classes=num_classes)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 갈래 매개 만들고 익히기
    print(f"\n3. Training model...")
    classifier = VideoClassifier(model, device=device)
    history = classifier.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        learning_rate=0.001
    )
    
    # 결과 그리기
    print("\n4. Visualizing training history...")
    plot_training_history(history)
    
    # 시험 예측
    print("\n5. Testing predictions on validation set...")
    classifier.model.eval()
    
    class_names = ['Horizontal', 'Vertical', 'Diagonal', 'Circular', 'Static']
    
    with torch.no_grad():
        videos, labels = next(iter(val_loader))
        videos = videos.to(device)
        outputs = classifier.model(videos)
        _, predicted = torch.max(outputs, 1)
        
        print(f"\nSample predictions:")
        for i in range(min(5, len(labels))):
            true_label = class_names[labels[i].item()]
            pred_label = class_names[predicted[i].item()]
            correct = "✓" if labels[i] == predicted[i] else "✗"
            print(f"  {correct} True: {true_label:12s} | Predicted: {pred_label:12s}")
    
    # 요약
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print(f"""
    1. 온전한 물길:
       - 움직임 무늬가 있는 자료 뭉치 만들기
       - 검증을 곁들인 모델 익히기
       - 성능 지켜보기와 그려 보기
    
    2. 모델 성능:
       - 가장 좋은 검증 정확도: {history['best_val_acc']:.2f}%
       - 단순한 3차원 누비기 신경망도 기본 움직임 무늬를 배울 수 있다
       - 인공 자료는 시제품 만들기에 좋다
    
    3. 익히기에서 얻은 눈썰미:
       - 3차원 누비기 신경망은 자리와 때의 특징을 끝에서 끝까지 배운다
       - 기억 공간 때문에 묶음 크기가 제한된다(3차원 텐서는 크다)
       - 배움 비율 일정 짜기가 모임을 돕는다
    
    4. 다음 걸음:
       - 실제 영상 자료 뭉치(UCF-101, Kinetics)를 써 보라
       - 튼튼함을 위해 자료 불리기 더하기
       - 더 깊은 얼개로 실험해 보라
       - 더 나은 성능을 위해 두 갈래 그물을 헤아려 보라
    """)


if __name__ == "__main__":
    main()```

## 논의

여기 짠 것은 함께 어울려 온전한 영상 이해 얼개를 이루는 클래스 3개(`Simple3DCNN`, `SyntheticVideoDataset`, `VideoClassifier`)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
붙박이 첫자리매김일 때 `Simple3DCNN`의 배울 수 있는 매개변수 전체 개수를 셈하여라. 무게와 치우침을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
층이나 덩이의 개수를 정할 수 있도록 `Simple3DCNN`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = Simple3DCNN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
