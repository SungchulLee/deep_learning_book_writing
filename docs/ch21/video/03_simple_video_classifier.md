# 합성 비디오를 분류하는 3D CNN

움직이는 색 사각형의 합성 클립을 만들고 간단한 3D CNN으로 분류하는 예제다. Dataset은 비디오·라벨을 제공하고 Simple3DCNN은 로짓을 계산하며 VideoClassifier는 트레인·검증 루프를 관리한다. 세 클래스가 모두 뉴럴 네트워크 구성 블록인 것은 아니다.

## 1. 코드

??? note "코드"

    ```python
    """
    영상 이해 — 입문 수준
    파일 03: 단순 영상 분류기 — 완전한 영상 분류 시스템 구축

    이 파일이 다루는 것:
    - 완전한 3차원 컨볼루션 뉴럴 네트워크 분류기 구축
    - 합성 영상 데이터셋 생성
    - 트레인 루프 구현
    - 영상 분류의 평가 지표
    - 모델 체크포인트와 추론

    수학적 바탕:
    분류 목표:
        영상-라벨 쌍에 걸쳐 교차 엔트로피 로스를 최소화한다

        L = -Σ y_i * log(p_i)

        여기서 각 기호는 다음과 같다.
        - y_i = 참 라벨(원핫 인코딩)
        - p_i = 클래스 i의 예측 확률
        - p = softmax(f(V)), 여기서 f는 3차원 컨볼루션 뉴럴 네트워크
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
    # 1부: 단순 3차원 컨볼루션 뉴럴 네트워크 분류기
    #=============================================================================

    class Simple3DCNN(nn.Module):
        """
        영상 분류를 위한 가벼운 3차원 컨볼루션 뉴럴 네트워크.

        구조:
            - 채널이 늘어나는 컨볼루션 블록 3개
            - 전역 평균 풀링
            - 완전 연결 분류기

        C3D보다 단순해 학습용 예제와 작은 데이터셋에 알맞다.
        """

        def __init__(self, 
                     num_classes: int = 10,
                     input_channels: int = 3,
                     dropout: float = 0.5):
            """
            단순 3차원 컨볼루션 뉴럴 네트워크를 초기화한다.

            인수:
                num_classes: 액션 클래스의 개수
                input_channels: 입력 채널의 개수(RGB는 3)
                dropout: 드롭아웃 확률
            """
            super().__init__()

            # 컨볼루션 블록 1: 3 → 32 채널
            # 입력: (B, 3, T, H, W)
            self.conv_block1 = nn.Sequential(
                nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            )
            # 출력: (B, 32, T/2, H/2, W/2)

            # 컨볼루션 블록 2: 32 → 64 채널
            self.conv_block2 = nn.Sequential(
                nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            )
            # 출력: (B, 64, T/4, H/4, W/4)

            # 컨볼루션 블록 3: 64 → 128 채널
            self.conv_block3 = nn.Sequential(
                nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            )
            # 출력: (B, 128, T/8, H/8, W/8)

            # 전역 평균 풀링
            # 공간과 시간 차원을 1x1x1로 줄인다
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
            뉴럴 네트워크를 통과하는 순전파.

            인수:
                x: 입력 영상 (B, C, T, H, W)

            반환값:
                클래스 로짓 (B, num_classes)
            """
            # 컨볼루션 블록을 거쳐 피처 추출
            x = self.conv_block1(x)  # (B, 32, T/2, H/2, W/2)
            x = self.conv_block2(x)  # (B, 64, T/4, H/4, W/4)
            x = self.conv_block3(x)  # (B, 128, T/8, H/8, W/8)

            # 전역 풀링: (B, 128, T/8, H/8, W/8) → (B, 128, 1, 1, 1)
            x = self.global_avg_pool(x)

            # 분류
            x = self.classifier(x)

            return x


    #=============================================================================
    # 2부: 합성 영상 데이터셋
    #=============================================================================

    class SyntheticVideoDataset(Dataset):
        """
        시연용 합성 영상 데이터셋.

        분류를 위해 움직임 패턴이 다른 영상을 만든다:
            - 클래스 0: 가로 움직임(왼쪽에서 오른쪽)
            - 클래스 1: 세로 움직임(위에서 아래)
            - 클래스 2: 대각선 움직임
            - 클래스 3: 도는 움직임
            - 클래스 4: 정지(움직임 없음)
        """

        def __init__(self,
                     num_samples: int = 1000,
                     num_frames: int = 16,
                     height: int = 64,
                     width: int = 64,
                     num_classes: int = 5):
            """
            합성 데이터셋을 초기화한다.

            인수:
                num_samples: 영상 표본의 개수
                num_frames: 영상마다의 프레임 개수
                height: 프레임 높이
                width: 프레임 너비
                num_classes: 클래스의 개수
            """
            self.num_samples = num_samples
            self.num_frames = num_frames
            self.height = height
            self.width = width
            self.num_classes = num_classes

            # 영상과 라벨을 모두 만들기
            self.videos, self.labels = self._generate_dataset()

        def _create_moving_square(self, 
                                motion_type: int,
                                t: int) -> np.ndarray:
            """
            움직임 클래스에 따라 움직이는 정사각형이 있는 프레임을 만든다.

            인수:
                motion_type: 움직임의 클래스(0~4)
                t: 시간 스텝(프레임 번호)

            반환값:
                numpy 배열로 된 프레임 (H, W, 3)
            """
            frame = np.zeros((self.height, self.width, 3), dtype=np.float32)

            # 정사각형 크기
            square_size = 8

            # 움직임 클래스에 따라 위치 계산하기
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

            # 정사각형 그리기(클래스마다 다른 색)
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
            전체 데이터셋을 만든다.

            반환값:
                videos: 형태가 (N, T, H, W, 3)인 텐서
                labels: 형태가 (N,)인 텐서
            """
            videos = []
            labels = []

            for i in range(self.num_samples):
                # 무작위 클래스
                label = np.random.randint(0, self.num_classes)

                # 영상 만들기
                video_frames = []
                for t in range(self.num_frames):
                    frame = self._create_moving_square(label, t)
                    video_frames.append(frame)

                video = np.array(video_frames)  # (T, H, W, 3)

                # 강건성을 위해 잡음을 조금 더하기
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
            영상-라벨 쌍 하나를 얻는다.

            인수:
                idx: 인덱스

            반환값:
                video: (3, T, H, W)
                label: 클래스 번호
            """
            return self.videos[idx], self.labels[idx]


    #=============================================================================
    # 3부: 트레인 파이프라인
    #=============================================================================

    class VideoClassifier:
        """
        영상 분류의 전체 트레인·평가 파이프라인.
        """

        def __init__(self,
                     model: nn.Module,
                     device: str = 'cpu'):
            """
            분류기를 초기화한다.

            인수:
                model: 3차원 컨볼루션 뉴럴 네트워크 모델
                device: 트레인에 쓸 장치('cpu' 또는 'cuda')
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
            한 에포크를 트레인한다.

            인수:
                train_loader: 트레인 데이터 로더
                optimizer: 옵티마이저
                criterion: 로스 함수

            반환값:
                그 에포크의 평균 로스와 정확도
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
            검증 세트로 평가한다.

            인수:
                val_loader: 검증 데이터 로더
                criterion: 로스 함수

            반환값:
                평균 로스와 정확도
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
            전체 트레인 루프.

            인수:
                train_loader: 트레인 데이터 로더
                val_loader: 검증 데이터 로더
                num_epochs: 트레인 에포크 수
                learning_rate: 러닝레이트

            반환값:
                트레인 이력 딕셔너리
            """
            # 트레인 준비
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
                # 트레인
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)

                # 검증
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                # 러닝레이트 스케줄링
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
    # 4부: 시각화
    #=============================================================================

    def plot_training_history(history: Dict):
        """
        트레인과 검증 지표를 그린다.

        인수:
            history: 트레인 이력 딕셔너리
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_losses']) + 1)

        # 로스 그래프
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
        plt.savefig('03_training_curves.png',
                    dpi=150, bbox_inches='tight')
        print(f"Training curves saved to 03_training_curves.png")
        plt.close()


    #=============================================================================
    # 5부: 메인 시연
    #=============================================================================

    def main():
        """
        단순 영상 분류기의 메인 시연.
        """
        print(__doc__)

        # 난수 시드 고정
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
            num_workers=0  # 디버깅 때는 0으로
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

        # 분류기를 만들고 트레인하기
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

        # 검증 세트에서 예측 확인
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
        1. 전체 파이프라인:
           - 움직임 패턴이 있는 데이터셋 생성
           - 검증을 곁들인 모델 트레인
           - 성능 모니터링과 시각화

        2. 모델 성능:
           - 가장 좋은 검증 정확도: {history['best_val_acc']:.2f}%
           - 단순한 3차원 컨볼루션 뉴럴 네트워크도 기본 움직임 패턴을 학습할 수 있다
           - 합성 데이터는 프로토타입 제작에 좋다

        3. 트레인에서 얻은 통찰:
           - 3차원 컨볼루션 뉴럴 네트워크는 공간과 시간의 피처를 엔드투엔드로 트레인한다
           - 메모리 때문에 배치 크기가 제한된다(3차원 텐서는 크다)
           - 러닝레이트 스케줄링이 수렴을 돕는다

        4. 다음 스텝:
           - 실제 영상 데이터셋(UCF-101, Kinetics)를 써 보라
           - 강건성을 위해 데이터 어그멘테이션 추가하기
           - 더 깊은 구조로 실험해 보라
           - 더 나은 성능을 위해 Two-Stream 네트워크를 고려해 보라
        """)


    if __name__ == "__main__":
        main()
    ```

## 2. 데이터와 클래스 지름길

기본 시연은 수평·수직·대각선·원형 움직임과 정지의 다섯 클래스를 사용한다. 한 표본의 모양은 3×16×64×64이고 배치는 B×3×16×64×64다. 정답은 0~4 정수다. 생성자는 num_classes를 받지만 색상 표가 다섯 개여서 5보다 크게 설정하면 인덱스 오류가 날 수 있다. 높이·너비도 최소한 8픽셀 사각형을 담을 수 있어야 한다.

각 움직임에 고유 색상을 주고 궤적도 클래스별로 고정한다. 트레인·검증은 같은 생성 규칙에서 잡음만 다르게 더한 데이터라 단일 프레임의 색만으로도 클래스를 맞힐 수 있다. 검증 정확도가 높아도 시간 순서나 움직임을 배웠다는 증거는 아니다. 색상을 클래스와 독립적으로 바꾸고 위치·속도·배경을 다양화하거나 시간 순서를 섞은 대조군 등으로 시간 정보의 기여를 검토해야 한다.

전체 데이터를 생성 시 메모리에 저장한다. 기본 시연의 800+200개 float32 클립만 약 750MiB이며 생성 중 NumPy 중간 배열·복사·모델 활성화 값은 추가다. 작은 모델이라는 사실만으로 데이터 로딩 메모리도 작은 것은 아니다.

## 3. 시공간 피처와 분류 head

각 3×3×3 컨볼루션은 padding 1로 크기를 유지하고 MaxPool3d(2)가 시간·공간을 절반으로 줄인다. 기본 입력의 피처는 32×8×32×32, 64×4×16×16, 128×2×8×8이다. 전역 평균 풀링 뒤 B×128을 64차원과 K클래스 로짓으로 변환한다. 세 풀링을 거쳐야 하므로 T·H·W가 충분히 커야 한다.

교차 엔트로피는 softmax를 거치지 않은 원시 로짓과 정수 정답을 사용한다. 트레인 전에 softmax를 따로 적용할 필요가 없다. 트레인 모드에서는 dropout·BN 통계를 사용하고 검증은 eval·no_grad로 동작을 바꾼다. 전역 평균은 피처를 요약하며 시간 순서 정보는 그 전에 3D 필터가 처리한 범위에 의존한다.

## 4. 트레인 기록과 검증 해석

옵티마이저는 Adam(lr=0.001) 하나뿐이므로 다른 옵티마이저와 비교한 실험이 아니다. 러닝레이트 스케줄러는 검증 로스를 보고 줄이고 best_val_acc는 최고 검증 정확도를 숫자로 기록한다. 최고 시점 웨이트를 저장·복원하지 않으므로 마지막 예측에 쓰는 모델이 최고 정확도 시점과 같다는 보장은 없다.

로스는 배치 평균에 배치 크기를 곱해 합한 뒤 표본 수로 나누므로 마지막 작은 배치를 적절히 반영한다. 다만 빈 데이터 로더에서는 total=0으로 나눈다. 검증 데이터를 러닝레이트 조정과 최고 점수 선택에 사용하므로 독립 시험 결과로 해석하지 않는다. 시연 마지막에도 같은 검증 배치의 일부를 보여 준다.

`ReduceLROnPlateau`의 verbose 인수 등은 설치 PyTorch 버전의 API와 맞춰야 한다. 출력 곡선은 해당 합성 문제의 기록일 뿐 실제 액션 인식 데이터의 수렴이나 일반화 성능을 보장하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
Simple3DCNN의 기본 num_classes=10과 시연 num_classes=5의 파라메터 수를 계산하라.

</div>

??? success "풀이"

    컨볼루션·BN 합계는 첫 블록 2688, 둘째 55488, 셋째 221568개다. 128→64 선형은 8256개이고 마지막 64→K 선형은 64K+K=65K개다. 총 288000+65K이므로 기본 10클래스는 288650개, 시연 5클래스는 288325개다. BN의 running 통계(이동 평균·분산)는 트레인 가능한 파라메터가 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 옵티마이저와 best_val_acc의 의미를 설명하라.

</div>

??? success "풀이"

    이미 Adam을 러닝레이트 0.001로 사용한다. 다른 옵티마이저와 비교하려면 명시적으로 대조 방법·초기화·데이터·러닝레이트 스케줄을 정한다. best_val_acc는 최고 기록 숫자이며 해당 웨이트를 보관하지 않는다. 마지막 모델을 최고 시점 모델이라고 부르면 안 된다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
좋은 검증 정확도가 움직임 트레인을 입증하지 못하는 이유를 설명하라.

</div>

??? success "풀이"

    색이 클래스마다 고정되고 궤적도 정형화되어 공간적 지름길이 가능하다. 트레인·검증 모두 같은 규칙이다. 클래스와 색을 분리하고 궤적·속도·배경을 다양화한 평가 및 시간 정보 제거 대조를 통해 어떤 신호를 사용하는지 확인한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList로 풀링 블록을 늘릴 때 시간 차원에 생길 수 있는 문제를 설명하라.

</div>

??? success "풀이"

    16프레임을 네 번 절반으로 줄이면 1이 되므로, 다섯 번째 MaxPool3d(2)는 유효하지 않다. 채널뿐 아니라 단계별 최소 시간·공간 크기를 추적해야 한다. YourBlock은 별도 정의가 필요한 자리 표시자이며 모듈 등록만으로 크기·입력 검증이 완성되지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

합성 클립의 3D CNN 트레인은 BCTHW 입력·시공간 풀링·로짓 로스와 트레인 기록을 이해하는 예다. 색상 지름길·고정 궤적·검증 재사용 때문에 액션 인식의 일반화를 입증하지 않으며 최고 기록과 마지막 웨이트도 구분한다.
