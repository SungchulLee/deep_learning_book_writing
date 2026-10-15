# 합성 비디오를 분류하는 3D CNN

움직이는 색 사각형의 합성 클립을 만들고 간단한 3D CNN으로 분류하는 예제다. Dataset은 비디오·레이블을 제공하고 Simple3DCNN은 로짓을 계산하며 VideoClassifier는 학습·검증 루프를 관리한다. 세 클래스가 모두 신경망 구성 블록인 것은 아니다.

## 1. 코드

??? note "코드"

    ```python
    """
    영상 이해 — 첫걸음 수준
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
        plt.savefig('03_training_curves.png',
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
           - 기억 공간 때문에 배치 크기가 제한된다(3차원 텐서는 크다)
           - 배움 비율 일정 짜기가 모임을 돕는다

        4. 다음 걸음:
           - 실제 영상 자료 뭉치(UCF-101, Kinetics)를 써 보라
           - 튼튼함을 위해 자료 불리기 더하기
           - 더 깊은 구조로 실험해 보라
           - 더 나은 성능을 위해 두 갈래 그물을 헤아려 보라
        """)


    if __name__ == "__main__":
        main()
    ```

## 2. 데이터와 클래스 지름길

기본 시연은 수평·수직·대각선·원형 움직임과 정지의 다섯 클래스를 사용한다. 한 표본의 모양은 3×16×64×64이고 배치는 B×3×16×64×64다. 정답은 0~4 정수다. 생성자는 num_classes를 받지만 색상 표가 다섯 개여서 5보다 크게 설정하면 인덱스 오류가 날 수 있다. 높이·너비도 최소한 8픽셀 사각형을 담을 수 있어야 한다.

각 움직임에 고유 색상을 주고 궤적도 클래스별로 고정한다. 학습·검증은 같은 생성 규칙에서 잡음만 다르게 더한 자료라 단일 프레임의 색만으로도 클래스를 맞힐 수 있다. 검증 정확도가 높아도 시간 순서나 움직임을 배웠다는 증거는 아니다. 색상을 클래스와 독립적으로 바꾸고 위치·속도·배경을 다양화하거나 시간 순서를 섞은 대조군 등으로 시간 정보의 기여를 검토해야 한다.

전체 자료를 생성 시 메모리에 저장한다. 기본 시연의 800+200개 float32 클립만 약 750MiB이며 생성 중 NumPy 중간 배열·복사·모델 활성값은 추가다. 작은 모델이라는 사실만으로 데이터 로딩 메모리도 작은 것은 아니다.

## 3. 시공간 특징과 분류 head

각 3×3×3 합성곱은 padding1로 크기를 유지하고 MaxPool3d(2)가 시간·공간을 절반으로 줄인다. 기본 입력의 특징은 32×8×32×32, 64×4×16×16, 128×2×8×8이다. 전역 평균 풀링 뒤 B×128을 64차원과 K클래스 로짓으로 변환한다. 세 풀링을 거쳐야 하므로 T·H·W가 충분히 커야 한다.

교차 엔트로피는 raw 로짓과 정수 정답을 사용한다. 학습 전에 softmax를 따로 적용할 필요가 없다. 학습 모드에서는 dropout·BN 통계를 사용하고 검증은 eval·no_grad로 동작을 바꾼다. 전역 평균은 특징을 요약하며 시간 순서 정보는 그 전에 3D 필터가 처리한 범위에 의존한다.

## 4. 학습 기록과 검증 해석

옵티마이저는 이미 Adam(lr=0.001)이다. ‘Adam으로 변경’은 다른 방법과의 비교가 아니다. 학습률 스케줄러는 검증 손실을 보고 줄이고 best_val_acc는 최고 검증 정확도를 숫자로 기록한다. 최고 시점 가중치를 저장·복원하지 않으므로 마지막 예측에 쓰는 모델이 최고 정확도 시점과 같다는 보장은 없다.

손실은 배치 평균에 배치 크기를 곱해 합한 뒤 표본 수로 나누므로 마지막 작은 배치를 적절히 반영한다. 다만 빈 loader에서는 total=0으로 나눈다. 검증 자료를 학습률 조정과 최고 점수 선택에 사용하므로 독립 시험 결과로 해석하지 않는다. 시연 마지막에도 같은 검증 배치의 일부를 보여 준다.

`ReduceLROnPlateau`의 verbose 인수 등은 설치 PyTorch 버전의 API와 맞춰야 한다. 출력 곡선은 해당 합성 문제의 기록일 뿐 실제 행동 인식 데이터의 수렴이나 일반화 성능을 보장하지 않는다. 여기서는 코드·출력을 유지하며 학습을 다시 실행하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
Simple3DCNN의 기본 num_classes=10과 시연 num_classes=5의 매개변수 수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    합성곱·BN 합계는 첫 블록2688, 둘째55488, 셋째221568개다. 128→64 선형은8256개이고 마지막은65K개다. 총288000+65K이므로 기본10클래스는288650개, 시연5클래스는288325개다. BN의 running 통계는 학습 가능한 매개변수가 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 옵티마이저와 best_val_acc의 의미를 설명하라.

</div>

??? success "연습문제 2 풀이"

    이미 Adam을 lr0.001로 사용한다. 다른 옵티마이저와 비교하려면 명시적으로 대조 방법·초기화·데이터·학습률 정책을 정한다. best_val_acc는 최고 기록 숫자이며 해당 가중치를 보관하지 않는다. 마지막 모델을 최고 시점 모델이라고 부르면 안 된다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
좋은 검증 정확도가 움직임 학습을 입증하지 못하는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    색이 클래스마다 고정되고 궤적도 정형화되어 공간적 지름길이 가능하다. 학습·검증 모두 같은 규칙이다. 클래스와 색을 분리하고 궤적·속도·배경을 다양화한 평가 및 시간 정보 제거 대조를 통해 어떤 신호를 사용하는지 확인한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList로 풀링 블록을 늘릴 때 시간 차원에 생길 수 있는 문제를 설명하라.

</div>

??? success "연습문제 4 풀이"

    16프레임을 네 번 절반으로 줄이면1, 다섯 번째 MaxPool3d(2)는 유효하지 않다. 채널뿐 아니라 단계별 최소 시간·공간 크기를 추적해야 한다. YourBlock은 별도 정의가 필요한 자리 표시자이며 모듈 등록만으로 크기·입력 검증이 완성되지 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

합성 클립의 3D CNN 학습은 BCTHW 입력·시공간 풀링·로짓 손실과 학습 기록을 이해하는 예다. 색상 지름길·고정 궤적·검증 재사용 때문에 움직임 인식의 일반화를 입증하지 않으며 최고 기록과 마지막 가중치도 구분한다.
