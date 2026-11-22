# 사용자 정의 데이터셋

분류 데이터를 CSV, 개별 파일, 메모리 텐서에서 읽는 여러 `Dataset` 구현을 비교한다. `Dataset`은 표본이 몇 개인지(`__len__`)와 번호 하나로 표본 하나를 꺼내는 법(`__getitem__`)만 정하는 PyTorch의 클래스이고, 이것을 `DataLoader`가 배치로 묶어 모델에 넘긴다. 데이터 접근과 모델의 순전파는 서로 다른 역할이다.

## 1. 코드

```python
"""
================================================================================
사용자 정의 Dataset 클래스 만들기
================================================================================

학습 목표:
- 어떤 데이터 형태에도 맞는 사용자 정의 Dataset 클래스를 만든다
- __len__과 __getitem__ 메서드를 구현한다
- 여러 데이터 형식(CSV, 이미지, 텍스트)을 다룬다
- 로딩하면서 변환을 적용한다
- 데이터 로딩의 모범 사례

소요 시간: 2시간쯤
난이도: ⭐⭐⭐⭐☆ (고급)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Callable
# 무작위로 뽑는 값이 아래에 나온다. 시드를 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)
np.random.seed(0)

print("="*80)
print("CUSTOM DATASET CLASSES")
print("="*80)

# =============================================================================
# 1부: 사용자 정의 데이터셋은 왜 필요한가?
# =============================================================================

print("\n" + "="*80)
print("PART 1: WHEN TO USE CUSTOM DATASETS")
print("="*80)

print("""
내장 TensorDataset은 단순한 경우에 쓰지만, 다음에는 사용자 정의 Dataset 클래스가
필요하다.

✓ 파일에서 데이터 로딩(메모리에 다 안 들어갈 때)
✓ 복잡한 데이터 형식(이미지, 텍스트, 오디오)
✓ 실시간 전처리와 어그멘테이션
✓ 여러 데이터 소스
✓ 사용자 정의 샘플링 전략
✓ 메모리 효율적인 로딩

사용자 정의 Dataset 템플릿:
------------------------
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 파일 경로, 메타데이터 등을 읽는다
        pass
    
    def __len__(self):
        # 표본의 총 개수를 반환한다
        return num_samples
    
    def __getitem__(self, idx):
        # 표본 하나를 읽어 반환한다
        # 여기서 변환을 적용할 수 있다
        return sample, label
""")

# =============================================================================
# 2부: 예제 1 - CSV 데이터셋
# =============================================================================

print("\n" + "="*80)
print("PART 2: CUSTOM DATASET FOR CSV FILES")
print("="*80)

class CSVDataset(Dataset):
    """
    CSV 파일에서 데이터를 불러오는 사용자 정의 Dataset
    
    Args:
        csv_file (str): CSV 파일의 경로
        feature_cols (list): 피처 열 이름의 목록
        target_col (str): 타깃 열의 이름
        transform (callable, 선택): 적용할 변환
    """
    
    def __init__(self, 
                 csv_file: str,
                 feature_cols: list,
                 target_col: str,
                 transform: Optional[Callable] = None):
        
        # CSV 파일을 읽는다
        self.data = pd.read_csv(csv_file)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.transform = transform
        
        # 피처와 목표를 꺼낸다
        self.X = self.data[feature_cols].values.astype(np.float32)
        self.y = self.data[target_col].values.astype(np.float32)
        
        print(f"Loaded CSV with {len(self)} samples")
        print(f"Features: {len(feature_cols)}")
    
    def __len__(self):
        """표본의 총 개수를 반환한다"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        표본 하나를 얻는다
        
        Args:
            idx (int): 가져올 표본의 번호
            
        Returns:
            튜플: (피처, 타깃)
        """
        # 피처와 목표를 얻는다
        features = torch.FloatTensor(self.X[idx])
        target = torch.FloatTensor([self.y[idx]])
        
        # 변환이 주어졌으면 적용한다
        if self.transform:
            features = self.transform(features)
        
        return features, target


# 예시 CSV 데이터를 만든다
print("\nCreating sample CSV data...")
sample_data = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.randn(1000),
    'target': np.random.randint(0, 2, 1000)
})

csv_path = "sample_data.csv"
sample_data.to_csv(csv_path, index=False)

# 사용자 정의 CSV 데이터셋을 쓴다
feature_cols = ['feature1', 'feature2', 'feature3']
target_col = 'target'

csv_dataset = CSVDataset(csv_path, feature_cols, target_col)

# DataLoader를 만든다
csv_loader = DataLoader(csv_dataset, batch_size=32, shuffle=True)

print(f"\nDataset length: {len(csv_dataset)}")
print(f"Number of batches: {len(csv_loader)}")

# 배치 하나를 읽어 시험한다
for batch_X, batch_y in csv_loader:
    print(f"\nFirst batch:")
    print(f"  Features shape: {batch_X.shape}")
    print(f"  Targets shape: {batch_y.shape}")
    break

# =============================================================================
# 3부: 예제 2 - 메모리 효율적인 데이터셋
# =============================================================================

print("\n" + "="*80)
print("PART 3: MEMORY-EFFICIENT DATASET")
print("="*80)

class MemoryEfficientDataset(Dataset):
    """
    필요할 때만 데이터를 불러오는 Dataset(지연 로딩)
    메모리에 다 안 들어가는 큰 데이터셋에 쓸모 있다
    """
    
    def __init__(self, data_dir: Path, file_extension: str = '.npy'):
        self.data_dir = Path(data_dir)
        self.file_extension = file_extension
        
        # 실제 데이터가 아니라 파일 경로만 저장한다
        self.file_paths = sorted(list(self.data_dir.glob(f'*{file_extension}')))
        
        print(f"Found {len(self.file_paths)} files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """필요할 때마다 디스크에서 데이터를 읽는다"""
        # 요청이 있을 때만 파일을 읽는다
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        # 마지막 열이 목표값이라고 가정한다
        features = torch.FloatTensor(data[:-1])
        target = torch.FloatTensor([data[-1]])
        
        return features, target


# =============================================================================
# 4부: 예제 3 - 어그멘테이션을 적용한 데이터셋
# =============================================================================

print("\n" + "="*80)
print("PART 4: DATASET WITH ON-THE-FLY AUGMENTATION")
print("="*80)

class AugmentedDataset(Dataset):
    """
    로딩하면서 데이터 어그멘테이션을 적용하는 Dataset
    """
    
    def __init__(self, X, y, augment: bool = True, noise_std: float = 0.1):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        self.augment = augment
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = self.X[idx].clone()
        target = self.y[idx]
        
        # 트레인 중에 어그멘테이션을 적용한다
        if self.augment:
            # 가우시안 잡음을 더한다
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
            
            # 다른 어그멘테이션을 더할 수도 있다:
            # - 무작위 스케일링
            # - 무작위 피처 드롭아웃
            # - 등등
        
        return features, target


# 어그멘테이션을 보여준다
print("\nDemonstrating augmentation...")
X_sample = np.random.randn(100, 5)
y_sample = np.random.randint(0, 2, 100)

aug_dataset = AugmentedDataset(X_sample, y_sample, augment=True, noise_std=0.1)
no_aug_dataset = AugmentedDataset(X_sample, y_sample, augment=False)

# 어그멘테이션한 경우와 하지 않은 경우의 같은 표본을 얻는다
orig_features, _ = no_aug_dataset[0]
aug_features, _ = aug_dataset[0]

print(f"Original sample: {orig_features[:3]}")
print(f"Augmented sample: {aug_features[:3]}")
print(f"Difference: {(aug_features - orig_features)[:3]}")

# =============================================================================
# 5부: 사용자 정의 데이터셋으로 트레인하기
# =============================================================================

print("\n" + "="*80)
print("PART 5: TRAINING WITH CUSTOM DATASET")
print("="*80)

# 모델 생성
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 트레인에 CSV 데이터셋을 쓴다
model = LogisticRegression(3)  # 3 features
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 20
print(f"Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in csv_loader:
        # 순전파
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 지표를 추적한다
        total_loss += loss.item() * len(batch_X)
        predicted_classes = (predictions >= 0.5).float()
        correct += (predicted_classes == batch_y).sum().item()
        total += len(batch_X)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} "
              f"Accuracy: {accuracy:.4f}")

print("\n✓ Training completed with custom dataset!")

# =============================================================================
# 핵심 요점
# =============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. 사용자 정의 Dataset의 구조
   ✓ torch.utils.data.Dataset을 상속해야 한다
   ✓ __len__과 __getitem__을 구현해야 한다
   ✓ 초기화 로직은 자유롭게 더할 수 있다

2. 언제 쓸까
   ✓ 데이터가 메모리에 다 안 들어갈 때
   ✓ 복잡한 데이터 형식(이미지, 텍스트, 오디오)
   ✓ 실시간으로 전처리할 때
   ✓ 트레인 중에 데이터를 어그멘테이션할 때
   ✓ 여러 데이터 소스

3. 모범 사례
   ✓ 데이터를 지연 로딩하라(__getitem__에서)
   ✓ 메모리에는 파일 경로만 담아라
   ✓ 가능하면 전처리한 데이터를 캐시하라
   ✓ 인덱싱을 올바르게 구현하라
   ✓ 일관된 텐서 자료형을 반환하라

4. 흔한 패턴
   ✓ CSV/엑셀: 파일 경로를 담고 __getitem__에서 읽는다
   ✓ 이미지: 이미지 경로를 담고 PIL/OpenCV로 불러온다
   ✓ 텍스트: 파일 경로를 담고 실시간으로 토큰화한다
   ✓ 큰 데이터셋: 메모리 매핑 배열

5. AUGMENTATION
   ✓ 트레인 중에 __getitem__에서 적용한다
   ✓ 플래그로 켜고 끈다
   ✓ 무작위 변환
   ✓ 데이터셋을 사실상 늘린다
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. 쉬움: CSVDataset에 표준화를 더하여라
   __init__에서 StandardScaler를 적용하라

2. 보통: ImageDataset 클래스를 만들어라.
   - 파일 경로에서 이미지를 불러온다
   - 변환을 적용한다(크기 조정, 정규화)
   - RGB와 그레이스케일을 다룬다

3. 보통: 캐싱을 구현하라.
   - 불러온 표본을 메모리에 캐시한다
   - 메모리가 차면 캐시를 비운다

4. 어려움: TextDataset을 만들어라.
   - 텍스트 파일을 불러온다
   - 실시간으로 토큰화한다
   - 어휘 사전을 만든다
   - 패딩한 시퀀스를 반환한다

5. 어려움: 가중 샘플링을 구현하라.
   - 불균형한 클래스의 균형을 맞춘다
   - WeightedRandomSampler를 쓴다
   - 단순한 오버샘플링과 비교한다
""")

print("\n" + "="*80)
print("다음: 「다중 클래스 분류」 쪽 - 이진 분류를 넘어서")
print("="*80)


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (141줄)"

    ```
    ================================================================================
    CUSTOM DATASET CLASSES
    ================================================================================

    ================================================================================
    PART 1: WHEN TO USE CUSTOM DATASETS
    ================================================================================

    내장 TensorDataset은 단순한 경우에 쓰지만, 다음에는 사용자 정의 Dataset 클래스가
    필요하다.

    ✓ 파일에서 데이터 로딩(메모리에 다 안 들어갈 때)
    ✓ 복잡한 데이터 형식(이미지, 텍스트, 오디오)
    ✓ 실시간 전처리와 어그멘테이션
    ✓ 여러 데이터 소스
    ✓ 사용자 정의 샘플링 전략
    ✓ 메모리 효율적인 로딩

    사용자 정의 Dataset 템플릿:
    ------------------------
    class CustomDataset(Dataset):
        def __init__(self, data_path, transform=None):
            # 파일 경로, 메타데이터 등을 읽는다
            pass
        
        def __len__(self):
            # 표본의 총 개수를 반환한다
            return num_samples
        
        def __getitem__(self, idx):
            # 표본 하나를 읽어 반환한다
            # 여기서 변환을 적용할 수 있다
            return sample, label


    ================================================================================
    PART 2: CUSTOM DATASET FOR CSV FILES
    ================================================================================

    Creating sample CSV data...
    Loaded CSV with 1000 samples
    Features: 3

    Dataset length: 1000
    Number of batches: 32

    First batch:
      Features shape: torch.Size([32, 3])
      Targets shape: torch.Size([32, 1])

    ================================================================================
    PART 3: MEMORY-EFFICIENT DATASET
    ================================================================================

    ================================================================================
    PART 4: DATASET WITH ON-THE-FLY AUGMENTATION
    ================================================================================

    Demonstrating augmentation...
    Original sample: tensor([ 0.9455, -0.2014,  1.5105])
    Augmented sample: tensor([ 0.7277, -0.1445,  1.4021])
    Difference: tensor([-0.2179,  0.0568, -0.1085])

    ================================================================================
    PART 5: TRAINING WITH CUSTOM DATASET
    ================================================================================
    Training for 20 epochs...
    Epoch [ 5/20] Loss: 0.6906 Accuracy: 0.5290
    Epoch [10/20] Loss: 0.6913 Accuracy: 0.5360
    Epoch [15/20] Loss: 0.6909 Accuracy: 0.5290
    Epoch [20/20] Loss: 0.6910 Accuracy: 0.5310

    ✓ Training completed with custom dataset!

    ================================================================================
    KEY TAKEAWAYS
    ================================================================================

    1. 사용자 정의 Dataset의 구조
       ✓ torch.utils.data.Dataset을 상속해야 한다
       ✓ __len__과 __getitem__을 구현해야 한다
       ✓ 초기화 로직은 자유롭게 더할 수 있다

    2. 언제 쓸까
       ✓ 데이터가 메모리에 다 안 들어갈 때
       ✓ 복잡한 데이터 형식(이미지, 텍스트, 오디오)
       ✓ 실시간으로 전처리할 때
       ✓ 트레인 중에 데이터를 어그멘테이션할 때
       ✓ 여러 데이터 소스

    3. 모범 사례
       ✓ 데이터를 지연 로딩하라(__getitem__에서)
       ✓ 메모리에는 파일 경로만 담아라
       ✓ 가능하면 전처리한 데이터를 캐시하라
       ✓ 인덱싱을 올바르게 구현하라
       ✓ 일관된 텐서 자료형을 반환하라

    4. 흔한 패턴
       ✓ CSV/엑셀: 파일 경로를 담고 __getitem__에서 읽는다
       ✓ 이미지: 이미지 경로를 담고 PIL/OpenCV로 불러온다
       ✓ 텍스트: 파일 경로를 담고 실시간으로 토큰화한다
       ✓ 큰 데이터셋: 메모리 매핑 배열

    5. AUGMENTATION
       ✓ 트레인 중에 __getitem__에서 적용한다
       ✓ 플래그로 켜고 끈다
       ✓ 무작위 변환
       ✓ 데이터셋을 사실상 늘린다


    ================================================================================
    EXERCISES
    ================================================================================

    1. 쉬움: CSVDataset에 표준화를 더하여라
       __init__에서 StandardScaler를 적용하라

    2. 보통: ImageDataset 클래스를 만들어라.
       - 파일 경로에서 이미지를 불러온다
       - 변환을 적용한다(크기 조정, 정규화)
       - RGB와 그레이스케일을 다룬다

    3. 보통: 캐싱을 구현하라.
       - 불러온 표본을 메모리에 캐시한다
       - 메모리가 차면 캐시를 비운다

    4. 어려움: TextDataset을 만들어라.
       - 텍스트 파일을 불러온다
       - 실시간으로 토큰화한다
       - 어휘 사전을 만든다
       - 패딩한 시퀀스를 반환한다

    5. 어려움: 가중 샘플링을 구현하라.
       - 불균형한 클래스의 균형을 맞춘다
       - WeightedRandomSampler를 쓴다
       - 단순한 오버샘플링과 비교한다


    ================================================================================
    다음: 「다중 클래스 분류」 쪽 - 이진 분류를 넘어서
    ================================================================================
    ```

## 2. 논의

### 표본의 구조를 통일하기

각 데이터셋은 `__len__`과 `__getitem__`으로 표본 수와 표본 접근을 정의한다. 피처와 정답을 같은 인덱스로 반환하고, DataLoader로 묶인 정답이 모델 출력과 같은 $(B,1)$ 형태인지 확인한다($B$는 배치 크기다 — 출력의 `torch.Size([32, 1])`이 그것이다). CSV 열 이름과 결측값, 파일별 피처 수 등은 데이터 읽기 단계에서 검증한다.

5부의 트레인에서 로스가 0.69 근처, 정확도가 0.53 근처에 머무는 것은 정상이다. 예시 CSV의 정답을 `np.random.randint`로 피처와 무관하게 만들었으므로 배울 관계가 없다. 그럴 때 모델이 할 수 있는 최선은 어느 표본에나 확률 0.5를 내는 것이고, 그때의 이진 교차 엔트로피가 $-\ln 0.5 = \ln 2 \approx 0.693$이다. 이 예제가 보이는 것은 Dataset이 트레인 루프에 제대로 물린다는 것이지 분류 성능이 아니다.

### 메모리와 디스크 접근

CSV 구현(`CSVDataset`)은 데이터를 생성자에서 미리 다 읽고, 파일 기반 구현(`MemoryEfficientDataset`)은 표본에 접근할 때 그 파일만 읽는다. 뒤엣것처럼 필요한 순간까지 읽기를 미루는 것을 지연 로딩(lazy loading)이라 한다. 지연 로딩은 메모리 사용을 줄일 수 있지만 디스크 I/O와 워커 프로세스 관리 비용이 생긴다. 파일마다 같은 형태와 자료형을 반환해야 배치를 구성할 수 있다.

### 어그멘테이션과 원본 보존

어그멘테이션은 표본을 조금씩 변형해 트레인 데이터를 늘리는 기법이다. `AugmentedDataset`은 피처를 복사한 뒤 가우시안 잡음을 더해 원본을 덮어쓰지 않는다. 잡음을 더해도 정답이 유지된다는 가정이 타당한지 확인하고, 평가용 데이터에는 트레인용 확률적 어그멘테이션을 그대로 적용하지 않는다. Dataset 자체를 트레인 가능한 파라메터를 가진 `nn.Module`로 취급하면 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
Dataset 구현과 `nn.Module` 모델의 차이를 설명하라. `CustomDataset` 자체에 선형 레이어의 파라메터 수 공식을 적용할 수 있는가?

</div>

??? success "풀이"
    Dataset은 데이터 접근 인터페이스이며 그 자체가 트레인되는 레이어는 아니다. 예제의 별도 로지스틱 모델은 입력 피처 수가 $d$일 때 웨이트 $d$개와 바이어스 1개를 가진다. 데이터셋 길이는 트레인 가능한 파라메터 수와 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 쪽은 `torch.optim.Adam`에 `lr=0.01`을 쓴다. 옵티마이저를 `torch.optim.SGD`(`lr=0.1`)로 바꾸고 원래 옵티마이저와 트레인 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.SGD(model.parameters(), lr=0.1)`로 바꾼다. Adam은 그래디언트와 그래디언트 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
파일 기반 데이터셋과 확률적 어그멘테이션에서 발생할 수 있는 문제를 설명하라.

</div>

??? success "풀이"
    파일 간 피처 수·정답 위치가 다르면 배치가 틀어질 수 있다. 잡음을 더한 피처가 원래 정답을 유지하는지 확인하고 평가 집합에는 의도하지 않은 어그멘테이션을 적용하지 않는다. 원본 텐서의 제자리 수정도 피한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`CSVDataset.__getitem__`은 정답을 `torch.FloatTensor([self.y[idx]])`처럼 대괄호로 감싸 형태 $(1,)$로 돌려준다. 감싸지 않고 형태 $()$인 스칼라로 돌려주면 `DataLoader`가 묶은 배치의 정답 형태는 어떻게 되는가? 그 배치를 5부의 `LogisticRegression(3)`과 `nn.BCELoss()`에 넣으면 무슨 일이 생기며, 정확도를 세는 줄 `(predicted_classes == batch_y)`는 왜 오류 없이 틀린 값을 내는가?

</div>

??? success "풀이"
    `DataLoader`는 표본들을 새 맨 앞 축으로 쌓는다. 표본의 정답이 $(1,)$이면 배치는 $(B,1)$이 되어 모델 출력 $(B,1)$과 맞지만, 정답이 스칼라 $()$이면 배치는 $(B,)$가 된다. `nn.BCELoss`는 예측과 정답의 형태가 다르면 `ValueError`를 내므로 이 어긋남은 로스에서 바로 드러난다.

    위험한 쪽은 오류가 나지 않는 줄이다. `predicted_classes`는 $(B,1)$, `batch_y`는 $(B,)$이므로 `==`가 브로드캐스팅으로 $(B,B)$ 표를 만든다. 표본 $i$의 예측을 **모든** 표본 $j$의 정답과 견준 표이고, `.sum()`은 그 $B^2$칸을 다 세므로 `correct`가 맞힌 표본 수가 아니게 된다. 로스에 넘기는 정답만 `batch_y.unsqueeze(1)`로 고쳐 오류를 없애고 이 줄을 그대로 두면 정확도가 조용히 틀린다. 형태를 Dataset 경계에서 맞추는 까닭이다.

    아래 코드는 이 쪽의 코드를 돌린 뒤에 이어 돌린다(`CSVDataset`, `csv_path`, `feature_cols`, `target_col`, `LogisticRegression`을 그대로 쓴다).

    ```python
    class ScalarTargetCSVDataset(CSVDataset):
        """정답을 (1,)이 아니라 스칼라 ()로 돌려주는 변형"""
        def __getitem__(self, idx):
            features, target = super().__getitem__(idx)
            return features, target.squeeze(0)       # (1,) -> ()

    scalar_dataset = ScalarTargetCSVDataset(csv_path, feature_cols, target_col)
    batch_X, batch_y = next(iter(DataLoader(scalar_dataset, batch_size=32)))
    print(batch_X.shape, batch_y.shape)   # torch.Size([32, 3]) torch.Size([32])

    model = LogisticRegression(3)
    predictions = model(batch_X)          # (32, 1)
    try:
        nn.BCELoss()(predictions, batch_y)
    except ValueError as e:
        print("ValueError:", e)           # 형태가 다르다는 오류

    predicted_classes = (predictions >= 0.5).float()
    print((predicted_classes == batch_y).shape)               # torch.Size([32, 32])
    print((predicted_classes == batch_y.unsqueeze(1)).shape)  # torch.Size([32, 1])
    ```

## 정리하며

Dataset은 표본 접근을, 모델은 예측 계산을 담당한다. 피처·정답의 정렬과 형태, 어그멘테이션의 타당성을 데이터 경계에서 확인한다.
