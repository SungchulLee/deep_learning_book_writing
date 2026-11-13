# 사용자 정의 데이터셋

분류 데이터를 CSV, 개별 파일, 메모리 텐서에서 읽는 여러 Dataset 구현을 비교한다. 데이터 접근과 모델의 순전파는 서로 다른 역할이다.

## 1. 코드

```python
"""
================================================================================
01_custom_dataset.py - 사용자 정의 Dataset 클래스 만들기
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
        
        # 학습 중에 어그멘테이션을 적용한다
        if self.augment:
            # 가우스 잡음을 더한다
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
# 5부: 사용자 정의 데이터셋으로 학습하기
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

# 학습에 CSV 데이터셋을 쓴다
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
   ✓ 학습 중에 데이터를 어그멘테이션할 때
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
   ✓ 학습 중에 __getitem__에서 적용한다
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
print("NEXT: 02_multiclass_classification.py - Beyond binary classification")
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
       ✓ 학습 중에 데이터를 어그멘테이션할 때
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
       ✓ 학습 중에 __getitem__에서 적용한다
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
    NEXT: 02_multiclass_classification.py - Beyond binary classification
    ================================================================================
    ```
## 2. 논의

### 표본의 구조를 통일하기

각 데이터셋은 `__len__`과 `__getitem__`으로 표본 수와 표본 접근을 정의한다. 피처와 정답을 같은 인덱스로 반환하고, DataLoader로 묶인 정답이 모델 출력과 같은 $(B,1)$ 형태인지 확인한다. CSV 열 이름과 결측값, 파일별 피처 수 등은 데이터 읽기 단계에서 검증한다.

### 메모리와 디스크 접근

CSV 구현은 데이터를 미리 읽고, 파일 기반 구현은 표본에 접근할 때 파일을 읽는다. 지연 로딩은 메모리 사용을 줄일 수 있지만 디스크 I/O와 작업자 관리 비용이 생긴다. 파일마다 같은 형태와 자료형을 반환해야 배치를 구성할 수 있다.

### 어그멘테이션과 원본 보존

`AugmentedDataset`은 피처를 복사한 뒤 잡음을 더해 원본을 덮어쓰지 않는다. 잡음을 더해도 정답이 유지된다는 가정이 타당한지 확인하고, 평가용 데이터에는 학습용 확률적 어그멘테이션을 그대로 적용하지 않는다. Dataset 자체를 학습 파라메터를 가진 `nn.Module`로 취급하면 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
Dataset 구현과 `nn.Module` 모델의 차이를 설명하라. `CustomDataset` 자체에 선형 레이어의 파라메터 수 공식을 적용할 수 있는가?

</div>

??? success "풀이"
    Dataset은 데이터 접근 인터페이스이며 그 자체가 학습 레이어는 아니다. 예제의 별도 로지스틱 모델은 입력 피처 수가 $d$일 때 웨이트 $d$개와 바이어스 1개를 가진다. 데이터셋 길이는 학습 파라메터 수와 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 그래디언트와 그래디언트 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
파일 기반 데이터셋과 확률적 어그멘테이션에서 발생할 수 있는 문제를 설명하라.

</div>

??? success "풀이"
    파일 간 피처 수·정답 위치가 다르면 배치가 틀어질 수 있다. 잡음을 더한 피처가 원래 정답을 유지하는지 확인하고 평가셋에는 의도하지 않은 어그멘테이션을 적용하지 않는다. 원본 텐서의 제자리 수정도 피한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반적인 레이어 확장 코드가 이 페이지의 원래 클래스에 그대로 적용되는지 검토하라. 모델과 데이터셋의 역할, 생성자 인수, 입력·출력 형태와 비선형성에서 필요한 수정 사항을 설명하라.

</div>

??? success "풀이"
    아래 코드는 일반적인 ModuleList 구성 패턴의 일부이며 이 페이지 클래스의 완성된 대체 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    적용하려면 생성자와 `forward`를 함께 정의하고 마지막 출력 형태와 로스의 입력 조건을 유지해야 한다. 선형 레이어만 여러 개 합성하면 여전히 하나의 선형 사상이므로 표현력을 늘리려면 적절한 비선형성이 필요하다. Dataset은 표본 접근 인터페이스이므로 레이어를 등록할 대상이 아니며 별도 모델 클래스에 이 구성을 적용한다.

## 정리하며

Dataset은 표본 접근을, 모델은 예측 계산을 담당한다. 피처·정답의 정렬과 형태, 어그멘테이션의 타당성을 데이터 경계에서 확인한다.
