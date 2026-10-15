# 사용자 정의 데이터셋

분류 데이터를 CSV, 개별 파일, 메모리 텐서에서 읽는 여러 Dataset 구현을 비교한다. 데이터 접근과 모델의 순전파는 서로 다른 역할이다.

## 1. 코드

```python
"""
================================================================================
01_custom_dataset.py - 맞춤 Dataset 클래스 만들기
================================================================================

학습 목표:
- 어떤 데이터 꼴에도 맞는 맞춤 Dataset 클래스를 만든다
- __len__과 __getitem__ 방법을 짠다
- 여러 데이터 클래스(CSV, 그림, 글월)를 다룬다
- 불러오면서 바꾸기를 건다
- 데이터 불러오기의 좋은 버릇

소요 시간: 2시간쯤
어려움: ⭐⭐⭐⭐☆ (앞선)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Callable
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
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
붙박이 TensorDataset은 단순한 자리에 쓰지만, 다음에는 맞춤 Dataset 클래스가
필요하다.

✓ 두루마리에서 데이터 불러오기(기억 자리에 다 안 들어갈 때)
✓ 복잡한 데이터 클래스(그림, 글월, 소리)
✓ 그때그때 미리 다듬기와 불리기
✓ 여러 데이터 밑동
✓ 남다른 뽑기 꾀
✓ 기억 자리를 아끼는 불러오기

맞춤 Dataset 본:
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
    CSV 두루마리에서 데이터를 불러오는 맞춤 Dataset
    
    Args:
        csv_file (str): CSV 두루마리의 길
        feature_cols (list): 특징 칸 이름의 목록
        target_col (str): 과녁 칸의 이름
        transform (callable, 골라 쓴다): 걸어 줄 바꾸기
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
        
        # 특징과 목표를 꺼낸다
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
            튜플: (특징, 과녁)
        """
        # 특징과 목표를 얻는다
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
    필요할 때만 데이터를 불러오는 Dataset(게으른 불러오기)
    기억 자리에 다 안 들어가는 큰 데이터셋에 쓸모 있다
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
# 4부: 예제 3 - 증강을 적용한 데이터셋
# =============================================================================

print("\n" + "="*80)
print("PART 4: DATASET WITH ON-THE-FLY AUGMENTATION")
print("="*80)

class AugmentedDataset(Dataset):
    """
    불러오면서 데이터 불리기를 거는 Dataset
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
        
        # 학습 중에 증강을 적용한다
        if self.augment:
            # 가우스 잡음을 더한다
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise
            
            # 다른 증강을 더할 수도 있다:
            # - 무작위 배율 조정
            # - 무작위 특징 드롭아웃
            # - 등등
        
        return features, target


# 증강을 보여준다
print("\nDemonstrating augmentation...")
X_sample = np.random.randn(100, 5)
y_sample = np.random.randint(0, 2, 100)

aug_dataset = AugmentedDataset(X_sample, y_sample, augment=True, noise_std=0.1)
no_aug_dataset = AugmentedDataset(X_sample, y_sample, augment=False)

# 증강한 경우와 하지 않은 경우의 같은 표본을 얻는다
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
1. 맞춤 Dataset의 짜임
   ✓ torch.utils.data.Dataset을 물려받아야 한다
   ✓ __len__과 __getitem__을 짜야 한다
   ✓ 초기값 잡는 논리는 마음대로 더할 수 있다

2. 언제 쓸까
   ✓ 데이터가 기억 자리에 다 안 들어갈 때
   ✓ 복잡한 데이터 클래스(그림, 글월, 소리)
   ✓ 그때그때 미리 다듬을 때
   ✓ 익히는 동안 데이터를 불릴 때
   ✓ 여러 데이터 밑동

3. 좋은 버릇
   ✓ 데이터를 게으르게 불러라(__getitem__에서)
   ✓ 기억 자리에는 두루마리 길만 담아라
   ✓ 될 수 있으면 미리 다듬은 데이터를 저장하라
   ✓ 자리 잡기를 제대로 하라
   ✓ 한결같은 텐서 클래스를 돌려주어라

4. 흔한 무늬
   ✓ CSV/엑셀: 두루마리 길을 담고 __getitem__에서 읽는다
   ✓ 그림: 그림 길을 담고 PIL/OpenCV으로 불러온다
   ✓ 글월: 두루마리 길을 담고 그때그때 토막으로 나눈다
   ✓ 큰 데이터셋: 기억 자리에 이어 붙인 배열

5. AUGMENTATION
   ✓ 익히는 동안 __getitem__에서 건다
   ✓ 깃발로 켜고 끈다
   ✓ 마구잡이 바꾸기
   ✓ 데이터셋을 사실상 늘린다
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. 쉬움: CSVDataset에 정규화를 더하여라
   __init__에서 StandardScaler을 걸어라

2. 보통: ImageDataset 클래스를 만들어라.
   - 자리표에서 그림을 불러온다
   - 바꾸기를 건다(크기 바꾸기, 정규화)
   - RGB과 잿빛을 다룬다

3. 보통: 저장을 짜라.
   - 불러온 표본을 기억 자리에 저장한다
   - 기억 자리가 차면 저장을 비운다

4. 어려움: TextDataset을 만들어라.
   - 글월 두루마리를 불러온다
   - 그때그때 토막으로 나눈다
   - 낱말 사전을 만든다
   - 덧댄 이음을 돌려준다

5. 어려움: 가중치 실은 뽑기를 짜라.
   - 치우친 클래스를 고르게 한다
   - WeightedRandomSampler을 쓴다
   - 단순한 늘려 뽑기와 견준다
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

    붙박이 TensorDataset은 단순한 자리에 쓰지만, 다음에는 맞춤 Dataset 클래스가
    필요하다.

    ✓ 두루마리에서 데이터 불러오기(기억 자리에 다 안 들어갈 때)
    ✓ 복잡한 데이터 클래스(그림, 글월, 소리)
    ✓ 그때그때 미리 다듬기와 불리기
    ✓ 여러 데이터 밑동
    ✓ 남다른 뽑기 꾀
    ✓ 기억 자리를 아끼는 불러오기

    맞춤 Dataset 본:
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

    1. 맞춤 Dataset의 짜임
       ✓ torch.utils.data.Dataset을 물려받아야 한다
       ✓ __len__과 __getitem__을 짜야 한다
       ✓ 초기값 잡는 논리는 마음대로 더할 수 있다

    2. 언제 쓸까
       ✓ 데이터가 기억 자리에 다 안 들어갈 때
       ✓ 복잡한 데이터 클래스(그림, 글월, 소리)
       ✓ 그때그때 미리 다듬을 때
       ✓ 익히는 동안 데이터를 불릴 때
       ✓ 여러 데이터 밑동

    3. 좋은 버릇
       ✓ 데이터를 게으르게 불러라(__getitem__에서)
       ✓ 기억 자리에는 두루마리 길만 담아라
       ✓ 될 수 있으면 미리 다듬은 데이터를 저장하라
       ✓ 자리 잡기를 제대로 하라
       ✓ 한결같은 텐서 클래스를 돌려주어라

    4. 흔한 무늬
       ✓ CSV/엑셀: 두루마리 길을 담고 __getitem__에서 읽는다
       ✓ 그림: 그림 길을 담고 PIL/OpenCV으로 불러온다
       ✓ 글월: 두루마리 길을 담고 그때그때 토막으로 나눈다
       ✓ 큰 데이터셋: 기억 자리에 이어 붙인 배열

    5. AUGMENTATION
       ✓ 익히는 동안 __getitem__에서 건다
       ✓ 깃발로 켜고 끈다
       ✓ 마구잡이 바꾸기
       ✓ 데이터셋을 사실상 늘린다


    ================================================================================
    EXERCISES
    ================================================================================

    1. 쉬움: CSVDataset에 정규화를 더하여라
       __init__에서 StandardScaler을 걸어라

    2. 보통: ImageDataset 클래스를 만들어라.
       - 자리표에서 그림을 불러온다
       - 바꾸기를 건다(크기 바꾸기, 정규화)
       - RGB과 잿빛을 다룬다

    3. 보통: 저장을 짜라.
       - 불러온 표본을 기억 자리에 저장한다
       - 기억 자리가 차면 저장을 비운다

    4. 어려움: TextDataset을 만들어라.
       - 글월 두루마리를 불러온다
       - 그때그때 토막으로 나눈다
       - 낱말 사전을 만든다
       - 덧댄 이음을 돌려준다

    5. 어려움: 가중치 실은 뽑기를 짜라.
       - 치우친 클래스를 고르게 한다
       - WeightedRandomSampler을 쓴다
       - 단순한 늘려 뽑기와 견준다


    ================================================================================
    NEXT: 02_multiclass_classification.py - Beyond binary classification
    ================================================================================
    ```
## 2. 논의

### 표본의 구조를 통일하기

각 데이터셋은 `__len__`과 `__getitem__`으로 표본 수와 표본 접근을 정의한다. 특징과 정답을 같은 인덱스로 반환하고, DataLoader로 묶인 정답이 모델 출력과 같은 $(B,1)$ 모양인지 확인한다. CSV 열 이름과 결측값, 파일별 특징 수 등은 데이터 읽기 단계에서 검증한다.

### 메모리와 디스크 접근

CSV 구현은 데이터를 미리 읽고, 파일 기반 구현은 표본에 접근할 때 파일을 읽는다. 지연 로딩은 메모리 사용을 줄일 수 있지만 디스크 I/O와 작업자 관리 비용이 생긴다. 파일마다 같은 모양과 자료형을 반환해야 배치를 구성할 수 있다.

### 증강과 원본 보존

`AugmentedDataset`은 특징을 복사한 뒤 잡음을 더해 원본을 덮어쓰지 않는다. 잡음을 더해도 정답이 유지된다는 가정이 타당한지 확인하고, 평가용 데이터에는 훈련용 확률적 증강을 그대로 적용하지 않는다. Dataset 자체를 학습 파라미터를 가진 `nn.Module`로 취급하면 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
Dataset 구현과 `nn.Module` 모델의 차이를 설명하라. `CustomDataset` 자체에 선형 계층의 파라미터 수 공식을 적용할 수 있는가?

</div>

??? success "연습문제 1 풀이"
    Dataset은 데이터 접근 인터페이스이며 그 자체가 학습 계층은 아니다. 예제의 별도 로지스틱 모델은 입력 특징 수가 $d$일 때 가중치 $d$개와 편향 1개를 가진다. 데이터셋 길이는 학습 파라미터 수와 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

</div>

??? success "연습문제 2 풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 경사와 경사 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 학습률에 달려 있다. 같은 초기 가중치와 데이터 순서, 평가 기준을 사용하고 각 학습률을 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
파일 기반 데이터셋과 확률적 증강에서 발생할 수 있는 문제를 설명하라.

</div>

??? success "연습문제 3 풀이"
    파일 간 특징 수·정답 위치가 다르면 배치가 틀어질 수 있다. 잡음을 더한 특징이 원래 정답을 유지하는지 확인하고 평가셋에는 의도하지 않은 증강을 적용하지 않는다. 원본 텐서의 제자리 수정도 피한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반적인 계층 확장 코드가 이 페이지의 원래 클래스에 그대로 적용되는지 검토하라. 모델과 데이터셋의 역할, 생성자 인수, 입력·출력 모양과 비선형성에서 필요한 수정 사항을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 일반적인 ModuleList 구성 패턴의 일부이며 이 페이지 클래스의 완성된 대체 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    적용하려면 생성자와 `forward`를 함께 정의하고 마지막 출력 모양과 손실의 입력 조건을 유지해야 한다. 선형 계층만 여러 개 합성하면 여전히 하나의 선형 사상이므로 표현력을 늘리려면 적절한 비선형성이 필요하다. Dataset은 표본 접근 인터페이스이므로 계층을 등록할 대상이 아니며 별도 모델 클래스에 이 구성을 적용한다.

## 정리하며

Dataset은 표본 접근을, 모델은 예측 계산을 담당한다. 특징·정답의 정렬과 모양, 증강의 타당성을 데이터 경계에서 확인한다.
