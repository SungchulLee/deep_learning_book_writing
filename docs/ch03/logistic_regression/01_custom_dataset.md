# 사용자 정의 데이터셋

01_custom_dataset.py - 사용자 정의 Dataset 클래스 만들기

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
================================================================================
01_custom_dataset.py - 맞춤 Dataset 갈래 만들기
================================================================================

배움 목표:
- 어떤 자료 꼴에도 맞는 맞춤 Dataset 갈래를 만든다
- __len__과 __getitem__ 방법을 짠다
- 여러 자료 갈래(CSV, 그림, 글월)를 다룬다
- 불러오면서 바꾸기를 건다
- 자료 불러오기의 좋은 버릇

마치는 데 드는 때: 2시간쯤
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
붙박이 TensorDataset은 단순한 자리에 쓰지만, 다음에는 맞춤 Dataset 갈래가
필요하다.

✓ 두루마리에서 자료 불러오기(기억 자리에 다 안 들어갈 때)
✓ 복잡한 자료 갈래(그림, 글월, 소리)
✓ 그때그때 미리 다듬기와 불리기
✓ 여러 자료 밑동
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
    CSV 두루마리에서 자료를 불러오는 맞춤 Dataset
    
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

csv_path = "/home/claude/pytorch_logistic_regression_tutorial/03_advanced/sample_data.csv"
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
    필요할 때만 자료를 불러오는 Dataset(게으른 불러오기)
    기억 자리에 다 안 들어가는 큰 자료 묶음에 쓸모 있다
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
    불러오면서 자료 불리기를 거는 Dataset
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
   ✓ 첫자리 잡는 논리는 마음대로 더할 수 있다

2. 언제 쓸까
   ✓ 자료가 기억 자리에 다 안 들어갈 때
   ✓ 복잡한 자료 갈래(그림, 글월, 소리)
   ✓ 그때그때 미리 다듬을 때
   ✓ Data augmentation during training
   ✓ 여러 자료 밑동

3. BEST PRACTICES
   ✓ Load data lazily (in __getitem__)
   ✓ Store only file paths in memory
   ✓ Cache preprocessed data if possible
   ✓ Use proper indexing
   ✓ Return consistent tensor types

4. COMMON PATTERNS
   ✓ CSV/Excel: Load file paths, read in __getitem__
   ✓ Images: Store image paths, load with PIL/OpenCV
   ✓ Text: Store file paths, tokenize on-the-fly
   ✓ Large datasets: Memory-mapped arrays

5. AUGMENTATION
   ✓ Apply in __getitem__ during training
   ✓ Use flags to enable/disable
   ✓ Random transformations
   ✓ Increases dataset size effectively
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. EASY: Add normalization to CSVDataset
   Apply StandardScaler in __init__

2. MEDIUM: Create ImageDataset class:
   - Load images from folder
   - Apply transforms (resize, normalize)
   - Handle RGB/grayscale

3. MEDIUM: Implement caching:
   - Cache loaded samples in memory
   - Clear cache when memory full

4. HARD: Create TextDataset:
   - Load text files
   - Tokenize on-the-fly
   - Create vocabulary
   - Return padded sequences

5. HARD: Implement weighted sampling:
   - Balance imbalanced classes
   - Use WeightedRandomSampler
   - Compare with simple oversampling
""")

print("\n" + "="*80)
print("NEXT: 02_multiclass_classification.py - Beyond binary classification")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

이 구현은 5개의 클래스(`CustomDataset`, `CSVDataset`, `MemoryEfficientDataset`, `AugmentedDataset`, 그리고 하나 더)를 정의하며, 이들이 함께 작동하여 완전한 로지스틱 회귀 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `CustomDataset`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `CustomDataset`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = CustomDataset(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
