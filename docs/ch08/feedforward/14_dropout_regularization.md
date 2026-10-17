# 드롭아웃 정규화

같은 선형 레이어 구조에 드롭아웃을 넣은 모델과 넣지 않은 모델을 비교한다. 두 클래스는 대안 모델이며, 서로 연결해 실행하는 부품이 아니다.

## 1. 코드

```python
"""
08_dropout_regularization.py - 과적합 막기

드롭아웃을 비롯한 정칙화 기법으로 모델이 학습 데이터를
외우지 못하게 하는 법을 배운다.

과적합: 학습 데이터에서는 잘하지만 시험 데이터에서는 못한다
해결책: 일반화를 북돋우는 정칙화 기법

소요 시간: 25~30분 | 난이도: ⭐⭐⭐☆☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================

print("="*70)
print("Dropout and Regularization")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST 불러오기
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

# 드롭아웃이 없는 모델 (과적합하기 쉽다)
class NoDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 드롭아웃이 있는 모델 (일반화가 더 좋다)
class DropoutNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        # 인스턴스 하나를 아래에서 두 번 쓴다. 드롭아웃은 배울 것도
        # 기억할 것도 없고 부를 때마다 새 마스크를 뽑을 뿐이라 안전하다.
        # 배치 정규화라면 이동 통계가 섞여 이렇게 쓰면 안 된다
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        # 마지막 레이어 앞에서 멈추고 출력에는 걸지 않는다. 로짓을 지우면
        # 그 표본의 예측이 통째로 망가진다.
        # 학습 때 살아남은 엑티베이션은 1/(1-p)배로 부풀려진다. 그래야
        # 평균이 유지되어, 평가 때 드롭아웃을 꺼도 눈금이 어긋나지 않는다
        x = self.dropout(x)  # 활성화 뒤의 드롭아웃
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def train_and_evaluate(model, name, epochs=10):
    """모델을 학습시키고 학습/시험 정확도를 기록한다."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 주의: 이 weight_decay가 두 모델 모두에 걸린다. 곧 "드롭아웃이
    # 없는 모델"도 정칙화가 아주 없는 것은 아니다. 드롭아웃만의 효과를
    # 보려면 이 값을 0으로 두고 견주어야 한다
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # L2 정칙화
    
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        # 학습
        model.train()  # 드롭아웃 켜기
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 주의: outputs는 드롭아웃이 켜진 채, 그것도 step() 앞에서
            # 계산한 값이다. 그래서 여기 나오는 학습 정확도는 드롭아웃
            # 모델 쪽이 실제보다 낮게 잡힌다.
            # 이 페이지가 보려는 것이 학습 정확도와 시험 정확도의
            # 간격이므로, 그 간격의 일부는 드롭아웃 자체가 만든 착시다.
            # 정확히 재려면 에포크 끝에 eval()로 학습 집합을 다시 훑어야 한다
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # 시험
        model.eval()  # 드롭아웃 끄기
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        
        print(f"{name} - Epoch {epoch+1}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    return train_accs, test_accs

# 모델을 만들기 직전마다 같은 씨앗을 심는다. 두 모델의 레이어 구성이 같으므로
# 초기 웨이트도 같아지고, 배치 순서까지 같은 자리에서 출발한다. 그래야
# 아래 표에 남는 차이를 드롭아웃 탓으로 돌릴 수 있고, 이 쪽에 실린 수도
# 다시 돌렸을 때 그대로 나온다
print("\nTraining WITHOUT Dropout:")
torch.manual_seed(42)
no_drop_train, no_drop_test = train_and_evaluate(NoDropoutNet(), "No Dropout")

print("\nTraining WITH Dropout:")
torch.manual_seed(42)
drop_train, drop_test = train_and_evaluate(DropoutNet(0.5), "Dropout 0.5")

# 비교 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(no_drop_train, 'b-', label='No Dropout', linewidth=2)
ax1.plot(drop_train, 'r-', label='With Dropout', linewidth=2)
ax1.set_title('Training Accuracy', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(no_drop_test, 'b-', label='No Dropout', linewidth=2)
ax2.plot(drop_test, 'r-', label='With Dropout', linewidth=2)
ax2.set_title('Test Accuracy', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_dropout_comparison.png', dpi=150)
print("\nPlot saved as '08_dropout_comparison.png'")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
DROPOUT:
  - 학습 중 뉴런을 무작위로 떨어뜨린다
  - 피처들이 함께 적응하는 것을 막는다
  - 여러 뉴럴 넷트웍의 앙상블처럼 작동한다
  - 은닉 레이어의 일반적인 비율: 0.2~0.5
  
USAGE:
  model.train()  # 학습 중에는 드롭아웃 켜기
  model.eval()   # 평가 중에는 드롭아웃 끄기
  
그 밖의 정칙화:
  - L2(최적화기의 weight_decay): 큰 웨이트에 벌점을 준다
  - L1: 희소성을 북돋운다
  - 조기 종료: 검증 로스가 늘면 멈춘다
  - 데이터 어그멘테이션: 데이터셋을 인위적으로 늘린다
  
언제 쓰는가:
  ✓ 작은 데이터셋에 큰 모델을 쓸 때
  ✓ 모델이 과적합한다(학습 정확도 >> 시험 정확도)
  ✓ 깊은 뉴럴 넷트웍
  
드롭아웃 비율:
  - 입력 레이어: 0.1~0.2(낮게)
  - 은닉 레이어: 0.3~0.5(높게)
  - 출력 레이어: 절대 쓰지 마라!
""")

print("\nEXERCISES:")
print("1. Try different dropout rates (0.1, 0.3, 0.7)")
print("2. Compare L1 vs L2 regularization")
print("3. Implement early stopping")
print("4. Add dropout to different positions in network")
plt.show()


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
Dropout and Regularization
======================================================================

Training WITHOUT Dropout:
No Dropout - Epoch 1: Train=91.8%, Test=96.2%
No Dropout - Epoch 2: Train=96.9%, Test=97.0%
No Dropout - Epoch 3: Train=97.8%, Test=97.6%
No Dropout - Epoch 4: Train=98.4%, Test=97.9%
No Dropout - Epoch 5: Train=98.7%, Test=97.9%
No Dropout - Epoch 6: Train=98.9%, Test=97.6%
No Dropout - Epoch 7: Train=99.1%, Test=97.8%
No Dropout - Epoch 8: Train=99.1%, Test=98.1%
No Dropout - Epoch 9: Train=99.2%, Test=97.7%
No Dropout - Epoch 10: Train=99.4%, Test=97.7%

Training WITH Dropout:
Dropout 0.5 - Epoch 1: Train=88.2%, Test=95.6%
Dropout 0.5 - Epoch 2: Train=94.8%, Test=96.8%
Dropout 0.5 - Epoch 3: Train=96.0%, Test=97.3%
Dropout 0.5 - Epoch 4: Train=96.4%, Test=97.4%
Dropout 0.5 - Epoch 5: Train=96.8%, Test=97.6%
Dropout 0.5 - Epoch 6: Train=97.1%, Test=97.8%
Dropout 0.5 - Epoch 7: Train=97.2%, Test=98.0%
Dropout 0.5 - Epoch 8: Train=97.4%, Test=98.0%
Dropout 0.5 - Epoch 9: Train=97.5%, Test=98.1%
Dropout 0.5 - Epoch 10: Train=97.6%, Test=97.8%

Plot saved as '08_dropout_comparison.png'

======================================================================
KEY TAKEAWAYS
======================================================================

DROPOUT:
  - 학습 중 뉴런을 무작위로 떨어뜨린다
  - 피처들이 함께 적응하는 것을 막는다
  - 여러 뉴럴 넷트웍의 앙상블처럼 작동한다
  - 은닉 레이어의 일반적인 비율: 0.2~0.5
  
USAGE:
  model.train()  # 학습 중에는 드롭아웃 켜기
  model.eval()   # 평가 중에는 드롭아웃 끄기
  
그 밖의 정칙화:
  - L2(최적화기의 weight_decay): 큰 웨이트에 벌점을 준다
  - L1: 희소성을 북돋운다
  - 조기 종료: 검증 로스가 늘면 멈춘다
  - 데이터 어그멘테이션: 데이터셋을 인위적으로 늘린다
  
언제 쓰는가:
  ✓ 작은 데이터셋에 큰 모델을 쓸 때
  ✓ 모델이 과적합한다(학습 정확도 >> 시험 정확도)
  ✓ 깊은 뉴럴 넷트웍
  
드롭아웃 비율:
  - 입력 레이어: 0.1~0.2(낮게)
  - 은닉 레이어: 0.3~0.5(높게)
  - 출력 레이어: 절대 쓰지 마라!


EXERCISES:
1. Try different dropout rates (0.1, 0.3, 0.7)
2. Compare L1 vs L2 regularization
3. Implement early stopping
4. Add dropout to different positions in network
```

## 2. 논의

두 모델은 `784→512→256→10` 구조로 파라메터 수가 같다. `DropoutNet`은 은닉 활성화에 무작위 마스크를 적용하지만 학습 가능한 계수를 추가하지 않는다.

PyTorch는 학습 중 유지된 값을 `1/(1-p)`배하고 평가 중에는 드롭아웃을 끈다. 따라서 학습 중 로스와 평가 모드의 로스는 서로 다른 동작 아래 측정될 수 있다.

이 코드의 학습 함수는 두 모델 모두에 weight_decay가 있는 Adam을 사용한다. 비교는 완전히 정규화가 없는 모델 대 정규화 모델이 아니라, 그 설정에 드롭아웃을 더한 효과이다. 테스트 결과를 보고 확률을 선택하려면 별도의 검증 분할이 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
두 모델의 파라메터 수와 드롭아웃에 의한 차이를 구하라.

</div>

??? success "풀이"
    선형 레이어별로 401920, 131328, 2570개이며 합계 535818개이다. 드롭아웃은 학습 가능한 파라메터가 없어 두 모델의 수가 같다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
p=0.5인 드롭아웃의 학습·평가 동작을 설명하라.

</div>

??? success "풀이"
    학습 중 각 원소가 0이 될 확률은 0.5이고 살아남은 값은 두 배가 된다. 평가 중에는 항등 함수이다. 매번 정확히 절반의 원소가 0이 된다는 뜻은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
드롭아웃을 추가한 모델의 학습 정확도가 낮아지면 실패인가?

</div>

??? success "풀이"
    학습 중 무작위 마스크가 켜져 있으면 정확도가 낮아질 수 있다. 같은 평가 모드와 분할에서 검증 성능을 비교해야 한다. 작은 학습·검증 격차만으로 좋은 일반화를 판단해서도 안 된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

드롭아웃은 모델 크기를 줄이지 않고 학습 중 활성화를 무작위로 바꾼다. 효과를 해석할 때 웨이트 감쇠 같은 다른 조건과 학습·평가 모드 차이를 함께 확인한다.
