# 파국적 잊음

**파국적 잊음**(파국적 방해라고도 한다)은 모든 이어 배우기 연구를 밀고 가는 근본 현상이다. 신경망을 여러 과제로 잇달아 익히면 새 과제를 배우면서 앞서 배운 정보를 크게 잊어버리기 일쑤다.

이 절은 수학적 정식화, 실험으로 보여 주기, 그리고 보통의 신경망이 왜 이렇게 굴러가는지에 대한 분석까지 이 문제를 빈틈없이 다룬다.

---

## 1. 지난 이야기

파국적 잊음 문제는 셈 사실을 배우는 연결주의 망을 살핀 McCloskey와 Cohen(1989)이 처음 짚어냈다. 이들은 새 덧셈 문제로 망을 익히면 앞서 배운 덧셈이 심하게 방해받는다는 것을 보았다. 새 앎이 옛 앎을 무너뜨리기보다 북돋우는 사람의 배움과는 사뭇 다른 모습이다.

!!! quote "처음의 관찰"
    "연결주의 망을 새 본새 묶음으로 익히면, 그 새 본새를 배우게 해 주는 이음 가중치의 변화가 대개 앞서 배운 본새를 '잊게' 만든다."
    — 매클로스키 & 코언, 1989

---

## 2. 수학적 틀

### 차례 학습의 얼개

신경망 $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$을 다음과 같은 잇단 과제로 익힌다고 하자.

$$
\mathcal{T}_1 \rightarrow \mathcal{T}_2 \rightarrow \cdots \rightarrow \mathcal{T}_T
$$

과제 $\tau$마다 다음이 딸린다.

- 학습 분포: $p_\tau(x, y)$
- 학습 데이터셋: $\mathcal{D}_\tau = \{(x_i^\tau, y_i^\tau)\}_{i=1}^{N_\tau}$
- 과제마다의 손실: $\mathcal{L}_\tau(\theta) = \mathbb{E}_{(x,y) \sim p_\tau}[\ell(f_\theta(x), y)]$

### 보통의 학습 절차

소박한 차례 학습에서는 과제 $\tau-1$을 마친 뒤 매개변수를 $\theta_{\tau-1}^*$으로 두고 다음을 최적화한다.

$$
\theta_\tau^* = \arg\min_\theta \mathcal{L}_\tau(\theta)
$$

이 최적화는 앞선 과제의 성능을 지키라는 어떤 제약도 없이, 지금 과제의 손실을 가장 작게 하는 방향으로 매개변수를 $\theta_{\tau-1}^*$에서 $\theta_\tau^*$으로 옮긴다.

### 잊음이 일어나는 까닭

걸음마다의 기울기 갱신은 다음과 같다.

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_\tau(\theta_t)
$$

이 갱신은 $\mathcal{L}_\tau$만 살피며, 매개변수의 변화가 $\mathcal{L}_1, \ldots, \mathcal{L}_{\tau-1}$에 어떤 영향을 주는지는 아예 아랑곳하지 않는다.

**핵심 통찰**: 서로 다른 과제의 손실 지형은 대체로 맞물려 있지 않다. $\mathcal{L}_\tau$의 가장 낮은 곳으로 다가가면 대개 앞선 과제 손실의 가장 낮은 곳에서 멀어진다.

---

## 3. 기하학적 해석

### 매개변수 공간에서 보기

매개변수 공간에서 잇단 두 과제의 손실 면을 생각해 보자.

```
           θ₁
            ↑
            |      Task 1 Minimum
            |           ●
            |          ╱ ╲
            |         ╱   ╲
            |        ╱     ╲
     ───────┼───────●───────────→ θ₂
            |   Task 2 Minimum
            |
```

과제 2로 익히면 매개변수가 과제 2의 가장 낮은 곳으로 다가가며 과제 1의 가장 낮은 곳에서 멀어진다. 잊음의 심함은 다음에 달렸다.

1. **가장 낮은 곳끼리의 거리**: 멀수록 → 더 많이 잊는다
2. **손실 면의 굽음**: 골이 날카로울수록 → 움직임에 더 민감하다
3. **차원**: 차원이 높으면 길도 많아지지만 방해도 늘어난다

### 표현의 겹침

신경망은 같은 매개변수가 여러 개념을 담는 흩어진 표현을 배운다. 새 과제로 익힐 때에는 다음이 일어난다.

1. **나누어 쓰는 특징**이 새 과제에 더 맞도록 바뀔 수 있다
2. 옛 과제의 **과제마다의 특징**이 덮어써질 수 있다
3. **용량 배분**이 최근 과제 쪽으로 쏠린다

---

## 4. 실험으로 보이기

### Split MNIST 잣대

표준 보기로 Split MNIST를 쓰는데, MNIST 숫자 10개를 이진 분류 과제 5개로 나눈다.

| 과제 | 부류 | 설명 |
|------|---------|-------------|
| $\mathcal{T}_1$ | {0, 1} | 0과 1을 가른다 |
| $\mathcal{T}_2$ | {2, 3} | 2와 3을 가른다 |
| $\mathcal{T}_3$ | {4, 5} | 4와 5를 가른다 |
| $\mathcal{T}_4$ | {6, 7} | 6과 7을 가른다 |
| $\mathcal{T}_5$ | {8, 9} | 8과 9를 가른다 |

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)

class SimpleNetwork(nn.Module):
    """
    파국적 잊음을 보여 주기 위한 단순한 앞먹임 망.
    
    구조:
        입력(784) → 숨은 층(256) → 숨은 층(256) → 출력(2)
    
    이 구조는 셈으로 다룰 만하면서도 파국적 잊음을
    또렷이 드러나게 한다.
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 편다
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def create_split_mnist_tasks(num_tasks=5):
    """
    Split MNIST 과제 설정을 만든다.
    
    반환값:
        리스트의 리스트이며 tasks[i]은 과제 i의 숫자 부류를 담는다.
    """
    all_digits = list(range(10))
    classes_per_task = 10 // num_tasks
    return [all_digits[i * classes_per_task:(i + 1) * classes_per_task] 
            for i in range(num_tasks)]

def create_task_dataset(full_dataset, task_classes):
    """
    특정 과제의 부류만 걸러 내고 이름표를 다시 매긴다.
    
    인수:
        full_dataset: 온전한 MNIST 데이터셋
        task_classes: 이 과제의 숫자 부류 목록
    
    반환값:
        걸러 낸 보기와 다시 매긴 이름표를 담은 TensorDataset
    """
    indices = [i for i in range(len(full_dataset)) 
               if full_dataset[i][1] in task_classes]
    
    data_list, label_list = [], []
    for idx in indices:
        img, label = full_dataset[idx]
        data_list.append(img)
        # 이진 분류를 위해 이름표를 [0, 1]로 다시 매긴다
        label_list.append(task_classes.index(label))
    
    return TensorDataset(
        torch.stack(data_list),
        torch.tensor(label_list, dtype=torch.long)
    )

def train_on_task(model, train_loader, device, epochs=5, lr=0.001):
    """과제 하나로 모델을 익힌다."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def evaluate_on_task(model, test_loader, device):
    """과제에서 모델의 정확도를 평가한다."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

def demonstrate_forgetting(num_tasks=5, epochs_per_task=5):
    """
    파국적 잊음의 주된 보여 주기.
    
    반환값:
        accuracy_matrix: 성분 [i,j]이 과제 j까지 익힌 뒤
                        과제 i의 정확도인 행렬
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # MNIST 불러오기
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    
    # 과제 데이터 로더를 만든다
    task_classes = create_split_mnist_tasks(num_tasks)
    train_loaders = [
        DataLoader(create_task_dataset(train_data, classes), 
                   batch_size=128, shuffle=True)
        for classes in task_classes
    ]
    test_loaders = [
        DataLoader(create_task_dataset(test_data, classes),
                   batch_size=128, shuffle=False)
        for classes in task_classes
    ]
    
    # 모형을 시작한다
    model = SimpleNetwork(num_classes=2).to(device)
    
    # 정확도를 좇는다: 행은 과제, 열은 학습 단계
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    # 차례 학습
    for task_id in range(num_tasks):
        print(f"\nTraining on Task {task_id} (classes {task_classes[task_id]})")
        
        # 지금 과제로 익힌다
        train_on_task(model, train_loaders[task_id], device, epochs_per_task)
        
        # 지금까지 본 모든 과제에서 평가한다
        for eval_id in range(task_id + 1):
            acc = evaluate_on_task(model, test_loaders[eval_id], device)
            accuracy_matrix[eval_id, task_id] = acc
            
            if eval_id < task_id:
                original = accuracy_matrix[eval_id, eval_id]
                forgetting = original - acc
                print(f"  Task {eval_id}: {acc:.1f}% "
                      f"(was {original:.1f}%, forgot {forgetting:.1f}%)")
            else:
                print(f"  Task {eval_id}: {acc:.1f}%")
    
    return accuracy_matrix, task_classes

# 보여 주기를 돌린다
if __name__ == "__main__":
    acc_matrix, tasks = demonstrate_forgetting()
    
    # 지표를 계산한다
    avg_forgetting = np.mean([
        acc_matrix[i, i] - acc_matrix[i, -1] 
        for i in range(len(tasks) - 1)
    ])
    avg_accuracy = np.mean(acc_matrix[:, -1])
    
    print(f"\n{'='*50}")
    print(f"Average Forgetting: {avg_forgetting:.1f}%")
    print(f"Final Average Accuracy: {avg_accuracy:.1f}%")
```

### 기대되는 결과

이 보기를 돌리면 대개 다음과 같은 정확도 행렬이 나온다.

| 과제 | T0 뒤 | T1 뒤 | T2 뒤 | T3 뒤 | T4 뒤 | 잊음 |
|------|----------|----------|----------|----------|----------|------------|
| 0    | **98.5%** | 62.3% | 55.1% | 51.8% | 49.2% | 49.3% |
| 1    | - | **97.8%** | 58.6% | 52.4% | 50.1% | 47.7% |
| 2    | - | - | **98.2%** | 61.3% | 53.7% | 44.5% |
| 3    | - | - | - | **97.5%** | 58.9% | 38.6% |
| 4    | - | - | - | - | **98.1%** | 해당 없음 |

**핵심 관찰**:

1. 과제마다 처음 배울 때는 정확도가 98% 남짓이다(대각선).
2. 뒤이은 과제를 배우고 나면 성능이 50% 남짓(거의 찍기 수준)으로 떨어진다.
3. 평균 잊음은 대개 40~50%이다.
4. 마지막 평균 정확도는 함께 익혔을 때보다 훨씬 낮다.

---

## 5. 잊음의 심함을 좌우하는 요인

### 망의 구조

| 요인 | 잊음에 미치는 영향 |
|--------|---------------------|
| **망의 깊이** | 깊을수록 더 많이 잊을 수 있다(흔들릴 매개변수가 많다) |
| **너비** | 층이 넓으면 용량은 커지지만 잊음이 반드시 줄지는 않는다 |
| **활성 함수** | ReLU 망이 시그모이드보다 더 흔들릴 수 있다 |
| **건너뛰는 이음** | 표현을 지키는 데 도움이 될 수 있다 |

### 과제의 성격

| 요인 | 잊음에 미치는 영향 |
|--------|---------------------|
| **과제의 닮음** | 닮은 과제일수록 더 많이 방해할 수 있다 |
| **과제의 어려움** | 새 과제가 어려울수록 매개변수를 더 많이 바꾸어야 한다 |
| **과제의 차례** | 차례가 마지막 성능을 크게 좌우할 수 있다 |
| **데이터셋 크기** | 과제마다 학습 데이터가 많을수록 덮어쓰기가 늘 수 있다 |

### 학습 초매개변수

| 요인 | 잊음에 미치는 영향 |
|--------|---------------------|
| **학습률** | 클수록 → 빨리 배우지만 더 많이 잊는다 |
| **시대 수** | 많을수록 → 과제 성능은 좋아지지만 더 많이 잊는다 |
| **배치 크기** | 다다르는 골의 날카로움을 좌우할 수 있다 |
| **최적화기** | 적응 방법은 다른 골로 모일 수 있다 |

---

## 6. 이론적 분석

### 선형 망

선형 망에서는 파국적 잊음을 정확히 뜯어볼 수 있다. 한 층짜리 선형 망을 생각해 보자.

$$
f_\theta(x) = Wx
$$

평균제곱오차 손실로 과제 $\tau$을 익히면 다음의 가장 좋은 해가 나온다.

$$
W_\tau^* = Y_\tau X_\tau^T (X_\tau X_\tau^T)^{-1}
$$

과제들의 입력 분포가 겹치지 않으면, 과제 $\tau$의 해가 그 부분 공간에서 과제 $\tau-1$의 해를 온전히 덮어쓴다.

### 기울기 방해와의 이음

잊음의 심함은 과제 사이의 기울기가 얼마나 맞물리는지와 이어진다. 매개변수 $\theta$에서 과제 $\tau$의 기울기를 다음과 같이 놓자.

$$
g_\tau(\theta) = \nabla_\theta \mathcal{L}_\tau(\theta)
$$

**기울기 방해**는 다음일 때 일어난다.

$$
g_{\tau_1}(\theta)^T g_{\tau_2}(\theta) < 0
$$

이는 한 과제가 나아지면 다른 과제가 나빠진다는 뜻이다. 이 음의 안곱이 클수록 잊음이 심하다.

---

## 7. 안정성과 말랑함의 딜레마

파국적 잊음은 배우는 체계에 깃든 근본적인 팽팽함을 비춘다.

**말랑함**: 새 앎을 얻는 힘
: 새 과제를 제대로 배우는 데 필요하다

**안정성**: 옛 앎을 지니는 힘
: 파국적 잊음을 피하는 데 필요하다

!!! info "딜레마"
    어떤 체계든 서로 겨루는 이 두 요구 사이에서 균형을 잡아야 한다. 안정성만 있으면 배움이 없고, 말랑함만 있으면 모조리 잊는다. 이어 배우기 방법들은 이 맞바꿈을 다스리는 저마다의 길이다.

### 생물학의 눈

흥미롭게도 생물의 신경망은 이만큼 파국적으로 잊지 않는다. 제안된 장치로는 다음이 있다.

1. **성긴 부호화**: 어떤 자극에도 뉴런의 작은 일부만 켜진다
2. **신경 조절**: 맥락에 따라 말랑함의 정도가 달라진다
3. **기억 다지기**: 최근 기억과 먼 기억을 다루는 체계가 따로 있다
4. **신경 생성**: 새 정보를 위해 새 뉴런이 자란다
5. **시냅스 다지기**: 중요한 시냅스를 가려 굳힌다

이런 생물학의 통찰이 여러 이어 배우기 알고리즘에 영감을 주었다.

---

## 8. 잊음 그려 보기

```python
def visualize_forgetting(accuracy_matrix, task_classes):
    """파국적 잊음을 두루 그려 본다."""
    num_tasks = len(task_classes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 열 지도
    ax1 = axes[0]
    im = ax1.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=100)
    ax1.set_xlabel('After Training Task')
    ax1.set_ylabel('Evaluated Task')
    ax1.set_title('Accuracy Matrix: Catastrophic Forgetting', fontweight='bold')
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    ax1.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    ax1.set_yticklabels([f'T{i}' for i in range(num_tasks)])
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:
                ax1.text(j, i, f'{accuracy_matrix[i, j]:.0f}',
                        ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
    
    # 성능이 떨어지는 모습을 보여 주는 선 그림
    ax2 = axes[1]
    for task_id in range(num_tasks):
        accs = [accuracy_matrix[task_id, j] 
                for j in range(task_id, num_tasks)]
        ax2.plot(range(task_id, num_tasks), accs, 
                marker='o', linewidth=2, markersize=8,
                label=f'Task {task_id} (classes {task_classes[task_id]})')
    
    ax2.set_xlabel('Training Stage')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Performance Degradation Over Time', fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(num_tasks))
    ax2.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_visualization.png', dpi=150)
    plt.show()
```

---

## 9. 실전에 주는 뜻

### 파국적 잊음이 중요할 때

1. **로봇 공학**: 새 조작 솜씨를 배운다고 기본 운동 제어가 지워져서는 안 된다
2. **의료 인공지능**: 새 질병을 넣는다고 이미 아는 질환의 알아내기가 나빠져서는 안 된다
3. **추천 체계**: 사용자의 취향은 바뀌지만 지난 본새도 중요하다
4. **자연 언어**: 모델은 두루 쓰이는 힘을 잃지 않고 새 영역에 맞추어야 한다
5. **자율 주행차**: 새 상황을 배운다고 기본 운전 규칙을 잊어서는 안 된다

### 받아들일 만할 때

1. **한 과제짜리 응용**: 차례 학습이 필요 없다
2. **넉넉한 되살리기**: 지난 데이터를 모두 담아 두고 되살릴 수 있을 때
3. **새로 시작하기**: 지난 앎을 일부러 버릴 때
4. **옮김만 중요할 때**: 앞으로의 옮김만 중요할 때(미세 조정)

---

## 연습문제

**연습문제 1.**
파국적 잊음을 정의하고 신경망에서 그것이 왜 일어나는지 설명하라.

??? success "연습문제 1 풀이"
    파국적 잊음이란 새 과제로 익힐 때 신경망이 앞서 배운 앎을 갑자기 잃어버리는 성향이다. 새 과제를 위한 기울기 내려가기 갱신이 옛 과제에 중요했던 가중치를 덮어쓰기 때문에 일어난다. 생물의 뇌와 달리 보통의 망에는 옛 앎을 지킬 장치가 없다.

---

**연습문제 2.**
간단한 실험으로 파국적 잊음을 보여라. 과제 A로 익히고 이어서 과제 B로 익힌 뒤 과제 A의 성능을 재라.

??? success "연습문제 2 풀이"
    MNIST 숫자 0~4(과제 A)로 MLP를 익히면 정확도가 98% 남짓 나온다. 이어서 숫자 5~9(과제 B)로 익힌다. B를 익힌 뒤 과제 A의 정확도는 20% 남짓(찍기 수준)으로 떨어진다. B에 맞추어진 가중치가 A에 필요한 특징을 덮어쓴 것이다.

---

**연습문제 3.**
잊음을 누그러뜨리는 접근법의 큰 세 갈래를 들어라.

??? success "연습문제 3 풀이"

    1. 벌주기 기반(EWC, SI): 중요한 가중치가 바뀌면 벌을 준다. 2. 되살리기 기반(경험 되살리기, A-GEM): 옛 과제의 보기를 담아 두거나 지어낸다. 3. 구조 기반(점진 확장 망, PackNet): 과제마다 따로 매개변수를 떼어 준다. 저마다 기억, 셈, 융통성에서 다른 맞바꿈을 한다.

---

**연습문제 4.**
이어 배우기에서 안정성과 말랑함의 딜레마란 무엇인가?

??? success "연습문제 4 풀이"
    모델은 새 과제를 배울 만큼 말랑하면서도 옛 과제를 지닐 만큼 안정되어야 한다. 안정성이 지나치면 새 과제를 배우지 못한다(모자란 맞춤). 말랑함이 지나치면 옛 과제를 잊는다(파국적 잊음). 모든 이어 배우기 방법은 이 두 목표를 맞바꾼다.

## 정리하며

파국적 잊음은 이어 배우기 방법이 반드시 다루어야 할 한가운데의 어려움이다.

- **정의**: 새 과제를 배울 때 옛 과제의 성능이 크게 떨어지는 일
- **까닭**: 옥죄지 않은 최적화가 중요한 가중치를 덮어쓴다
- **심함**: Split MNIST 잣대에서 대개 정확도가 40~60% 떨어진다
- **근본**: 배우는 체계의 안정성-말랑함 딜레마를 비춘다

뒤이은 절들은 파국적 잊음을 누그러뜨리는 여러 접근법을 내보이는데, 저마다 다음 사이에서 다른 맞바꿈을 준다.

- 기억 효율(지난 보기를 담아 두느냐 마느냐)
- 셈 비용(앞먹임과 되돌림을 더 하느냐)
- 사생활 지키기(데이터를 담아 두지 않기)
- 규모 확장성(잇단 과제를 많이 다루기)

**참고 문헌**

1. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

2. French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

3. Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2013). An empirical investigation of catastrophic forgetting in gradient-based neural networks. *arXiv preprint arXiv:1312.6211*.

4. Kemker, R., McClure, M., Abitino, A., Hayes, T., & Kanan, C. (2018). Measuring catastrophic forgetting in neural networks. *AAAI Conference on Artificial Intelligence*.
