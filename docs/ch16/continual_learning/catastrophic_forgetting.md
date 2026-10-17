# 파국적 망각

파국적 망각은 새 과제를 학습한 뒤 이전 과제의 성능이 크게 떨어지는 현상이다. 새 데이터의 로스만 줄이는 갱신은 이전 과제의 성능을 보존하도록 설계되어 있지 않다. 이 페이지는 순차 학습의 수식과 Split MNIST 예제로 성능 변화를 해석한다.

## 1. 지난 이야기

McCloskey와 Cohen(1989)은 연결주의 뉴럴 넷트웍에서 새로운 산술 문제의 학습이 이전 학습을 방해하는 현상을 분석했다. 이후 순차적으로 바뀌는 데이터에서 이전 성능을 유지하는 문제가 지속 학습의 주요 연구 주제가 되었다. 사람과 뉴럴 넷트웍의 기억을 비교할 수는 있지만 사람에게 간섭이나 망각이 없다고 가정하지 않는다.

## 2. 수학적 틀

### 순차 학습 설정

뉴럴 넷트웍 $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$을 다음과 같은 잇단 과제로 익힌다고 하자.

$$
\mathcal{T}_1 \rightarrow \mathcal{T}_2 \rightarrow \cdots \rightarrow \mathcal{T}_T
$$

과제 $\tau$마다 다음이 딸린다.

- 학습 분포: $p_\tau(x, y)$
- 학습 데이터셋: $\mathcal{D}_\tau = \{(x_i^\tau, y_i^\tau)\}_{i=1}^{N_\tau}$
- 과제마다의 로스: $\mathcal{L}_\tau(\theta) = \mathbb{E}_{(x,y) \sim p_\tau}[\ell(f_\theta(x), y)]$

### 단순 순차 학습

단순 순차 학습에서는 과제 $\tau-1$을 마친 뒤 파라메터를 $\theta_{\tau-1}^*$으로 두고 다음을 최적화한다.

$$
\theta_\tau^* = \arg\min_\theta \mathcal{L}_\tau(\theta)
$$

이 최적화는 앞선 과제의 성능을 지키라는 어떤 제약도 없이, 지금 과제의 로스를 가장 작게 하는 방향으로 파라메터를 $\theta_{\tau-1}^*$에서 $\theta_\tau^*$으로 옮긴다.

### 새 과제 갱신의 영향

스텝마다의 그래디언트 갱신은 다음과 같다.

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_\tau(\theta_t)
$$

이 갱신은 $\mathcal{L}_\tau$만 살피며, 파라메터의 변화가 $\mathcal{L}_1, \ldots, \mathcal{L}_{\tau-1}$에 어떤 영향을 주는지는 아예 아랑곳하지 않는다.

새 과제의 하강 방향이 이전 로스를 높일 수 있다. 반대로 공유할 수 있는 피처를 배우면 이전 과제에도 도움이 될 수 있으므로 모든 과제 전환이 망각으로 이어지는 것은 아니다.

---

## 3. 기하학적 해석

### 파라메터 공간에서 보기

파라메터 공간에서 잇단 두 과제의 로스 면을 생각해 보자.

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

그림은 두 과제의 좋은 해가 다를 수 있음을 보여 주는 도식이다. 최적점 사이의 거리만으로 망각을 결정할 수는 없다. 이전 로스의 곡률과 실제 갱신 방향, 두 과제의 표현 공유와 학습 경로가 함께 영향을 준다.

이전 로스를 크게 바꾸지 않는 방향이 있을 수도 있고, 작은 변화가 큰 로스 증가를 만드는 방향도 있다. 파라메터 수가 많거나 네트워크가 깊다는 사실만으로 간섭의 크기를 단정하지 않는다.

### 표현의 겹침

여러 입력·과제가 같은 파라메터와 피처를 공유한다. 새 과제의 갱신이 이 표현을 바꾸면 이전 출력도 달라질 수 있다. 공유가 전이를 돕는지 간섭을 만드는지는 과제와 로스, 모델 용량에 따라 달라진다.

공유 피처의 변화, 과제별 출력 관계와 모델 용량을 함께 살핀다. 표현 변화가 있다는 사실만으로 이전 과제에 해롭다고 결론내리지 말고 실제 이전 성능을 측정한다.

---

## 4. 실험으로 보이기

### Split MNIST 벤치마크

이 예제는 숫자 10개를 {0,1}, {2,3}, {4,5}, {6,7}, {8,9}의 다섯 과제로 나눈다. 각 과제의 라벨을 0·1로 다시 매기고 같은 2개 출력 레이어를 사용한다. 전체 숫자 10개 중 하나를 고르는 클래스 증분 평가와는 다른 프로토콜이다.

| 과제 | 클래스 | 설명 |
|------|---------|-------------|
| $\mathcal{T}_1$ | {0, 1} | 0과 1을 가른다 |
| $\mathcal{T}_2$ | {2, 3} | 2와 3을 가른다 |
| $\mathcal{T}_3$ | {4, 5} | 4와 5를 가른다 |
| $\mathcal{T}_4$ | {6, 7} | 6과 7을 가른다 |
| $\mathcal{T}_5$ | {8, 9} | 8과 9를 가른다 |

### PyTorch 구현

기본 다섯 과제에서만 각 과제가 이진 분류가 된다. num_tasks를 임의로 바꾸면 정수 나눗셈 때문에 숫자가 누락되거나 2개 출력과 라벨 범위가 맞지 않을 수 있다. 각 과제의 Adam은 새로 만들어지며, 정확도 행렬은 행=평가 과제, 열=학습 단계다. 반환 정확도는 백분율이고 대각선과 마지막 열의 차이는 퍼센트포인트다.

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
    파국적 잊음을 보여 주기 위한 단순한 피드포워드 망.
    
    구조:
        입력(784) → 숨은 레이어(256) → 숨은 레이어(256) → 출력(2)
    
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

**출력:**

```
Training on Task 0 (classes [0, 1])
  Task 0: 99.9%

Training on Task 1 (classes [2, 3])
  Task 0: 75.0% (was 99.9%, forgot 24.9%)
  Task 1: 99.5%

Training on Task 2 (classes [4, 5])
  Task 0: 13.2% (was 99.9%, forgot 86.7%)
  Task 1: 88.6% (was 99.5%, forgot 10.9%)
  Task 2: 99.9%

Training on Task 3 (classes [6, 7])
  Task 0: 76.4% (was 99.9%, forgot 23.5%)
  Task 1: 81.4% (was 99.5%, forgot 18.0%)
  Task 2: 49.3% (was 99.9%, forgot 50.6%)
  Task 3: 99.8%

Training on Task 4 (classes [8, 9])
  Task 0: 23.0% (was 99.9%, forgot 76.9%)
  Task 1: 74.0% (was 99.5%, forgot 25.4%)
  Task 2: 15.5% (was 99.9%, forgot 84.4%)
  Task 3: 82.4% (was 99.8%, forgot 17.4%)
  Task 4: 99.4%

==================================================
Average Forgetting: 51.0%
Final Average Accuracy: 58.9%
```

### 결과를 읽는 방법

학습 직후 대각선 성능과 이후 같은 행의 성능을 비교한다. 마지막 열의 평균은 최종 평균 정확도다. 코드가 출력하는 average forgetting은 대각선에서 최종 값을 뺀 평균으로, 전체 학습 경로의 최댓값을 기준으로 한 망각 지표와 다르다.

정확도가 얼마나 떨어지는지는 초기화·과제 순서·학습 예산과 출력 규약에 따라 달라진다. 40~50퍼센트포인트 같은 고정 범위를 일반적 결과로 제시하지 않는다. 표시된 출력은 보존된 예시이며 이번 편집에서 다시 측정하지 않았다.

---

## 5. 잊음의 심함을 좌우하는 요인

모델의 표현 공유와 용량, 과제 유사도·순서, 데이터 양과 러닝레이트·학습 단계 수가 망각에 영향을 준다. 비슷한 과제는 공유 피처를 통해 도움을 줄 수도 있고 출력 관계가 충돌해 간섭할 수도 있다.

같은 모델과 데이터 순서에서 하나의 설정을 바꾸고 새 과제의 학습 성능과 이전 성능을 함께 측정한다. 네트워크 깊이, 엑티베이션 함수, 에포크 수만으로 망각이 일정 방향으로 변한다고 단정할 수 없다.

## 6. 이론적 분석

### 선형 최소제곱 예시

입력을 열로 모은 X와 정답 Y에 대해 선형 예측은 WX다. XXᵀ가 가역이면 최소제곱 해는 다음과 같다.

$$W^*=YX^\top(XX^\top)^{-1}$$

가역이 아니면 해가 유일하지 않을 수 있으며 의사역행렬과 초기화·최적화 방식이 중요해진다. 새 과제 입력이 이전 입력과 직교하는 부분공간에 있고 경사 갱신이 새 입력의 부분공간에만 작용하면 이전 입력의 예측을 유지할 수도 있다. 입력 분포가 다르다는 사실만으로 이전 해가 완전히 덮어써지는 것은 아니다.

### 경사 간섭의 국소적 해석

이전·새 과제 경사를 g_old, g_new라 하면 작은 러닝레이트 η의 새 과제 갱신에서 다음과 같다.

$$L_{\rm old}(\theta-\eta g_{\rm new})-L_{\rm old}(\theta)\approx-\eta g_{\rm old}^\top g_{\rm new}$$

내적이 음수이면 이전 로스가 증가하는 1차 경향이 있다. 이 관계는 현재 웨이트와 작은 갱신에 대한 근사다. 곡률과 여러 갱신의 누적 때문에 최종 망각을 내적 하나로 결정할 수는 없다.

## 7. 안정성과 가소성의 딜레마

**가소성**은 새 데이터에 맞춰 변하는 능력이고 **안정성**은 이전 과제의 유용한 성능을 유지하는 능력이다. 중요한 갱신을 강하게 제한하면 새 과제 학습이 어려워질 수 있고, 새 로스만 최적화하면 이전 과제를 잊을 수 있다.

생물학의 희소 활성화, 기억 재생과 시냅스 변화에 대한 연구는 지속 학습 설계에 영감을 주었다. 이런 대응 관계가 인공 모델의 망각 방지를 자동으로 보장하는 것은 아니다.

## 8. 망각 시각화

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

로봇·추천·언어 모델처럼 데이터가 순차적으로 바뀌는 시스템에서는 새 조건에 적응한 뒤에도 유지해야 할 이전 능력을 정의한다. 반대로 오래된 관계가 더 이상 유효하지 않다면 선택적인 망각이 필요할 수 있다.

정규화는 웨이트 변화를 제한하고, 리플레이는 과거 표본이나 생성 표본을 다시 사용하며, 증류는 이전 모델의 출력을 유지하도록 학습한다. 구조 기반 방법은 과제별 경로나 파라메터를 분리할 수 있다. 저장·계산 비용과 테스트 시 과제 정보의 필요성을 함께 평가한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
파국적 망각과 새 과제의 학습 실패를 구분하라.

</div>

??? success "풀이"
    파국적 망각은 이전에 잘 수행하던 과제의 성능이 이후 학습으로 크게 낮아진 것이다. 새 과제를 처음부터 잘 배우지 못하는 문제와 다르다. 각 과제의 학습 직후 정확도와 이후 정확도를 함께 기록해야 구분할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
MNIST의 두 과제로 망각을 평가할 때 출력 공간을 어떻게 정해야 하는가?

</div>

??? success "풀이"
    과제별 로컬 라벨·분리 헤드를 쓰는지 전체 10개 클래스의 단일 출력을 쓰는지 먼저 정한다. 첫 과제 학습 직후와 두 번째 과제 학습 후 같은 테스트셋을 평가한다. 기회 수준과 예상 정확도는 후보 수에 따라 달라지므로 20% 등 고정 결과를 미리 정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정규화·리플레이·구조 기반 방법의 주요 자원 차이를 설명하라.

</div>

??? success "풀이"
    정규화는 기준 웨이트와 중요도 등의 통계를 저장할 수 있다. 리플레이는 표본 버퍼나 생성기를 저장하고 추가 학습을 수행한다. 구조 기반 방법은 과제별 파라메터나 마스크를 추가할 수 있다. 각 범주 안에서도 비용과 가정이 달라 실제 저장량과 계산량을 비교한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
안정성과 가소성을 함께 평가해야 하는 이유를 설명하라.

</div>

??? success "풀이"
    웨이트를 전혀 갱신하지 않으면 이전 출력은 유지되지만 새 과제를 배우지 못할 수 있다. 새 과제만 최적화하면 이전 성능을 잃을 수 있다. 따라서 최종 평균 정확도뿐 아니라 새 과제 학습 성능과 과제별 성능 변화를 함께 평가한다.

## 정리하며

새 로스를 줄이는 갱신이 이전 과제의 성능을 손상시킬 수 있다. 망각을 측정하려면 과제·출력 공간과 행렬 축을 고정하고 학습 직후 및 이후 성능을 비교한다. 경사 간섭은 국소적 설명이며 최종 결과는 표현 공유·곡률·학습 경로와 데이터 조건에 따라 달라진다.

**참고 문헌**

1. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

2. French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

3. Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2013). An empirical investigation of catastrophic forgetting in gradient-based neural networks. *arXiv preprint arXiv:1312.6211*.

4. Kemker, R., McClure, M., Abitino, A., Hayes, T., & Kanan, C. (2018). Measuring catastrophic forgetting in neural networks. *AAAI Conference on Artificial Intelligence*.
