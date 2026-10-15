# CNN 유틸리티

cnn_utils.py는 데이터 로딩, CNN 두 종류, 학습·평가와 시각화·저장을 공통 함수로 묶은 모듈이다. [MNIST 데이터셋](01_mnist_dataset.md), [Fashion-MNIST 데이터셋](02_fashion_mnist_dataset.md), [CIFAR-10 데이터셋](03_cifar10_dataset.md), [Fashion-MNIST 분류기](05_fashion_mnist_classifier.md)와 [CIFAR-10 심화](07_cifar10_advanced.md)에서 사용한다. 함수의 입력 규약과 상태 변경을 이해하면 호출하는 예제들의 동작도 확인할 수 있다.

## 1. 코드

```python
"""
cnn_utils.py
============
CNN 실습을 위한 종합 유틸리티 모듈

이 모듈은 CNN 학습에 필요한 공통 함수를 모두 마련해 준다.
- 인자 구문 분석과 설정
- MNIST, Fashion-MNIST, CIFAR-10 데이터 적재
- 모델 구조
- 학습과 평가 반복문
- 시각화 도구
- 모델 저장과 불러오기

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random


# ===================================================================
# 설정과 준비
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CNN Training')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--no-mps', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=False)
    parser.add_argument('--path', type=str, default='./model.pth')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        args.device = torch.device("cuda")
    elif use_mps:
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")
    return args


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================================================================
# 데이터 적재
# ===================================================================

def load_data(train_kwargs, test_kwargs, fashion_mnist=False, cifar10=False):
    if cifar10:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_class = datasets.FashionMNIST if fashion_mnist else datasets.MNIST
        train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
        test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)
    return train_loader, test_loader


# ===================================================================
# 모델 구조
# ===================================================================

class CNN(nn.Module):
    """MNIST와 Fashion-MNIST를 위한 기본 CNN."""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        x = self.dropout1(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CNN_CIFAR10(nn.Module):
    """CIFAR-10을 위한 심화 CNN."""
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.dropout1(self.pool(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.dropout1(self.pool(F.relu(self.conv4(F.relu(self.conv3(x))))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ===================================================================
# 학습과 평가
# ===================================================================

def train(model, train_loader, loss_fn, optimizer, scheduler, device, epochs, log_interval=10, dry_run=False):
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if batch_idx % log_interval == 0:
                print(f'Epoch: {epoch}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if dry_run:
                break
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch}: Avg Loss: {running_loss / len(train_loader):.4f}, Acc: {epoch_acc:.2f}%\n')
        scheduler.step()


def compute_accuracy(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return accuracy


# ===================================================================
# 시각화와 모델 저장
# ===================================================================

def show_batch_or_ten_images_with_label_and_predict(test_loader, model, device,
                                                      classes=None, n=10, cifar10=False):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predictions = outputs.max(1)
    images, labels, predictions = images.cpu(), labels.cpu(), predictions.cpu()
    n_display = min(n, len(images))
    cols = min(5, n_display)
    rows = (n_display + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    if n_display == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for idx in range(n_display):
        ax = axes[idx]
        img = images[idx]
        if cifar10:
            img = (img / 2 + 0.5).permute(1, 2, 0)
            ax.imshow(img.numpy())
        else:
            img = img.squeeze() / 2 + 0.5
            ax.imshow(img.numpy(), cmap='gray')
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        title = f'True: {classes[true_label]}\nPred: {classes[pred_label]}' if classes else f'True: {true_label}\nPred: {pred_label}'
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(title, fontsize=8, color=color)
        ax.axis('off')
    for idx in range(n_display, len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model_class, device, path):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    pass
```

## 2. 모듈 점검

모듈 본문은 함수와 클래스를 정의하며 main 부분은 pass이다. 다음은 별도의 점검 스크립트로, 데이터 로딩 인수와 출력 범위, 두 모델의 차원·매개변수 수와 일부 난수의 재현을 확인한다. 이 점검이 학습·평가·저장의 모든 동작을 검증하는 것은 아니다.

```python
"""cnn_utils 점검 - 이 모듈을 들여오는 쪽들이 기대는 약속을 확인한다.

이 코드는 모듈에 들어 있지 않다. cnn_utils.py 를 그대로 두고 바깥에서
불러 보는 점검용 스크립트다.
"""

import inspect

import torch

import cnn_utils as utils

# =============================================================================
# 1절: load_data 의 서명과 세 갈래
# =============================================================================
# 다섯 쪽이 이 함수 하나를 부른다. 서명이 바뀌면 다섯 쪽이 함께 깨지므로
# 여기에 찍어 둔다.
print("load_data", inspect.signature(utils.load_data))

train_kwargs = {'batch_size': 64, 'shuffle': True}
test_kwargs = {'batch_size': 1000, 'shuffle': False}

utils.set_seed(1)
for name, flag in [("MNIST", {}),
                   ("Fashion-MNIST", {"fashion_mnist": True}),
                   ("CIFAR-10", {"cifar10": True})]:
    trainloader, testloader = utils.load_data(train_kwargs, test_kwargs, **flag)
    images, _ = next(iter(trainloader))
    print(f"{name:13s} train {len(trainloader.dataset):5d}  test {len(testloader.dataset):5d}  "
          f"batch {tuple(images.shape)}  range [{images.min():.1f}, {images.max():.1f}]")

# =============================================================================
# 2절: 두 모델의 모양과 매개변수 수
# =============================================================================
# fc1.in_features 는 손으로 센 값과 맞아야 한다.
#   CNN         : 28 -> 14 -> 7,  64 x 7 x 7 = 3136
#   CNN_CIFAR10 : 32 -> 16 -> 8,  64 x 8 x 8 = 4096
print()
for model_class, shape in [(utils.CNN, (2, 1, 28, 28)),
                           (utils.CNN_CIFAR10, (2, 3, 32, 32))]:
    model = model_class()
    output = model(torch.zeros(shape))
    n_param = sum(p.numel() for p in model.parameters())
    print(f"{model_class.__name__:12s} {shape} -> {tuple(output.shape)}  "
          f"flatten {model.fc1.in_features}  params {n_param:,}")

# =============================================================================
# 3절: set_seed 가 정말 되풀이되는가
# =============================================================================
print()
utils.set_seed(1)
a = torch.randn(3)
utils.set_seed(1)
b = torch.randn(3)
print("set_seed(1) twice ->", torch.equal(a, b), [f"{v:.4f}" for v in a.tolist()])
```

**출력:**

```
load_data (train_kwargs, test_kwargs, fashion_mnist=False, cifar10=False)
MNIST         train 60000  test 10000  batch (64, 1, 28, 28)  range [-1.0, 1.0]
Fashion-MNIST train 60000  test 10000  batch (64, 1, 28, 28)  range [-1.0, 1.0]
Files already downloaded and verified
Files already downloaded and verified
CIFAR-10      train 50000  test 10000  batch (64, 3, 32, 32)  range [-1.0, 1.0]

CNN          (2, 1, 28, 28) -> (2, 10)  flatten 3136  params 421,642
CNN_CIFAR10  (2, 3, 32, 32) -> (2, 10)  flatten 4096  params 2,168,362

set_seed(1) twice -> True ['0.6614', '0.2669', '0.0617']
```

!!! note "첫 실행에서만 보이는 줄"
    데이터가 아직 `./data`에 없으면 `Files already downloaded and verified` 두 줄 자리에 `Downloading ...` 과 `Extracting ...` 이 대신 찍힌다. 내려받기가 한 번 끝나면 다시 나오지 않으므로 여기에는 싣지 않았다. 두 줄인 까닭은 `load_data`가 CIFAR-10의 학습 집합과 시험 집합을 따로 만들면서 저마다 `download=True`로 부르기 때문이다. MNIST와 Fashion-MNIST는 이미 내려받은 뒤에는 아무 줄도 찍지 않는다.

출력에서 읽을 것이 셋이다. 첫째, 서명 줄이 세 데이터셋을 가르는 것은 뒤의 옵션 두 개뿐임을 보인다. 앞의 두 인자는 `DataLoader`에 그대로 펼쳐 넣는 사전이라 데이터셋과 무관하다. 둘째, `range [-1.0, 1.0]`이 세 줄 모두에 나타난다 — 회색조는 `Normalize((0.5,), (0.5,))`, CIFAR-10은 채널마다 되풀이한 `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`를 쓰지만 옮겨 가는 범위는 같다. 셋째, `flatten 3136`과 `flatten 4096`은 손으로 센 $64 \times 7 \times 7$과 $64 \times 8 \times 8$과 맞는다.

CNN은 421642개, CNN_CIFAR10은 2168362개의 매개변수를 갖는다. 차이의 대부분은 fc1 입력과 출력 차원이 커진 데서 온다. 합성곱 깊이도 늘었으므로 차이가 완전 연결층에서만 생겼다는 뜻은 아니다. 매개변수의 비율과 각 층의 연산·활성값 메모리 비율도 구분한다.

## 3. 논의

공통 함수는 중복을 줄이고 설정의 일관성을 유지하는 데 도움이 된다. 반대로 함수 규약이나 상태 처리 변경이 여러 예제에 영향을 주므로 호출부를 함께 점검해야 한다. 한 파일에 모았다는 사실만으로 유지보수성이 보장되지는 않는다.

`load_data`의 서명은 이 모듈에서 가장 널리 쓰이는 약속이다. `load_data(train_kwargs, test_kwargs, fashion_mnist=False, cifar10=False)`에서 앞의 두 인자는 `DataLoader`에 `**` 로 펼쳐 넣는 사전이고, 뒤의 두 옵션이 데이터셋을 고른다. 아무것도 주지 않으면 MNIST, `fashion_mnist=True`면 Fashion-MNIST, `cifar10=True`면 CIFAR-10이다. 옵션이 둘로 나뉘어 있어 둘 다 참인 부름이 문법상 막혀 있지 않은데, 코드가 `if cifar10:`을 먼저 보므로 이때는 CIFAR-10이 조용히 이긴다(연습문제 2).

set_seed는 PyTorch·CUDA·NumPy·random의 난수 시드를 정하고 cuDNN의 결정적 알고리즘 선택을 요청한다. 같은 환경과 호출 순서의 일부 난수를 재현할 수 있지만 모든 연산·장치·버전의 재현성을 보장하지 않는다. DataLoader는 worker별 PyTorch 시드를 설정하며, 별도의 NumPy·사용자 정의 난수 생성기와 변환을 쓰면 worker_init_fn과 generator 설정도 검토한다. cuDNN 설정은 CUDA의 해당 연산에만 적용된다.

train은 시작할 때 train()을, compute_accuracy는 eval()을 호출한다. 모델 두 종류에는 드롭아웃이 있고 BatchNorm은 없다. 드롭아웃의 학습·평가 동작과 no_grad의 그래프 비활성화는 별개이다. compute_accuracy는 종료 후 이전 모드를 복원하지 않으므로 뒤에 학습을 연결하면 다시 train()을 설정해야 한다. train 함수는 시작 때 이를 수행한다.

손실 기록은 배치 평균의 단순 평균이므로 작은 마지막 배치까지 표본별로 동일 가중한 값과 차이가 난다. dry_run으로 한 배치만 처리해도 전체 len(train_loader)로 나누므로 그 모드의 출력 평균은 특히 주의해야 한다. train은 scheduler가 있다는 가정으로 매 에포크 step()을 호출한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`CNN`은 `x.view(-1, 64 * 7 * 7)`로, `CNN_CIFAR10`은 `x.view(-1, 64 * 8 * 8)`로 평평하게 편다. 두 상수 3136과 4096이 입력 크기에서 어떻게 나오는지 층을 따라가며 보여라. 이어서 `CNN`에 $32 \times 32$ 회색조 이미지를 넣으면 어떤 일이 벌어지는지 답하라.

</div>

??? success "연습문제 1 풀이"
    합성곱은 커널 3에 `padding=1`이라 크기를 바꾸지 않는다. 출력 크기가

    $$H_{\text{out}} = H + 2 \times 1 - 3 + 1 = H$$

    이기 때문이다. 크기를 줄이는 것은 `MaxPool2d(2, 2)` 뿐이고 한 번에 절반이 된다.

    - `CNN`: $28 \to 14 \to 7$, 채널 64 — $64 \times 7 \times 7 = 3136$
    - `CNN_CIFAR10`: $32 \to 16 \to 8$, 채널 64 — $64 \times 8 \times 8 = 4096$

    2절 출력의 `flatten 3136`과 `flatten 4096`이 이 값이다.

    $32 \times 32$ 회색조를 `CNN`에 넣으면 두 번 반씩 줄어 $8 \times 8$이 되고 채널 64와 합쳐 한 장에 값이 4096개가 된다. `view(-1, 3136)`은 전체 원소 수가 3136의 배수이길 바라는데 4096은 그렇지 않으므로 여기서 멈춘다.

    ```
    RuntimeError: shape '[-1, 3136]' is invalid for input of size 4096
    ```

    3채널 컬러 이미지를 넣으면 그보다 먼저 `conv1`에서 걸린다.

    ```
    RuntimeError: Given groups=1, weight of size [32, 1, 3, 3], expected input[1, 3, 32, 32] to have 1 channels, but got 3 channels instead
    ```

    CNN은 28×28 회색조 입력에 맞춰 설계되어 있다. 잘못된 크기에서 view(-1,3136)는 보통 오류를 내지만, 전체 원소 수가 우연히 나누어떨어지면 배치 축을 잘못 해석할 수도 있다. 예를 들어 32×32 표본 49개는 4096×49/3136=64로 재해석되어 표본 수가 바뀐다. 원래 배치 크기와 특징 크기를 명시적으로 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`load_data(train_kwargs, test_kwargs, fashion_mnist=True, cifar10=True)`처럼 두 옵션을 모두 참으로 주면 어떤 데이터셋이 돌아오는가? 코드에서 까닭을 짚고, 이런 부름을 소리 내어 막으려면 함수를 어떻게 고치면 되는지 보여라.

</div>

??? success "연습문제 2 풀이"
    CIFAR-10이 돌아온다. `load_data`는 `if cifar10:` 가지를 먼저 보고 그 안에서 CIFAR-10 두 집합을 만든 뒤 함수를 빠져나가므로, `fashion_mnist`는 읽히지도 않는다.

    ```python
    trainloader, _ = utils.load_data({'batch_size': 4}, {'batch_size': 4},
                                     fashion_mnist=True, cifar10=True)
    print(type(trainloader.dataset).__name__, len(trainloader.dataset))
    ```

    ```
    Files already downloaded and verified
    Files already downloaded and verified
    CIFAR10 50000
    ```

    조용히 이기는 대신 멈추게 하려면 함수 소개에 두 줄이면 된다.

    ```python
    if fashion_mnist and cifar10:
        raise ValueError("fashion_mnist 와 cifar10 을 함께 참으로 줄 수 없다")
    ```

    더 깔끔한 쪽은 옵션 둘을 `dataset='mnist' | 'fashion_mnist' | 'cifar10'` 하나로 합치는 것이다. 그러면 어긋난 상태가 아예 표현되지 않는다. 다만 이 절의 다섯 쪽이 모두 지금 서명으로 부르고 있으므로, 고치려면 다섯 쪽을 함께 고쳐야 한다. 공용 모듈의 서명을 바꾸는 값이 이런 것이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
load_data에 사용자 지정 평균·표준편차를 받는 인터페이스를 설계하라. 아래 경험적 통계량이 어떤 집계 규약에 따른 값인지 설명하고, 평균 0·분산 1의 표준화와 성능 개선이 같은 주장인지 답하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    def load_data(train_kwargs, test_kwargs, fashion_mnist=False, cifar10=False,
                  mean=None, std=None):
        if cifar10:
            if mean is None: mean = (0.4914, 0.4822, 0.4465)
            if std is None: std = (0.2470, 0.2435, 0.2616)
        elif fashion_mnist:
            if mean is None: mean = (0.2860,)
            if std is None: std = (0.3530,)
        else:
            if mean is None: mean = (0.1307,)
            if std is None: std = (0.3081,)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # ... 함수의 나머지
    ```

    세 쌍 모두 학습 집합의 모든 픽셀를 한 무더기로 놓고 잰 값이다(MNIST 60,000장, Fashion-MNIST 60,000장, CIFAR-10 50,000장).

    같은 학습셋의 픽셀 전체에서 정한 통계량으로 표준화하면 그 집계 기준의 평균과 분산을 조정할 수 있다. 반올림한 값이나 다른 분할에서는 정확히 0과 1이 되지 않는다. 이것이 모든 모델에서 성능 개선을 보장하는 것은 아니다. 아래 표는 고정된 (0.5,0.5) 변환이 남기는 평균과 표준편차를 보여 준다.

    | 데이터셋 | 원래 평균 | 원래 표준편차 | (0.5, 0.5) 뒤 평균 | (0.5, 0.5) 뒤 표준편차 |
    |---|---|---|---|---|
    | MNIST | 0.1307 | 0.3081 | −0.7386 | 0.6162 |
    | Fashion-MNIST | 0.2860 | 0.3530 | −0.4280 | 0.7060 |
    | CIFAR-10 (R) | 0.4914 | 0.2470 | −0.0172 | 0.4940 |
    | CIFAR-10 (G) | 0.4822 | 0.2435 | −0.0356 | 0.4870 |
    | CIFAR-10 (B) | 0.4465 | 0.2616 | −0.1070 | 0.5232 |

    표에서 MNIST는 배경 픽셀의 비중이 커 (0.5,0.5) 변환 뒤 평균이 약 −0.7386이다. 채널마다 남는 평균·스케일 차이는 표준화로 조절할 수 있지만, 편향 항 하나가 스케일 차이까지 보정한다는 설명은 부정확하다. 실제 최적화의 영향은 구조와 설정에 달려 있다.

    CIFAR-10에서 0.2023 등의 값은 이미지별 표준편차를 평균한 통계이고 0.2470 등의 값은 전체 픽셀을 집계한 표준편차이다. 다른 양을 측정한 것이므로 목적과 규약을 명시한다. Normalize는 제공된 상수로 변환할 뿐 어느 집계 규약을 강제하지 않는다. 자세한 비교는 [CIFAR-10 데이터셋](03_cifar10_dataset.md)에 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
train을 두 번 호출할 때 옵티마이저·스케줄러 객체를 유지하거나 새로 만들면 무엇이 달라지는가?

</div>

??? success "연습문제 4 풀이"
    같은 객체를 유지하면 옵티마이저 모멘텀·모멘트와 스케줄러 진행 상태를 이어 간다. 새 옵티마이저를 lr=0.01로 만들면 학습률과 그 상태를 초기화한다. 그러나 기존 옵티마이저에 스케줄러만 새로 만든다고 항상 학습률이 0.01로 복구되는 것은 아니다. 현재 param_group의 lr과 initial_lr, 스케줄러 종류·초기화에 따라 달라진다. 스케줄러가 실제 갱신에 쓰는 옵티마이저를 참조하는지 확인하고 정확한 재개에는 둘의 state_dict와 반복 위치를 복원한다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
클래스 이름을 저마다의 정확도로 잇는 사전을 돌려주는 `compute_per_class_accuracy` 함수를 유틸리티 모듈에 더하라. $10 \times 10$ 혼동 행렬을 들고 있지 말고 클래스마다 맞힌 수와 전체 수만 세는 방식으로 구현한 뒤, 그렇게 해서 무엇을 잃는지 적어라.

</div>

??? success "연습문제 5 풀이"
    ```python
    def compute_per_class_accuracy(model, test_loader, device, classes=None):
        model.eval()
        num_classes = 10
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                for i in range(len(target)):
                    label = target[i].item()
                    class_correct[label] += (predicted[i] == target[i]).item()
                    class_total[label] += 1

        result = {}
        for i in range(num_classes):
            name = classes[i] if classes else str(i)
            acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            result[name] = acc
        return result
    ```

    시험 집합을 한 번 훑으며 클래스마다 맞힌 수와 전체 수를 쌓고, 마지막에 나눈다. `compute_accuracy`가 하나의 수로 뭉뚱그리는 것을 클래스별로 갈라 놓은 것이다.

    클래스별 정답 수와 전체 수만 저장하면 어느 다른 클래스로 오분류했는지 알 수 없다. 공간 비용은 클래스 수 K에 대해 O(K), 혼동 행렬은 O(K²)이다. K=10에서는 차이가 작지만 클래스 수가 매우 크면 비용의 차이도 의미가 있다. 표본이 없는 클래스의 분모 0 처리 방식도 정의해야 한다.

    혼동 행렬이 필요하면 안쪽 반복문을 이렇게 바꾸면 된다.

    ```python
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    # ... 배치마다
    for t, p in zip(target.view(-1).cpu(), predicted.view(-1).cpu()):
        confusion[t, p] += 1
    ```

    클래스별 정확도는 여기서 `100.0 * confusion.diag() / confusion.sum(dim=1)`로 바로 나오므로, 혼동 행렬 쪽이 위 함수의 결과를 포함한다. 어느 클래스가 어느 클래스와 헷갈리는지까지 보려면 이쪽을 골라야 한다.

---

## 정리하며

유틸리티의 함수 인수뿐 아니라 모드·난수·옵티마이저와 스케줄러 상태가 호출 결과에 영향을 준다. 데이터셋 옵션의 우선순위, 고정 입력 차원과 평균 손실의 집계 방식을 명시하고, 점검 출력이 확인하는 범위를 넘어 재현성이나 학습의 올바름을 보장하지 않는다.
