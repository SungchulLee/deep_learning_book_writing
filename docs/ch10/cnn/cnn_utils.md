# CNN 유틸리티

이 쪽은 `cnn_utils.py` 한 파일을 통째로 싣는다. 인자 구문 분석, MNIST/Fashion-MNIST/CIFAR-10 적재, 두 가지 CNN 구조, 학습 반복문, 평가, 시각화, 모델 저장이 여기에 모여 있다. 이 절의 다섯 쪽 — [MNIST 데이터셋](01_mnist_dataset.md), [Fashion-MNIST 데이터셋](02_fashion_mnist_dataset.md), [CIFAR-10 데이터셋](03_cifar10_dataset.md), [Fashion-MNIST 분류기](05_fashion_mnist_classifier.md), [CIFAR-10 심화](07_cifar10_advanced.md) — 이 모두 `import cnn_utils as utils` 로 이 모듈을 들여온다. 그래서 여기의 서명 하나가 바뀌면 다섯 쪽이 함께 깨진다. 2절은 그 약속이 아직 그대로인지를 찍어서 확인한다.

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

1절의 코드는 함수와 클래스를 정의할 뿐 스스로는 아무것도 찍지 않는다(`if __name__ == "__main__": pass`). 그래서 이 모듈만 놓고는 쪽에 실을 출력이 없고, 실린 수가 아직 맞는지 확인할 길도 없다. 아래 스크립트는 모듈 바깥에서 모듈을 불러 세 가지를 찍는다. 다섯 쪽이 부르는 `load_data`의 서명, 두 모델이 평평하게 펴는 차원과 매개변수 수, 그리고 `set_seed`가 정말 같은 난수를 되돌려 주는지다. 이 코드는 `cnn_utils.py`에 들어 있지 않다 — 모듈을 그대로 두고 바깥에서 불러 보는 점검용이다.

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

출력에서 읽을 것이 셋이다. 첫째, 서명 줄이 세 데이터셋을 가르는 것은 뒤의 깃발 두 개뿐임을 보인다. 앞의 두 인자는 `DataLoader`에 그대로 펼쳐 넣는 사전이라 데이터셋과 무관하다. 둘째, `range [-1.0, 1.0]`이 세 줄 모두에 나타난다 — 회색조는 `Normalize((0.5,), (0.5,))`, CIFAR-10은 채널마다 되풀이한 `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`를 쓰지만 옮겨 가는 범위는 같다. 셋째, `flatten 3136`과 `flatten 4096`은 손으로 센 $64 \times 7 \times 7$과 $64 \times 8 \times 8$과 맞는다.

매개변수 수는 두 모델을 견주게 해 준다. `CNN_CIFAR10`은 421,642개의 `CNN`보다 5.14배 크지만, 그 차이를 만든 것은 합성곱 층을 둘에서 넷으로 늘린 것이 아니라 완전 연결 층이다. `CNN`은 매개변수의 95.2%(401,536 / 421,642)가 `fc1` 하나에 몰려 있고 `CNN_CIFAR10`은 96.7%(2,097,664 / 2,168,362)다. 합성곱 넷을 모두 합쳐도 65,568개로 전체의 3%에 지나지 않는다.

## 3. 논의

잘 설계된 유틸리티 모듈은 손보기 좋은 기계 학습 프로젝트에 꼭 필요하다. 인자 구문 분석, 데이터 적재, 모델 정의, 학습 반복문을 한 파일에 모으면 모든 실습 스크립트가 한결같이 움직인다. 학습 절차나 모델 구조를 고치면 그 모듈을 들여오는 모든 스크립트에 저절로 퍼지므로, 코드를 베껴 쓸 때 생기는 어긋남의 위험이 줄어든다. 값은 그 반대쪽에 있다 — 서명 하나를 고치면 부르는 쪽 다섯을 함께 고쳐야 한다.

`load_data`의 서명은 이 모듈에서 가장 널리 쓰이는 약속이다. `load_data(train_kwargs, test_kwargs, fashion_mnist=False, cifar10=False)`에서 앞의 두 인자는 `DataLoader`에 `**` 로 펼쳐 넣는 사전이고, 뒤의 두 깃발이 데이터셋을 고른다. 아무것도 주지 않으면 MNIST, `fashion_mnist=True`면 Fashion-MNIST, `cifar10=True`면 CIFAR-10이다. 깃발이 둘로 나뉘어 있어 둘 다 참인 부름이 문법상 막혀 있지 않은데, 코드가 `if cifar10:`을 먼저 보므로 이때는 CIFAR-10이 조용히 이긴다(연습문제 2).

`set_seed`는 재현성을 챙기는 손길을 한자리에 모아 둔다. PyTorch, CUDA, NumPy, 파이썬 내장 `random`의 씨앗을 함께 정하므로 이 네 곳에서 나오는 난수는 되풀이된다 — 2절의 마지막 줄이 그 확인이다. 다만 무작위성의 *모든* 원천이 잡히는 것은 아니다. `torch.backends.cudnn.deterministic = True`는 이름 그대로 cuDNN, 곧 CUDA 위에서만 뜻이 있어 CPU나 애플 MPS로 돌릴 때는 아무 일도 하지 않는다. `DataLoader`를 `num_workers > 0`으로 쓰면 일꾼 프로세스마다 씨앗을 따로 심어야 하고(`worker_init_fn`), 원자적 덧셈을 쓰는 몇몇 GPU 커널은 `torch.use_deterministic_algorithms(True)`까지 켜야 잡힌다. 이 모듈이 가리는 범위는 일꾼 없는 한 프로세스 실행이며, 이 절의 쪽들은 모두 그 범위 안에 있다.

학습 반복문은 PyTorch의 모범 관행을 따른다. `train`은 첫머리에서 `model.train()`을, `compute_accuracy`는 첫머리에서 `model.eval()`을 부른다. 이 모듈에서 두 방식이 갈리는 층은 드롭아웃뿐이다 — `CNN`과 `CNN_CIFAR10` 모두 `nn.Dropout(0.25)`와 `nn.Dropout(0.5)`를 쓰고 배치 정규화 층은 하나도 없다. 학습 방식에서 드롭아웃은 값을 확률 $p$로 0으로 만들고 남은 값을 $1 / (1 - p)$로 키워 기댓값을 맞추며, 평가 방식에서는 아무것도 떨어뜨리지 않고 그대로 흘려보낸다. (배치 정규화를 쓰는 신경망이라면 `eval()`이 그 층을 끄는 것이 아니라, 지금 배치의 통계량 대신 학습하며 쌓아 둔 이동 평균을 쓰게 만든다. 이 모듈에는 해당 층이 없으니 여기서는 드롭아웃만 생각하면 된다.) 평가 중의 `torch.no_grad()` 문맥 관리자는 계산 그래프를 만들지 않아 메모리를 아끼고 추론을 빠르게 한다.

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

    곧 `CNN`은 $28 \times 28$ 회색조 전용이다. 완전 연결 층이 입력 크기를 못으로 박아 두기 때문이며, CIFAR-10 쪽이 `CNN_CIFAR10`을 따로 두는 까닭이 여기에 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`load_data(train_kwargs, test_kwargs, fashion_mnist=True, cifar10=True)`처럼 두 깃발을 모두 참으로 주면 어떤 데이터셋이 돌아오는가? 코드에서 까닭을 짚고, 이런 부름을 소리 내어 막으려면 함수를 어떻게 고치면 되는지 보여라.

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

    조용히 이기는 대신 멈추게 하려면 함수 첫머리에 두 줄이면 된다.

    ```python
    if fashion_mnist and cifar10:
        raise ValueError("fashion_mnist 와 cifar10 을 함께 참으로 줄 수 없다")
    ```

    더 깔끔한 쪽은 깃발 둘을 `dataset='mnist' | 'fashion_mnist' | 'cifar10'` 하나로 합치는 것이다. 그러면 어긋난 상태가 아예 표현되지 않는다. 다만 이 절의 다섯 쪽이 모두 지금 서명으로 부르고 있으므로, 고치려면 다섯 쪽을 함께 고쳐야 한다. 공용 모듈의 서명을 바꾸는 값이 이런 것이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`load_data`는 세 데이터셋을 모두 평균 0.5, 표준편차 0.5로 정규화한다. 사용자가 정한 통계량을 받도록 고치고, 참된 데이터셋 통계량이 더 나은 까닭을 수로 보여라.

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

    세 쌍 모두 학습 집합의 모든 화소를 한 무더기로 놓고 잰 값이다(MNIST 60,000장, Fashion-MNIST 60,000장, CIFAR-10 50,000장).

    참된 통계량을 쓰면 정의상 평균 0, 표준편차 1이 된다. (0.5, 0.5)를 쓰면 그 자리에서 얼마나 밀려나는지는 데이터셋마다 다르다.

    | 데이터셋 | 원래 평균 | 원래 표준편차 | (0.5, 0.5) 뒤 평균 | (0.5, 0.5) 뒤 표준편차 |
    |---|---|---|---|---|
    | MNIST | 0.1307 | 0.3081 | −0.7386 | 0.6162 |
    | Fashion-MNIST | 0.2860 | 0.3530 | −0.4280 | 0.7060 |
    | CIFAR-10 (R) | 0.4914 | 0.2470 | −0.0172 | 0.4940 |
    | CIFAR-10 (G) | 0.4822 | 0.2435 | −0.0356 | 0.4870 |
    | CIFAR-10 (B) | 0.4465 | 0.2616 | −0.1070 | 0.5232 |

    흔한 짐작과 달리 평균이 0에서 가장 멀리 밀려나는 쪽은 CIFAR-10이 아니라 MNIST다. 거의 검은 화소로 이루어져 밝기 평균이 0.1307밖에 되지 않기 때문이고, 자연 사진인 CIFAR-10은 평균이 이미 0.5 언저리라 (0.5, 0.5)만으로도 거의 가운데에 온다. CIFAR-10에서 통계량이 걸리는 곳은 평균이 아니라 채널이다. 세 채널의 퍼짐이 0.2470 / 0.2435 / 0.2616으로 서로 달라, 하나의 상수 0.5로 나누면 채널마다 조금씩 다른 눈금이 남는다. 신경망은 이렇게 남은 치우침과 눈금을 첫 층의 치우침 항으로 메우는 일을 따로 배워야 한다.

    한마디 덧붙일 것이 있다. CIFAR-10의 표준편차로 (0.2023, 0.1994, 0.2010)을 적어 놓은 코드를 아주 자주 보게 된다. 이 값은 틀린 것이 아니라 **다른 것을 잰 값**이다 — 이미지마다 표준편차를 구한 뒤 50,000장에 걸쳐 평균 낸 값(`x.std(dim=(1, 2)).mean(dim=0)`)으로, 네 자리까지 그대로 나온다. `transforms.Normalize`가 나누는 것은 데이터셋 전체를 한 무더기로 본 표준편차이므로 여기서는 0.2470 / 0.2435 / 0.2616 쪽이 맞다. 두 값의 비가 1.22배에 그쳐 어느 쪽으로 학습해도 결과가 크게 갈리지는 않지만, 같은 이름으로 다른 양을 부르는 것은 다른 일이다. 자세한 것은 [CIFAR-10 데이터셋](03_cifar10_dataset.md)에 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
`train` 함수는 최적화기와 스케줄러를 스스로 만들지 않고 인자로 받는다. 이 둘을 함수 안에서 새로 만들면 무엇이 달라지는지, `train`을 두 번 이어 부르는 경우를 두고 설명하라.

</div>

??? success "연습문제 4 풀이"
    스케줄러부터 보자. 스케줄러는 `step()`이 몇 번 불렸는지 세는 계수기를 안에 지니고, 그 계수기가 지금의 학습률을 정한다. 이 절의 두 학습 쪽은 `optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.gamma)`를 쓰고 `cfg.gamma`의 기본값은 0.7이므로, 세대마다 학습률이 0.7배가 된다. 기본값대로 `epochs=14`로 한 번 부르고 나면 학습률은 처음의

    $$0.01 \times 0.7^{14} \approx 6.8 \times 10^{-5}$$

    까지 내려가 있다. 같은 스케줄러 객체를 그대로 넘겨 두 번째로 부르면 학습은 그 자리에서 이어지지만, `train` 안에서 스케줄러를 새로 만들면 계수기가 0으로 되돌아가 학습률이 다시 0.01에서 시작한다. 잘 좁혀 오던 학습이 두 번째 부름에서 처음의 147배 되는 걸음으로 되돌아가는 셈이다.

    최적화기도 같다. SGD의 관성 완충기나 Adam의 1차·2차 적률 추정값은 최적화기 객체 안에 쌓이는 상태다. 새로 만들면 이 상태가 버려져 관성이 0에서 다시 붙기 시작하고, Adam이라면 초기 몇 걸음이 편향 보정 구간으로 되돌아간다. 더 나쁜 것은 `train` 안에서 만든 최적화기가 밖에서 만든 스케줄러와 다른 객체를 가리키게 되어, `scheduler.step()`이 아무 데도 닿지 않는 학습률만 고치게 되는 경우다. 밖에서 짝지어 만들어 함께 넘기면 이런 어긋남이 생길 자리가 없다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
부류 이름을 저마다의 정확도로 잇는 사전을 돌려주는 `compute_per_class_accuracy` 함수를 유틸리티 모듈에 더하라. $10 \times 10$ 혼동 행렬을 들고 있지 말고 부류마다 맞힌 수와 전체 수만 세는 방식으로 구현한 뒤, 그렇게 해서 무엇을 잃는지 적어라.

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

    시험 집합을 한 번 훑으며 부류마다 맞힌 수와 전체 수를 쌓고, 마지막에 나눈다. `compute_accuracy`가 하나의 수로 뭉뚱그리는 것을 부류별로 갈라 놓은 것이다.

    잃는 것은 **어디로** 틀렸는지다. 이 방식은 혼동 행렬의 대각선만 세므로, 5를 3으로 읽는지 8로 읽는지 구분할 수 없다. 메모리가 아까워서 피하는 것은 아니라는 점을 분명히 해 두자 — 혼동 행렬은 정수 $10 \times 10 = 100$개일 뿐이고 그 크기는 시험 집합이 아무리 커져도 그대로다. 이득은 오직 코드가 짧다는 것이며, 보고할 지표가 부류별 정확도 하나뿐일 때만 남는 이득이다.

    혼동 행렬이 필요하면 안쪽 반복문을 이렇게 바꾸면 된다.

    ```python
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    # ... 배치마다
    for t, p in zip(target.view(-1).cpu(), predicted.view(-1).cpu()):
        confusion[t, p] += 1
    ```

    부류별 정확도는 여기서 `100.0 * confusion.diag() / confusion.sum(dim=1)`로 바로 나오므로, 혼동 행렬 쪽이 위 함수의 결과를 포함한다. 어느 부류가 어느 부류와 헷갈리는지까지 보려면 이쪽을 골라야 한다.

---

## 정리하며

**다룬 것** — CNN 유틸리티

이 모듈은 다섯 쪽이 함께 기대는 바탕이라, 여기의 약속이 곧 그 다섯 쪽의 약속이다. `load_data(train_kwargs, test_kwargs, fashion_mnist=False, cifar10=False)`는 깃발 하나로 세 데이터셋을 가르고 어느 쪽이든 화소를 $[-1, 1]$로 옮긴다(둘 다 참이면 CIFAR-10이 이긴다). `CNN`은 $28 \times 28$ 회색조 전용으로 3136차원에서 평평해지며 매개변수가 421,642개, `CNN_CIFAR10`은 $32 \times 32$ RGB용으로 4096차원에 2,168,362개이고, 둘 다 95% 넘는 매개변수가 첫 완전 연결 층 하나에 몰려 있다. `set_seed`는 PyTorch·CUDA·NumPy·`random` 네 곳을 함께 잡지만 cuDNN 깃발은 CUDA에서만 뜻이 있고 `DataLoader` 일꾼은 따로 챙겨야 한다.

2절의 점검 스크립트가 이 수들을 모두 찍는다. 모듈이 조용히 바뀌면 이 쪽의 출력이 먼저 어긋나므로, 다섯 쪽이 깨지기 전에 여기서 걸린다.

앞의 연습문제 5개로 직접 확인할 수 있다.
