# CNN 유틸리티

이 유틸리티 모듈은 CNN 실습 묶음이 함께 쓰는 바탕을 마련해 준다. 인자 구문 분석, MNIST/Fashion-MNIST/CIFAR-10 데이터 적재, 모델 구조, 학습 반복문, 평가, 시각화, 모델 저장이 여기에 든다. 이 함수들을 한 모듈에 모으면 코드를 다시 쓰기 좋고 모든 실습 스크립트가 한결같이 움직인다.

## 코드

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

## 논의

잘 설계된 유틸리티 모듈은 손보기 좋은 기계 학습 프로젝트에 꼭 필요하다. 인자 구문 분석, 데이터 적재, 모델 정의, 학습 반복문을 한 파일에 모으면 모든 실습 스크립트가 한결같이 움직인다. 학습 절차나 모델 구조를 고치면 그 모듈을 들여오는 모든 스크립트에 저절로 퍼지므로, 코드를 베껴 쓸 때 생기는 어긋남의 위험이 줄어든다.

`set_seed` 함수는 재현성을 꼼꼼히 챙기는 방법을 보여 준다. PyTorch, CUDA, NumPy, 파이썬 내장 random 모듈의 씨앗을 모두 정하면 무작위성의 모든 원천이 통제된다. `torch.backends.cudnn.deterministic = True` 깃발은 성능을 조금 내주는 대신 cuDNN이 결정적인 알고리즘을 쓰게 하는데, 이는 벌레잡이와 실험 비교에 매우 중요하다.

학습 반복문은 PyTorch의 모범 관행을 따른다. 학습 전에 `model.train()`을 불러 (드롭아웃과 배치 정규화를 켜고) 평가 전에 `model.eval()`을 불러 (끈다). 평가 중의 `torch.no_grad()` 문맥 관리자는 쓸데없는 기울기 계산을 막아 메모리를 아끼고 추론을 빠르게 한다.

## 연습문제

**연습문제 1.**
`load_data` 함수는 모든 데이터셋을 평균 0.5, 표준편차 0.5로 정규화한다. 사용자가 정한 정규화 통계량을 받도록 고치고, 참된 데이터셋 통계량(예를 들어 MNIST의 평균 0.1307, 표준편차 0.3081)이 더 나은 결과를 주는 까닭을 설명하라.

??? success "연습문제 1 풀이"
    ```python
    def load_data(train_kwargs, test_kwargs, fashion_mnist=False, cifar10=False,
                  mean=None, std=None):
        if cifar10:
            if mean is None: mean = (0.4914, 0.4822, 0.4465)
            if std is None: std = (0.2023, 0.1994, 0.2010)
        else:
            if mean is None: mean = (0.1307,)
            if std is None: std = (0.3081,)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        # ... 함수의 나머지
    ```
    참된 데이터셋 통계량을 쓰면 평균 0, 분산 1인 분포가 되는데 이것이 기울기 기반 최적화에 최적이다. 뭉뚱그린 값(0.5, 0.5)은 평균이 0이 아니고 분산도 1이 아닌 분포를 주어, 신경망이 치우침을 메우는 일을 따로 배워야 한다. (거의 검은 화소인) MNIST에서는 차이가 작지만, 채널별 통계량이 크게 다른 CIFAR-10에서는 차이가 더 뚜렷해진다.

---

**연습문제 2.**
`train` 함수는 매개변수를 통해 최적화기를 암묵적으로 새로 만든다. 스케줄러를 최적화기와 함께 넘겨주어야 하는 까닭과, 함수 안에서 스케줄러를 새로 만들면 어떻게 될지 설명하라.

??? success "연습문제 2 풀이"
    스케줄러를 넘겨주어야 하는 까닭은 그것이 내부 상태, 곧 `step()` 호출 횟수를 세는 계수기를 지니기 때문이다. 이 계수기가 감쇠 일정에 따라 지금의 학습률을 정한다. `train` 함수 안에서 스케줄러를 새로 만들면 `train`을 부를 때마다 계수기가 0으로 되돌아가 학습률 일정이 사실상 초기화된다.

    마찬가지로 최적화기도 밖에서 만들어 넘겨주어야 한다(적어도 그 상태는 지켜야 한다). (SGD의) 관성 완충기나 (Adam의) 기울기 이동 평균 같은 내부 상태를 지닐 수 있기 때문이다. 최적화기를 새로 만들면 이렇게 쌓인 상태가 버려져 학습이 불안정해질 수 있다.

---

**연습문제 3.**
부류 이름을 저마다의 정확도로 잇는 사전을 돌려주는 `compute_per_class_accuracy` 함수를 유틸리티 모듈에 더하라. 혼동 행렬 방식으로 구현하라.

??? success "연습문제 3 풀이"
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
    이 함수는 시험 집합을 한 번 훑으며 부류마다 맞힌 수와 전체 수를 쌓는다. ($10 \times 10$ 행렬이 될) 온전한 혼동 행렬을 메모리에 만들지 않으므로 시험 집합이 커도 효율적이다. 혼동 행렬 방식은 어떤 부류가 어떤 부류와 헷갈리는지까지 알려 주지만, 가장 흔히 보고하는 지표는 부류별 정확도이다.
