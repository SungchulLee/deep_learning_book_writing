"""cnn_utils — cnn_utils 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch08/cnn/cnn_utils.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

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
