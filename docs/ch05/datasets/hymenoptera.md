# Hymenoptera

전이 학습은 미리 학습된 모델을 이용하여 데이터가 적은 새 과제에서도 좋은 성능을 낸다. 이 스크립트는 디렉터리 구조에서 레이블을 자동으로 붙여 주는 `ImageFolder`를 써서 Hymenoptera 데이터셋(개미와 벌)에 미리 학습된 ResNet-18을 미세 조정하며, 뼈대 얼리기, 데이터 증강, 학습률 스케줄러를 쓰는 학습을 보인다.

## 코드

```python
"""
Hymenoptera — ImageFolder와 ResNet 전이 학습
=====================================================
``torchvision.datasets.ImageFolder``로 불러온 Hymenoptera(개미와 벌) 데이터셋에
미리 학습된 ResNet-18을 미세 조정한다.

사용법
-----
    python hymenoptera.py
    python hymenoptera.py --epochs 10 --lr 0.001
"""

import argparse
import copy
import os
import time
import urllib.request
import zipfile

# ========================================================================
# 메인
# ========================================================================

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# ---------------------------------------------------------------------------
# 전역 설정
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Hymenoptera Transfer Learning")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--test-batch-size", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--no-cuda", action="store_true", default=False)
parser.add_argument("--no-mps", action="store_true", default=False)
ARGS = parser.parse_args()

torch.manual_seed(ARGS.seed)

ARGS.use_cuda = not ARGS.no_cuda and torch.cuda.is_available()
ARGS.use_mps = not ARGS.no_mps and torch.backends.mps.is_available()
if ARGS.use_cuda:
    DEVICE = torch.device("cuda")
elif ARGS.use_mps:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

TRAIN_KWARGS = {"batch_size": ARGS.batch_size}
TEST_KWARGS = {"batch_size": ARGS.test_batch_size}
if ARGS.use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    TRAIN_KWARGS.update(cuda_kwargs)
    TEST_KWARGS.update(cuda_kwargs)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASSES = ("ants", "bees")
PATH = "./model.pth"

# ---------------------------------------------------------------------------
# 내려받기
# ---------------------------------------------------------------------------


def download_dataset():
    if os.path.isdir("./hymenoptera_data"):
        return
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    zip_path = "hymenoptera_data.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")
    os.remove(zip_path)
    print("Hymenoptera dataset downloaded and extracted.")


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------


def load_data():
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    }
    data_dir = "./hymenoptera_data"
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], **TRAIN_KWARGS)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------


def show_batch(dataloader, model=None):
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(MEAN, STD)],
        std=[1 / s for s in STD],
    )
    images, labels = next(iter(dataloader))

    if model is not None:
        outputs = model(images.to(DEVICE))
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu()
    else:
        preds = labels

    n = min(ARGS.batch_size, 10)
    _, axes = plt.subplots(1, n, figsize=(12, 3))
    for i in range(n):
        img = denorm(images[i]).permute(1, 2, 0).numpy().clip(0, 1)
        axes[i].imshow(img)
        axes[i].axis("off")
        title = f"label: {CLASSES[labels[i]]}"
        if model is not None:
            title += f"\npred: {CLASSES[preds[i]]}"
        axes[i].set_title(title)
    plt.show()


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False


def train(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes):
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(ARGS.epochs):
        print(f"Epoch {epoch}/{ARGS.epochs - 1}\n" + "-" * 10)
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f"  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
        print()

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------


def compute_accuracy(model, testloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct // total} %")

    correct_pred = {c: 0 for c in CLASSES}
    total_pred = {c: 0 for c in CLASSES}
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, preds = torch.max(model(images), 1)
            for lab, pred in zip(labels, preds):
                if lab == pred:
                    correct_pred[CLASSES[lab]] += 1
                total_pred[CLASSES[lab]] += 1
    for cls, cnt in correct_pred.items():
        print(f"  class {cls:>5s}: {100 * cnt / total_pred[cls]:.1f} %")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main():
    download_dataset()
    dataloaders, dataset_sizes, class_names = load_data()

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    freeze_backbone(model)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=ARGS.momentum)
    exp_lr = lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)

    train(model, criterion, opt, exp_lr, dataloaders, dataset_sizes)

    torch.save(model.state_dict(), PATH)
    loaded = models.resnet18()
    loaded.fc = nn.Linear(loaded.fc.in_features, 2)
    loaded = loaded.to(DEVICE)
    loaded.load_state_dict(torch.load(PATH))

    show_batch(dataloaders["val"], loaded)
    compute_accuracy(loaded, dataloaders["val"])


if __name__ == "__main__":
    main()```

## 논의

ResNet-18을 쓰는 전이 학습은 미리 학습된 특징의 힘을 보여 준다. ImageNet의 120만 장으로 학습된 합성곱 뼈대는 모서리, 결, 모양 같은 일반적인 시각 특징을 뽑아낸다. 새 이진 분류 과제를 위해서는 마지막 완전 연결층만 바꾸어 학습시킨다.

뼈대를 얼리면(`param.requires_grad = False`) 학습 가능한 매개변수가 1100만 개 남짓에서 마지막 선형층의 1,000개 남짓으로 크게 줄어든다. 이는 미리 학습된 특징을 파국적으로 잊는 일을 막고 학습 데이터와 계산량도 훨씬 적게 든다.

데이터 증강 파이프라인은 학습 이미지에는 무작위 크기 조정 잘라내기와 좌우 뒤집기를 적용하고, 검증에는 결정적인 가운데 잘라내기를 쓴다. 두 파이프라인 모두 ImageNet의 통계(평균과 표준편차)로 정규화하는데, 미리 학습된 모델이 그렇게 정규화된 입력을 받도록 되어 있으므로 꼭 필요하다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

