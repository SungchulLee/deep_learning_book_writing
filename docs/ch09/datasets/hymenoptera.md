# Hymenoptera

ImageFolder로 개미·벌 이미지를 불러오고 사전학습 ResNet-18의 출력층을 교체한다. 백본 계수를 동결한 전이 학습과 검증 체크포인트 선택을 살펴본다.

## 1. 코드

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
    main()
```

??? note "전체 출력 (131줄)"

    ```
    Hymenoptera dataset downloaded and extracted.
    Epoch 0/24
    ----------
      train Loss: 0.5599 Acc: 0.9139
      val Loss: 12.7522 Acc: 0.5425

    Epoch 1/24
    ----------
      train Loss: 9.4083 Acc: 0.6189
      val Loss: 20.5794 Acc: 0.5425

    Epoch 2/24
    ----------
      train Loss: 14.6956 Acc: 0.5369
      val Loss: 20.6320 Acc: 0.5425

    Epoch 3/24
    ----------
      train Loss: 14.8269 Acc: 0.5369
      val Loss: 20.5971 Acc: 0.5425

    Epoch 4/24
    ----------
      train Loss: 14.8024 Acc: 0.5369
      val Loss: 20.6695 Acc: 0.5425

    Epoch 5/24
    ----------
      train Loss: 14.7878 Acc: 0.5369
      val Loss: 20.1131 Acc: 0.5425

    Epoch 6/24
    ----------
      train Loss: 14.7704 Acc: 0.5328
      val Loss: 20.1344 Acc: 0.5425

    Epoch 7/24
    ----------
      train Loss: 21.0102 Acc: 0.4959
      val Loss: 10.0418 Acc: 0.5425

    Epoch 8/24
    ----------
      train Loss: 8.7442 Acc: 0.4877
      val Loss: 0.6372 Acc: 0.6471

    Epoch 9/24
    ----------
      train Loss: 1.1150 Acc: 0.6066
      val Loss: 1.6665 Acc: 0.5425

    Epoch 10/24
    ----------
      train Loss: 1.7852 Acc: 0.4836
      val Loss: 1.6086 Acc: 0.5425

    Epoch 11/24
    ----------
      train Loss: 1.7942 Acc: 0.5000
      val Loss: 1.5750 Acc: 0.5425

    Epoch 12/24
    ----------
      train Loss: 1.7860 Acc: 0.5082
      val Loss: 1.6646 Acc: 0.5425

    Epoch 13/24
    ----------
      train Loss: 1.7920 Acc: 0.4918
      val Loss: 1.6256 Acc: 0.5425

    Epoch 14/24
    ----------
      train Loss: 2.2034 Acc: 0.4959
      val Loss: 0.8266 Acc: 0.5686

    Epoch 15/24
    ----------
      train Loss: 1.3222 Acc: 0.4590
      val Loss: 0.5155 Acc: 0.7516

    Epoch 16/24
    ----------
      train Loss: 0.9417 Acc: 0.4016
      val Loss: 0.5145 Acc: 0.7712

    Epoch 17/24
    ----------
      train Loss: 0.8511 Acc: 0.4139
      val Loss: 0.5100 Acc: 0.7647

    Epoch 18/24
    ----------
      train Loss: 0.8309 Acc: 0.4016
      val Loss: 0.5179 Acc: 0.7778

    Epoch 19/24
    ----------
      train Loss: 0.8300 Acc: 0.4303
      val Loss: 0.5271 Acc: 0.7778

    Epoch 20/24
    ----------
      train Loss: 0.8314 Acc: 0.4303
      val Loss: 0.5195 Acc: 0.7778

    Epoch 21/24
    ----------
      train Loss: 0.7638 Acc: 0.5328
      val Loss: 0.5195 Acc: 0.7647

    Epoch 22/24
    ----------
      train Loss: 0.7515 Acc: 0.5123
      val Loss: 0.5244 Acc: 0.7778

    Epoch 23/24
    ----------
      train Loss: 0.7578 Acc: 0.5205
      val Loss: 0.5215 Acc: 0.7647

    Epoch 24/24
    ----------
      train Loss: 0.7524 Acc: 0.5205
      val Loss: 0.5186 Acc: 0.7647

    Training complete in 16m 24s
    Best val Acc: 0.7778
    Accuracy: 79 %
      class  ants: 78.6 %
      class  bees: 79.5 %
    ```


## 2. 논의

백본의 requires_grad를 False로 둔 뒤 새 Linear(512, 2)를 연결하므로 기본 출력 헤드에는 1026개의 학습 매개변수가 있다. 새 헤드는 교체 후 생성되어 학습 가능하다.

requires_grad=False는 계수 경사를 막지만 train 모드의 배치 정규화 이동 통계량까지 동결하지는 않는다. 이 코드처럼 전체 모델에 train()을 호출하면 백본의 영속 버퍼가 바뀔 수 있다. 완전히 고정된 특징 추출을 원하면 해당 층의 모드도 설계해야 한다.

학습에는 무작위 변환을, 검증에는 정한 전처리를 사용한다. 최고 검증 정확도의 상태를 깊은 복사해 복원하지만 같은 검증셋에서 보고한 최선 성능은 선택에 사용된 지표이며 독립 테스트 성능은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
새 출력 헤드의 학습 매개변수 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    $512\times2+2=1026$개이다. 백본 가중치는 등록되어 있어도 requires_grad=False이면 학습 가능한 수에서는 제외한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
백본 계수를 동결한 뒤에도 state_dict가 바뀔 수 있는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    train 모드의 BatchNorm은 running_mean·running_var 등의 버퍼를 갱신할 수 있다. 계수의 경사 설정과 모듈의 학습·평가 모드는 다른 제어이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
ImageFolder로 만든 두 분할의 클래스 대응을 확인해야 하는 이유는 무엇인가?

</div>

??? success "연습문제 3 풀이"
    폴더 이름으로 class_to_idx가 정해진다. 분할에 일부 클래스가 빠지거나 이름이 다르면 인덱스 대응이 달라질 수 있으므로 학습·검증과 복원 후에 같은 사전을 사용해야 한다.

## 정리하며

전이 학습에서는 새 헤드, 동결 계수와 변경 가능한 버퍼를 구분한다. 사전학습 입력 전처리를 맞추고 최고 검증 체크포인트와 독립 평가의 차이를 확인한다.
