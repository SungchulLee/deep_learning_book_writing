# 실전 예제

앞의 두 쪽은 잔차 블록을 만들고([기본 잔차 블록](01_basic_residual_block.md)) 그것을 단계별로 쌓아 ResNet-18부터 ResNet-152까지를 정의했다([ResNet 구현](02_resnet_implementation.md)). 어느 쪽도 실제 데이터로 학습시키지는 않았다. 이 쪽이 그 일을 한다 — CIFAR-10 5만 장으로 ResNet-18을 처음부터 학습시키고, 데이터 증강·학습률 일정·이음매 저장과 되읽기까지 갖춘 파이프라인 전체를 보인다.

구조가 정해진 뒤에도 남는 선택이 있고, 32×32 이미지에서 그 가운데 눈에 띄게 결과를 가르는 것이 **첫 층**(stem)이다. ResNet 구현 쪽의 `ResNet`은 224×224 ImageNet 이미지를 겨냥한 것이라 7×7 보폭 2 합성곱 뒤에 보폭 2 최대 풀링을 둔다. 이 stem을 32×32 입력에 그대로 대면 잔차 단계가 시작되기도 전에 해상도가 8×8로 줄어든다. 아래 코드는 두 stem을 같은 씨앗·같은 일정으로 나란히 학습시켜, 매개변수 0.07%밖에 차이 나지 않는 이 선택이 정확도를 얼마나 움직이는지 잰다.

!!! note "이 쪽은 홀로 돈다 — 앞 쪽의 파일을 따로 만들어 둘 필요가 없다"
    아래 코드는 `BasicBlock`과 `ResNet`을 이 쪽 안에서 다시 정의한다. 블록은 [ResNet 구현](02_resnet_implementation.md)의 것과 같고, `stem` 인자가 하나 붙어 첫 층을 ImageNet용과 CIFAR용 가운데 고를 수 있다는 점만 다르다. 곧 `stem='imagenet'`이 앞 쪽의 `ResNet` 그대로이며, 출력 [1]번 표의 첫 줄에 찍히는 매개변수 11,181,642개가 앞 쪽 출력의 `ResNet-18`과 같은 값이라는 것으로 그 점을 확인할 수 있다.

    `torchvision.models.resnet18`도 7×7 보폭 2 합성곱과 최대 풀링으로 된 같은 stem을 쓴다. `torchvision.models.resnet18(weights=None, num_classes=10)`의 매개변수 또한 11,181,642개로 같은 값이다. 즉 여기서 `stem='imagenet'`으로 재는 것은 torchvision의 모델을 그대로 가져다 쓸 때 일어나는 일이기도 하다.

## 1. 코드

```python
"""
실전 예제: CIFAR-10에서 ResNet-18 학습시키기
==============================================
데이터 적재, 학습 반복문, 평가, 이음매 저장과 되읽기까지 담은 완전한
파이프라인. 앞 쪽의 `BasicBlock`을 그대로 쓰되 첫 층(stem)만 두 가지로
고를 수 있게 하여, 32x32 입력에 ImageNet용 stem을 쓰면 무엇을 잃는지 잰다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ========================================================================
# 모델
# ========================================================================


class BasicBlock(nn.Module):
    """기본 잔차 블록 — 3x3 합성곱 둘과 지름길 하나.

    02_resnet_implementation.md 의 `BasicBlock` 과 같은 것이다. 뒤에 배치
    정규화가 오므로 두 합성곱 모두 bias=False 로 둔다.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out)


class ResNet(nn.Module):
    """stem 을 고를 수 있는 ResNet.

    stem='imagenet' — 7x7 보폭 2 합성곱 다음에 3x3 보폭 2 최대 풀링.
                      224x224 입력을 겨냥한 설계이며 02_resnet_implementation.md
                      의 ResNet 이 쓰는 것이 이것이다.
    stem='cifar'    — 3x3 보폭 1 합성곱 하나, 풀링 없음. 32x32 에서는 이쪽이다.
    """

    def __init__(self, block, layers, num_classes=10, stem='cifar'):
        super().__init__()
        self.in_channels = 64

        if stem == 'imagenet':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                   bias=False)
            self.maxpool = None
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """단계 하나를 만든다. 사영 지름길은 첫 블록에만 붙는다."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion))
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnet18(num_classes=10, stem='cifar'):
    """ResNet-18: 단계마다 기본 블록 [2, 2, 2, 2]개."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, stem)


def stage_sizes(model, size=32):
    """size x size 입력이 단계를 지나며 남기는 공간 크기를 따라간다."""
    x = torch.zeros(2, 3, size, size)
    sizes = []
    # 모양만 보면 되므로 평가 상태로 둔다. 학습 상태의 배치 정규화는 지금
    # 배치의 통계로 정규화하면서 이동 평균까지 갱신한다(표본이 하나뿐인
    # 배치에서는 분산을 낼 수 없어 아예 터진다).
    model.eval()
    with torch.no_grad():
        x = F.relu(model.bn1(model.conv1(x)))
        if model.maxpool is not None:
            x = model.maxpool(x)
        for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
            x = layer(x)
            sizes.append(x.shape[-1])
    return sizes


# ========================================================================
# 데이터
# ========================================================================

# CIFAR-10 전체의 채널별 평균과 표준편차. 입력을 평균 0, 표준편차 1 근처로
# 옮겨 놓아야 첫 층의 기울기 크기가 채널마다 들쭉날쭉해지지 않는다.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def get_cifar10_datasets(root='./data'):
    """CIFAR-10 을 한 번만 내려받아 학습용·시험용 데이터셋을 만든다."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    # 시험 집합에는 증강을 걸지 않는다. 증강은 학습 집합을 넓히는 장치이지
    # 평가 대상을 바꾸는 장치가 아니다.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    return trainset, testset


# num_workers 를 0으로 둔다. 1 이상이면 맥과 윈도가 일꾼 프로세스를
# **새로 띄우며**(spawn) 이 각본을 처음부터 다시 읽는다. 맨 바깥에서 자료를
# 돌리고 있으면 일꾼이 또 일꾼을 띄워 끝내 죽는다. 1 이상을 쓰려면 돌리는
# 부분을 모두 `if __name__ == "__main__":` 안으로 넣어야 한다.
def make_loaders(trainset, testset, seed=0, batch_size=128,
                 test_batch_size=256, num_workers=0):
    """씨앗 하나에 섞는 차례까지 매달아 적재기를 만든다.

    generator 를 주지 않으면 같은 씨앗을 심어도 배치의 차례가 달라져
    아래 수가 다시 나오지 않는다.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, generator=g)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                            num_workers=num_workers)
    return trainloader, testloader


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ========================================================================
# 학습과 평가
# ========================================================================


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 세대를 학습시키고 (배치 평균 손실, 정확도)를 돌려준다."""
    model.train()      # 배치 정규화가 배치 통계를 쓰고 이동 평균을 갱신한다
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()          # 기울기는 쌓이므로 매번 지워야 한다
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """시험 집합에서 (배치 평균 손실, 정확도)를 잰다."""
    model.eval()       # 배치 정규화가 이동 평균을 쓴다
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, targets).item()
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return running_loss / len(dataloader), 100. * correct / total


def evaluate_per_class(model, dataloader, classes, device):
    """부류마다 맞힌 수와 정확도를 적어 준다."""
    model.eval()
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predicted = model(inputs).argmax(1)
            for label, hit in zip(targets.tolist(),
                                  predicted.eq(targets).tolist()):
                class_correct[label] += hit
                class_total[label] += 1

    print(f"{'class':10}{'correct':>9}{'total':>7}{'accuracy':>11}")
    print("-" * 37)
    for i, name in enumerate(classes):
        acc = 100. * class_correct[i] / class_total[i]
        print(f"{name:10}{class_correct[i]:9d}{class_total[i]:7d}{acc:10.2f}%")
    print("-" * 37)
    overall = 100. * sum(class_correct) / sum(class_total)
    print(f"{'overall':10}{sum(class_correct):9d}{sum(class_total):7d}"
          f"{overall:10.2f}%")


def train_resnet_cifar10(trainset, testset, stem, seed, num_epochs, device,
                         learning_rate=0.1, weight_decay=5e-4):
    """씨앗 하나로 ResNet-18 을 처음부터 학습시킨다.

    돌려주는 것: (가장 좋았던 시험 정확도, 마지막 시험 정확도, 마지막 모델)
    """
    # 가중치 초기화와 증강(RandomCrop, RandomHorizontalFlip)의 난수가 이 한 줄에
    # 매달려 있다. 배치를 섞는 차례는 make_loaders 가 따로 씨앗 붙인 Generator 로
    # 정하므로, 둘을 함께 심어야 아래 수가 다시 나온다.
    torch.manual_seed(seed)

    trainloader, testloader = make_loaders(trainset, testset, seed=seed)
    model = resnet18(num_classes=10, stem=stem).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc, last_acc = 0.0, 0.0
    for epoch in range(num_epochs):
        lr = scheduler.get_last_lr()[0]
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()
        last_acc = test_acc

        # 가장 좋았던 이음매만 남긴다. 마지막 세대가 가장 좋으리라는 보장이 없다.
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'test_acc': test_acc},
                       f'resnet18_{stem}_seed{seed}_best.pth')

        print(f"  epoch {epoch + 1}/{num_epochs}  lr {lr:.4f} | "
              f"train loss {train_loss:.4f} acc {train_acc:5.2f}% | "
              f"test loss {test_loss:.4f} acc {test_acc:5.2f}%")

    return best_acc, last_acc, model


# ========================================================================
# 메인
# ========================================================================

if __name__ == "__main__":
    EPOCHS = 4
    SEEDS = (0, 1, 2)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 72)
    print("PRACTICAL EXAMPLE: ResNet-18 on CIFAR-10")
    print("=" * 72)
    print(f"torch {torch.__version__} | device {device}")

    # --- [1] stem 이 남기는 해상도 ---------------------------------------
    print("\n[1] 32x32 입력에서 stem 두 가지가 남기는 해상도")
    print(f"{'stem':22}{'stage1':>8}{'stage2':>8}{'stage3':>8}"
          f"{'stage4':>8}{'params':>13}")
    print("-" * 67)
    counts = {}
    for stem, label in (('imagenet', 'imagenet 7x7/2+pool'),
                        ('cifar', 'cifar 3x3/1')):
        m = resnet18(num_classes=10, stem=stem)
        counts[stem] = sum(p.numel() for p in m.parameters())
        sizes = stage_sizes(m)
        print(f"{label:22}" + "".join(f"{f'{v}x{v}':>8}" for v in sizes)
              + f"{counts[stem]:>13,}")
    diff = counts['imagenet'] - counts['cifar']
    print(f"\nparameter difference {diff:,} = (7*7 - 3*3) * 3 * 64 "
          f"= {(49 - 9) * 3 * 64:,}, i.e. "
          f"{100. * diff / counts['imagenet']:.2f}% of the model")

    # --- [2] 데이터 -------------------------------------------------------
    trainset, testset = get_cifar10_datasets()
    trainloader, testloader = make_loaders(trainset, testset, seed=0)
    print("\n[2] 데이터")
    print(f"train {len(trainset):,} samples in {len(trainloader)} "
          f"batches of 128")
    print(f"test  {len(testset):,} samples in {len(testloader)} "
          f"batches of 256, last batch "
          f"{len(testset) - (len(testloader) - 1) * 256}")
    print(f"classes {CLASSES}")

    # --- [3] 학습 ---------------------------------------------------------
    print(f"\n[3] 학습 — {EPOCHS} epochs, SGD(lr 0.1, momentum 0.9, "
          f"weight decay 5e-4), cosine schedule")
    results = {}
    seed0_model = None
    for stem in ('cifar', 'imagenet'):
        results[stem] = []
        for seed in SEEDS:
            print(f"\nstem={stem}  seed={seed}")
            best, last, model = train_resnet_cifar10(
                trainset, testset, stem, seed, EPOCHS, device)
            results[stem].append((best, last))
            if stem == 'cifar' and seed == 0:
                seed0_model = model

    # --- [4] 씨앗 셋의 폭 --------------------------------------------------
    print("\n[4] 씨앗 셋에서 얻은 시험 정확도의 폭")
    for stem in ('cifar', 'imagenet'):
        lasts = [v for _, v in results[stem]]
        bests = [v for v, _ in results[stem]]
        print(f"  {stem:9} last {min(lasts):5.2f} ~ {max(lasts):5.2f}%"
              f"   best {min(bests):5.2f} ~ {max(bests):5.2f}%")
    gap = (min(v for _, v in results['cifar'])
           - max(v for _, v in results['imagenet']))
    print(f"  cifar 의 최저 - imagenet 의 최고 = {gap:+.2f} percentage points")

    # --- [5] 부류별 정확도 -------------------------------------------------
    print("\n[5] 부류별 정확도 (stem=cifar, seed=0, 마지막 세대)")
    evaluate_per_class(seed0_model, testloader, CLASSES, device)

    # --- [6] 이음매 되읽기 -------------------------------------------------
    print("\n[6] 저장해 둔 이음매 되읽기")
    ckpt = torch.load('resnet18_cifar_seed0_best.pth', weights_only=True)
    restored = resnet18(num_classes=10, stem='cifar').to(device)
    restored.load_state_dict(ckpt['model_state_dict'])
    _, acc = evaluate(restored, testloader, nn.CrossEntropyLoss(), device)
    print(f"  saved at epoch {ckpt['epoch']}, recorded test acc "
          f"{ckpt['test_acc']:.2f}%")
    print(f"  reloaded model test acc {acc:.2f}% -> identical: "
          f"{acc == ckpt['test_acc']}")
    print("=" * 72)
```

**출력:**

```
========================================================================
PRACTICAL EXAMPLE: ResNet-18 on CIFAR-10
========================================================================
torch 2.5.1 | device mps

[1] 32x32 입력에서 stem 두 가지가 남기는 해상도
stem                    stage1  stage2  stage3  stage4       params
-------------------------------------------------------------------
imagenet 7x7/2+pool        8x8     4x4     2x2     1x1   11,181,642
cifar 3x3/1              32x32   16x16     8x8     4x4   11,173,962

parameter difference 7,680 = (7*7 - 3*3) * 3 * 64 = 7,680, i.e. 0.07% of the model
Files already downloaded and verified
Files already downloaded and verified

[2] 데이터
train 50,000 samples in 391 batches of 128
test  10,000 samples in 40 batches of 256, last batch 16
classes ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

[3] 학습 — 4 epochs, SGD(lr 0.1, momentum 0.9, weight decay 5e-4), cosine schedule

stem=cifar  seed=0
  epoch 1/4  lr 0.1000 | train loss 2.0853 acc 25.62% | test loss 1.6905 acc 36.98%
  epoch 2/4  lr 0.0854 | train loss 1.5372 acc 43.20% | test loss 1.4713 acc 46.18%
  epoch 3/4  lr 0.0500 | train loss 1.2425 acc 54.81% | test loss 1.1880 acc 58.48%
  epoch 4/4  lr 0.0146 | train loss 0.9827 acc 65.10% | test loss 0.9353 acc 66.58%

stem=cifar  seed=1
  epoch 1/4  lr 0.1000 | train loss 1.8994 acc 31.57% | test loss 1.5650 acc 42.98%
  epoch 2/4  lr 0.0854 | train loss 1.3726 acc 49.48% | test loss 1.2050 acc 56.79%
  epoch 3/4  lr 0.0500 | train loss 1.0350 acc 62.88% | test loss 0.9281 acc 66.87%
  epoch 4/4  lr 0.0146 | train loss 0.7758 acc 72.44% | test loss 0.7532 acc 73.57%

stem=cifar  seed=2
  epoch 1/4  lr 0.1000 | train loss 1.9684 acc 28.48% | test loss 1.6564 acc 38.69%
  epoch 2/4  lr 0.0854 | train loss 1.4489 acc 46.78% | test loss 1.5295 acc 45.28%
  epoch 3/4  lr 0.0500 | train loss 1.1249 acc 59.51% | test loss 1.0016 acc 64.17%
  epoch 4/4  lr 0.0146 | train loss 0.8665 acc 69.06% | test loss 0.8234 acc 70.82%

stem=imagenet  seed=0
  epoch 1/4  lr 0.1000 | train loss 2.1363 acc 29.85% | test loss 1.5782 acc 41.18%
  epoch 2/4  lr 0.0854 | train loss 1.4921 acc 45.24% | test loss 1.3435 acc 50.82%
  epoch 3/4  lr 0.0500 | train loss 1.2816 acc 53.48% | test loss 1.1645 acc 57.87%
  epoch 4/4  lr 0.0146 | train loss 1.0930 acc 60.85% | test loss 1.0531 acc 62.37%

stem=imagenet  seed=1
  epoch 1/4  lr 0.1000 | train loss 2.2414 acc 26.54% | test loss 1.6037 acc 40.82%
  epoch 2/4  lr 0.0854 | train loss 1.5561 acc 42.69% | test loss 1.4267 acc 47.98%
  epoch 3/4  lr 0.0500 | train loss 1.3385 acc 51.34% | test loss 1.2564 acc 55.31%
  epoch 4/4  lr 0.0146 | train loss 1.1505 acc 58.63% | test loss 1.0861 acc 61.31%

stem=imagenet  seed=2
  epoch 1/4  lr 0.1000 | train loss 2.1037 acc 29.96% | test loss 1.6807 acc 38.35%
  epoch 2/4  lr 0.0854 | train loss 1.5225 acc 44.10% | test loss 1.4046 acc 48.24%
  epoch 3/4  lr 0.0500 | train loss 1.2911 acc 53.17% | test loss 1.1480 acc 58.18%
  epoch 4/4  lr 0.0146 | train loss 1.1032 acc 60.46% | test loss 1.0281 acc 63.67%

[4] 씨앗 셋에서 얻은 시험 정확도의 폭
  cifar     last 66.58 ~ 73.57%   best 66.58 ~ 73.57%
  imagenet  last 61.31 ~ 63.67%   best 61.31 ~ 63.67%
  cifar 의 최저 - imagenet 의 최고 = +2.91 percentage points

[5] 부류별 정확도 (stem=cifar, seed=0, 마지막 세대)
class       correct  total   accuracy
-------------------------------------
plane           656   1000     65.60%
car             829   1000     82.90%
bird            573   1000     57.30%
cat             393   1000     39.30%
deer            657   1000     65.70%
dog             576   1000     57.60%
frog            741   1000     74.10%
horse           660   1000     66.00%
ship            818   1000     81.80%
truck           755   1000     75.50%
-------------------------------------
overall        6658  10000     66.58%

[6] 저장해 둔 이음매 되읽기
  saved at epoch 4, recorded test acc 66.58%
  reloaded model test acc 66.58% -> identical: True
========================================================================
```

## 2. 논의

**stem 하나가 해상도의 대부분을 삼킨다.** 출력 [1]번 표의 두 줄은 매개변수가 11,181,642개와 11,173,962개로 0.07%밖에 다르지 않은데, 32×32 입력이 단계를 지나며 남기는 크기는 8×8·4×4·2×2·1×1 대 32×32·16×16·8×8·4×4로 한 축마다 네 배가 차이 난다. 공간 크기 공식

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

에 ImageNet stem의 두 층을 차례로 넣어 보면 그 까닭이 나온다. 7×7 합성곱($k=7$, $p=3$, $s=2$)은 $\lfloor (32 + 6 - 7)/2 \rfloor + 1 = 16$을, 이어지는 3×3 최대 풀링($k=3$, $p=1$, $s=2$)은 $\lfloor (16 + 2 - 3)/2 \rfloor + 1 = 8$을 준다. 잔차 단계가 한 번도 돌기 전에 이미 네 배가 줄어 있는 것이다. 224×224에서는 같은 stem이 56×56을 남기므로 아무 문제가 없다. 문제는 stem이 아니라 stem과 입력 크기의 짝이다.

매개변수로 치르는 값은 $7^2 \cdot 3 \cdot 64 = 9{,}408$개와 $3^2 \cdot 3 \cdot 64 = 1{,}728$개의 차이, 곧 7,680개뿐이다. 모델 전체의 0.07%다. 값이 이토록 작은데도 뒤에 오는 모든 단계가 네 배 작은 격자 위에서 돌게 되고, 마지막 단계에 이르면 ImageNet stem 쪽의 특징 맵은 1×1이 되어 뒤따르는 전역 평균 풀링이 평균 낼 것이 한 칸밖에 남지 않는다.

**나란히 재면 CIFAR용 stem이 이긴다 — 다만 정해지는 것은 방향뿐이다.** 출력 [3]번은 두 stem을 씨앗 0·1·2에서 각각 4세대씩, 같은 최적화기와 같은 일정으로 학습시킨 것이다. 마지막 세대의 시험 정확도를 모으면 다음과 같다.

| stem | 씨앗 0 | 씨앗 1 | 씨앗 2 | 폭 |
|---|---|---|---|---|
| cifar 3×3/1 | 66.58% | 73.57% | 70.82% | 66.58 ~ 73.57% |
| imagenet 7×7/2+pool | 62.37% | 61.31% | 63.67% | 61.31 ~ 63.67% |

두 폭이 겹치지 않으므로($66.58 > 63.67$) CIFAR용 stem이 낫다는 것은 씨앗 잡음으로 설명되지 않는다. 그러나 **차이의 크기는 아직 정해지지 않았다.** CIFAR용 쪽의 폭만 해도 $73.57 - 66.58 = 6.99$%포인트로, 두 무리 사이의 간격 $66.58 - 63.67 = 2.91$%포인트보다 넓기 때문이다. 씨앗 0끼리만 견주어 "stem을 바꾸니 4.21%포인트가 좋아졌다"고 적었다면 그 수의 상당 부분은 씨앗이 만든 것이다. 한 번 돌린 것은 재어 본 것이 아니다.

해상도를 축마다 네 배 잃고도 차이가 이 정도에 그치는 데에는 까닭이 있다. 4세대는 두 모델 모두에게 이르다. 세 씨앗 모두 마지막 세대까지 학습 정확도와 시험 정확도가 나란히 오르는 중이고(씨앗 1의 시험 정확도는 42.98 → 56.79 → 66.87 → 73.57%다), 어느 쪽도 아직 용량에 부딪히지 않았다. 여기서 재는 것은 "다 배우고 난 뒤의 차이"가 아니라 "같은 예산에서 얼마나 나아갔는가의 차이"다.

**이 66~74%를 "ResNet의 CIFAR-10 정확도"로 옮겨 적어서는 안 된다.** He 등이 CIFAR-10에서 실제로 보고한 값은 다음과 같다(「Identity Mappings in Deep Residual Networks」, 2016, 표 3의 시험 오차).

| 모델 | 시험 오차 | 정확도 |
|---|---|---|
| ResNet-110 | 6.61% | 93.39% |
| ResNet-164 | 5.93% | 94.07% |
| ResNet-1001 | 7.61% | 92.39% |

이 셋은 CIFAR-10 전용 구조다. 단계가 셋이고 채널이 16·32·64이며, 이 쪽이 쓰는 ResNet-18과는 다른 모델이다. **He 등은 ResNet-18·34·50을 CIFAR-10에서 보고한 적이 없다.** 그 셋은 ImageNet 구조이고, 논문의 CIFAR-10 실험에 나오는 것은 20·32·44·56·110층짜리다. 그러니 "CIFAR-10에서 ResNet-50은 94~95%"류의 수는 그 논문에 근거가 없다. 게다가 표의 셋째 줄이 보이듯 깊이가 곧 이득도 아니다. 1001층은 164층보다 1.68%포인트 **나쁘다** — 이 역전과 사전 활성화가 그것을 어떻게 뒤집는지는 [항등 사상](identity_mapping.md) 쪽에서 다룬다.

학습 예산도 자릿수가 다르다. 위의 93.39%는 He 등(2015)이 정한 CIFAR-10 일정 — 반복 64,000번 — 에 걸쳐 얻은 값이고, 배치 128로 5만 장을 훑으면 그것은 약 164세대에 해당한다. 이 쪽의 4세대는 그 40분의 1이다. 두 수는 견줄 수 있는 것이 아니다. 이 쪽이 세우는 주장은 정확도의 값이 아니라 **파이프라인이 끝까지 돌아가고, stem의 선택이 측정할 수 있는 차이를 만들며, 씨앗 하나로는 그 차이를 잴 수 없다**는 것이다.

**탈것이 동물보다 쉽다.** 출력 [5]번에서 가장 잘 맞힌 것은 car 82.90%와 ship 81.80%, 가장 못 맞힌 것은 cat 39.30%와 bird 57.30%다. 부류를 둘로 나누어 평균 내면 탈것 넷(plane, car, ship, truck)이 76.45%, 동물 여섯(bird, cat, deer, dog, frog, horse)이 60.00%로 16.45%포인트 차이가 난다. 부류마다 시험 표본이 1,000장으로 같으므로 전체 정확도는 열 값의 단순 평균이고, 실제로 $(305.80 + 360.00)/10 = 66.58$%가 [3]번 씨앗 0의 마지막 세대 값과 정확히 같다. 다만 이 표는 **무엇을 무엇으로 잘못 보았는지**는 말해 주지 않는다. 그것을 알려면 혼동 행렬을 따로 재야 한다.

**model.train()과 model.eval()을 가르는 것은 배치 정규화다.** `train_epoch`은 `model.train()`으로, `evaluate`와 `evaluate_per_class`는 `model.eval()`로 시작한다. 드롭아웃이 없는 이 모델에서 두 상태를 가르는 것은 배치 정규화 하나다. 학습 상태에서는 지금 배치의 평균과 분산으로 정규화하면서 이동 평균을 갱신하고, 평가 상태에서는 그 이동 평균을 쓴다. `model.eval()`을 빠뜨리면 시험 정확도가 배치의 구성에 따라 달라지고, 배치 크기가 1이면 분산이 0이 되어 아예 터진다. 같은 이유로 `stage_sizes`도 `model.eval()`을 먼저 부른다.

**코사인 담금질은 학습률을 반주기 코사인으로 떨어뜨린다.** `CosineAnnealingLR(optimizer, T_max=T)`은 $t$번째 세대의 학습률을

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

로 정한다. 여기서는 $\eta_{\max} = 0.1$, $\eta_{\min} = 0$, $T = 4$이므로 $t = 0, 1, 2, 3$에서 각각 0.1000, 0.0854, 0.0500, 0.0146이 되고, 출력의 `lr` 열이 정확히 이 네 값이다. 마지막 세대의 학습률이 처음의 15% 아래로 내려가는 덕분에 마지막 값이 요동 없이 자리를 잡는다. $T$를 실제 세대 수와 맞추는 것이 중요하다. $T$를 100으로 두고 4세대만 돌리면 학습률이 0.1 언저리에 머문 채 끝나 마지막 값이 세대마다 크게 흔들린다.

**시험 손실만 마지막 배치에 조금 기울어 있다.** 시험 집합 10,000장을 256장씩 끊으면 $10{,}000 = 39 \cdot 256 + 16$이라 배치가 40개 나오고 마지막 배치에는 16장뿐이다. `evaluate`가 돌려주는 손실은 배치마다의 평균을 다시 배치 개수로 나눈 값이므로, 16장짜리 배치가 256장짜리 배치와 같은 무게를 갖는다. 표본 단위 평균과는 조금 다르다. 반면 정확도는 맞힌 개수를 전체 개수로 나눈 것이라 이 편향이 없다. 출력의 정확도 열은 그대로 믿어도 되고, 손실 열은 같은 코드로 잰 값끼리만 견주어야 한다.

**이음매는 가장 좋았던 세대의 것을 남긴다.** `test_acc > best_acc`일 때만 저장하므로 파일에는 마지막 세대가 아니라 가장 좋았던 세대의 가중치가 들어간다. 최적화기의 상태까지 함께 담아 두면 학습을 이어 갈 수 있다. 출력 [6]번은 그 파일을 새 모델에 되읽어 다시 평가한 것으로, 정확도가 저장할 때 적어 둔 값과 소수점까지 같다(`identical: True`). 되읽은 모델이 다른 값을 내놓는다면 대개 `model.eval()`을 빠뜨렸거나 이동 평균이 든 버퍼를 함께 저장하지 않은 것이다.

**씨앗을 어디에 심는가.** `train_resnet_cifar10`은 `torch.manual_seed(seed)`를 모델을 만들기 **전에** 부른다. 카이밍 초기화가 여기에 매달려 있기 때문이다. 그것만으로는 모자라서, `make_loaders`가 `torch.Generator`를 따로 씨앗 붙여 `DataLoader`에 넘긴다. 이것이 없으면 배치의 차례가 실행마다 달라져 같은 씨앗을 심어도 위의 수가 다시 나오지 않는다. `RandomCrop`과 `RandomHorizontalFlip`은 전역 난수 발생기를 쓰므로 `manual_seed` 하나로 함께 묶인다(`num_workers=0`이기 때문이다. 일꾼을 쓰면 씨앗이 프로세스마다 갈라져 따로 챙겨야 한다).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
두 stem의 매개변수 차이 7,680개를 손으로 구하고, 그것이 모델 전체에서 차지하는 몫을 구하라. 출력 [1]번의 값과 맞는가?

</div>

??? success "연습문제 1 풀이"
    stem의 합성곱은 둘 다 치우침이 없고 입력 채널 3개, 출력 채널 64개다. 그러므로 매개변수는 커널 크기에만 달려 있다.

    $$7^2 \cdot 3 \cdot 64 = 9{,}408, \qquad 3^2 \cdot 3 \cdot 64 = 1{,}728$$

    차이는 $9{,}408 - 1{,}728 = 7{,}680$개이고, 이는 $(7^2 - 3^2) \cdot 3 \cdot 64 = 40 \cdot 192$로도 같다. 전체에서 차지하는 몫은 $7{,}680 / 11{,}181{,}642 = 0.000687$, 곧 0.07%다. 나머지 부품(잔차 단계 넷, 배치 정규화, 마지막 선형층)은 두 모델에서 글자 그대로 같으므로 차이는 이것뿐이다. 출력 [1]번의 `parameter difference 7,680 ... i.e. 0.07% of the model`이 이 값이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
크기 공식 $H_{\text{out}} = \lfloor (H_{\text{in}} + 2p - k)/s \rfloor + 1$로 ImageNet stem이 32×32 입력에 남기는 크기를 두 층 모두 따라가라. 224×224 입력이었다면 얼마가 되는가?

</div>

??? success "연습문제 2 풀이"
    7×7 합성곱은 $k = 7$, $p = 3$, $s = 2$이므로

    $$\left\lfloor \frac{32 + 6 - 7}{2} \right\rfloor + 1 = \lfloor 15.5 \rfloor + 1 = 16$$

    이고, 이어지는 3×3 최대 풀링은 $k = 3$, $p = 1$, $s = 2$이므로

    $$\left\lfloor \frac{16 + 2 - 3}{2} \right\rfloor + 1 = \lfloor 7.5 \rfloor + 1 = 8$$

    이다. 잔차 단계에 8×8이 들어가며, 출력 [1]번의 `stage1` 열이 그 값이다. 같은 계산을 224에 하면 $\lfloor 223/2 \rfloor + 1 = 112$, 이어서 $\lfloor 111/2 \rfloor + 1 = 56$이 되어 단계들이 56×56·28×28·14×14·7×7 위에서 돈다. stem 자체가 잘못된 것이 아니라 stem과 입력 크기의 짝이 어긋난 것임을 여기서 알 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
코사인 담금질 식으로 $\eta_{\max} = 0.1$, $\eta_{\min} = 0$, $T = 4$일 때 네 세대의 학습률을 구하고 출력의 `lr` 열과 맞추어라. `T_max=100`으로 두고 4세대만 돌리면 무엇이 달라지는가?

</div>

??? success "연습문제 3 풀이"
    $\eta_t = \tfrac{1}{2} \cdot 0.1 \cdot (1 + \cos(\pi t / 4))$에 $t = 0, 1, 2, 3$을 넣는다.

    $$\eta_0 = 0.1, \quad \eta_1 = \frac{0.1(1 + \tfrac{\sqrt 2}{2})}{2} = 0.0854, \quad \eta_2 = \frac{0.1}{2} = 0.05, \quad \eta_3 = \frac{0.1(1 - \tfrac{\sqrt 2}{2})}{2} = 0.0146$$

    출력의 네 세대에 찍힌 `lr 0.1000`, `lr 0.0854`, `lr 0.0500`, `lr 0.0146`이 이것이다.

    `T_max=100`으로 두면 $t = 3$에서도 $\eta_3 = 0.05 \cdot (1 + \cos(0.03\pi)) = 0.0998$이라 학습률이 사실상 처음 값 그대로다. 큰 학습률에서 멈추면 가중치가 아직 골짜기 바닥에 내려앉지 못한 상태라 마지막 정확도가 세대마다 몇 퍼센트포인트씩 튄다. 일정의 길이는 실제로 돌릴 세대 수와 맞추어야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
출력 [5]번에서 탈것 넷(plane, car, ship, truck)의 평균 정확도와 동물 여섯(bird, cat, deer, dog, frog, horse)의 평균 정확도를 구하라. 두 값에서 전체 66.58%가 어떻게 나오는가? 이 표만으로 말할 수 없는 것은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    탈것 넷은

    $$\frac{65.60 + 82.90 + 81.80 + 75.50}{4} = \frac{305.80}{4} = 76.45\%$$

    이고, 동물 여섯은

    $$\frac{57.30 + 39.30 + 65.70 + 57.60 + 74.10 + 66.00}{6} = \frac{360.00}{6} = 60.00\%$$

    이다. 차이는 16.45%포인트다. 부류마다 시험 표본이 1,000장으로 같으므로 전체 정확도는 열 값의 단순 평균이며, 두 무리의 합을 그대로 써서

    $$\frac{305.80 + 360.00}{10} = 66.58\%$$

    를 얻는다. 출력 [5]번의 `overall` 줄과 같은 값이다. 표본 수가 부류마다 달랐다면 가중 평균을 써야 했을 것이다.

    이 표가 말해 주지 않는 것은 **혼동의 방향**이다. cat이 39.30%라는 것은 1,000장 가운데 607장을 틀렸다는 뜻일 뿐, 그것을 dog로 보았는지 deer로 보았는지는 알 수 없다. 알려면 $10 \times 10$ 혼동 행렬을 따로 쌓아야 한다. `evaluate_per_class`의 반복문에서 `confusion[label][pred] += 1`을 세는 것으로 충분하다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
이 쪽이 얻은 66.58\~73.57%를 "ResNet은 CIFAR-10에서 93%를 넘는다"는 흔한 말과 나란히 놓아도 되는가? He 등이 실제로 보고한 값을 적고, 견줄 수 없는 까닭을 세 가지 들어라.

</div>

??? success "연습문제 5 풀이"
    He 등(2016)의 항등 사상 논문 표 3에 실린 CIFAR-10 시험 오차는 ResNet-110이 6.61%, ResNet-164가 5.93%, ResNet-1001이 7.61%다. 정확도로 바꾸면 각각 93.39%, 94.07%, 92.39%이고, 사전 활성화 판본의 1001층은 4.92% 오차 곧 95.08%다. 나란히 놓을 수 없는 까닭은 셋이다.

    1. **구조가 다르다.** 논문의 CIFAR-10 모델은 단계 셋에 채널 16·32·64인 전용 구조(20·32·44·56·110층)다. 이 쪽의 ResNet-18은 단계 넷에 채널 64·128·256·512인 ImageNet 구조이며, He 등은 그것을 CIFAR-10에서 보고한 적이 없다. "CIFAR-10에서 ResNet-50은 94~95%" 같은 수는 이 논문에서 나온 것이 아니다.

    2. **학습 예산이 다르다.** 93.39%는 He 등(2015)이 정한 반복 64,000번의 결과이고, 배치 128로 5만 장을 훑으면 $64{,}000 \times 128 / 50{,}000 \approx 164$세대에 해당한다. 이 쪽은 4세대이므로 40분의 1이다. 출력 [3]번에서 세 씨앗 모두 마지막 세대까지 정확도가 오르는 중이라는 것이 그 증거다.

    3. **한 번 돌린 값이 아니다.** He 등(2015)은 110층을 다섯 번 돌려 최고값 6.43%와 평균±표준편차 6.61±0.16%를 함께 적었다. 이 쪽도 마찬가지로 씨앗 셋의 폭 66.58\~73.57%로 적어야 하며, 그 폭은 6.99%포인트로 두 stem 사이의 간격 2.91%포인트보다도 넓다.

    덧붙여, "깊을수록 좋다"는 직관도 이 표에서 깨진다. 1001층(7.61%)은 164층(5.93%)보다 1.68%포인트 나쁘다. 이 역전을 사전 활성화가 어떻게 뒤집는지는 [항등 사상](identity_mapping.md) 쪽이 다룬다.

---

## 정리하며

**다룬 것** — 실전 예제

CIFAR-10에서 ResNet-18을 처음부터 학습시키는 파이프라인 전체를 한 각본에 담았다. 데이터 증강(4칸 덧대기 뒤 무작위 자르기, 좌우 뒤집기)과 채널별 정규화, SGD(운동량 0.9, 가중치 감쇠 $5 \times 10^{-4}$)와 코사인 담금질, 세대마다의 평가, 가장 좋았던 이음매의 저장과 되읽기가 그것이다.

수로 남는 것은 셋이다. 첫째, stem을 ImageNet용에서 CIFAR용으로 바꾸는 데 드는 매개변수는 7,680개 — 전체의 0.07% — 인데 잔차 단계가 보는 해상도는 8×8에서 32×32로 달라진다. 둘째, 같은 씨앗 셋으로 4세대씩 재면 마지막 시험 정확도가 CIFAR용 66.58\~73.57%, ImageNet용 61.31\~63.67%로 두 폭이 겹치지 않는다. 다만 CIFAR용 쪽의 폭(6.99%포인트)이 두 무리 사이의 간격(2.91%포인트)보다 넓으므로, 정해진 것은 차이의 방향이지 크기가 아니다. 셋째, 되읽은 이음매가 저장할 때 적어 둔 66.58%를 소수점까지 그대로 낸다.

그리고 이 쪽이 적지 않는 수가 하나 있다. 93%다. He 등이 CIFAR-10에서 얻은 93.39%(ResNet-110, 오차 6.61%)는 다른 구조로 40배 긴 학습을 한 결과이며, ResNet-18·34·50의 CIFAR-10 값은 그 논문에 아예 없다. 잔차 연결이 학습 곡선 자체를 어떻게 바꾸는지는 [학습 비교](03_training_comparison.md)가, 더하기 뒤의 ReLU를 치워 지름길을 참된 항등으로 만드는 변형은 [항등 사상](identity_mapping.md)이 이어 본다.

**참고 문헌**

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *CVPR 2016*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016*.
