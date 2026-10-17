# 실전 예제

CIFAR-10으로 ResNet-18 변형을 학습하며 어그멘테이션·평가·체크포인트 저장과 복원을 연결한다. 첫 컨볼루션과 풀링으로 구성한 스템(stem)을 CIFAR용 3×3 스트라이드 1과 ImageNet용 7×7 스트라이드 2+풀링으로 나누어 비교한다.

코드에는 BasicBlock과 ResNet 정의가 포함되어 있다. 같은 시드 번호와 에포크 수를 사용한 비교이지만 입력 해상도에 따른 연산 예산과 초기화·어그멘테이션 난수까지 완전히 같다는 뜻은 아니다. 관찰된 성능과 비교 설계의 범위를 함께 살펴본다.

## 1. 코드

```python
"""
실전 예제: CIFAR-10에서 ResNet-18 학습시키기
==============================================
데이터 적재, 학습 반복문, 평가, 이음매 저장과 되읽기까지 담은 완전한
파이프라인. 앞 쪽의 `BasicBlock`을 그대로 쓰되 첫 레이어(stem)만 두 가지로
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
    """기본 잔차 블록 — 3x3 컨볼루션 둘과 지름길 하나.

    02_resnet_implementation.md 의 `BasicBlock` 과 같은 것이다. 뒤에 배치
    정규화가 오므로 두 컨볼루션 모두 bias=False 로 둔다.
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

    stem='imagenet' — 7x7 스트라이드 2 컨볼루션 다음에 3x3 스트라이드 2 최대 풀링.
                      224x224 입력을 겨냥한 설계이며 02_resnet_implementation.md
                      의 ResNet 이 쓰는 것이 이것이다.
    stem='cifar'    — 3x3 스트라이드 1 컨볼루션 하나, 풀링 없음. 32x32 에서는 이쪽이다.
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
# 옮겨 놓아야 첫 레이어의 그래디언트 크기가 채널마다 들쭉날쭉해지지 않는다.
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
    # 시험 집합에는 어그멘테이션을 걸지 않는다. 어그멘테이션은 학습 집합을 넓히는 장치이지
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
    """한 세대를 학습시키고 (배치 평균 로스, 정확도)를 돌려준다."""
    model.train()      # 배치 정규화가 배치 통계를 쓰고 이동 평균을 갱신한다
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()          # 그래디언트는 쌓이므로 매번 지워야 한다
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """시험 집합에서 (배치 평균 로스, 정확도)를 잰다."""
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
    # 웨이트 초기화와 어그멘테이션(RandomCrop, RandomHorizontalFlip)의 난수가 이 한 줄에
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

**stem 하나가 해상도의 대부분을 삼킨다.** 출력 [1]번 표의 두 줄은 파라메터가 11,181,642개와 11,173,962개로 0.07%밖에 다르지 않은데, 32×32 입력이 단계를 지나며 남기는 크기는 8×8·4×4·2×2·1×1 대 32×32·16×16·8×8·4×4로 한 축마다 네 배가 차이 난다. 공간 크기 공식

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

에 ImageNet stem의 두 레이어를 차례로 넣어 보면 그 까닭이 나온다. 7×7 컨볼루션($k=7$, $p=3$, $s=2$)은 $\lfloor (32 + 6 - 7)/2 \rfloor + 1 = 16$을, 이어지는 3×3 최대 풀링($k=3$, $p=1$, $s=2$)은 $\lfloor (16 + 2 - 3)/2 \rfloor + 1 = 8$을 준다. 잔차 단계가 한 번도 돌기 전에 이미 네 배가 줄어 있는 것이다. 224×224에서는 같은 stem이 56×56을 남기므로 아무 문제가 없다. 문제는 stem이 아니라 stem과 입력 크기의 짝이다.

스템 컨볼루션의 웨이트 차이는 9408−1728=7680개로 전체의 약 0.07%이다. 그러나 각 축의 피처 크기가 4배 차이 나므로 대응 단계의 공간 위치 수는 대략 16배 다르다. 파라메터 차이가 작아도 연산량·엑티베이션 메모리와 정보 처리 방식의 차이는 크다. ImageNet 스템의 마지막 1×1 피처 맵에서는 전역 평균 풀링이 한 위치만 결합한다.

**이 실험의 비교**: 아래 표는 두 스템을 세 시드에서 각각 4에포크 학습한 마지막 테스트 정확도이다. 같은 에포크 수는 같은 표본 처리 횟수의 기준이며 동일 FLOP이나 시간 예산의 비교는 아니다. 다른 크기의 첫 커널은 난수 소비량도 달라 뒤쪽 초기 웨이트와 전역 난수에 의존한 어그멘테이션까지 같은 값으로 고정된다고 가정할 수 없다.

| stem | 시드 0 | 시드 1 | 시드 2 | 폭 |
|---|---|---|---|---|
| cifar 3×3/1 | 66.58% | 73.57% | 70.82% | 66.58 ~ 73.57% |
| imagenet 7×7/2+pool | 62.37% | 61.31% | 63.67% | 61.31 ~ 63.67% |

세 실행에서 CIFAR용 스템의 정확도가 더 높다. 이는 해당 실행들의 관찰이며, 세 시드 범위가 겹치지 않는다는 이유만으로 시드 변동을 통계적으로 배제할 수는 없다. 시드별 차이와 평균·산포를 함께 보고 더 많은 반복이나 적절한 불확실성 평가로 비교한다. 범위가 넓다는 사실이 효과 크기가 전혀 추정되지 않았다는 뜻도 아니다.

두 모델은 4에포크라는 짧은 예산에서 학습했고 마지막까지 정확도가 변하고 있다. 이 결과는 그 예산에서의 성능 비교이며 충분히 조율한 최종 성능의 비교는 아니다. 현재 곡선만으로 어느 모델도 용량 한계에 도달하지 않았다고 증명할 수는 없다.

**이 66~74%를 "ResNet의 CIFAR-10 정확도"로 옮겨 적어서는 안 된다.** He 등이 CIFAR-10에서 실제로 보고한 값은 다음과 같다(「Identity Mappings in Deep Residual Networks」, 2016, 표 3의 테스트 오차).

| 모델 | 테스트 오차 | 정확도 |
|---|---|---|
| ResNet-110 | 6.61% | 93.39% |
| ResNet-164 | 5.93% | 94.07% |
| ResNet-1001 | 7.61% | 92.39% |

이 표는 인용한 논문의 CIFAR 설정에서 나온 결과이며 이 예제의 ResNet-18 변형과 구조·깊이·학습 조건이 다르다. 논문에 없는 ResNet-18·34·50의 CIFAR 정확도를 이 값에서 추정하지 않는다. 사전 활성화 조건과 기본 조건도 구분해서 인용해야 한다.

학습 기간과 조율 절차도 다르다. 예를 들어 반복 64000번·배치 128의 표본 처리량은 5만 장 기준 약 164에포크에 해당한다. 이것과 4에포크 결과의 차이를 구조만의 차이로 해석할 수 없다. 현재 파이프라인의 작동 예시와 논문의 최종 성능 결과를 구분한다.

**클래스별 결과**: 기록된 시드 0 모델에서 탈것 네 클래스의 평균 재현율은 76.45%, 동물 여섯은 60.00%이다. 이는 이 모델의 해당 평가값이며 탈것이 모든 조건에서 본질적으로 더 쉽다는 뜻은 아니다. 클래스당 표본 수가 같아 전체 66.58%는 열 재현율의 단순 평균이다. 어떤 클래스로 틀렸는지는 혼동 행렬이 필요하다.

**학습·평가 모드**: 이 모델에는 드롭아웃이 없고 BatchNorm이 모드에 따라 달라진다. 학습에서는 배치·공간 통계로 계산하고 이동 통계를 갱신하며 평가에서는 기본적으로 이동 통계를 사용한다. 배치 크기 1이라고 항상 오류가 나는 것은 아니다. 채널당 N×H×W가 1인 학습 모드처럼 통계에 필요한 관측이 부족한 경우를 구분한다.

**코사인 어닐링**: 매 에포크 말 한 번 스케줄러를 호출할 때 초기 상태의 t=0부터 다음 식을 따른다. T_max는 호출 횟수 기준이며, 첫 T_max 구간에서 최솟값으로 감소한다.

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

여기서는 최고 러닝레이트 0.1, 최솟값 0, T=4여서 네 에포크가 사용하는 값은 약 0.1000, 0.0854, 0.0500, 0.0146이다. 이런 감소가 마지막 정확도를 안정시켰다는 원인 설명은 다른 일정과의 비교가 필요하다. T_max=100이면 이 기간의 감쇠가 작다는 수식적 차이만 알 수 있고 정확도가 반드시 크게 요동친다는 결론은 나오지 않는다.

**시험 로스만 마지막 배치에 조금 기울어 있다.** 시험 집합 10,000장을 256장씩 끊으면 $10{,}000 = 39 \cdot 256 + 16$이라 배치가 40개 나오고 마지막 배치에는 16장뿐이다. `evaluate`가 돌려주는 로스는 배치마다의 평균을 다시 배치 개수로 나눈 값이므로, 16장짜리 배치가 256장짜리 배치와 같은 무게를 갖는다. 표본 단위 평균과는 조금 다르다. 반면 정확도는 맞힌 개수를 전체 개수로 나눈 것이라 이 편향이 없다. 출력의 정확도 열은 그대로 믿어도 되고, 로스 열은 같은 코드로 잰 값끼리만 견주어야 한다.

**체크포인트 선택과 복원**: 코드는 테스트 정확도가 이전 최고값을 넘을 때 저장한다. 이 테스트셋이 에포크 선택에 사용되므로 선택된 정확도를 독립 최종 평가값으로 해석해서는 안 된다. 학습셋에서 검증 분할을 만들고 마지막 테스트를 따로 남기는 것이 필요하다. 저장된 모델·옵티마이저·에포크 정보만으로 정확한 재개가 완성되지 않으며 스케줄러·난수·데이터 순서 등도 복원해야 한다. 같은 정확도로 다시 평가했다는 사실은 해당 복원 경로의 점검이며 모든 예측값이 비트 단위로 같다는 증명은 아니다.

**난수 상태**: 모델 초기화 전의 manual_seed와 셔플용 generator는 서로 다른 난수 흐름을 관리한다. 별도 generator가 없어도 적절히 전체 호출 순서를 고정하면 데이터 순서를 재현할 수 있지만, 초기화와 로더 난수를 분리하면 통제가 쉬워진다. num_workers와 변환, 별도 NumPy·사용자 정의 난수 생성기까지 고려해야 하며 같은 시드 번호만으로 다른 장치·버전의 동일 결과가 보장되지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
두 stem의 파라메터 차이 7,680개를 손으로 구하고, 그것이 모델 전체에서 차지하는 몫을 구하라. 출력 [1]번의 값과 맞는가?

</div>

??? success "연습문제 1 풀이"
    stem의 컨볼루션은 둘 다 편향이 없고 입력 채널 3개, 출력 채널 64개다. 그러므로 파라메터는 커널 크기에만 달려 있다.

    $$7^2 \cdot 3 \cdot 64 = 9{,}408, \qquad 3^2 \cdot 3 \cdot 64 = 1{,}728$$

    차이는 $9{,}408 - 1{,}728 = 7{,}680$개이고, 이는 $(7^2 - 3^2) \cdot 3 \cdot 64 = 40 \cdot 192$로도 같다. 전체에서 차지하는 몫은 $7{,}680 / 11{,}181{,}642 = 0.000687$, 곧 0.07%다. 나머지 부품(잔차 단계 넷, 배치 정규화, 마지막 선형 레이어)은 두 모델에서 글자 그대로 같으므로 차이는 이것뿐이다. 출력 [1]번의 `parameter difference 7,680 ... i.e. 0.07% of the model`이 이 값이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
크기 공식 $H_{\text{out}} = \lfloor (H_{\text{in}} + 2p - k)/s \rfloor + 1$로 ImageNet stem이 32×32 입력에 남기는 크기를 두 레이어 모두 따라가라. 224×224 입력이었다면 얼마가 되는가?

</div>

??? success "연습문제 2 풀이"
    7×7 컨볼루션은 $k = 7$, $p = 3$, $s = 2$이므로

    $$\left\lfloor \frac{32 + 6 - 7}{2} \right\rfloor + 1 = \lfloor 15.5 \rfloor + 1 = 16$$

    이고, 이어지는 3×3 최대 풀링은 $k = 3$, $p = 1$, $s = 2$이므로

    $$\left\lfloor \frac{16 + 2 - 3}{2} \right\rfloor + 1 = \lfloor 7.5 \rfloor + 1 = 8$$

    이다. 잔차 단계에 8×8이 들어가며, 출력 [1]번의 `stage1` 열이 그 값이다. 같은 계산을 224에 하면 $\lfloor 223/2 \rfloor + 1 = 112$, 이어서 $\lfloor 111/2 \rfloor + 1 = 56$이 되어 단계들이 56×56·28×28·14×14·7×7 위에서 돈다. stem 자체가 잘못된 것이 아니라 stem과 입력 크기의 짝이 어긋난 것임을 여기서 알 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
T_max=4와 T_max=100인 코사인 일정의 초기 네 에포크 러닝레이트를 비교하고 성능에 대해 알 수 있는 범위를 설명하라.

</div>

??? success "연습문제 3 풀이"
    최고 0.1·최저 0에서 $\eta_t=0.05[1+\cos(\pi t/T)]$이다. T=4의 t=0,1,2,3 값은 약 0.1,0.0854,0.05,0.0146이고 T=100은 마지막 t=3에서도 약 0.0998이다. 짧은 학습 기간의 감쇠 속도가 다르지만 어느 쪽의 정확도가 높거나 안정적인지는 실제 검증 비교가 필요하다.

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

    이다. 차이는 16.45%포인트다. 클래스마다 시험 표본이 1,000장으로 같으므로 전체 정확도는 열 값의 단순 평균이며, 두 무리의 합을 그대로 써서

    $$\frac{305.80 + 360.00}{10} = 66.58\%$$

    를 얻는다. 출력 [5]번의 `overall` 줄과 같은 값이다. 표본 수가 클래스마다 달랐다면 가중 평균을 써야 했을 것이다.

    이 표가 말해 주지 않는 것은 **혼동의 방향**이다. cat이 39.30%라는 것은 1,000장 가운데 607장을 틀렸다는 뜻일 뿐, 그것을 dog로 보았는지 deer로 보았는지는 알 수 없다. 알려면 $10 \times 10$ 혼동 행렬을 따로 쌓아야 한다. `evaluate_per_class`의 반복문에서 `confusion[label][pred] += 1`을 세는 것으로 충분하다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
이 예제의 정확도를 논문의 ResNet 결과와 직접 비교할 때 통제되지 않은 조건을 설명하라.

</div>

??? success "연습문제 5 풀이"
    모델의 스템·단계·깊이와 폭, 학습 기간·옵티마이저·어그멘테이션과 조율 예산, 반복 실행과 결과 집계가 다르다. 현재 결과는 테스트셋으로 체크포인트도 선택하므로 독립적인 최종 평가와 구분해야 한다. 논문 수치를 인용할 때에는 모델명뿐 아니라 데이터셋과 기본·사전 활성화 조건을 함께 명시한다. 성능 차이 하나를 스템이나 깊이만의 효과로 돌리지 않는다.

---

## 정리하며

스템 변경은 파라메터 수보다 피처 해상도와 후속 연산량에 더 큰 차이를 만들 수 있다. 세 시드의 정확도는 해당 학습 예산에서의 관찰이며 초기화·배치·어그멘테이션의 통제와 평가 분할을 함께 살펴야 한다. 테스트 기반 최적 체크포인트 선택과 독립 테스트 평가, 모델 복원과 완전한 학습 재개도 구분한다.

**참고 문헌**

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *CVPR 2016*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016*.
