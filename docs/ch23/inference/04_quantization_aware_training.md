# 양자화 인지 학습(QAT)

양자화 인지 학습은 순전파에서 저정밀도 표현의 오차를 모사하고 역전파로 실수 웨이트를 갱신하는 방법이다. 이 예제는 CIFAR-10용 잔차 CNN에 PyTorch의 eager mode QAT 준비와 변환 절차를 적용하려는 코드다. 준비된 모델의 학습과 실제 정수 모델의 실행은 서로 다른 단계이며, 아래 구현에는 배포 가능한 정수 경로를 완성하는 데 필요한 작업이 남아 있다.

## 1. 예제 코드

```python
"""
가운데 수준: 양자화를 헤아린 익히기(QAT)

이 각본은 익히는 동안 양자화를 흉내내어 모델이 양자화에 튼튼한 무게를
배우게 하는 양자화를 헤아린 익히기를 보인다.

다루는 주제:
- 익히는 동안의 흉내 양자화
- 양자화를 헤아린 익히기 과정
- 익힌 뒤 양자화와의 견줌
- 채널마다와 텐서마다의 양자화 견줌

수학적 바탕:
- 피드포워드: 양자화한 무게를 쓴다
- 뒤먹임: 곧바로 지나가기 어림개를 쓴다
  ∂L/∂w_float ≈ ∂L/∂w_quant (양자화를 항등으로 친다)
- 곧바로 지나가기 어림개는 미분할 수 없는 양자화를 그래디언트가 지나가게 한다

먼저 알아야 할 것:
- 양자화 기본
- 뒤먹임 퍼뜨리기에 대한 이해
- 익히기 되풀이에 익숙함
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

# ========================================================================
# 메인
# ========================================================================

from utils import (
    evaluate_accuracy,
    compare_model_sizes,
    compare_accuracies,
    seed_everything
)


class SimpleResNet(nn.Module):
    """CIFAR-10을 위한 ResNet 꼴 모델."""
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 잔차 블록
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 첫 컨볼루션
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 남는 이음 덩이 1
        identity = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + identity
        
        # 남는 이음 덩이 2(내림 표집 곁들임)
        x = self.relu(self.bn3(self.conv3(x)))
        identity = x
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + identity
        
        # 분류 머리
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def get_cifar10_dataloaders(batch_size=128):
    """CIFAR-10 데이터셋을 불러온다."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_with_qat(model, train_loader, test_loader, epochs=20, lr=0.01, device='cpu'):
    """
    양자화를 헤아린 익히기로 모델을 익힌다.
    
    양자화를 헤아린 익히기 과정:
    1. 흉내 양자화 모듈을 끼운다
    2. 양자화 흉내와 함께 익힌다
    3. 그래디언트가 곧바로 지나가기 어림개를 지난다
    4. 익힌 뒤 실제 양자화 모델로 바꾼다
    """
    model = model.to(device)
    
    # 양자화를 헤아린 익히기를 위해 모델을 마련한다
    model.train()
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\nTraining with Quantization-Aware Training...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # 몸풀기 뒤 배치 고르게 맞추기를 얼린다
        if epoch > epochs // 2:
            model.apply(torch.quantization.disable_observer)
        if epoch > epochs * 3 // 4:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        scheduler.step()
        
        if epoch % 5 == 0:
            test_acc = evaluate_accuracy(model, test_loader, device)
            print(f"Epoch [{epoch+1}/{epochs}] Test Acc: {test_acc*100:.2f}%")
    
    # 양자화된 모델로 바꾸기
    model.eval()
    model_quantized = quant.convert(model, inplace=False)
    
    return model_quantized


def main():
    """양자화를 헤아린 익히기 시범의 으뜸 함수."""
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("QUANTIZATION-AWARE TRAINING DEMONSTRATION")
    print("="*60)
    
    # 데이터를 불러온다
    train_loader, test_loader = get_cifar10_dataloaders()
    
    # 양자화를 헤아린 익히기로 모델을 만들고 익힌다
    model = SimpleResNet(num_classes=10)
    model_qat = train_with_qat(model, train_loader, test_loader, epochs=20, device=device)
    
    # 평가한다
    qat_acc = evaluate_accuracy(model_qat, test_loader, device)
    print(f"\nQAT Model Accuracy: {qat_acc*100:.2f}%")
    
    print("\nQAT typically achieves:")
    print("- Better accuracy than PTQ (1-2% improvement)")
    print("- Same size reduction (4x for INT8)")
    print("- Requires retraining (longer time)")
    print("- Best for models sensitive to quantization")


if __name__ == "__main__":
    main()
```

## 2. 입력과 잔차 연결

기본 입력은 B×3×32×32다. `conv1`과 `conv2`는 B×64×32×32를 유지하므로 첫 잔차 덧셈이 가능하다. `conv3`은 stride=2로 B×128×16×16을 만들고 `conv4`가 이를 유지해 두 번째 덧셈을 한다. 전역 평균 풀링 후 B×128, 최종 로짓은 B×10이다. 교차 엔트로피에는 소프트맥스 전 로짓을 전달한다.

64×64 RGB 입력도 첫 컨볼루션을 바꿀 필요가 없다. 중간 공간 크기는 64×64에서 32×32로 줄고 `AdaptiveAvgPool2d((1,1))`가 마지막 피처 크기를 128로 고정한다. 채널이나 stride를 바꾸는 잔차 블록에서는 더할 두 경로의 차원을 맞춰야 한다.

## 3. QAT의 준비·학습·변환

`prepare_qat`는 설정된 모듈에 fake quantization과 관측기를 연결한다. fake quantization은 저비트 값으로 반올림·클리핑한 효과를 실수 텐서로 모사하므로, 학습 중 모델이 이미 INT8 웨이트만으로 저장·실행되는 것은 아니다. 옵티마이저는 준비 후 생성되며, 학습이 끝난 뒤 `eval()`과 `convert`로 정수 모듈로 변환하려 한다.

관측기를 끄는 것과 fake quantization을 끄는 것은 다르다. `disable_observer`는 범위 추정을 멈추고 이미 추정한 스케일을 사용하게 한다. 기본 20에포크에서 조건 `epoch > epochs // 2`는 0부터 세는 인덱스 11, 즉 12번째 에포크부터 적용된다. `freeze_bn_stats` 호출은 17번째 에포크부터다.

다만 이 코드는 Conv-BN-ReLU를 QAT용 결합 모듈로 구성하지 않았다. 결합된 QAT 모듈을 대상으로 하는 `freeze_bn_stats` 호출이 일반 `BatchNorm2d`의 통계까지 동결한다고 설명할 수 없다. 실제로 어떤 모듈이 바뀌는지 확인해야 한다.

## 4. 정수 추론 경로에서 필요한 작업

이 eager mode 예제에는 입력·출력의 `QuantStub`·`DeQuantStub`, 잔차 덧셈을 위한 양자화 연산 경로, 계층 결합이 빠져 있다. `prepare_qat`를 호출했다고 일반 실수 입력과 `x + identity`가 변환 후 자동으로 올바른 정수 경로가 되는 것은 아니다. 관련 구성은 [PyTorch 양자화 레시피](https://docs.pytorch.org/tutorials/recipes/quantization.html)를 참고할 수 있다.

또한 `fbgemm` 설정과 변환된 정수 모듈의 실행 장치를 맞추어야 한다. `main`이 CUDA를 선택하고 변환 뒤 같은 CUDA 장치에서 평가하는 흐름을 그대로 사용하면 CPU 정수 백엔드의 실행 조건을 충족하지 않는다. QAT 학습에 사용하는 장치와 변환 후 배포 장치를 구분해야 한다.

## 5. 비교 결과의 해석

이 코드는 무작위 초기화 모델을 QAT 방식으로 학습하며, 별도의 FP32 기준 모델이나 PTQ 모델을 학습·평가하지 않는다. 출력문의 “PTQ보다 1~2% 개선”은 이 실행에서 얻은 비교 결과가 아니다. 또한 같은 디렉터리의 `utils.evaluate_accuracy`는 이미 0~100의 백분율을 반환하지만 호출부는 여기에 다시 100을 곱한다. 따라서 표시되는 정확도에는 단위 오류가 있다. INT8 웨이트 데이터의 이상적인 4배 감소도 전체 직렬화 파일과 실행 중 메모리의 감소율을 뜻하지 않는다.

품질 비교에서는 같은 데이터 분할과 전처리, 공정한 학습 예산을 사용하고 기준 모델·PTQ·QAT를 구분한다. 이 예제는 테스트 세트를 주기적으로 조회하므로 설정 선택에 그 결과를 사용한다면 별도의 검증 세트가 필요하다. 출력 정확도만으로 지연 시간이나 배포 성공 여부까지 판단할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
배치 크기 4, RGB 32×32 입력에서 각 잔차 블록과 풀링·선형 계층의 출력 모양을 구하라.

</div>

??? success "풀이"

    첫 두 컨볼루션과 첫 덧셈은 4×64×32×32다. stride=2인 conv3 이후와 두 번째 덧셈은 4×128×16×16이다. 전역 평균 풀링은 4×128×1×1, 평탄화는 4×128, 분류 로짓은 4×10이다. 잔차 덧셈의 두 입력 모양이 각각 일치한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
RGB 64×64 입력으로 바꿀 때 첫 컨볼루션과 분류기의 차원을 수정해야 하는지 설명하라.

</div>

??? success "풀이"

    첫 컨볼루션은 이미 입력 채널 3을 받는다. 공간 크기는 두 번째 블록에서 32×32가 되지만 적응형 평균 풀링이 1×1로 줄이므로 fc의 입력 128도 유지된다. 따라서 이 입력 크기 변경만으로 계층 차원을 수정할 필요는 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
관측기 동결, BatchNorm 통계 동결, fake quantization 비활성화의 목적을 비교하라.

</div>

??? success "풀이"

    관측기 동결은 양자화 범위와 스케일의 갱신을 멈춘다. BatchNorm 통계 동결은 실행 평균·분산의 변화를 멈추는 별도 작업이다. fake quantization 비활성화는 반올림·클리핑 오차의 모사를 끈다. 세 작업은 서로 대신할 수 없으며 이 예제의 BN 동결 함수는 결합된 QAT 모듈을 전제로 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
잔차 블록의 수를 설정 가능하게 구성할 때 필요한 조건을 설명하라. 아래 코드 조각의 역할과 한계도 서술하라.

</div>

??? success "풀이"

    ModuleList는 하위 모듈과 파라미터를 등록하는 컨테이너다. 아래 YourBlock은 정의되지 않은 자리표시자이며 완성된 확장 코드가 아니다. 각 블록의 입출력 채널과 공간 크기, 잔차 투영을 정하고 forward에서 순서대로 호출해야 한다. QAT에서는 새 블록의 결합·양자화 경계와 덧셈 경로도 준비해야 한다. 레이어 수를 늘리는 것만으로 변환 호환성이 보장되지는 않는다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

QAT는 양자화 오차를 모사하며 웨이트를 학습하는 단계와 실제 저비트 연산으로 변환하는 단계를 구분해야 한다. 관측기와 BatchNorm 통계의 동결은 서로 다르고, 잔차 연결·입출력 경계·백엔드까지 맞추어야 정수 추론 경로가 완성된다. 이 예제의 출력 안내문을 측정된 PTQ 대비 성능으로 해석하지 않는다.
