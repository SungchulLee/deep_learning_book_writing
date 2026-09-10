# CIFAR-10 심화

이 실습은 CIFAR-10 분류를 위해 합성곱 층 네 개와 더 많은 필터, 더 넓은 완전 연결층을 갖춘 깊은 CNN을 구현한다. 기본 모델보다 나아진 구조는 신경망의 깊이와 너비가 늘면 특징 표현이 좋아지고 정확도가 곧바로 오른다는 것을 보여 주며, 기본 모델의 60~70%에 견주어 대개 75~80%에 이른다.

## 1. 코드

```python
"""
07_cifar10_advanced.py
======================
성능이 더 좋은 CIFAR-10용 심화 CNN

난이도: 어려움
예상 시간: 2시간

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import torch.nn as nn
import torch.optim as optim
import cnn_utils as utils

cfg = utils.parse_args()
utils.set_seed(seed=cfg.seed)

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

# 데이터 적재
train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}
trainloader, testloader = utils.load_data(train_kwargs, test_kwargs, cifar10=True)

# 모델 준비
model = utils.CNN_CIFAR10().to(cfg.device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

loss_ftn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.gamma)

# 학습 전
utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, classes=CIFAR10_CLASSES, n=10, cifar10=True
)

# 학습
utils.train(
    model, trainloader, loss_ftn, optimizer, scheduler,
    cfg.device, cfg.epochs, cfg.log_interval, cfg.dry_run
)

# 학습 후
utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, classes=CIFAR10_CLASSES, n=10, cifar10=True
)
test_accuracy = utils.compute_accuracy(model, testloader, cfg.device)

if cfg.save_model:
    utils.save_model(model, cfg.path)

print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    pass
```

## 2. 논의

심화 구조는 풀링 연산마다 그 앞에 합성곱 층을 두 개씩 두어 합성곱 뼈대의 깊이를 두 배로 만든다. 첫 블록은 padding=1인 Conv2d(3, 32, 3) 층 두 개를 써서 공간 차원을 지키면서 특징 채널을 32개로 늘린다. 둘째 블록도 마찬가지로 Conv2d(32, 64, 3) 층 두 개를 쓴다. 이 "합성곱 둘 뒤 풀링" 방식은 블록마다 합성곱 하나보다 넓은 실효 수용 영역을 주어, 하향 표본화에 앞서 더 복잡한 공간 무늬를 붙잡게 해 준다.

기본 모델의 6개/16개 필터에서 32개/64개로 늘리면 용량이 훨씬 커진다. 필터마다 서로 다른 시각 무늬를 잡는 법을 배우므로 첫 층의 필터 32개는 32가지 모서리와 질감과 색 기울기를 나타낼 수 있는데, 이는 기본 모델의 6가지보다 훨씬 많다. (기본 모델의 120개에 견주어) 뉴런 512개짜리 완전 연결층은 이렇게 배운 특징을 엮어 부류를 결정할 용량을 더 많이 준다.

약 65%에서 78%로의 성능 향상은 모델의 용량과 데이터셋의 복잡함 사이의 관계를 보여 준다. 그렇지만 이 구조는 아직 최상급($>95\%$)과는 거리가 멀며, 그 수준에 이르려면 잔차 연결, 데이터 증강, 층마다의 배치 정규화, 꼼꼼한 초매개변수 조정 같은 기법이 필요하다. 이 간격이 ResNet 같은 요즘 구조를 공부할 까닭이 된다.

## 연습문제

**연습문제 1.**
CIFAR-10 기본 모델과 심화 모델의 매개변수 수를 견주어라. 비를 계산하고 정확도 향상이 매개변수 증가에 비례하는지 논하라.

??? success "연습문제 1 풀이"
    기본 모델(SimpleCNN)의 매개변수는 약 62,006개이다. 심화 모델(CNN_CIFAR10)은 약 220만 개로 대략 35배 많다. 그런데 정확도는 약 65%에서 약 78%로, 상대적으로 20%쯤 오를 뿐이다. 이 수확 체감은 매개변수를 늘린다고 정확도가 선형으로 오르지는 않음을 보여 준다. 구조 설계(깊이, 연결 방식)와 학습 기법(증강, 정규화, 규제)이 순수한 매개변수 수만큼이나 중요하다. 최상급 모델은 매개변수가 더 많아서가 아니라 구조를 꼼꼼히 설계하여 95%를 넘긴다.

---

**연습문제 2.**
심화 모델은 (합성곱 블록 뒤에) 0.25, (완전 연결층 뒤에) 0.5의 드롭아웃 비율을 쓴다. 깊이에 따라 다른 비율을 쓰는 까닭과 둘 다 0.5로 두면 어떻게 될지 설명하라.

??? success "연습문제 2 풀이"
    합성곱 층은 공간 위치에 걸쳐 매개변수를 나누어 쓰므로 활성값의 25%만 떨어뜨려도 공간 구조를 크게 망가뜨리지 않으면서 충분한 규제가 된다. 완전 연결층은 출력당 매개변수가 훨씬 많고(뉴런 512개짜리 층 하나만 해도 매개변수가 200만 개가 넘는다) 과적합에 더 잘 빠지므로 0.5라는 높은 비율이 알맞다.

    둘 다 0.5로 두면 합성곱 층이 학습 중에 공간 정보를 너무 많이 잃는다. 특징 맵이 지나치게 성겨져 뒤따르는 층이 일관된 무늬를 잡기 어려워진다. 그러면 수렴이 느려지고 마지막 정확도도 떨어질 가능성이 크다. 깊이에 따라 달리 매긴 드롭아웃 비율은 규제의 세기와 신경망을 지나는 정보 흐름을 지킬 필요 사이에서 균형을 잡는다.

---

**연습문제 3.**
CIFAR-10 학습 변환에 데이터 증강(무작위 좌우 뒤집기와 padding=4인 무작위 잘라내기)을 더하라. 좌우 뒤집기가 CIFAR-10의 모든 부류에 안전한 까닭을 설명하고 정확도가 얼마나 오를지 어림하라.

??? success "연습문제 3 풀이"
    ```python
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ```
    좌우 뒤집기는 CIFAR-10의 10개 부류 모두에 안전한데, 좌우를 바꾸면 범주가 달라지는 비대칭이 어느 부류에도 없기 때문이다. 뒤집힌 비행기, 자동차, 새, 고양이 따위도 여전히 같은 물체이다. 이는 "b"를 뒤집으면 "d"가 되는 문자 인식과 대조된다.

    데이터 증강은 대개 CIFAR-10 정확도를 3~5%p 올린다. 덧댄 뒤 무작위로 잘라 내면 물체가 화면 안에서 옮겨져 평행 이동 불변성을 흉내 내고, 좌우 뒤집기는 실질적인 데이터셋 크기를 공짜로 두 배로 만든다. 둘이 함께 모델이 화소의 정확한 자리와 방향을 외우지 못하게 하여 과적합을 줄인다. 심화 구조에서 기대되는 정확도는 약 80~83%이다.

## 정리하며

**다룬 것** — CIFAR-10 심화

심화 구조는 풀링 연산마다 그 앞에 합성곱 층을 두 개씩 두어 합성곱 뼈대의 깊이를 두 배로 만든다.

앞의 연습문제 3개로 직접 확인할 수 있다.
