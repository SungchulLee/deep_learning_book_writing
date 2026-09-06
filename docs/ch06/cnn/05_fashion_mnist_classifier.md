# Fashion-MNIST 분류기

같은 CNN 구조를 Fashion-MNIST에 적용해 보면 데이터셋의 복잡함이 분류 성능에 곧바로 미치는 영향이 드러난다. 모델의 용량, 학습 절차, 초매개변수가 모두 같은데도 정확도가 숫자 MNIST의 약 99%에서 Fashion-MNIST의 90~92%로 떨어진다. 이 비교는 부류 사이의 시각적 유사성과 부류 안의 변화가 분류 난이도를 좌우하는 주된 요인임을 뚜렷하게 보여 준다.

## 코드

```python
"""
05_fashion_mnist_classifier.py
================================
Fashion-MNIST 데이터셋으로 CNN 학습시키기

같은 모델, 다른 데이터 = 다른 난이도!

난이도: 중간
예상 시간: 1~2시간

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import torch.nn as nn
import torch.optim as optim
import cnn_utils as utils

# =============================================================================
# 설정
# =============================================================================

cfg = utils.parse_args()
utils.set_seed(seed=cfg.seed)

FASHION_LABELS = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

# =============================================================================
# 데이터 적재
# =============================================================================

train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}
trainloader, testloader = utils.load_data(
    train_kwargs, test_kwargs, fashion_mnist=True
)

# =============================================================================
# 모델 준비
# =============================================================================

model = utils.CNN().to(cfg.device)
loss_ftn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.gamma)

# =============================================================================
# 학습
# =============================================================================

utils.train(
    model, trainloader, loss_ftn, optimizer, scheduler,
    cfg.device, cfg.epochs, cfg.log_interval, cfg.dry_run
)

# =============================================================================
# 평가
# =============================================================================

utils.show_batch_or_ten_images_with_label_and_predict(
    testloader, model, cfg.device, classes=FASHION_LABELS, n=10
)
test_accuracy = utils.compute_accuracy(model, testloader, cfg.device)
print(f"Test Accuracy: {test_accuracy:.2f}%")

if cfg.save_model:
    utils.save_model(model, cfg.path)


if __name__ == "__main__":
    pass
```

## 논의

이 실험에서 가장 많은 것을 말해 주는 대목이 정확도의 차이이다. 숫자에서 99%를 내는 바로 그 두 층짜리 CNN이 옷가지에서는 대개 90~92%에 그친다. 이 차이는 모델의 한계가 아니라 데이터 자체의 성질에서 온다. 숫자는 모양이 매우 뚜렷하여 "1"과 "8"은 화소 수준에서 근본적으로 다르다. 반면 옷가지는 구조적인 특징을 함께 지녀서, 티셔츠와 셔츠는 둘 다 소매가 있는 윗옷이고 깃 모양 같은 미묘한 세부만 다르다.

Fashion-MNIST의 혼동 행렬에는 특유의 오류 무늬가 드러난다. 가장 흔히 헷갈리는 쌍은 티셔츠/윗옷과 셔츠, 풀오버와 코트, 운동화와 앵클부츠이다. 이런 혼동은 세밀한 구별 특징이 사라지는 $28 \times 28$ 해상도에서의 진짜 시각적 모호함을 드러낸다. 이 어려운 쌍에서 성능을 올리려면 해상도를 높이거나, 더 정교한 구조(더 깊은 신경망, 어텐션 장치)를 쓰거나, 데이터 증강 전략을 써야 한다.

이 실험은 여러 데이터셋에 같은 코드 바탕을 쓰는 일의 값어치도 보여 준다. `fashion_mnist=True` 깃발 하나만 바꿈으로써 데이터의 복잡함이 미치는 영향을 다른 모든 변수에서 떼어 낸다. 이런 통제된 비교는 경험적 기계 학습의 근본 원리이다. 한 번에 한 요인만 바꾸어 그것이 성능에 얼마나 이바지하는지 살핀다.

## 연습문제

**연습문제 1.**
모델을 학습시킨 뒤 Fashion-MNIST의 10개 부류마다 정확도를 계산하라. 정확도가 가장 낮은 세 부류를 찾고 왜 가장 분류하기 어려운지 가설을 세워라.

??? success "연습문제 1 풀이"
    ```python
    import torch
    class_correct = [0] * 10
    class_total = [0] * 10
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{FASHION_LABELS[i]:15s}: {acc:.2f}%")
    ```
    보통 가장 어려운 세 부류는 셔츠(약 70~75%), 티셔츠/윗옷(약 80~85%), 코트(약 82~86%)이다. 셔츠가 가장 어려운 까닭은 윤곽이 비슷해 티셔츠나 풀오버와 쉽게 헷갈리기 때문이다. $28 \times 28$ 해상도에서는 구별에 쓰이는 특징(깃, 단추, 옷감 무늬)이 너무 작아 믿을 만하게 가려낼 수 없다.

---

**연습문제 2.**
난이도가 다른데도 이 모델은 MNIST와 Fashion-MNIST에 같은 구조를 쓴다. MNIST 쪽 구조는 그대로 두면서 Fashion-MNIST의 정확도를 높일 만한 구조 변경 두 가지를 제안하고 그 까닭을 밝혀라.

??? success "연습문제 2 풀이"
    근거 있는 변경 두 가지는 다음과 같다.

    1. **필터 128개짜리 셋째 합성곱 층을 더한다.** Fashion-MNIST는 비슷한 부류를 가르는 데 더 추상적인 특징이 필요하다. 특징 위계가 깊어지면 두 층으로는 담을 수 없는 티셔츠와 셔츠의 미묘한 차이를 붙잡을 수 있다. 층이 하나 늘면 수용 영역도 넓어져 신경망이 더 큰 공간 맥락을 살필 수 있다.

    2. **완전 연결층의 너비를 뉴런 128개에서 256개로 늘린다.** Fashion-MNIST의 10개 부류는 특징 공간에서의 결정 경계가 숫자 10개 부류보다 복잡하다. 완전 연결층이 넓어지면 이런 비선형 경계를 다룰 용량이 커진다. 임베딩 공간에서 결정 경계가 얽혀 있는 헷갈리는 부류 쌍에 특히 중요하다.

---

**연습문제 3.**
세대마다 검증 정확도를 좇아 조기 종료를 구현하라. 검증 정확도가 세 세대 잇달아 나아지지 않으면 학습을 멈춘다. 이 기법이 일반화에 도움이 되는 까닭을 설명하라.

??? success "연습문제 3 풀이"
    ```python
    best_acc = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(cfg.epochs):
        # 한 세대 학습
        model.train()
        for data, target in trainloader:
            data, target = data.to(cfg.device), target.to(cfg.device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_ftn(output, target)
            loss.backward()
            optimizer.step()

        # 평가한다
        val_acc = utils.compute_accuracy(model, testloader, cfg.device)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # 가장 좋은 모델 가중치 저장
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step()
    ```
    조기 종료는 모델이 학습 집합을 외우기 시작하기 전에 학습을 멈추어 과적합을 막는다. 학습이 이어지면 학습 손실은 계속 줄지만 검증 정확도는 정체되거나 떨어질 수 있는데, 이렇게 갈라지는 것이 과적합의 징후이다. 검증 성능이 가장 좋은 지점에서 멈추면 처음 보는 데이터에 더 잘 일반화하는 모델을 얻는다.
