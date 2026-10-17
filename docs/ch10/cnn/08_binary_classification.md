# 이진 분류

MNIST의 10개 숫자에서 두 클래스를 고르는 조합은 45개이다. 각 조합에 별도 BinaryCNN을 학습해 학습 로스와 테스트 정확도를 비교한다. 이진 부분 문제의 순위는 모델·학습 설정과 표본에 의존하며, 10클래스 분류기의 혼동이나 데이터셋의 고유 난이도와 같지 않다.

## 1. 코드

```python
"""
08_binary_classification.py
============================
MNIST 이진 분류 분석

난이도: 고급
예상 시간: 2~3시간

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

print("=" * 70)
print("Binary Classification Analysis on MNIST")
print("=" * 70)


# =============================================================================
# 1절: 이진 분류용 CNN
# =============================================================================

class BinaryCNN(nn.Module):
    """이진 분류를 위한 단순한 CNN.

    클래스가 둘뿐인데도 마지막 레이어의 출력 단위는 **둘**이다. 그래야 로짓을
    `nn.CrossEntropyLoss`에 그대로 넘길 수 있고, 이는 출력을 하나만 두고
    `nn.BCEWithLogitsLoss`를 쓰는 방식과 똑같은 로스를 준다(2.4절).

    무게 개수: conv1 160 + conv2 4,640 + fc1 100,416 + fc2 130 = 105,346.
    fc1 하나가 100,416으로 전체의 95.32%를 차지한다.
    """
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2)          # 클래스가 둘이므로 로짓도 둘

    def forward(self, x):
        # 28x28 -> conv -> 28x28 -> pool -> 14x14 -> conv -> 14x14 -> pool -> 7x7
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)           # 32 x 7 x 7 = 1568
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # 로짓을 그대로 돌려준다. 시그모이드도 소프트맥스도 여기서 씌우지 않는다
        return x


# =============================================================================
# 2절: MNIST를 불러 숫자별로 나누기
# =============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 배치 크기를 데이터셋 크기로 잡아 6만 장을 한 번에 메모리로 올린다
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
all_images, all_labels = next(iter(train_loader))

# 숫자별로 데이터 정리.
# 부류별 장수는 고르지 않다 — 5,421장(숫자 5)에서 6,742장(숫자 1)까지 벌어진다.
# 그래서 쌍마다 학습 장수가 11,263장에서 13,007장 사이로 달라진다.
digit_images = {}
digit_labels = {}
for i in range(10):
    mask = all_labels == i
    digit_images[i] = all_images[mask]
    digit_labels[i] = all_labels[mask]


# =============================================================================
# 3절: 두 숫자로 학습시키기
# =============================================================================

def train_binary(model, images_0, labels_0, images_1, labels_1, epochs=10):
    """두 부류로 학습시키고 로스의 자취를 돌려준다.

    원래의 MNIST 이름표(`labels_0`, `labels_1`)는 **개수만** 쓰고 버린다.
    이진 문제의 이름표는 0과 1로 새로 붙이며, 앞에 온 쪽이 0이다.
    """
    images = torch.cat([images_0, images_1])
    labels = torch.cat([torch.zeros(len(labels_0), dtype=torch.long),
                        torch.ones(len(labels_1), dtype=torch.long)])
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()        # 로짓을 받는다. 소프트맥스는 이 안에서 걸린다
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_trace = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_trace.append(total_loss / len(loader))
    return loss_trace


# =============================================================================
# 4절: 45개 쌍을 모두 돌리고 학습 곡선을 그리기
# =============================================================================

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
final_loss = {}

for i in range(10):
    for j in range(10):
        if i >= j:
            axes[i, j].axis('off')
            continue

        # 쌍마다 씨앗을 되돌린다. 45개 모델이 모두 같은 초기 무게에서 떠나므로
        # 쌍 사이의 차이는 오직 데이터에서 온다. 쌍 하나만 따로 다시 돌려도
        # 같은 값이 나온다 — 처음에 한 번만 씨앗을 주면 어느 쌍의 값이든
        # 그 앞에 돈 쌍들의 난수에 딸리게 되어 따로 확인할 수 없다.
        torch.manual_seed(42)
        model = BinaryCNN()
        n_train = len(digit_images[i]) + len(digit_images[j])
        print(f"Training classifier: {i} vs {j}  (train {n_train}, batches {(n_train + 63) // 64})")
        loss_trace = train_binary(
            model, digit_images[i], digit_labels[i],
            digit_images[j], digit_labels[j], epochs=10
        )
        print("  loss " + " ".join(f"{v:.6f}" for v in loss_trace))
        final_loss[(i, j)] = loss_trace[-1]

        # 45개 쌍을 통틀어 로스가 0.000009에서 0.184까지, 곧 2만 배에 걸쳐 있다.
        # 선형 축에서는 모든 곡선이 바닥에 붙어 한 줄로 보이므로 로그 축을 쓴다.
        axes[i, j].plot(loss_trace, 'r-', linewidth=2)
        axes[i, j].set_yscale('log')
        axes[i, j].set_ylim([1e-6, 1])
        axes[i, j].set_title(f'{i} vs {j}', fontsize=8)
        axes[i, j].axis('off')
        # 아래쪽 삼각형은 바로 위와 같은 자취를 색만 바꾸어 되비친 것이다.
        # 따로 학습한 결과가 아니므로 45개 실험이 90칸에 그려진다.
        axes[j, i].plot(loss_trace, 'b-', linewidth=2)
        axes[j, i].set_yscale('log')
        axes[j, i].set_ylim([1e-6, 1])
        axes[j, i].set_title(f'{j} vs {i}', fontsize=8)
        axes[j, i].axis('off')

plt.suptitle('MNIST Binary Classification Learning Curves', fontsize=16)
plt.tight_layout()
plt.show()


# =============================================================================
# 5절: 마지막 로스로 45개 쌍 줄 세우기
# =============================================================================

print()
print("=" * 70)
print("Final-epoch loss, all 45 pairs (hardest first)")
print("=" * 70)
for (i, j), v in sorted(final_loss.items(), key=lambda kv: -kv[1]):
    print(f"  {i} vs {j}   {v:.6f}")


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
Binary Classification Analysis on MNIST
======================================================================
Training classifier: 0 vs 1  (train 12665, batches 198)
  loss 0.032393 0.003089 0.003016 0.001516 0.002052 0.001460 0.000663 0.000379 0.000167 0.000173
Training classifier: 0 vs 2  (train 11881, batches 186)
  loss 0.078485 0.013402 0.008397 0.004665 0.002318 0.001945 0.001911 0.001372 0.001321 0.000380
Training classifier: 0 vs 3  (train 12054, batches 189)
  loss 0.070485 0.007552 0.004139 0.002698 0.001764 0.001243 0.001368 0.000467 0.000150 0.000060
Training classifier: 0 vs 4  (train 11765, batches 184)
  loss 0.047532 0.007825 0.004021 0.002596 0.002723 0.000973 0.000249 0.000064 0.000042 0.000031
Training classifier: 0 vs 5  (train 11344, batches 178)
  loss 0.073260 0.012755 0.006494 0.006198 0.003700 0.002488 0.001917 0.000809 0.000318 0.000120
Training classifier: 0 vs 6  (train 11841, batches 186)
  loss 0.061560 0.013753 0.010517 0.005914 0.006262 0.004419 0.003179 0.003779 0.005392 0.002538
Training classifier: 0 vs 7  (train 12188, batches 191)
  loss 0.052641 0.005469 0.002517 0.001655 0.000467 0.000182 0.000078 0.000056 0.000038 0.000027
Training classifier: 0 vs 8  (train 11774, batches 184)
  loss 0.074121 0.014819 0.008427 0.007036 0.006231 0.002896 0.002482 0.001359 0.000388 0.000331
Training classifier: 0 vs 9  (train 11872, batches 186)
  loss 0.058303 0.012442 0.008701 0.005904 0.003238 0.003320 0.001578 0.001470 0.000560 0.000282
Training classifier: 1 vs 2  (train 12700, batches 199)
  loss 0.068073 0.017471 0.011291 0.007909 0.006300 0.004940 0.004322 0.003362 0.003808 0.002363
Training classifier: 1 vs 3  (train 12873, batches 202)
  loss 0.071319 0.010101 0.005260 0.001650 0.001511 0.000474 0.000316 0.001043 0.000716 0.000190
Training classifier: 1 vs 4  (train 12584, batches 197)
  loss 0.057436 0.009861 0.007417 0.005261 0.003698 0.002675 0.002744 0.002641 0.001686 0.000841
Training classifier: 1 vs 5  (train 12163, batches 191)
  loss 0.054885 0.007360 0.002723 0.001425 0.000732 0.000257 0.000208 0.000204 0.000068 0.000039
Training classifier: 1 vs 6  (train 12660, batches 198)
  loss 0.049770 0.004099 0.002481 0.001179 0.001512 0.002893 0.000254 0.000607 0.000178 0.000109
Training classifier: 1 vs 7  (train 13007, batches 204)
  loss 0.054830 0.013194 0.008910 0.006940 0.005725 0.006865 0.003627 0.005104 0.003393 0.002791
Training classifier: 1 vs 8  (train 12593, batches 197)
  loss 0.066208 0.011110 0.009450 0.006240 0.004005 0.005848 0.002219 0.002448 0.001730 0.001350
Training classifier: 1 vs 9  (train 12691, batches 199)
  loss 0.054709 0.010092 0.007295 0.004359 0.003470 0.002395 0.001196 0.001245 0.000952 0.000278
Training classifier: 2 vs 3  (train 12089, batches 189)
  loss 0.113350 0.028250 0.017437 0.010322 0.007409 0.007396 0.003248 0.002638 0.001094 0.000447
Training classifier: 2 vs 4  (train 11800, batches 185)
  loss 0.073368 0.009565 0.006013 0.003859 0.003054 0.002395 0.001703 0.000742 0.000600 0.000235
Training classifier: 2 vs 5  (train 11379, batches 178)
  loss 0.085433 0.007024 0.002854 0.002211 0.000707 0.000307 0.000153 0.000077 0.000058 0.000045
Training classifier: 2 vs 6  (train 11876, batches 186)
  loss 0.076312 0.005438 0.004194 0.001228 0.001365 0.000855 0.000173 0.000081 0.000055 0.000041
Training classifier: 2 vs 7  (train 12223, batches 191)
  loss 0.080861 0.025920 0.016630 0.013734 0.011873 0.009500 0.006444 0.005753 0.004876 0.002274
Training classifier: 2 vs 8  (train 11809, batches 185)
  loss 0.098119 0.023497 0.015144 0.009260 0.006888 0.005188 0.002386 0.003505 0.002843 0.002161
Training classifier: 2 vs 9  (train 11907, batches 187)
  loss 0.080824 0.016877 0.009033 0.005474 0.003561 0.002405 0.001732 0.002453 0.000231 0.000115
Training classifier: 3 vs 4  (train 11973, batches 188)
  loss 0.072044 0.009092 0.004500 0.002301 0.000816 0.001016 0.001715 0.000114 0.000051 0.000039
Training classifier: 3 vs 5  (train 11552, batches 181)
  loss 0.144390 0.026336 0.019941 0.013380 0.009353 0.008643 0.006654 0.006307 0.006818 0.005721
Training classifier: 3 vs 6  (train 12049, batches 189)
  loss 0.066734 0.005208 0.001998 0.001815 0.000295 0.000133 0.000112 0.000055 0.000037 0.000027
Training classifier: 3 vs 7  (train 12396, batches 194)
  loss 0.077247 0.018474 0.010298 0.006538 0.005074 0.005717 0.003410 0.002085 0.000605 0.000321
Training classifier: 3 vs 8  (train 11982, batches 188)
  loss 0.124614 0.022258 0.009254 0.006004 0.006049 0.002417 0.002951 0.001656 0.003333 0.000490
Training classifier: 3 vs 9  (train 12080, batches 189)
  loss 0.092637 0.019176 0.009591 0.007668 0.005028 0.002828 0.002053 0.003086 0.001437 0.000527
Training classifier: 4 vs 5  (train 11263, batches 176)
  loss 0.068632 0.005336 0.004062 0.002554 0.000734 0.003479 0.001493 0.000212 0.000054 0.000040
Training classifier: 4 vs 6  (train 11760, batches 184)
  loss 0.064811 0.008751 0.005402 0.003171 0.003521 0.001235 0.002065 0.001083 0.000283 0.000122
Training classifier: 4 vs 7  (train 12107, batches 190)
  loss 0.077894 0.012509 0.006790 0.005120 0.003373 0.003196 0.002933 0.001995 0.001194 0.001113
Training classifier: 4 vs 8  (train 11693, batches 183)
  loss 0.077317 0.014525 0.010547 0.007649 0.004974 0.005211 0.004546 0.002613 0.001150 0.000553
Training classifier: 4 vs 9  (train 11791, batches 185)
  loss 0.183971 0.029795 0.021497 0.017828 0.015283 0.013028 0.011840 0.010084 0.011959 0.008196
Training classifier: 5 vs 6  (train 11339, batches 178)
  loss 0.086559 0.021631 0.013369 0.010868 0.006288 0.004952 0.006677 0.002231 0.001246 0.002496
Training classifier: 5 vs 7  (train 11686, batches 183)
  loss 0.068226 0.008000 0.003909 0.002315 0.001305 0.001413 0.000582 0.000204 0.000072 0.000042
Training classifier: 5 vs 8  (train 11272, batches 177)
  loss 0.116046 0.024139 0.015555 0.012343 0.009408 0.009304 0.006778 0.004849 0.003911 0.004137
Training classifier: 5 vs 9  (train 11370, batches 178)
  loss 0.100244 0.020203 0.013908 0.009291 0.007315 0.005887 0.005138 0.002646 0.002069 0.001494
Training classifier: 6 vs 7  (train 12183, batches 191)
  loss 0.036739 0.002392 0.001053 0.000200 0.000083 0.000253 0.000541 0.000028 0.000012 0.000009
Training classifier: 6 vs 8  (train 11769, batches 184)
  loss 0.097057 0.018833 0.012004 0.007634 0.006091 0.005165 0.004591 0.002729 0.001113 0.000708
Training classifier: 6 vs 9  (train 11867, batches 186)
  loss 0.061716 0.005549 0.003213 0.001560 0.000886 0.000486 0.000101 0.000058 0.000046 0.000040
Training classifier: 7 vs 8  (train 12116, batches 190)
  loss 0.098721 0.012813 0.011696 0.006185 0.006317 0.003999 0.003304 0.001973 0.001203 0.001310
Training classifier: 7 vs 9  (train 12214, batches 191)
  loss 0.133957 0.025467 0.017293 0.012486 0.009580 0.007802 0.006302 0.005175 0.004225 0.001813
Training classifier: 8 vs 9  (train 11800, batches 185)
  loss 0.108190 0.024777 0.016567 0.013194 0.011417 0.009197 0.005796 0.004172 0.003013 0.002990

======================================================================
Final-epoch loss, all 45 pairs (hardest first)
======================================================================
  4 vs 9   0.008196
  3 vs 5   0.005721
  5 vs 8   0.004137
  8 vs 9   0.002990
  1 vs 7   0.002791
  0 vs 6   0.002538
  5 vs 6   0.002496
  1 vs 2   0.002363
  2 vs 7   0.002274
  2 vs 8   0.002161
  7 vs 9   0.001813
  5 vs 9   0.001494
  1 vs 8   0.001350
  7 vs 8   0.001310
  4 vs 7   0.001113
  1 vs 4   0.000841
  6 vs 8   0.000708
  4 vs 8   0.000553
  3 vs 9   0.000527
  3 vs 8   0.000490
  2 vs 3   0.000447
  0 vs 2   0.000380
  0 vs 8   0.000331
  3 vs 7   0.000321
  0 vs 9   0.000282
  1 vs 9   0.000278
  2 vs 4   0.000235
  1 vs 3   0.000190
  0 vs 1   0.000173
  4 vs 6   0.000122
  0 vs 5   0.000120
  2 vs 9   0.000115
  1 vs 6   0.000109
  0 vs 3   0.000060
  2 vs 5   0.000045
  5 vs 7   0.000042
  2 vs 6   0.000041
  6 vs 9   0.000040
  4 vs 5   0.000040
  3 vs 4   0.000039
  1 vs 5   0.000039
  0 vs 4   0.000031
  0 vs 7   0.000027
  3 vs 6   0.000027
  6 vs 7   0.000009
```

## 2. 논의

### 2.1 45개 쌍이 모두 쉽다

기록된 테스트 정확도는 99.4175~100%이며 이 설정에서 오류가 적다. 45개 문제의 테스트 처리 건수를 합하면 90000건이지만 같은 원본 이미지가 9개 쌍에 반복 사용된다. 따라서 90000개의 독립적인 이미지에 대한 평가처럼 불확실성을 계산하면 안 된다.

다수 클래스만 고르는 기준선을 함께 보면 클래스 비율의 영향을 확인할 수 있다. 1 대 5 테스트셋은 1135장과 892장으로 상수 1 예측의 정확도가 약 55.99%이다. 이는 정확도의 최솟값이 아니라 단순 기준선이다. 쌍별 클래스 수와 표본 수가 달라 균형 정확도와 오류 건수도 함께 비교할 수 있다.

높은 정확도에서 작은 차이를 비교할 때에는 한두 표본과 시드 변동이 순위에 영향을 줄 수 있다. 테스트 100%도 모집단 오류가 0이라는 증거는 아니다. 학습 로스 순위와 테스트 오류 순위는 별도로 해석한다.

### 2.2 어려운 쌍은 짐작과 반쯤만 맞는다

마지막 에포크의 로스로 줄을 세우면 출력 블록의 끝에 있는 차례가 나온다. 가장 어려운 다섯은 **4 대 9**(0.008196), **3 대 5**(0.005721), **5 대 8**(0.004137), **8 대 9**(0.002990), **1 대 7**(0.002791)이고, 가장 쉬운 넷은 **6 대 7**(0.000009), **3 대 6**과 **0 대 7**(여섯째 자리까지 똑같이 0.000027), **0 대 4**(0.000031)이다. 양 끝의 비는 반올림하지 않은 값으로 901배여서, 45개 쌍이 모두 쉽다고 해도 그 안의 폭은 세 자릿수에 이른다.

4와 9, 3과 5처럼 일부 필체가 비슷할 수 있다는 시각적 가설을 세울 수 있다. 하지만 숫자 전체를 특정 획 하나의 차이로 정의할 수는 없다. 0 대 6에서 한 방향의 오류가 많다는 관찰도 실제 오분류 표본을 살펴 원인을 검토해야 한다.

3 대 8은 초기 로스에 비해 최종 로스 순위가 낮아진다. 학습 진행에 따라 순위가 바뀐다는 관찰이며, 어떤 피처를 언제 배웠는지 로스만으로 확인한 것은 아니다.

### 2.3 로스 곡선은 난이도를 얼마나 알려 주는가

기록된 45개 쌍에서 마지막 학습 로스와 테스트 오류율의 스피어만 상관은 0.647이다. 이 표의 순위가 어느 정도 함께 움직인다는 기술 통계이지 인과관계나 모든 시드에서의 관계를 보장하지 않는다. 쌍들이 같은 숫자·이미지를 공유하므로 45개 관측을 완전히 독립으로 취급하지 않는다. 학습 로스는 확률의 크기와 오분류의 확신 정도에 민감하고, 정확도는 결정 경계를 넘었는지에 의존한다.

기록된 45개 로스 곡선 중 29개가 적어도 한 번 증가한다. 미니배치 순서, 최적화 상태와 집계 방식 등이 영향을 줄 수 있으며 스케줄러 부재가 유일한 원인이라고 할 수 없다. 배치 평균의 단순 평균은 쌍별 마지막 배치 크기의 영향도 받는다. 같은 초기 웨이트를 사용해도 쌍별 표본 수와 갱신 수가 달라 조건 전체가 동일해지는 것은 아니다.

이 분석은 각 이진 문제를 별도로 푼 결과이다. one-vs-one 다중 분류기로 배포하려면 45개의 결정을 일관된 투표나 점수 결합으로 통합하는 절차가 필요하다. 이진 정확도의 평균이 그 통합 분류기의 정확도와 같지는 않다.

### 2.4 클래스가 둘인데 출력 단위가 둘인 까닭

`BinaryCNN`의 마지막 레이어는 `nn.Linear(64, 2)`이다. 클래스가 둘뿐인데 로짓을 둘 두는 것이 낭비처럼 보이지만, 로스를 적어 보면 하나를 두는 방식과 정확히 같다. 로짓을 $z_0, z_1$이라 하고 참값을 $y \in \{0, 1\}$이라 하면 `nn.CrossEntropyLoss`가 재는 값은

$$L = -\log \frac{e^{z_y}}{e^{z_0} + e^{z_1}}$$

분자와 분모를 $e^{z_0}$으로 나누고 $z=z_1-z_0$로 두면 이진 교차 엔트로피가 된다. 이 로스값의 동등성은 같은 유효 로짓 z를 비교한 결과이다. 서로 독립적으로 초기화하고 학습한 1출력·2출력 모델의 최적화 경로까지 같다는 뜻은 아니다. 정규화와 파라메터화도 경로에 영향을 줄 수 있다.

여기서 조심할 것은 `nn.BCEWithLogitsLoss`가 **로짓**을 받는다는 점이다. 시그모이드를 안에서 걸어 주므로 `forward`에서 다시 `torch.sigmoid`를 씌우면 시그모이드가 두 번 걸린다. 이것은 오류를 내지 않고 조용히 학습만 망가뜨리는 종류의 실수이며, 연습문제 4에서 그 결과를 수로 확인한다. 확률을 직접 넘기고 싶다면 `nn.BCELoss`를 써야 하고, [7.2 이진 분류](../../ch07/logistic_regression/binary_classification.md)가 바로 그 짝(`torch.sigmoid` + `nn.BCELoss`)으로 로지스틱 회귀를 세운다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
1 대 5 문제의 상수 분류기 정확도를 구하고 이를 쌍별 비교에 어떻게 사용할지 설명하라.

</div>

??? success "풀이"
    학습에서는 6742/12163≈55.43%, 테스트에서는 1135/2027≈55.99%이다. 99.9507%는 이 단순 기준선보다 높지만 기준선이 정확도의 수학적 하한은 아니다. 쌍별 클래스 수·비율이 달라 전체 정확도뿐 아니라 클래스별 재현율과 균형 정확도도 비교할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
마지막 학습 로스 순위와 초기 로스 순위의 차이를 어떻게 해석해야 하는가?

</div>

??? success "풀이"
    기록된 마지막 로스 상위 쌍은 4 대 9, 3 대 5, 5 대 8이고 하위에는 6 대 7, 3 대 6, 0 대 7이 있다. 이는 이 모델과 학습 예산에서의 순위이다. 필체 유사성은 가능한 설명이지만 로스만으로 어떤 획이 원인인지 알 수 없다. 초기 순위와 최종 순위의 차이에는 학습 동역학·표본 수·업데이트 수 등이 영향을 줄 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
45개 이진 모델과 마지막 출력만 10개인 모델 하나의 표본 처리량·저장 파라메터 수를 비교하라.

</div>

??? success "풀이"
    각 숫자는 9개 쌍에 쓰이므로 같은 에포크 수에서 이진 모델들의 처리 건수 합은 $9\times60000=540000$으로 10클래스 모델의 9배이다. 이것은 벽시계 학습 시간이 정확히 9배라는 뜻은 아니다. 저장 계수는 $45\times105346=4740570$개 대 105866개로 약 44.78배이다. 전체 one-vs-one 투표에는 여러 모델의 순전파와 결합이 필요하지만 실제 추론 시간 비율은 병렬화·배치와 구현에 달려 있다. 별도 이진 모델은 각 쌍에 맞출 수 있고 하나의 모델은 표현을 공유하지만 어느 쪽의 정확도가 높은지는 별도 평가해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
BCEWithLogitsLoss 앞에 시그모이드를 적용하면 로스와 예측이 어떻게 달라지는가? 3 대 5의 클래스 비율로 이론적 로스 하한을 구하라.

</div>

??? success "풀이"
    모델이 $p=\sigma(z)$를 반환하면 로스는 p를 로짓으로 해석해 $q=\sigma(p)$를 사용한다. 실수 연산에서 $q\in(0.5,\sigma(1))\approx(0.5,0.7311)$이므로 음성의 확률을 0.5 아래로 낮출 수 없다. 양성 로스의 하한은 약 0.3133, 음성은 약 0.6931이다. 양성 비율을 $5421/(6131+5421)\approx0.46927$로 두면 표본별로 완전히 분리할 수 있다는 가정의 평균 로스 하한은 약 0.5149이다. 실제 유한 모델과 최적화가 이 하한에 도달한다고 보장하지 않는다.

    예측에서 p 자체를 0.5로 자르면 원래 z를 0으로 자르는 것과 같아 높은 정확도가 나올 수도 있다. 반면 다시 시그모이드를 적용한 q를 0.5로 자르면 이상적인 실수 계산에서는 모두 양성이다. 부동소수점 포화와 임계값의 > 또는 >= 규약은 끝값에서 별도로 고려한다. 로짓까지의 기울기는 $(q-y)p(1-p)$이므로 p의 포화가 학습을 느리게 할 수 있다. 올바른 조합은 BCEWithLogitsLoss에 원래 로짓을, BCELoss에는 확률을 전달하는 것이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
45개 이진 분류기마다 테스트 정확도를 재고, 성분 $(i, j)$가 숫자 $i$와 $j$를 가르는 테스트 정확도인 $10 \times 10$ 히트맵를 그려라. 대각선을 0으로 두면 왜 안 되는지 밝히고, 나온 무늬를 2.2절의 로스 차례와 비교하여라.

</div>

??? success "풀이"
    아래 코드는 본문 학습을 재실행해 테스트 지표와 히트맵을 만든다. 같은 환경·초기 상태·데이터 순서를 재현하면 같은 결과를 기대할 수 있지만 시드 42만으로 모든 환경에서 같은 웨이트를 보장하지는 않는다. 본문이 학습한 모델 객체를 보존해 평가하는 방식과도 구분한다.

    ```python
    import numpy as np

    # 대각선은 0이 아니라 NaN이어야 한다 — 아래에서 그 까닭을 밝힌다
    accuracy_matrix = np.full((10, 10), np.nan)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    test_images, test_labels = next(iter(test_loader))

    for i in range(10):
        for j in range(i + 1, 10):
            torch.manual_seed(42)              # 본문의 4절과 같은 씨앗
            model = BinaryCNN()
            train_binary(model, digit_images[i], digit_labels[i],
                         digit_images[j], digit_labels[j], epochs=10)

            # 시험 집합에서 평가. j가 1, i가 0 — train_binary가 붙인 차례와 같다
            mask = (test_labels == i) | (test_labels == j)
            test_data = test_images[mask]
            test_labs = (test_labels[mask] == j).long()
            model.eval()
            with torch.no_grad():
                outputs = model(test_data)
                _, preds = torch.max(outputs, 1)
                acc = (preds == test_labs).float().mean().item() * 100
            accuracy_matrix[i, j] = acc
            accuracy_matrix[j, i] = acc

    plt.imshow(accuracy_matrix, cmap='viridis')
    plt.colorbar(label='test accuracy (%)')
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.title('Pairwise binary test accuracy')
    plt.show()
    ```

    **대각선의 의미**: 같은 숫자끼리의 이진 문제는 정의하지 않았으므로 0%라는 관측값처럼 표시하지 않고 NaN이나 마스크로 비워 둔다. 좁은 정확도 차이를 확대해 표시할 때에는 색상 범위를 명시해 작은 차이가 과장되어 보이지 않도록 한다.

    **나온 값.** 낮은 쪽부터 2 대 7 99.4175%(2,060건 가운데 12건 틀림), 3 대 9 99.4552%(11/2,019), 3 대 5 99.4742%(10/1,902), 0 대 6 99.4840%(10/1,938), 7 대 9 99.5582%(9/2,037)이고, 4 대 5, 4 대 7, 4 대 8, 1 대 3, 1 대 4, 2 대 5, 3 대 6, 0 대 4의 여덟 쌍이 100.0000%이다. 45개 평균은 99.8280%이다.

    **오류의 방향.** `preds`와 `test_labs`를 두 방향으로 따로 세면(`((test_labs == 0) & (preds == 1)).sum()`과 그 반대) 더 볼 것이 나온다. 위의 다섯 쌍 가운데 네 쌍에서 오류가 한쪽으로 쏠린다.

    | 쌍 | 오류 | 방향 |
    |---|---|---|
    | 2 대 7 | 12 | 2 → 7이 6, 7 → 2가 6 |
    | 3 대 9 | 11 | 9 → 3이 10, 3 → 9가 1 |
    | 3 대 5 | 10 | 5 → 3이 9, 3 → 5가 1 |
    | 0 대 6 | 10 | 6 → 0이 10, 0 → 6이 0 |
    | 7 대 9 | 9 | 9 → 7이 7, 7 → 9가 2 |

    오류가 어느 방향으로 발생하는지 함께 보면 대칭적인 쌍별 정확도 표보다 상세한 정보를 얻을 수 있다. 여러 쌍의 오류를 합친 숫자별 건수에는 같은 원본 이미지의 반복 사용이 포함되므로 독립적인 실패 이미지 수로 해석하지 않는다.

    **로스 순위와 정확도 순위**: 기록된 상관은 0.647이며 두 순위는 일치하지 않는다. 학습과 테스트의 분할 차이뿐 아니라 로스가 확신 정도를 반영하고 정확도가 경계 통과만 세는 차이도 있다. 작은 오류 건수의 변동과 시드 변화가 순위에 미치는 영향도 확인한다.

    한편 흔히 짐작하는 3 대 8은 여기서 99.8992%(2/1,984)로 45개 가운데 위에서 스물한째, 곧 한가운데다. "3과 8은 비슷하니 어렵다"는 짐작은 이 실험에서 지지되지 않는다. 다만 이 모든 차례는 시드 하나로 잰 값이므로, 주장으로 삼으려면 시드를 바꾸어 여러 번 재고 변동과 함께 적어야 한다.

---

## 정리하며

이진 쌍별 분석은 해당 부분 문제의 학습 로스와 테스트 오류를 보여 준다. 높은 정확도와 로스 순위를 데이터 고유 난이도로 단정하지 않고, 반복 사용한 테스트 이미지와 쌍별 표본 수를 고려한다. 1로짓 BCE와 2로짓 CE는 유효 로짓의 표현을 맞추면 로스가 같지만 학습 경로까지 같지는 않다. 어느 표현이든 로스의 입력 규약을 지켜야 한다.
