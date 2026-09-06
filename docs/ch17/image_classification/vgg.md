# VGGNet
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- VGGNet(2014)이 들여온 핵심 새로움을 이해한다
- VGGNet이 뒤이은 얼개 꾸밈에 어떤 영향을 주었는지 가려낸다

## 개요

**해**: 2014 | **매개변수**: 1억 3800만 | **핵심 새로움**: 한결같은 3×3 누비기, 아주 깊은 그물

VGGNet(Simonyan & Zisserman, 2014)은 그물의 깊이가 성능에 결정적임을 보여 주었다. 작은 3×3 누비기 거르개만을 깊게 쌓아 씀으로써 VGG는 단순하고 한결같은 얼개로 센 결과를 냈다.

## 핵심 눈썰미: 작은 거르개, 더 깊게

3×3 누비기 둘을 쌓으면 5×5 누비기 하나와 받는 자리가 같지만 매개변수는 더 적고 비선형은 더 많다:

$$\text{Two 3×3: } 2 \times (3^2 C^2) = 18C^2 \text{ params}$$

$$\text{One 5×5: } 5^2 C^2 = 25C^2 \text{ params}$$

3×3 누비기 셋을 쌓으면 7×7 하나와 같으며 아끼는 몫은 더 크다.

## 얼개 변종

| 모델 | 깊이 | 매개변수 | 상위 5 어긋남 |
|-------|-------|-----------|-------------|
| VGG-11 | 11층 | 1억 3300만 | 10.4% |
| VGG-16 | 16층 | 1억 3800만 | 7.4% |
| VGG-19 | 19층 | 1억 4400만 | 7.3% |

```python
import torchvision.models as models

model = models.vgg16(weights='DEFAULT')
# 알아내기의 등뼈로 아주 흔히 쓰인다(더 빠른 R-CNN)
# 그리고 옮겨 배우기로 나누기(FCN)
```

VGG의 한결같은 꾸밈 덕분에 여러 뒤이은 일(FCN, 더 빠른 R-CNN)의 붙박이 등뼈가 되었다. 그러나 매개변수 1억 3800만 개와 큰 기억 공간 씀씀이 때문에 더 효율적인 얼개가 나오게 되었다.

## 참고 문헌

1. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.

## 연습문제

**연습문제 1.**
Explain the key architectural contribution of VGGNet. Why did using $3 \times 3$ convolutions throughout represent a significant advance?

??? success "연습문제 1 풀이"
    VGGNet demonstrated that deep networks with exclusively $3 \times 3$ convolution filters can achieve superior performance compared to networks using larger filters. Two stacked $3 \times 3$ conv layers have the same effective receptive field as a single $5 \times 5$ layer, but with fewer parameters ($2 \times 3^2 C^2 = 18C^2$ vs. $25C^2$) and an additional nonlinearity between them, increasing the model's representational power. This principle of using small filters with greater depth became a cornerstone of subsequent CNN architectures.

---

**연습문제 2.**
VGG-16의 온전히 이은 층의 매개변수 개수를 셈하고, 이 층이 왜 매개변수 수를 좌우하는지 설명하여라.

??? success "연습문제 2 풀이"
    The final conv layer produces $7 \times 7 \times 512 = 25{,}088$ features. The first FC layer maps this to 4096 units: $25{,}088 \times 4{,}096 = 102{,}760{,}448$ parameters. The second FC layer: $4{,}096 \times 4{,}096 = 16{,}777{,}216$. The classification layer: $4{,}096 \times 1{,}000 = 4{,}096{,}000$. Total FC parameters: $\approx 124M$ out of $\approx 138M$ total (about 90%). This is because fully connected layers connect every input to every output, while convolutional layers share weights spatially.

---

**연습문제 3.**
VGG-16은 VGG-19과 어떻게 다른가? 깊이를 더하면 늘 성능이 나아지는가?

??? success "연습문제 3 풀이"
    VGG-16은 누비기 13층 + 온전히 이은 3층이고, VGG-19은 누비기 16층 + 온전히 이은 3층으로 마지막 세 덩이마다 누비기 층을 하나씩 더한 것이다. VGG-19은 ImageNet에서 정확도가 아주 조금 낫지만 셈 값이 더 든다. VGG-19을 넘어 잔차 이음 없이 층을 더 쌓으면 He 외(2015)가 보인 대로 오히려 나빠지며(익힘 어긋남이 커지며), 이것이 ResNet을 낳았다. 이 줄어드는 이득은 깊이만으로는 모자라며 건너뛰는 이음 같은 얼개의 새로움이 필요함을 일러 준다.

---

**연습문제 4.**
VGGNet은 흔히 결 옮기기나 느낌 손실 같은 일에서 특징 뽑개로 쓰인다. 가운데 층의 특징이 이런 쓰임새에 왜 쓸모 있는지 설명하여라.

??? success "연습문제 4 풀이"
    VGGNet의 층마다 추상 수준이 다른 특징을 담아낸다. 앞쪽 층(conv1, conv2)은 테두리나 결 같은 낮은 수준의 특징을, 깊은 층(conv4, conv5)은 물체의 부분이나 자리 배치 같은 높은 수준의 뜻을 담아낸다. **결 옮기기**에서는 앞쪽 층 깨어남의 그람 행렬이 결과 결풍을 담아내고 깊은 층 깨어남이 내용을 담아낸다. **느낌 손실**에서는 만든 그림과 목표 그림을 화소 공간이 아니라 특징 공간에서 견주는데, 손실이 화소 하나하나의 맞춤이 아니라 짜임과 뜻의 닮음에 민감하므로 더 또렷하고 느낌상 실제 같은 결과가 나온다.
