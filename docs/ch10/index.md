# 11장: 합성곱 신경망

이 장은 공간적으로 짜인 데이터를 다루는 바탕 구조를 살펴본다. 주로 이미지이지만, 시계열처럼 격자 모양으로 늘어놓을 수 있는 것은 모두 여기에 든다. 고전적인 합성곱 신경망에서 잔차 구조를 거쳐 비전 트랜스포머로 나아가는데, 그 흐름을 꿰는 물음은 하나다. **공간에 대한 앞선 믿음을 사람이 손으로 넣을 것인가, 자료에서 배우게 할 것인가.**

합성곱은 [3.4절](../ch03/mnist/04_cnn.md)에서 MNIST를 풀며 한 번 만났다. 거기서는 쓰는 법을 보았고, 이 장에서는 **왜 그렇게 생겼는지**를 본다.

---

## 11.1 합성곱 신경망

합성곱 연산에서 시작하여, 효율과 넓은 수용 영역을 위한 변형들을 거쳐, 실제 데이터셋에 태워 보는 데까지 간다.

**개념**

- [합성곱 연산](cnn/convolution.md) — 특징 추출을 위한 이산 합성곱과 상호상관의 수학적 바탕
- [특징 맵](cnn/feature_maps.md) — 특징 맵 텐서의 기하, 계산, 해석
- [수용 영역](cnn/receptive_field.md) — 뉴런 하나가 얼마만큼의 공간적 맥락에 닿는지에 대한 분석
- [밑바닥부터 만드는 합성곱](cnn/conv_from_scratch.md) — 합성곱을 손으로 짜서 라이브러리와 맞추어 보기

**변형**

- [1차원 합성곱](cnn/conv1d.md) — 시계열, 음향, 텍스트 같은 순차 데이터에 합성곱 적용하기
- [팽창 합성곱](cnn/dilated_convolutions.md) — 매개변수를 늘리지 않고 수용 영역을 넓히는 아트루스 합성곱
- [깊이별 분리 합성곱](cnn/depthwise_separable.md) — MobileNet, EfficientNet, ShuffleNet이 쓰는 효율적인 합성곱 분해
- [전치 합성곱](cnn/transposed_conv.md) — 인코더-디코더 구조, GAN, 초해상도를 위한 학습 가능한 상향 표본화

**데이터셋과 분류기**

- [MNIST 데이터셋](cnn/01_mnist_dataset.md) — 손글씨 숫자를 읽어 들이고 살펴보기
- [Fashion-MNIST 데이터셋](cnn/02_fashion_mnist_dataset.md) — 같은 크기, 더 어려운 일감
- [CIFAR-10 데이터셋](cnn/03_cifar10_dataset.md) — 색이 있는 자연 영상으로 옮겨 가기
- [Fashion-MNIST 분류기](cnn/05_fashion_mnist_classifier.md) — 앞의 개념을 모아 첫 분류기를 세운다
- [CIFAR-10 기본](cnn/06_cifar10_basic.md) — 자연 영상에서의 바탕 성능
- [CIFAR-10 심화](cnn/07_cifar10_advanced.md) — 합성곱을 넷으로 늘리고 학습률을 깎아 가며 재기
- [이진 분류](cnn/08_binary_classification.md) — 두 갈래만 가를 때 달라지는 것
- [CNN 유틸리티](cnn/cnn_utils.md) — 이 절의 예제들이 함께 쓰는 도구

---

## 11.2 잔차 연결

기울기가 곧바로 흐르는 길을 내어 아주 깊은 신경망의 학습을 가능하게 하는 건너뛰기 연결과 잔차 학습.

- [기본 잔차 블록](residual/01_basic_residual_block.md) — 지름길 하나가 무엇을 바꾸는지
- [항등 사상](residual/identity_mapping.md) — 순수한 항등 지름길과 깨끗한 기울기 흐름을 위한 사전 활성화 블록 설계
- [ResNet 구현](residual/02_resnet_implementation.md) — 기본 블록과 병목 블록으로 ResNet 계열 세우기
- [학습 비교](residual/03_training_comparison.md) — 지름길이 있을 때와 없을 때를 나란히 학습시켜 견주기
- [실전 예제](residual/06_practical_example.md) — 실제 과제에 잔차 구조를 태워 보기

---

## 11.3 CNN에서 ViT로

이미지를 조각의 순차열로 다루는 쪽으로 옮겨 가는 길, 그리고 그 중간에 놓인 장치들.

- [합성곱 신경망에서 트랜스포머로](vit/cnn_to_vit_bridge.md) — 지역성 기반에서 어텐션 기반 처리로 나아간 길
- [압축-여기](vit/squeeze_excitation.md) — 계산 부담을 거의 늘리지 않는 적응형 채널 재보정
- [비국소 신경망](vit/non_local.md) — 합성곱을 쌓지 않고 먼 거리의 의존을 곧바로 계산하기

---

## 정리하며

세 절은 같은 것을 점점 덜 가정하는 차례다. 합성곱은 **가까운 것끼리 관계있다**는 믿음을 구조에 박아 넣고, 잔차 연결은 그 구조를 깊게 쌓을 수 있게 하며, 비국소·어텐션 장치는 **무엇이 무엇과 관계있는지를 자료에서 배우게** 한다.

그 마지막 걸음을 끝까지 밀고 간 것이 [12장](../ch11/index.md)과 [13장](../ch12/index.md)의 순차열 모델과 트랜스포머다.
