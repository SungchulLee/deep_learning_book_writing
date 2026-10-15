# 11장: 합성곱 신경망

합성곱 신경망(CNN)은 국소적인 연결과 가중치 공유를 이용해 이미지·시퀀스의 특징을 학습한다. 이 장에서는 합성곱의 계산과 수용 영역을 설명하고, 잔차 연결과 채널·공간 어텐션으로 확장한다.

[3.4절](../ch03/mnist/04_cnn.md)의 MNIST 예제에서 사용한 합성곱을 수식과 구현 관점에서 다시 살펴본다. 어텐션도 구조와 위치 표현에 따른 가정을 가지므로 CNN에서 어텐션으로의 변화를 단순히 모든 사전 가정을 제거하는 과정으로 보지는 않는다.

## 11.1 합성곱 신경망

합성곱 연산에서 시작하여, 효율과 넓은 수용 영역을 위한 변형들을 거쳐, 실제 데이터셋의 분류 예제에 적용한다.

**개념**

- [합성곱 연산](cnn/convolution.md) — 특징 추출을 위한 이산 합성곱과 상호상관의 수학적 기초
- [특징 맵](cnn/feature_maps.md) — 특징 맵 텐서의 기하, 계산, 해석
- [수용 영역](cnn/receptive_field.md) — 뉴런 하나가 얼마만큼의 공간적 맥락에 닿는지에 대한 분석
- [밑바닥부터 만드는 합성곱](cnn/conv_from_scratch.md) — 합성곱을 직접 구현해 라이브러리와 맞추어 보기

**변형**

- [1차원 합성곱](cnn/conv1d.md) — 시계열, 음향, 텍스트 같은 순차 데이터에 합성곱 적용하기
- [팽창 합성곱](cnn/dilated_convolutions.md) — 매개변수를 늘리지 않고 수용 영역을 넓히는 아트루스 합성곱
- [깊이별 분리 합성곱(Depthwise Separable Convolution)](cnn/depthwise_separable.md) — MobileNet, EfficientNet, ShuffleNet이 쓰는 효율적인 합성곱 분해
- [전치 합성곱](cnn/transposed_conv.md) — 인코더-디코더 구조, GAN, 초해상도를 위한 학습 가능한 업샘플링

**데이터셋과 분류기**

- [MNIST 데이터셋](cnn/01_mnist_dataset.md) — 손글씨 숫자를 읽어 들이고 살펴보기
- [Fashion-MNIST 데이터셋](cnn/02_fashion_mnist_dataset.md) — 같은 크기, 더 어려운 과제
- [CIFAR-10 데이터셋](cnn/03_cifar10_dataset.md) — 색이 있는 자연 영상으로 옮겨 가기
- [Fashion-MNIST 분류기](cnn/05_fashion_mnist_classifier.md) — 앞의 개념을 모아 첫 분류기를 세운다
- [CIFAR-10 기본](cnn/06_cifar10_basic.md) — 자연 영상에서의 기초 성능
- [CIFAR-10 심화](cnn/07_cifar10_advanced.md) — 여러 합성곱층과 학습률 일정으로 학습하기
- [이진 분류](cnn/08_binary_classification.md) — 두 갈래만 가를 때 달라지는 것
- [CNN 유틸리티](cnn/cnn_utils.md) — 이 절의 예제들이 함께 쓰는 도구

---

## 11.2 잔차 연결

잔차 연결은 입력을 뒤의 층에 더하는 경로를 만들어 깊은 모델의 최적화를 돕는다. 기울기 소실이나 폭발을 자동으로 없애는 것은 아니다.

- [기본 잔차 블록](residual/01_basic_residual_block.md) — 지름길 하나가 무엇을 바꾸는지
- [항등 사상](residual/identity_mapping.md) — 항등 지름길과 사전 활성화 블록의 기울기 전달
- [ResNet 구현](residual/02_resnet_implementation.md) — 기본 블록과 병목 블록으로 ResNet 계열 세우기
- [학습 비교](residual/03_training_comparison.md) — 지름길이 있을 때와 없을 때를 나란히 학습시켜 비교
- [실전 예제](residual/06_practical_example.md) — 분류 예제로 잔차 구조의 구성 살펴보기

---

## 11.3 CNN에서 ViT로

이미지를 패치 시퀀스로 표현하는 관점과 CNN에 어텐션을 결합하는 방법을 살펴본다.

- [합성곱 신경망에서 트랜스포머로](vit/cnn_to_vit_bridge.md) — 지역성 기반에서 어텐션 기반 처리로 나아간 길
- [Squeeze-and-Excitation](vit/squeeze_excitation.md) — 전역 통계를 이용한 채널별 가중치 조정
- [비국소 신경망](vit/non_local.md) — 합성곱을 쌓지 않고 먼 거리의 의존을 곧바로 계산하기

---

## 정리하며

합성곱은 국소 연결과 가중치 공유로 격자 데이터의 특징을 계산하고, 잔차 연결은 깊은 모델의 최적화를 돕는다. 채널 어텐션과 비국소 연산은 입력에 따라 특징의 가중치나 위치 간 관계를 조절한다. 각 방법의 표현력과 계산 비용을 구분해 이해한 뒤 [12장](../ch11/index.md)의 시퀀스 모델과 [13장](../ch12/index.md)의 트랜스포머로 이어 간다.
