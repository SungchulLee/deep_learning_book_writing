# 9장: 뉴럴 네트워크의 구성 요소

이 장에서는 다층 퍼셉트론을 구성하고 학습시키는 데 필요한 요소를 살펴본다. 엑티베이션 함수로 비선형성을 만들고, 순전파와 역전파로 예측과 경사를 계산한다. 이어서 초기화, 정규화 레이어, 과적합을 줄이는 기법이 학습에 어떤 영향을 주는지 설명한다.

## 1. 엑티베이션 함수

엑티베이션 함수는 선형 변환만으로 표현할 수 없는 관계를 학습하게 한다. 함수의 모양뿐 아니라 도함수, 출력 범위, 로스과의 조합을 함께 살펴본다.

- [ELU](activations/elu.md): 음수 입력에서도 값을 내는 활성화
- [회귀 예제](activations/07_regression_with_activations.md): 출력 범위와 활성화 선택
- [다중 클래스 분류](activations/06_multiclass_classification.md): 로짓과 확률의 구분
- [활성화 비교](activations/09_activation_comparison.md): 같은 실험 조건에서의 비교

## 2. 순방향 뉴럴 네트워크

먼저 각 레이어의 차원과 파라메터 개수를 정리하고, 데이터와 경사가 레이어를 통과하는 과정을 이해한다. 구현 예제는 NumPy의 수동 계산에서 PyTorch의 학습 루프와 응용으로 이어진다.

이 절이 **이 장에서 가장 길다**(29쪽). 이론 여섯 쪽 뒤로는 **같은 것을 여러 번 짓는** 자리이므로, 아는 도구가 나오면 건너뛰어도 된다.

**구조와 그 성질**

- [MLP 구조](feedforward/mlp_architecture.md) — 레이어마다의 차원과 파라메터 수
- [순전파](feedforward/forward_pass.md) · [역전파](feedforward/backpropagation.md)
- [경사의 흐름](feedforward/gradient_flow.md) — 레이어마다의 인수가 곱으로 쌓여 소실·폭주가 된다
- [보편 근사 정리](feedforward/universal_approximation.md) — 너비만으로 충분하다는 말이 무엇을 **보장하지 않는가**
- [깊이와 너비](feedforward/depth_vs_width.md) — 용량을 정하는 두 축

**바닥부터 올라가기** (아래 여섯은 차례로 읽도록 되어 있다)

1. [순수 NumPy로 만드는 선형 회귀](feedforward/01_linear_regression_numpy.md) — 순전파·로스·경사·갱신을 전부 손으로
2. [PyTorch 텐서로 만드는 선형 회귀](feedforward/02_linear_regression_pytorch.md) — 텐서로 옮기되 autograd 는 아직 끈 채로
3. [역전파를 직접 구현한 간단한 뉴럴 네트워크](feedforward/03_simple_nn_manual.md) — ReLU 은닉 레이어를 더해 동심원을 가른다
4. [Autograd 소개](feedforward/04_autograd_introduction.md) — 수동 역전파를 `backward()` 로 바꾼다
5. [간단한 퍼셉트론](feedforward/05_simple_perceptron.md) — `nn.Linear(1, 1)`. 고전 퍼셉트론의 계단 규칙은 **아니다**
6. [2레이어 뉴럴 네트워크](feedforward/06_two_layer_network.md) — XOR 네 점

**nn 모듈로 짓기**

- [nn.Module과 옵티마이저](feedforward/07_nn_module_and_optimizers.md) · [Sequential 사용하기](feedforward/10_using_sequential.md) · [사용자 정의 Module](feedforward/11_custom_module.md)

**부품 고르기**

- [엑티베이션 함수](feedforward/12_activation_functions.md) — 여섯을 같은 구간에서 그려 견준다(학습 성능 비교는 아니다)
- [로스 함수](feedforward/13_loss_functions.md) — 입력이 로짓인지 확률인지, 목표의 모양과 자료형이 무엇인지

**학습을 다듬기**

- [웨이트 초기화](feedforward/19_weight_initialization.md) — Xavier·He·작은 정규분포·0 을 견준다
- [배치 정규화](feedforward/16_batch_normalization.md) · [배치 정규화 자세히 보기](feedforward/17_batch_normalization_detailed.md) — 뒤엣것이 분할을 지켜 가며 견준다
- [드롭아웃 정칙화](feedforward/14_dropout_regularization.md) · [정칙화 기법 자세히 보기](feedforward/15_regularization_techniques_detailed.md)
- [러닝레이트 일정 조절](feedforward/18_learning_rate_scheduling.md) — 곡선을 그려 보이는 시연이다

**과제에 올리기**

- [MNIST 분류 자세히 보기](feedforward/09_mnist_classification_detailed.md) · [CIFAR-10 분류기](feedforward/20_cifar10_classifier.md)
- [회귀 과제](feedforward/21_regression_task.md) — California Housing · [다중 출력 뉴럴 네트워크](feedforward/22_multi_output_network.md) — 헤드 둘
- [깊은 뉴럴 네트워크](feedforward/23_deep_network.md) — 잔차 블록을 `ModuleList` 로 쌓는다

**곁가지**

- [캡슐 뉴럴 네트워크 (CapsNet)](feedforward/49_capsule_net.md) — 순방향 계열이 아닌 다른 갈래다. 구조 정의와 파라메터 수까지만 보인다

## 3. 웨이트 초기화

초기화는 레이어를 지나는 활성화와 경사의 크기에 영향을 준다. 분산 보존에 관한 근사 가정을 이해하고 활성화에 맞는 방법을 선택한다.

- [웨이트 초기화의 원리](weight_initialization/weight_initialization.md)
- [He 초기화](weight_initialization/he_init.md)

## 4. 정규화 레이어

정규화 레이어(normalization)은 정한 축의 통계량으로 활성화의 크기를 조정한다. 배치 통계량을 쓰는지, 표본마다 계산하는지, 학습과 평가에서 동작이 달라지는지 구분한다.

- [배치 정규화](normalization/batch_normalization.md)
- [레이어 정규화](normalization/layer_normalization.md)
- [인스턴스 정규화](normalization/instance_normalization.md)
- [RMSNorm](normalization/rms_norm.md)
- [정규화 레이어 비교](normalization/normalization_comparison.md)

## 5. 과적합을 줄이는 기법

일반화 성능을 높이려는 **정칙화**(regularization)는 앞 절의 **정규화 레이어**(normalization)와 **다른 것이다** — 영어가 다르고 하는 일도 다른데 우리말이 비슷해 가장 자주 헷갈리는 짝이다. 검증 데이터로 강도를 고르고, 각 기법의 가정이 과제에 맞는지 확인한다.

**무엇이며 언제 쓰나**

- [정칙화 개요](regularization/regularization_overview.md) — 과소적합과 과적합 사이에서 무엇을 제약하는가. 먼저 읽는 지도다

**패널티를 더하기**

- [L1 정칙화 (라쏘)](regularization/l1_regularization.md) — 희소한 해 · [L2 정칙화 (릿지)](regularization/l2_regularization.md) — 노름을 줄인다
- [엘라스틱 넷 정칙화](regularization/elastic_net.md) — 둘을 섞는다
- [L1/L2 정칙화](regularization/l1_l2_regularization.md) — 선형 회귀와 회귀 MLP 에서 실제로 재어 본다

**뉴런과 연결을 끄기**

- [드롭아웃](regularization/dropout.md) — 활성화를 끈다 · [드롭커넥트](regularization/dropconnect.md) — 웨이트를 끈다
- [드롭아웃 예제](regularization/dropout_example.md) — 같은 MLP 로 견준다. 판정은 **같은 평가 모드**에서 해야 한다

**데이터를 늘리고 섞기**

- [데이터 어그멘테이션](regularization/data_augmentation.md) — 새 표본을 모은 것과 **같지 않다**
- [믹스업](regularization/mixup.md) · [컷믹스](regularization/cutmix.md) · [컷아웃](regularization/cutout.md)
- [레이블 스무딩](regularization/label_smoothing.md) — 정확도와 보정이 늘 좋아지지는 않는다

**학습을 멈추거나 흔들기**

- [조기 종료](regularization/early_stopping.md) · [조기 종료 예제](regularization/early_stopping_example.md) — 검증으로 멈출 때를 고르고 **별도 테스트셋**에서 잰다
- [잡음 주입](regularization/noise_injection.md) — 입력·웨이트·활성화·경사 어디에나 줄 수 있다

## 정리하며

뉴럴 네트워크 설계에서는 구조, 활성화, 로스, 초기화, 학습과 평가 모드를 함께 고려해야 한다. 각 요소의 효과는 데이터와 학습 조건에 따라 달라지므로 검증 성능과 계산 비용을 기준으로 판단한다.
