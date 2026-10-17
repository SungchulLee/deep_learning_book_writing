# 9장: 뉴럴 넷트웍의 구성 요소

이 장에서는 다층 퍼셉트론을 구성하고 학습시키는 데 필요한 요소를 살펴본다. 엑티베이션 함수로 비선형성을 만들고, 순전파와 역전파로 예측과 경사를 계산한다. 이어서 초기화, 정규화 레이어, 과적합을 줄이는 기법이 학습에 어떤 영향을 주는지 설명한다.

## 1. 엑티베이션 함수

엑티베이션 함수는 선형 변환만으로 표현할 수 없는 관계를 학습하게 한다. 함수의 모양뿐 아니라 도함수, 출력 범위, 로스과의 조합을 함께 살펴본다.

- [ELU](activations/elu.md): 음수 입력에서도 값을 내는 활성화
- [회귀 예제](activations/07_regression_with_activations.md): 출력 범위와 활성화 선택
- [다중 클래스 분류](activations/06_multiclass_classification.md): 로짓과 확률의 구분
- [활성화 비교](activations/09_activation_comparison.md): 같은 실험 조건에서의 비교

## 2. 순방향 뉴럴 넷트웍

먼저 각 레이어의 차원과 파라메터 개수를 정리하고, 데이터와 경사가 레이어를 통과하는 과정을 이해한다. 구현 예제는 NumPy의 수동 계산에서 PyTorch의 학습 루프와 응용으로 이어진다.

- [MLP 구조](feedforward/mlp_architecture.md)
- [순전파](feedforward/forward_pass.md)와 [역전파](feedforward/backpropagation.md)
- [보편 근사 정리](feedforward/universal_approximation.md)
- [깊이와 너비](feedforward/depth_vs_width.md)
- [경사의 흐름](feedforward/gradient_flow.md)

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

일반화 성능을 높이려는 정규화(regularization)는 정규화 레이어와 구별한다. 검증 데이터로 강도를 선택하고, 각 기법의 가정이 과제에 맞는지 확인한다.

- [정규화 개요](regularization/regularization_overview.md)
- [드롭아웃](regularization/dropout.md)과 [드롭커넥트](regularization/dropconnect.md)
- [조기 종료](regularization/early_stopping.md)
- [L1](regularization/l1_regularization.md), [L2](regularization/l2_regularization.md), [엘라스틱 넷](regularization/elastic_net.md)
- [데이터 어그멘테이션](regularization/data_augmentation.md)과 [레이블 스무딩](regularization/label_smoothing.md)
- [믹스업](regularization/mixup.md), [컷믹스](regularization/cutmix.md), [컷아웃](regularization/cutout.md)
- [잡음 주입](regularization/noise_injection.md)

## 정리하며

뉴럴 넷트웍 설계에서는 구조, 활성화, 로스, 초기화, 학습과 평가 모드를 함께 고려해야 한다. 각 요소의 효과는 데이터와 학습 조건에 따라 달라지므로 검증 성능과 계산 비용을 기준으로 판단한다.
