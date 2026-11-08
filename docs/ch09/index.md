# 10장: 학습 파이프라인

이 장에서는 데이터를 읽고 배치로 묶는 단계부터 학습, 검증, 모델 선택과 저장까지 연결한다. 각 구성 요소를 따로 이해하는 것과 함께, 전처리·분할·모드·평가 규약이 전체 과정에서 일관적인지 확인한다.

## 1. 정확성과 효율성

[구현의 정확성](foundations/correctness.md)과 [계산 효율](foundations/efficiency.md)을 구분한다. 올바른 구현이 일반화 성능을 보장하지 않고, 점근적 복잡도가 실제 시간을 완전히 설명하지도 않는다.

## 2. 데이터셋과 데이터로더

텐서, 이미지 폴더, 좌표 레이블과 순차 데이터에 맞는 인터페이스를 살펴본다. 표본의 대응을 지키고 학습·검증·테스트를 먼저 분리한 뒤 필요한 통계량과 변환을 학습한다.

- [TensorDataset](datasets/dataset_from_tensor.md)과 [IterableDataset 안내](datasets/11_iterable_style_minimal.md)
- [MNIST](datasets/mnist.md), [FashionMNIST](datasets/fashion_mnist.md), [CIFAR-10](datasets/cifar10.md)
- [얼굴 랜드마크](datasets/face_landmarks.md)와 [Hymenoptera 전이 학습](datasets/hymenoptera.md)
- [텍스트 어그멘테이션](datasets/nlp_augmentation.md)과 [약지도 학습](datasets/weak_supervision_snorkel.md)
- [균형 샘플링](dataloaders/balanced_dataloader.md), [변환](dataloaders/dataloader_transforms.md), [배치 시각화](dataloaders/draw_batch_images.md)

## 3. 학습 루프와 기록

[MNIST 전체 학습](training/01_complete_mnist_training.md)과 [TensorBoard 기록](training/tensorboard_mnist_commented.md)을 통해 순전파, 로스, 역전파와 갱신을 연결한다. [실전 학습 요령](training/training_recipes.md)에서는 초기 점검, 혼합 정밀도와 메모리·처리량을 다룬다.

## 4. 로스와 옵티마이저

출력 형식과 목표에 맞는 [로스 선택](loss/loss_selection.md)이 먼저이다. [후버 로스](loss/huber_loss.md), [Focal Loss](loss/focal_loss.md)와 [사용자 정의 로스](loss/01_custom_loss_functions.md)을 살펴보고 각 가정과 집계 방식을 비교한다.

옵티마이저는 로스에서 계산한 경사를 사용해 갱신한다. **어느 방법의 우열도 러닝레이트와 평가 조건에서 떼어 판단할 수 없다.**

**모멘텀 계열**

- [모멘텀을 쓰는 SGD](optimizers/momentum.md) — 지난 경사의 지수 감쇠 평균을 쌓는다
- [네스테로프 가속 경사](optimizers/nesterov.md) — 먼저 가 보고 그 자리에서 경사를 잰다

**좌표마다 보폭을 바꾸기**

- [Adagrad](optimizers/adagrad.md) — 경사 제곱을 **누적**한다. 그래서 보폭이 한없이 줄어든다
- [Adadelta](optimizers/adadelta.md) · [RMSprop](optimizers/rmsprop.md) — 누적 대신 **지수 감쇠 평균**으로 그 문제를 푼다

**둘을 합친 Adam 계열**

- [Adam](optimizers/adam.md) — 경사와 경사 제곱의 지수 평균, 그리고 초기화 바이어스 보정
- [AdamW](optimizers/adamw.md) — 웨이트 감쇠를 적응형 변환에서 **떼어낸다**
- [LAMB](optimizers/lamb.md) — 텐서별 신뢰 비율. 원 논문의 큰 배치 결과는 **특정 하드웨어·설정**의 것이다

**NumPy 로 직접 짜 보기**

- [Adagrad 실습](optimizers/adagrad_optimizer.md) · [RMSprop 실습](optimizers/rmsprop_optimizer.md) · [Adam 실습](optimizers/adam_optimizer.md) — 자동 미분 없이 상태 갱신만 본다

**견주기**

- [기준 함수에서 옵티마이저 비교](optimizers/optimizer_comparison.md) — 2변수 목적함수 넷
- [뉴럴 네트워크 학습에서 옵티마이저 비교](optimizers/01_optimizer_comparison.md) — 합성 회귀의 **학습 로스**를 본다

**옵티마이저 둘레**

- [러닝레이트 스케줄러](optimizers/02_learning_rate_schedulers.md) — 곡선 넷을 만들고 학습 루프에 거는 법
- [그래디언트 관리](optimizers/03_gradient_management.md) — 클리핑과 누적의 **적용 순서**가 결과를 바꾼다

## 5. 러닝레이트와 과적합

[워밍업](schedulers/lr_warmup.md)과 [순환 러닝레이트](schedulers/cyclical_lr.md)은 갱신의 시간별 스케일을 바꾼다. 호출 단위와 스케줄러 상태를 확인한다. [편향–분산](overfitting/bias_variance_tradeoff.md), [학습 곡선](overfitting/learning_curves_demo.md), [과적합·과소적합](overfitting/overfitting_underfitting_demo.md)을 함께 보며 학습 부족과 일반화 문제를 구분한다.

## 6. 평가와 하이퍼파라미터 선택

[분류 지표](evaluation/02_classification_metrics.md), [회귀 지표](evaluation/03_regression_metrics.md)와 [혼동 행렬](evaluation/05_confusion_matrix.md)은 서로 다른 오류를 보여 준다. [교차 검증](evaluation/04_cross_validation.md)과 [통계적 추론](evaluation/statistical_inference.md)에서는 선택 효과와 평가의 불확실성을 고려한다.

[격자 탐색](hyperparameters/01_grid_search.md), [무작위 탐색](hyperparameters/02_random_search.md), [베이즈 최적화](hyperparameters/03_bayesian_optimization.md), [연속 반감법](hyperparameters/06_successive_halving.md)의 탐색 비용과 조건을 비교한다. 최종 테스트셋은 탐색에 반복 사용하지 않는다.

## 7. 저장과 배포

[저장과 복원](save_load/01_basic_save_load.md)은 모델 상태와 구조·전처리를 함께 맞추는 작업이다. [ONNX](save_load/04_onnx_export.md), [양자화](save_load/2_quantization.md), [배포 파이프라인](save_load/4_deployment_pipeline.md)에서는 내보낸 결과의 입출력 규약과 수치적 일치를 확인한다.

## 정리하며

학습 파이프라인은 데이터를 나누는 기준부터 평가와 복원까지 이어진다. 각 예제의 실행 범위와 가정을 읽고, 한 단계의 올바른 동작이 전체 과정의 신뢰성을 자동으로 보장하지 않는다는 점을 기억한다.
