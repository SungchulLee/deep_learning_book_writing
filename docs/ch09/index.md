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

옵티마이저는 로스에서 계산한 경사를 사용해 갱신한다. [모멘텀](optimizers/momentum.md), [네스테로프](optimizers/nesterov.md), [Adagrad](optimizers/adagrad.md), [Adadelta](optimizers/adadelta.md), [RMSprop](optimizers/rmsprop.md), [Adam](optimizers/adam.md), [AdamW](optimizers/adamw.md), [LAMB](optimizers/lamb.md)의 상태와 갱신 차이를 이해한다. 어느 방법의 우열도 러닝레이트와 평가 조건에서 분리해 판단할 수는 없다.

## 5. 러닝레이트와 과적합

[워밍업](schedulers/lr_warmup.md)과 [순환 러닝레이트](schedulers/cyclical_lr.md)은 갱신의 시간별 스케일을 바꾼다. 호출 단위와 스케줄러 상태를 확인한다. [편향–분산](overfitting/bias_variance_tradeoff.md), [학습 곡선](overfitting/learning_curves_demo.md), [과적합·과소적합](overfitting/overfitting_underfitting_demo.md)을 함께 보며 학습 부족과 일반화 문제를 구분한다.

## 6. 평가와 하이퍼파라미터 선택

[분류 지표](evaluation/02_classification_metrics.md), [회귀 지표](evaluation/03_regression_metrics.md)와 [혼동 행렬](evaluation/05_confusion_matrix.md)은 서로 다른 오류를 보여 준다. [교차 검증](evaluation/04_cross_validation.md)과 [통계적 추론](evaluation/statistical_inference.md)에서는 선택 효과와 평가의 불확실성을 고려한다.

[격자 탐색](hyperparameters/01_grid_search.md), [무작위 탐색](hyperparameters/02_random_search.md), [베이즈 최적화](hyperparameters/03_bayesian_optimization.md), [연속 반감법](hyperparameters/06_successive_halving.md)의 탐색 비용과 조건을 비교한다. 최종 테스트셋은 탐색에 반복 사용하지 않는다.

## 7. 저장과 배포

[저장과 복원](save_load/01_basic_save_load.md)은 모델 상태와 구조·전처리를 함께 맞추는 작업이다. [ONNX](save_load/04_onnx_export.md), [양자화](save_load/2_quantization.md), [배포 파이프라인](save_load/4_deployment_pipeline.md)에서는 내보낸 결과의 입출력 규약과 수치적 일치를 확인한다.

## 정리하며

학습 파이프라인은 데이터를 나누는 기준부터 평가와 복원까지 이어진다. 각 예제의 실행 범위와 가정을 읽고, 한 단계의 올바른 동작이 전체 과정의 신뢰성을 자동으로 보장하지 않는다는 점을 기억한다.
