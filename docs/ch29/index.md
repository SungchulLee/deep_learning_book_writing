# 30장: 생성적 적대 신경망(GAN)

GAN은 생성기와 판별기를 번갈아 학습해 데이터 분포를 근사하는 생성 모델이다. 생성기는 잠재 변수로 샘플을 만들고 판별기는 실제·생성 샘플을 구분한다. 두 모델의 경쟁은 단순한 손실 최소화와 다른 학습 동역학을 만들며, 그럴듯한 샘플과 충분한 다양성을 함께 얻는 것이 과제다.

## 1. 적대적 학습과 균형

- [적대적 학습](gan_foundations/adversarial.md): 생성기·판별기의 역할, 최적 판별기와 분포 일치의 의미를 설명한다.
- [내시 균형](gan_foundations/nash.md): 균형 조건과 학습 알고리즘의 수렴을 구분하고 순환·안정성 문제를 살펴본다.
- [기본 GAN 손실 함수](gan_losses/original.md): 원래 최소최대 목적과 비포화 생성기 손실, 확률·로짓 인터페이스를 비교한다.

## 2. 구조와 학습 예제

- [기본 GAN 모듈](gan_architectures/45_gan.md): 완전연결 생성기와 판별기의 차원 및 파라미터를 확인한다.
- [DCGAN 모듈](gan_architectures/46_dcgan.md): 업샘플링·합성곱 구조와 입력 크기 조건을 살펴본다.
- [DCGAN 구조](gan_architectures/dcgan.md): 합성곱 생성기의 해상도 변화, 정규화와 샘플링을 설명한다.
- [GAN 유틸리티](gan_architectures/gan_utils.md): 손실, 시각화, 체크포인트와 평가 인터페이스를 정리한다.
- [2차원 GAN 예제](gan_architectures/example_2d_gan.md): 인공 데이터에서 학습 단계와 분포 시각화를 읽는다.
- [MNIST DCGAN 예제](gan_architectures/example_mnist_dcgan.md): 데이터·모델·손실 사이의 입력 크기와 출력 계약을 확인한다.

코드에 표시된 이미지 크기, 판별기 출력과 손실의 전제가 서로 맞지 않는 사례는 각 페이지 본문에서 설명한다. 예제의 이름만으로 실행 가능성이나 생성 품질을 가정하지 않는다.

## 3. 생성 모델 평가

- [평가의 기초](gan_evaluation/01_evaluation_basics.md): 분포 적합도와 샘플 품질이 서로 다른 질문임을 살펴본다.
- [우도 기반 평가](gan_evaluation/02_likelihood_metrics.md): NLL, 차원당 비트 수, 퍼플렉서티의 단위와 집계 방식을 구분한다.
- [샘플 시각화](gan_evaluation/03_sample_visualization.md): 격자, 잠재 보간, 재구성 오차를 목적에 맞게 사용한다.
- [인셉션 점수 계산](gan_evaluation/01_inception_score.md)과 [인셉션 점수의 원리](gan_evaluation/inception_score.md): 예측 확신과 클래스 분산, 분할과 한계를 설명한다.
- [FID 계산](gan_evaluation/02_frechet_inception_distance.md)과 [FID의 원리](gan_evaluation/fid.md): 특징 모멘트의 거리와 유한 표본·전처리의 영향을 다룬다.
- [생성 정밀도와 재현율](gan_evaluation/precision_recall.md): 특징 공간의 구 합집합을 이용해 품질과 포괄성을 나누어 살펴본다.
- [종합 평가 예제](gan_evaluation/complete_evaluation_example.md): mock 통계와 실제 이미지 평가를 구분하고 평가 절차를 정리한다.

평가 코드에는 실제 이미지를 처리하는 구현, 임의 특징을 사용하는 시연, 단순 통계 mock 예제가 함께 있다. 각각의 결과가 무엇을 측정하는지 확인하고, 같은 조건에서 얻은 값만 비교한다. 낮은 FID나 높은 IS만으로 다양성과 비암기를 보장할 수 없다.

## 4. 적대적 공격

[적대적 공격의 기초](adversarial_robustness/attacks.md)에서는 분류기 입력을 바꾸는 FGSM·PGD와 표적 공격을 다룬다. GAN 학습과는 최적화 변수와 목적이 다르며, 입력 전처리와 섭동 제약·성공률의 분모를 일치시켜야 한다.

## 정리하며

이 장의 핵심은 생성기·판별기의 학습 목표, 구조와 데이터의 인터페이스, 생성 분포의 여러 평가 측면을 함께 이해하는 것이다. 균형의 존재와 실제 학습 수렴, 계산 예제와 실측 성능을 구분해 읽는다.
