# 31장: 확산 모델

확산 모델은 데이터에 노이즈를 더하는 과정을 정의하고 그에 대응하는 역과정으로 새로운 표본을 생성한다. 이 장에서는 로그 밀도의 입력 기울기인 스코어에서 출발해 노이즈 예측 학습, DDPM·DDIM 샘플링과 조건부 생성을 연결한다. 이론의 분포·시간 관례와 각 코드가 실제로 구현하는 범위를 함께 읽는 것이 목표다.

## 1. 권장 학습 순서

먼저 [스코어 함수](score_based/01_score_functions_basics.md)와 [스코어 매칭](score_based/02_score_matching_theory.md)을 익힌다. 이어 [DSM 이론](score_based/denoising_score_matching.md)과 [SSM 이론](score_based/sliced_score_matching.md)에서 알려진 노이즈 목표와 무작위 방향 미분이 어떻게 학습 목적을 만드는지 확인한다.

그다음 [역과정 유도](foundations/reverse_process.md), [노이즈 스케줄](ddpm/noise_schedule.md)과 [DDPM 샘플링](ddpm/sampling.md)을 읽는다. 마지막으로 [DDIM](ddim/fundamentals.md)과 [CFG](conditional/classifier_free.md)를 통해 생성 단계 수와 조건 강도를 조절하는 방법을 살펴본다.

## 2. 스코어 기반 생성의 단계별 예제

| 주제 | 페이지 |
|---|---|
| 랑주뱅 동역학 | [기본 예제](score_based/03_langevin_dynamics.md), [간단한 구현](score_based/03_langevin_dynamics_2.md) |
| DSM | [고정 노이즈 학습](score_based/04_denoising_score_matching.md), [배치 학습](score_based/04_implementing_dsm_practical.md) |
| SSM | [방향 미분 예제](score_based/05_sliced_score_matching.md), [분산과 구현](score_based/05_sliced_score_matching_2.md) |
| 조건부 신경망 | [시간·노이즈 임베딩](score_based/06_score_network_architectures.md), [잔차 구조](score_based/06_score_networks.md) |
| 여러 노이즈 수준 | [NCSN 예제](score_based/07_noise_conditional_scores.md), [다중 수준 구현](score_based/07_noise_conditional_scores_2.md) |
| 연속시간 | [스코어 SDE](score_based/08_score_sde.md), [SDE 프레임워크](score_based/08_score_sde_framework.md) |
| 이미지 예제 | [노이즈 예측 CNN](score_based/09_image_generation.md), [U-Net 골격](score_based/09_image_generation_unet.md) |
| 확장 | [조건부 응용](score_based/10_advanced_applications.md), [DDPM과의 연결](score_based/10_complete_diffusion_connection.md) |

여러 예제는 교육용 골격이며 목표 스케일, 브로드캐스팅이나 역시간 부호가 이론과 다른 부분도 있다. 각 페이지에서 설명하는 제한을 확인한다. 클래스 이름이나 출력 모양이 맞는다는 사실만으로 올바른 분포를 생성한다고 판단하지 않는다.

## 3. DDPM에서 잠재 확산까지

[간단한 DDPM](ddpm/01_ddpm_toy.md)은 최소 CNN의 학습 흐름을, [확장된 DDPM](ddpm/02_ddpm.md)은 잔차 U-Net·EMA·체크포인트를 보여 준다. [클래스 조건 예제](ddpm/03_conditional_ddpm.md)와 [빠른 DDIM 예제](ddpm/04_ddim_fast_sampling.md)는 조건·시간 처리뿐 아니라 코드의 차원 및 끝점 문제도 설명한다. [잠재 확산](ddpm/05_latent_diffusion.md)은 압축된 표현에서 노이즈 예측을 학습하는 방법과 원본 픽셀 공간 확산의 차이를 다룬다.

모듈별로 읽으려면 [2차원 확산](ddpm/simple_diffusion.md), [공통 유틸리티](ddpm/diffusion_utils.md), [U-Net 구조](ddpm/unet_architecture.md)를 거쳐 [MNIST 파이프라인](ddpm/mnist_diffusion.md)으로 연결한다. 호출 진입점은 [학습 예제](ddpm/example_train.md)와 [샘플링 예제](ddpm/example_sample.md)에 있다. [방법 비교](ddpm/compare_methods.md)는 학습되지 않은 모델로 루프를 비교하므로 생성 품질 실험과 구분한다.

## 4. 평가에서 확인할 질문

[FID](evaluation/fid.md)는 실제·생성 특징의 평균과 공분산을 비교하고, [인셉션 점수](evaluation/inception_score.md)는 생성 이미지의 분류 예측을 요약한다. [우도 기반 평가](evaluation/likelihood.md)는 테스트 데이터에 부여한 확률을 살핀다. 이 값들은 서로 다른 질문에 답한다.

샘플러 비교에서는 가중치와 전처리·표본 수를 고정하고 실제 시간 목록, 모델 호출 수, 가이던스 관례와 벽시계 시간을 기록한다. 단순 노이즈 MSE를 정확한 NLL로 보고하거나, 낮은 FID를 모든 조건 충실도·다양성·새로움의 증거로 사용하지 않는다.

## 정리하며

확산 학습은 무작위 시간의 노이즈 입력으로 예측기를 맞추고, 생성은 그 예측을 시간에 맞는 역전이로 변환한다. 스코어와 노이즈의 스케일, 역시간의 방향, 조건 표현과 깨끗한 끝점을 일치시켜야 두 과정이 연결된다. 구현의 범위와 평가의 의미를 함께 확인한다.
