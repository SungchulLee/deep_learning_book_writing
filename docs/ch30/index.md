# 31장: 확산 모델

확산 모델은 데이터에 노이즈를 더하는 과정을 정의하고 그에 대응하는 역과정으로 새로운 표본을 생성한다. 이 장에서는 로그 밀도의 입력 그래디언트인 스코어에서 출발해 노이즈 예측 학습, DDPM·DDIM 샘플링과 조건부 생성을 연결한다. 이론의 분포·시간 관례와 각 코드가 실제로 구현하는 범위를 함께 읽는 것이 목표다.

## 1. 권장 학습 순서

먼저 [스코어 함수](score_based/01_score_functions_basics.md)와 [스코어 매칭](score_based/02_score_matching_theory.md)을 익힌다. 이어 [DSM 이론](score_based/denoising_score_matching.md)과 [SSM 이론](score_based/sliced_score_matching.md)에서 알려진 노이즈 목표와 무작위 방향 미분이 어떻게 학습 목적을 만드는지 확인한다.

그다음 [역과정 유도](foundations/reverse_process.md), [노이즈 스케줄](ddpm/noise_schedule.md)과 [DDPM 샘플링](ddpm/sampling.md)을 읽는다. 마지막으로 [DDIM](ddim/fundamentals.md)과 [CFG](conditional/classifier_free.md)를 통해 생성 단계 수와 조건 강도를 조절하는 방법을 살펴본다.

## 2. 스코어 기반 생성의 단계별 예제

이 절은 **이론 쪽과 예제 쪽이 짝을 이룬다.** 주제마다 먼저 원리를 적고 뒤에 돌려 보는 꼴이므로, 둘을 나란히 두었다.

**스코어란 무엇인가**

- [스코어 함수 기초](score_based/01_score_functions_basics.md) — 로그 밀도의 입력 그래디언트. **정규화 상수를 몰라도** 된다는 것이 요점이다
- [스코어 매칭 이론](score_based/02_score_matching_theory.md) — 부분적분으로 모르는 항을 지운다

**스코어로 샘플링하기 — 랑주뱅**

- [랑주뱅 동역학](score_based/03_langevin_dynamics.md) — 그래디언트 이동 + 가우시안 잡음. 보정 없는 이산화의 한계도 함께
- [랑주뱅 샘플링의 최소 예제](score_based/03_langevin_dynamics_2.md) — 스코어를 $-x$ 로 **알고 있을 때**의 2차원 시연

**스코어를 배우는 두 길**

- [디노이징 스코어 매칭의 원리](score_based/denoising_score_matching.md) → [고정 노이즈로 학습](score_based/04_denoising_score_matching.md) → [미니배치 학습 구성](score_based/04_implementing_dsm_practical.md)
- [슬라이스드 스코어 매칭의 원리](score_based/sliced_score_matching.md) → [방향 미분 예제](score_based/05_sliced_score_matching.md) → [가우시안 투영을 쓸 때](score_based/05_sliced_score_matching_2.md) — 단위 구면 방향과 **투영 공분산이 다르다**

**네트워크에 노이즈 수준을 알려 주기**

- [노이즈 조건 스코어 네트워크 구조](score_based/06_score_network_architectures.md) · [조건 입력과 잔차 구조](score_based/06_score_networks.md)
- [노이즈 조건부 스코어와 어닐링된 샘플링](score_based/07_noise_conditional_scores.md) · [다중 노이즈의 목표와 브로드캐스팅](score_based/07_noise_conditional_scores_2.md) — **배율과 축**이 틀리기 쉬운 자리다

**연속시간으로 — SDE**

- [스코어 기반 확률미분방정식](score_based/08_score_sde.md) · [스코어 기반 SDE 프레임워크](score_based/08_score_sde_framework.md) — VE·VP 계수와 오일러 샘플러

**이미지로, 그리고 DDPM 과 잇기**

- [이미지 스코어 학습의 간단한 예제](score_based/09_image_generation.md) · [시간 조건 U-Net과 노이즈 예측](score_based/09_image_generation_unet.md) — 출력이 스코어인지 노이즈인지는 **구조가 아니라 학습 목표**가 정한다
- [스코어 모델과 DDPM의 연결](score_based/10_complete_diffusion_connection.md) — 둘이 **알려진 배율**로 오간다
- [클래스 조건 생성과 인페인팅의 기초](score_based/10_advanced_applications.md)

여러 예제는 교육용 골격이며 목표 스케일, 브로드캐스팅이나 역시간 부호가 이론과 다른 부분도 있다. 각 페이지에서 설명하는 제한을 확인한다. 클래스 이름이나 출력 모양이 맞는다는 사실만으로 올바른 분포를 생성한다고 판단하지 않는다.

## 3. DDPM에서 잠재 확산까지

**정방향과 역방향의 규칙**

- [노이즈 스케줄](ddpm/noise_schedule.md) — 같은 시간 인덱스라도 스케줄이 다르면 모델이 보는 신호·노이즈 비율이 다르다
- [DDPM 샘플링](ddpm/sampling.md) — 노이즈 예측을 역전이 평균으로 바꾸는 식, 분산 선택, 마지막 단계 처리

**공통 부품**

- [확산 모델 유틸리티](ddpm/diffusion_utils.md) — $\beta$ 스케줄·누적곱·입력 축의 관례를 한 자리에 맞춘다
- [확산 모델을 위한 U-Net 구조](ddpm/unet_architecture.md) — 관측과 시간을 받아 **같은 크기**를 낸다

**작게 지어 보기**

- [2차원 데이터로 이해하는 확산 모델](ddpm/simple_diffusion.md) — 산점도로 정방향과 생성을 **눈으로** 견준다
- [MNIST로 살펴보는 간단한 DDPM](ddpm/01_ddpm_toy.md) — 작은 CNN 으로 학습과 역전이를 잇는다

**MNIST 파이프라인**

- [MNIST 확산 모델 학습 파이프라인](ddpm/mnist_diffusion.md) — U-Net 과 유틸리티를 연결하고 EMA 로 생성한다
- [학습 실행 예제](ddpm/example_train.md) · [체크포인트를 이용한 샘플링](ddpm/example_sample.md) — 웨이트만이 아니라 **구조와 스케줄도** 학습 때와 맞아야 한다

**키우고 조건 걸기**

- [잔차 U-Net을 사용하는 DDPM](ddpm/02_ddpm.md) — 시간 조건 잔차 블록, 공간 어텐션, 코사인 스케줄, EMA
- [클래스 조건 DDPM과 분류기 없는 가이던스](ddpm/03_conditional_ddpm.md) — CFG

**빠르게, 그리고 잠재 공간에서**

- [DDIM으로 생성 단계 줄이기](ddpm/04_ddim_fast_sampling.md) — **같은 학습**에 다른 전이. $\eta=0$ 이면 경로가 결정적이다
- [DDPM과 DDIM의 샘플링 호출 비교](ddpm/compare_methods.md) — 호출 수와 확률성을 견준다. **학습되지 않은 모델**로 루프만 재므로 생성 품질 실험이 아니다
- [잠재 공간에서 학습하는 확산 모델](ddpm/05_latent_diffusion.md) — 픽셀 대신 압축 표현에서

## 4. 평가에서 확인할 질문

[FID](evaluation/fid.md)는 실제·생성 피처의 평균과 공분산을 비교하고, [인셉션 점수](evaluation/inception_score.md)는 생성 이미지의 분류 예측을 요약한다. [우도 기반 평가](evaluation/likelihood.md)는 테스트 데이터에 부여한 확률을 살핀다. 이 값들은 서로 다른 질문에 답한다.

샘플러 비교에서는 웨이트와 전처리·표본 수를 고정하고 실제 시간 목록, 모델 호출 수, 가이던스 관례와 벽시계 시간을 기록한다. 단순 노이즈 MSE를 정확한 NLL로 보고하거나, 낮은 FID를 모든 조건 충실도·다양성·새로움의 증거로 사용하지 않는다.

## 정리하며

확산 학습은 무작위 시간의 노이즈 입력으로 예측기를 맞추고, 생성은 그 예측을 시간에 맞는 역전이로 변환한다. 스코어와 노이즈의 스케일, 역시간의 방향, 조건 표현과 깨끗한 끝점을 일치시켜야 두 과정이 연결된다. 구현의 범위와 평가의 의미를 함께 확인한다.
