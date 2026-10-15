# 27장: 변분 오토인코더(VAE)

VAE는 잠재 사전과 확률 디코더로 생성 모형을 정의하고, 인코더로 근사 사후를 예측한다. [앞 장의 오토인코더](../ch25/index.md)에 확률적 추론과 KL 항을 더하면서 재구성·잠재 정보·사전 일치의 관계를 살펴본다. 좋은 재구성만으로 사전 생성의 품질이 보장되지는 않으며 KL도 입력 정보량 하나만을 측정하지 않는다.

## 1. 모형과 KL의 이해

- [VAE의 기본 원리](architecture/47_vae.md): 잠재 모형, ELBO와 재매개변수화의 연결
- [KL 항의 의미와 계산](theory/kl_term.md): 사전 KL과 사후 간격, 가우시안 공식과 정보 분해
- [인코더 네트워크](architecture/encoder.md): 평균·로그분산 예측과 상각 추론
- [사전 분포 선택](architecture/prior.md): 표준정규·혼합 사전과 집계 사후의 관계

## 2. 모듈과 변형

- [완전 연결 VAE](architecture/vae.md): 입력·출력, 손실 합산과 생성 메서드
- [합성곱 VAE](architecture/conv_vae.md): 이미지 특징과 고정 해상도 구조
- [β-VAE](architecture/beta_vae.md): KL 가중치, 표현 분리와 평가의 한계
- [조건부 VAE](architecture/conditional_vae.md): 클래스 조건과 조건부 ELBO
- [합성곱 조건부 VAE](architecture/conv_cvae.md): 공간 임베딩과 디코더 조건 연결

β를 조절하거나 조건을 추가해도 의미 요인 분리·스타일 보존·생성 품질 개선이 자동으로 보장되지는 않는다. 구조와 목표가 실제로 무엇을 제약하는지 확인하고 별도의 평가로 검증한다.

## 3. 학습과 생성 평가

- [VAE 학습 절차](training/train_vae.md): 배치 합 손실, 검증과 체크포인트
- [β-VAE 학습](training/train_beta_vae.md): 가중치 적용과 설정 간 공통 평가
- [조건부 VAE 학습](training/train_cvae.md): 레이블 계약과 클래스별 진단
- [Free Bits](training/free_bits.md): 작은 KL에 대한 비용 완화와 실제 KL 기록
- [체크포인트 복원과 샘플 생성](training/generate_samples.md): 사전 생성·보간·잠재 축 탐색의 구분

제공된 예시에는 누락된 보조 함수나 메서드 간 인수 차이가 있는 경로도 있다. 각 절은 그 한계를 설명하며 표시된 코드만으로 모든 학습·시각화가 완료된다고 가정하지 않는다. 검증용 표준 ELBO, 생성 품질·다양성과 목적별 표현 평가를 구분하고 최종 테스트 자료는 모델 선택과 분리한다.

## 정리하며

이 장의 핵심은 확률적 잠재 모형의 정의와 근사 추론·평가의 구분이다. 재구성 항과 KL의 단위, 실제 샘플링 분포와 관측 모형을 일치시켜야 VAE의 학습과 생성 결과를 해석할 수 있다.
