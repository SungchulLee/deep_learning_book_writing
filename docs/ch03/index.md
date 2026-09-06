# 장 개요


이 장은 딥러닝을 떠받치는 통계적 기초를 다룬다. 최대가능도 추정, 선형 회귀, 로지스틱 회귀, 소프트맥스 회귀, 손실 함수가 그것이다. 각 주제를 제1원리에서 출발하여 엄밀한 수학적 유도, 확률적 해석, 그에 따르는 PyTorch 구현과 함께 제시한다. 딥러닝의 사실상 모든 손실 함수를 최대가능도 추정의 원리로부터 유도할 수 있으므로 이 기초를 이해하는 것이 필수적이다.

## 최대가능도 추정

손실 함수를 확률 모형과 연결하는 기초적인 모수 추정 틀.

- MLE 개요 -- 기초 확률에서 신경망 응용까지, PyTorch로 배우는 MLE 튜토리얼 개요
- 빠른 시작 -- 5분 만에 끝내는 설치와 첫 예제 안내
- [최대가능도 추정](mle/mle.md) -- MLE의 핵심 이론: 가능도 함수, 로그가능도, 동전 던지기로 얻는 직관
- [확률적 해석](mle/probabilistic_interpretation.md) -- 통합된 관점: 모든 손실은 음의 로그가능도이고 모든 정칙화 항은 사전분포이다
- [회귀를 위한 MLE](mle/mle_regression.md) -- 가우시안 NLL로서의 MSE, 라플라스 NLL로서의 MAE, 이분산 모형
- [분류를 위한 MLE](mle/mle_classification.md) -- 베르누이 NLL로서의 BCE, 범주형 NLL로서의 교차 엔트로피, MLE에서 유도하는 소프트맥스

## 선형 회귀

닫힌 형태의 해와 반복적 해를 모두 가진 기초적인 지도 학습 알고리즘.

- 선형 회귀 개요 -- 기초부터 고급 개념까지의 튜토리얼 시리즈 안내
- 빠른 시작 -- 설치와 첫 튜토리얼 안내
- [선형 회귀](linear_regression/linear_regression.md) -- 모형 명세, 단변량 및 다변량 정식화, 확률적 해석
- [닫힌 형태의 해](linear_regression/closed_form.md) -- 정규방정식, 벡터 미적분 유도, 기하학적 해석, NumPy/PyTorch 구현
- [경사 하강법에 의한 해](linear_regression/gd_solution.md) -- MSE 경사 유도와 네 단계의 PyTorch 구현
- [다항 특징](linear_regression/polynomial_features.md) -- 비선형 특징 사상, 편향-분산 절충, 교차 검증에 의한 모형 선택
- [릿지 회귀](linear_regression/ridge_regression.md) -- L2 벌점, 닫힌 형태의 해, 기하학적 해석과 베이즈 해석
- [라쏘 회귀](linear_regression/lasso_regression.md) -- L1 벌점, 희소성과 특징 선택, 좌표 하강법, 엘라스틱넷

## 로지스틱 회귀

확률 모형의 관점에서 본 이진 분류와 다중 클래스 분류.

- 로지스틱 회귀 개요 -- 네 단계로 이어지는 튜토리얼 시리즈
- 시작하기 -- 빠른 시작 안내와 권장 학습 순서
- 기초 개요 -- 1단계 튜토리얼: PyTorch 텐서, 시그모이드, 이진 분류
- 중급 개요 -- 2단계 튜토리얼: 제대로 된 학습 루프, DataLoader, 배치 구성
- 고급 개요 -- 3단계 튜토리얼: 사용자 정의 데이터셋, 다중 클래스, 고급 기법
- 응용 개요 -- 4단계 튜토리얼: 의료 진단을 비롯한 실전 파이프라인
- [시그모이드 함수](logistic_regression/sigmoid.md) -- 로그 오즈로부터의 유도, 성질, 오즈비, PyTorch 시각화
- [이진 분류](logistic_regression/binary_classification.md) -- 베르누이 모형, GLM 틀, BCE와 MLE의 연결
- [결정 경계](logistic_regression/decision_boundary.md) -- 경계 방정식, 가중치 벡터의 기하, `BCEWithLogitsLoss`
- [경사 계산](logistic_regression/gradient.md) -- BCE 경사 유도, 헤세 행렬, 볼록성 증명, 뉴턴 방법
- [정칙화된 로지스틱 회귀](logistic_regression/regularized.md) -- L2, L1, 엘라스틱넷 벌점과 완전한 PyTorch 파이프라인

## 소프트맥스 회귀

소프트맥스 함수를 사용해 로지스틱 회귀를 다중 클래스 분류로 확장하기.

- 소프트맥스 회귀 개요 -- PyTorch로 하는 다중 클래스 분류 튜토리얼 시리즈
- 빠른 시작 -- 설치와 출발점 안내
- [다중 클래스 분류](softmax_regression/multiclass.md) -- 범주형 분포, 원핫 부호화, 베르누이로부터의 일반화
- 소프트맥스 함수 -- 유도, 야코비 행렬, 온도 조정, 수치적으로 안정한 구현
- 수치적 안정성 -- log-sum-exp 기법, `nn.Module` 분류기, 완전한 학습 루프
- 교차 엔트로피 손실 -- MLE 유도, PyTorch 인터페이스, N그램 언어 모형 시연

## 손실 함수

정보 이론에서 실용적인 PyTorch 사용까지 아우르는 손실 함수의 수학적 기초.

- 손실 함수 개요 -- 손실 함수란 무엇인가, 최적화에서의 역할, PyTorch에서의 계산 방법
- 정보 이론 -- 딥러닝을 위한 자기정보량, 엔트로피, 교차 엔트로피, 상호정보량
- MSE와 MAE -- 가우시안 및 라플라스 잡음 모형, MLE 유도, PyTorch 학습 파이프라인
- 이진 교차 엔트로피 -- 베르누이 가능도, `BCELoss`와 `BCEWithLogitsLoss` 비교, 실무적 사용법
- [교차 엔트로피 손실](loss/cross_entropy.md) -- 정보 이론적 유도, 전체 경사 유도, KL 발산과의 연결
- [KL 발산](loss/kl_divergence.md) -- 정의, 성질, PyTorch 인터페이스, VAE와 증류에서의 응용
- KL 발산과 거리 공리 -- 거리 공리 분석과 대칭화된 대안
- 가우시안에 대한 KL 발산 -- 단변량, 다변량, 대각 공분산의 경우에 대한 닫힌 형태 유도
- KL 발산과 피셔 정보 -- 국소 이차 근사, 자연 경사, 정보 기하
