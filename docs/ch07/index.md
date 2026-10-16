# 통계적 학습 — 장 개요

이 장에서는 확률 모델에서 학습 목표를 정하는 과정을 살펴본다. 최대가능도 추정(MLE)을 출발점으로 선형 회귀, 로지스틱 회귀, 소프트맥스 회귀를 연결하고, 각 모델의 로스 함수와 추정값을 해석한다. MSE와 교차 엔트로피는 적절한 분포 가정 아래에서 음의 로그우도로 유도할 수 있지만, 모든 학습 목적이 이런 해석을 갖는 것은 아니다.

## 1. 최대가능도 추정

관측 데이터를 잘 설명하는 파라미터를 찾는 방법이다. 우도는 데이터를 고정하고 파라미터를 바꾸어 비교한다. 추정량의 성질과 수치 최적화의 수렴은 서로 다른 문제이므로 구분해서 읽는다.

- [최대가능도 추정](mle/mle.md): 우도, 로그우도, 피셔 정보와 점근적 성질
- [회귀에서의 MLE](mle/mle_regression.md): 고정 분산 가우스 모델과 MSE, 라플라스 모델과 MAE
- [분류에서의 MLE](mle/mle_classification.md): 베르누이 모델과 BCE, 범주형 모델과 교차 엔트로피
- [확률적 해석](mle/probabilistic_interpretation.md): 로스와 분포 가정, 사전분포와 MAP 추정

## 2. 선형 회귀

선형 회귀는 예측값이 파라미터에 대해 선형인 모델이다. 정규 방정식과 경사 하강법을 비교하고, 피처 변환과 정규화가 추정에 미치는 영향을 살펴본다.

- [선형 회귀](linear_regression/linear_regression.md): 모델, 로스와 확률적 해석
- [해석적 해](linear_regression/closed_form.md): 최소제곱 문제와 행렬 표현
- [경사 하강법](linear_regression/gd_solution.md): 반복 업데이트와 러닝레이트
- [다항 피처](linear_regression/polynomial_features.md): 비선형 피처와 모델 선택
- [릿지 회귀](linear_regression/ridge_regression.md), [라쏘 회귀](linear_regression/lasso_regression.md): L2·L1 패널티와 해의 차이

## 3. 로지스틱 회귀

로지스틱 회귀는 선형 점수를 시그모이드로 변환해 이진 클래스의 확률을 모델링한다. 확률 추정과 임곗값에 따른 분류 결정을 구분한다.

- [시그모이드](logistic_regression/sigmoid.md): 로짓, 오즈와 확률
- [이진 분류](logistic_regression/binary_classification.md): 베르누이 우도와 BCE
- [결정 경계](logistic_regression/decision_boundary.md): 파라미터와 경계의 기하
- [경사 계산](logistic_regression/gradient.md): 미분, 곡률과 최적화
- [정규화](logistic_regression/regularized.md): L1·L2 패널티와 일반화

## 4. 소프트맥스 회귀

서로 배타적인 여러 클래스를 예측할 때 로짓 벡터를 확률 벡터로 변환한다. 클래스 인덱스와 원핫 정답, 로짓과 확률을 구분하는 것이 구현의 핵심이다.

- [기초](softmax_regression/01_fundamentals.md): 소프트맥스와 교차 엔트로피
- [다중 클래스 분류](softmax_regression/multiclass.md): 범주형 모델과 학습
- [소프트맥스 회귀의 한계](softmax_regression/02_limits.md): 선형 결정 경계와 표현력

## 5. 로스 함수

로스의 수학적 정의와 라이브러리 입력 조건을 함께 확인한다. 로스를 합할지 평균할지에 따라 경사의 스케일과 정규화 항의 상대적 비중도 달라진다.

- [교차 엔트로피](loss/cross_entropy.md): 분포 적합과 분류 로스
- [KL 발산](loss/kl_divergence.md): 두 분포의 차이와 계산 방향
- [L2 로스](loss/l2_loss.md): 제곱오차와 평균제곱오차

## 정리하며

확률 모델은 로스를 선택하고 추정값을 해석하는 근거를 제공한다. 좋은 학습 결과를 판단하려면 모델 가정, 최적화 상태, 검증 데이터의 성능을 함께 확인해야 한다.
