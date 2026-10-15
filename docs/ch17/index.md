# 18장: 베이즈 기초

베이즈 추론은 사전분포와 관측 모형을 결합해 매개변수의 사후분포를 구하고 예측·의사결정에 사용한다. 이 장은 켤레 갱신, 점·구간 요약, 계층 모형과 모델 비교를 다룬다. 계산한 확률의 의미는 모형과 사전분포에 조건부라는 점을 유지하고, 해석적 결과와 대입·수치 근사의 범위를 구분한다.

## 1. 베이즈 추론의 기본 요소

- [사전분포·우도·사후분포](bayesian_foundations/prior_likelihood_posterior.md): 베이즈 정리, 충분통계량, 순차 갱신
- [켤레 사전분포](bayesian_foundations/conjugate_priors.md): 베타·감마·정규·정규-역감마 모형의 갱신과 예측
- [MAP 추정](bayesian_foundations/map_estimation.md): 손실 함수, 정규화와 좌표 의존성
- [베이즈 신용구간](bayesian_foundations/credible_intervals.md): 동일 꼬리 구간과 HPD 집합의 의미·조건

## 2. 회귀와 함수 사전분포

- [베이즈 선형 회귀](bayesian_distributions/bayesian_linear_regression.md): 계수 불확실성과 관측 예측분산
- [베이즈 로지스틱 회귀](bayesian_distributions/bayesian_logistic_regression.md): 라플라스 사후 근사와 예측 적분
- [가우시안 프로세스](bayesian_distributions/gaussian_processes.md): 평균 함수·커널과 정규 회귀의 예측

## 3. 계층 모델과 수축

- [계층 베이즈 모델](hierarchical/hierarchical_bayes.md): 비풀링·완전 풀링·부분 풀링
- [다층 모델](hierarchical/multilevel.md): 그룹별 효과와 Gibbs 조건부 갱신
- [경험적 베이즈](hierarchical/empirical_bayes.md): 공통 하이퍼파라미터 추정과 대입의 한계
- [계층 모델 계산 예제](bayesian_foundations/07_hierarchical_models.md): 조건부 수축과 단순 대입
- [경험적 베이즈 계산 예제](bayesian_foundations/08_empirical_bayes.md): 적률 추정과 제임스-스타인 수축의 조건

## 4. 모델 비교

- [모델 증거·주변 우도](model_comparison/selection.md): 평균 우도와 순차 예측, 수치 적분
- [베이즈 인자](model_comparison/bayes_factors.md): 사후 오즈, 사전 민감도와 근사
- [베이즈 가설 검정](model_comparison/hypothesis_testing.md): 점 귀무가설과 새비지-디키 비의 조건
- [정보 기준](model_comparison/information_criteria.md): AIC·BIC·DIC·WAIC·LOO의 서로 다른 목표와 스케일

## 5. 응용과 예측

- [선형 회귀 계산 예제](bayesian_foundations/09_bayesian_linear_regression.md): 알려진 잡음분산 아래 회귀와 구간 비교
- [A/B 비교와 순차 의사결정 예제](bayesian_foundations/10_advanced_applications.md): 차이의 사후분포와 Thompson 샘플링
- [금융의 매개변수 불확실성](finance/parameter_uncertainty.md): 단위·연율화·예측 공분산
- [베이즈 포트폴리오 최적화](finance/portfolio.md): 목적함수와 블랙-리터먼 평균 추론
- [국면 탐지와 전략 평가](finance/regime.md): 상태 필터링·의존성·중단 규칙

## 정리하며

사후분포는 지정한 모형 아래의 불확실성을 나타낸다. 매개변수 분포와 관측 예측분포, 조건부 계산과 하이퍼파라미터를 적분한 계산을 구분한다. 모델 비교에서도 적합성 점검·계산 진단·사전 민감도를 함께 살피며 상대 점수를 절대적인 모델 보장으로 읽지 않는다.
