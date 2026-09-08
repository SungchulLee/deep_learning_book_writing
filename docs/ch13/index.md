# 13장: 베이즈의 바탕

모든 통계 추론은 하나의 물음에서 시작한다. 관찰한 데이터가 주어졌을 때 그것을 낳은 밑바탕의 과정에 대해 무엇을 알 수 있는가? **베이즈의 틀**은 베이즈 정리로 앞선 믿음과 관찰한 증거를 어우러 모르는 양 위의 온전한 뒤확률 분포를 내놓아 이에 답한다. 점 어림값과 달리 뒤확률은 불확실함의 상태를 온전히 담아, 원칙 있는 예측과 모형 견줌, 불확실함 속의 결정을 가능하게 한다.

이 장은 베이즈 정리와 켤레 앞확률에서 층층 모형과 모형 견줌까지 나아가며 베이즈 추론의 이론적 바탕을 세운다. 이 개념들이 뒤이은 장에서 다루는 어림 추론 방법, 표집 알고리즘, 베이즈 신경망의 수학적 등뼈를 이룬다.

---

## 1. 베이즈의 바탕

- [앞확률, 가능도, 뒤확률](bayesian_foundations/prior_likelihood_posterior.md) -- 근본이 되는 세 양과 베이즈 정리로 어우러지는 모습을 빈틈없이 다루기
- [켤레 앞확률](bayesian_foundations/conjugate_priors.md) -- 해석적 해를 주는 베타-이항, 감마-푸아송, 정규-정규 족과 켤레성의 이론
- [최대 뒤확률 어림](bayesian_foundations/map_estimation.md) -- 최대 뒤확률 점 어림값, 최대 가능도 및 뒤확률의 평균과의 견줌, 벌주기와의 이음
- [믿음 구간](bayesian_foundations/credible_intervals.md) -- 양 꼬리가 같은 구간과 최고 뒤확률 밀도 구간으로 베이즈식 불확실성 재기

---

## 2. 베이즈 분포

- [베이즈 선형 회귀](bayesian_distributions/bayesian_linear_regression.md) -- 켤레 정규-정규 모형으로 매개변수와 예측 위의 온전한 뒤확률 분포
- [베이즈 로지스틱 회귀](bayesian_distributions/bayesian_logistic_regression.md) -- 어림 추론이 필요한 켤레 아닌 뒤확률로, 해석적 방법과 셈 방법을 잇기
- [가우스 과정](bayesian_distributions/gaussian_processes.md) -- 알맹이에 담은 가정으로 함수 위에 곧바로 앞확률을 정하는 비모수 베이즈 회귀

---

## 3. 층층 모형

- [층층 베이즈 모형](hierarchical/hierarchical_bayes.md) -- 반쯤 모으기, 무리 수준의 변동, 오그라들기 현상을 갖춘 여러 층 추론
- [다층 모형](hierarchical/multilevel.md) -- 겹친 데이터를 뜯어보기 위한, 짜임새 있는 확률 효과를 갖춘 섞인 효과 모형
- [경험적 베이즈](hierarchical/empirical_bayes.md) -- 온전한 베이즈와 빈도주의 사이의 실전적인 중간 지대로서 데이터에서 초매개변수 어림하기

---

## 4. 모형 견줌

- [모형 증거(주변 가능도)](model_comparison/selection.md) -- 모형 아래에서 데이터의 확률을 셈하여 베이즈판 오컴의 면도날을 구현하기
- [베이즈 인자](model_comparison/bayes_factors.md) -- 겨루는 모형을 원칙 있게 견주기 위한 모형 증거의 비
- [베이즈 가설 검정](model_comparison/hypothesis_testing.md) -- 겨루는 가설 사이의 증거를 수로 나타내는 데 베이즈 인자와 뒤확률 승산 쓰기
- [정보 기준](model_comparison/information_criteria.md) -- 빈도주의와 베이즈의 모형 고르기를 잇는, 셈으로 다룰 수 있는 어림(AIC, BIC, WAIC)

---

## 5. 금융에서의 쓰임새

- [베이즈 포트폴리오 최적화](finance/portfolio.md) -- 더 튼튼한 포트폴리오를 위해 평균-분산 최적화에 매개변수 불확실성 아우르기
- [금융에서의 매개변수 불확실성](finance/parameter_uncertainty.md) -- 포트폴리오 짜기와 위험 다스리기 전반에 어림 위험을 수로 나타내고 퍼뜨리기
- [국면 알아채기와 전략 평가](finance/regime.md) -- 시장 국면을 알아채는 온라인 베이즈 갱신과 전략을 견주는 베이즈 A/B 시험

---

## 정리하며

이 마당은 베이즈의 바탕、베이즈 분포、층층 모형、모형 견줌을 차례로 짚었다.
