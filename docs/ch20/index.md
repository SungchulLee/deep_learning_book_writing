# 21장: 근사 추론

정확한 베이즈 추론은 사후확률 분포 $p(\theta \mid \mathcal{D})$을 셈해야 하고, 그러려면 다시 주변 우도 $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\,p(\theta)\,d\theta$의 값을 매겨야 한다. 가장 단순한 모델을 빼면 이 적분은 다룰 수 없다. 근사 추론 방법은 정확한 사후확률을 다룰 수 있는 어림으로 바꿔 베이즈 계산을 감당할 수 있게 만든다.

이 장은 근사 추론의 큰 갈래 셋을 다룬다. **변분 추론**은 사후확률 추정을 최적화 문제로 바꾸어, 다룰 수 있는 집안 가운데 KL 발산으로 참 사후확률에 가장 가까운 것을 찾는다. **EM 알고리즘**은 기댓값 걸음과 최대화 걸음을 되풀이해 잠재변수 모델을 다루며, 온전한 사후확률을 곧바로 셈할 필요를 없앤다. **베이즈 신경망**은 이 생각을 딥러닝으로 넓혀, 가중치에 분포를 두고 원리 있게 불확실성을 잰다.

---

## 1. 변분 추론

- [변분 추론 구조](variational_inference/framework.md) -- 변분 추론을 최적화로 세우기, KL 발산과의 관계, 다른 방법과의 비교
- [증거 하한(ELBO)](variational_inference/elbo.md) -- 서로 보완하는 세 가지 유도, 틈 살피기, 팽팽함의 조건, EM 및 VAE과의 이음
- [평균장 변분 추론](variational_inference/mean_field.md) -- 온전히 인수로 나눈 어림, 가장 좋은 인수 유도, 상관을 무시할 때의 한계
- [매개변수 바꾸기와 블랙박스 변분 추론](variational_inference/reparameterization.md) -- 점수 함수 어림꼴, 분산 줄이기, 기울기를 쓰는 변분 추론을 위한 매개변수 바꾸기 재주
- [Amortized Variational Inference](variational_inference/amortized.md) -- 규모를 키울 수 있는 사후확률 추정을 위한 추론 망, VAE, 나눠 갚기 틈

---

## 2. 기댓값 최대화(EM)

- [EM의 기초](em/foundations.md) -- 잠재변수 모델, 잠재변수가 하는 일, EM 알고리즘이 필요한 까닭
- [E 단계와 M 단계](em/e_step_m_step.md) -- 사후확률 계산과 기댓값 충분 통계량을 비롯한 걸음마다의 자세한 구조
- [가우시안 혼합 모델](em/gmm.md) -- 온전한 유도, 구현, 금융으로의 넓힘을 갖춘 EM의 대표 쓰임새
- [EM의 갈래](em/variants.md) -- 넓힌 EM, 변분 EM, 그리고 E 걸음이나 M 걸음이 닫힌 형태가 아닌 모델로의 넓힘

---

## 3. 베이즈 신경망

- [베이즈 신경망의 기초](bnn/fundamentals.md) -- 원리 있게 불확실성을 재려고 가중치에 확률 분포 두기
- [가중치 불확실성](bnn/weight_uncertainty.md) -- 가중치에 사전확률 정하기, 가중치 공간의 기하, 주요 추론 방법
- [Bayes by Backprop](bnn/bayes_by_backprop.md) -- ELBO 최적화와 매개변수 바꾸기 재주를 쓴 변분 추론으로 베이즈 신경망 학습
- [사전분포 선택](bnn/prior_selection.md) -- 가우스, 라플라스, 편자, 못과 판 사전확률과 층 베이즈 및 경험 베이즈 전략

---

## 정리하며

이 마당은 변분 추론, 기댓값 최대화(EM), 베이즈 신경망을 차례로 짚었다.
