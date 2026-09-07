# 15장: 표집과 추론
이 장은 표집에 기댄 추론 방법의 이론과 실전을 다룬다. 바탕이 되는 마르코프 사슬 이론에서 시작해 몬테카를로 적분, MCMC 알고리즘, 랑주뱅 동역학, 가능도 없는 추론까지 나아간다. 이 방법들은 해석으로 풀 수 없을 때 베이즈 추론을 셈으로 굴리는 장치를 주며, 매개변수가 많은 복잡한 모형에서도 뒤확률을 어림할 수 있게 한다.

---

## 마르코프 사슬

- [바탕](markov_chains/fundamentals.md) — 핵심 정의, 마르코프 성질, 옮김 행렬, 여러 걸음 동역학
- [멈춘 분포](markov_chains/stationary.md) — 긴 눈으로 본 평형 분포와, 마르코프 사슬 이론을 MCMC에 잇는 그 중심 몫
- [에르고드성](markov_chains/ergodicity.md) — 상태 갈래 나누기, 쪼갤 수 없음, 주기 없음, 되돌아옴, 그리고 MCMC가 맞음을 보장하는 모임
- [숨은 마르코프 모형](markov_chains/hmm.md) — 관측 모형을 갖춘 숨은 마르코프 사슬로, 사슬 이론과 통계 추론을 잇는다

## 몬테카를로 방법

- [몬테카를로 적분](monte_carlo/integration.md) — 무작위 표집으로 감당하기 어려운 적분을 어림하기, 흩어짐 줄이기 기법과 함께
- [실효 표본 크기](monte_carlo/ess.md) — 무게 준 표본이나 얽힌 표본의 정보량을 재는 근본 진단
- 물리치기 표집 — 위에서 눌러 주는 제안 분포로 과녁 분포의 표본 만들기

### 중요도 표집

- [바탕](monte_carlo/importance_sampling/fundamentals.md) — 다른 분포에서 표집하고 중요도 무게를 주어 어떤 분포 아래의 기댓값 셈하기
- [중요도 표집의 실효 표본 크기](monte_carlo/importance_sampling/ess.md) — 무게 준 중요도 표본이 과녁 분포의 독립 표본 몇 개에 맞먹는지 재기
- [스스로 고르게 하는 중요도 표집](monte_carlo/importance_sampling/self_normalized.md) — 분자와 분모를 함께 어림해 고르게 하지 않은 과녁 밀도 다루기
- [제안 분포 설계](monte_carlo/importance_sampling/proposal_design.md) — 좋은 제안을 짜는 원칙과 전략, 가장 좋은 제안 이끌어 내기까지

## MCMC 방법

- [메트로폴리스-헤이스팅스](mcmc/metropolis_hastings.md) — 고르게 하는 상수까지만 아는 분포에서 표집하는 바탕 MCMC 알고리즘
- [깁스 표집](mcmc/gibbs_sampling.md) — 온전한 조건부 분포를 써서 받아들임 확률이 늘 1인 메트로폴리스-헤이스팅스의 특별한 경우
- [MCMC 진단](mcmc/diagnostics.md) — 끝이 있는 MCMC 결과에서 모임, 섞임의 질, 뒤확률 간추림의 미더움 살피기
- [NUTS](mcmc/nuts.md) — 되돌아섬을 알아채 HMC 자취의 길이를 스스로 맞추는 U턴 없는 표집기

### 흉내 담금질

- [바탕](mcmc/simulated_annealing/fundamentals.md) — 볼츠만 분포를 써서 온 세상 최적화를 하는 멈추지 않는 메트로폴리스-헤이스팅스
- [온도 일정](mcmc/simulated_annealing/schedules.md) — 살펴보기와 써먹기의 주고받음을 다스리는 식힘 일정 짜기
- [하나로 꿰는 개념으로서의 온도](mcmc/simulated_annealing/temperature_unifying.md) — MCMC의 온도와 소프트맥스, 퍼짐 모형, 강화 학습 사이의 이음
- [멈추지 않는 MCMC로 본 SA](mcmc/simulated_annealing/sa_as_mcmc.md) — 과녁 분포가 바뀌는, 시간에 따라 달라지는 메트로폴리스-헤이스팅스로 흉내 담금질 이해하기
- [모임 이론](mcmc/simulated_annealing/convergence.md) — SA이 언제 왜 온 세상 최적점을 찾는지의 수학 이론, 에너지 벽, 그리고 실전에서의 뜻
- [EM을 위한 정해진 담금질](mcmc/simulated_annealing/annealed_em.md) — 가능도 최적화에서 국소 최적점을 벗어나려고 EM 알고리즘에 온도 쓰기

### 해밀턴 몬테카를로

- [훑어보기](mcmc/hmc/overview.md) — HMC의 온전한 이론. 곧 무작위 걸음 방법을 크게 앞지르는, 물리가 이끄는 제안
- [해밀턴 동역학](mcmc/hmc/hamiltonian_dynamics.md) — 라그랑주 역학에서 해밀턴 역학까지의 물리 바탕, 심플렉틱 짜임, 보존 법칙
- [위상 공간](mcmc/hmc/phase_space.md) — 정해진 동역학을 가능하게 하는, 자리와 운동량 변수의 넓힌 상태 공간
- [HMC 알고리즘](mcmc/hmc/algorithm.md) — 운동량 덧붙이기, 개구리뜀 적분, 메트로폴리스 바로잡기를 갖춘 온전한 알고리즘
- [개구리뜀 적분기](mcmc/hmc/leapfrog_integrator.md) — HMC를 위해 부피와 시간 뒤집힘을 지키는 심플렉틱 수치 적분기
- [질량 행렬](mcmc/hmc/mass_matrix.md) — 운동량이 속도로 어떻게 옮겨지는지 맞추기, 기하로 풀이하기와 어림 전략과 함께
- [기하로 풀이하기](mcmc/hmc/geometric_interpretation.md) — 미분 기하, 정보 기하, 물리적 직관으로 HMC 이해하기

## 랑주뱅 동역학

- [바탕](langevin/fundamentals.md) — 랑주뱅 확률 미분 방정식으로 MCMC 표집과 기울기 기반 최적화를 잇는 이어진 시간 얼개
- [바로잡지 않은 랑주뱅 알고리즘(ULA)](langevin/ula.md) — 메트로폴리스 바로잡기 없이 잘게 나눈 랑주뱅 동역학으로, 확률 기울기와 잘 맞는다
- [MALA](langevin/mala.md) — 기울기를 담은 제안과 받아들임-물리침 바로잡기를 합친 메트로폴리스 바로잡은 랑주뱅 알고리즘
- [점수 맞추기와 퍼짐](langevin/score_and_diffusion.md) — 랑주뱅 동역학, 밀도 어림, 낳는 모형을 하나로 꿰는 개념으로서의 점수 함수

## 어림 베이즈 셈하기

- [가능도 없는 추론](abc/likelihood_free.md) — 가능도는 없지만 흉내내기 장치가 있을 때의 흉내 기반 추론 들여오기
- [ABC 물리치기 표집](abc/rejection_sampling.md) — 간추린 통계량과 너그러움 문턱값을 쓰는 가장 단순한 가능도 없는 알고리즘
- [ABC-MCMC](abc/abc_mcmc.md) — 매개변수 공간을 더 효율적으로 살펴보려고 ABC과 마르코프 사슬 몬테카를로 합치기
- [ABC-SMC](abc/abc_smc.md) — 너그러움을 알아서 고르고 알갱이로 표집하는 잇단 몬테카를로 ABC

## MCMC 방법의 견줌

- [훑어보기](mcmc_comparison/overview.md) — MH, 깁스, 랑주뱅, HMC를 두루 견주고 실전에서 방법 고르는 길잡이
- [이론으로 견주기](mcmc_comparison/theoretical.md) — 모임 속도, 스펙트럼 분석, 가장 좋은 눈금 잡기 이론을 아우르는 엄밀한 견줌
- [차원에 따른 눈금](mcmc_comparison/scaling.md) — 차원이 커질 때 MCMC 방법마다 어떻게 굴러가는지와 효율을 지키는 전략
- [실전에서 방법 고르기](mcmc_comparison/method_selection.md) — 미분 가능함, 차원, 얽힘 짜임, 셈 예산에 기댄 결정 얼개
