# 20장: 샘플링과 추론

이 장은 해석적 적분·직접 생성이 어려운 분포의 기댓값과 사후 요약을 계산하는 방법을 다룬다. 마르코프 사슬의 불변성과 수렴 조건에서 출발해 몬테카를로 적분·중요도 샘플링·MCMC와 시뮬레이션 기반 추론을 연결한다. 최적화·고정 목표 샘플링·학습한 생성모형의 목적을 구분하고 실제 코드의 계산 범위와 유한 실행 오차를 함께 살핀다.

## 1. 마르코프 사슬

**사슬이란 무엇인가**

- [마르코프 사슬의 기초](markov_chains/fundamentals.md) — 전이 커널·여러 단계 분포와 시간 평균
- [기본 마르코프 사슬](markov_chains/basic_markov_chain.md) — 유한 상태, 시간에 따라 변하지 않는 전이
- [랜덤 워크](markov_chains/random_walk.md) — 1차원의 평균·분산, 2차원의 격자 경로

**정상분포와 수렴**

- [정상분포](markov_chains/stationary.md) — 불변성과 **극한분포는 다른 말**이다
- [정상분포 구하기](markov_chains/stationary_distribution.md) — 고유벡터·선형방정식·거듭제곱·점유율
- [에르고드성](markov_chains/ergodicity.md) — 기약성·주기·재귀성 · [수렴 정리](markov_chains/convergence_theorems.md) — 총변동 거리와 혼합 시간

**행렬로 셈하기**

- [전이 행렬](markov_chains/transition_matrices.md) — 거듭제곱과 Chapman–Kolmogorov
- [흡수 사슬](markov_chains/absorbing_chains.md) — 과도 상태의 방문 횟수와 흡수 확률
- [연속시간 마르코프 과정](markov_chains/continuous_time_markov.md) — 생성행렬의 비대각은 확률이 아니라 **전이율**이다

**얹은 모델**

- [은닉 마르코프 모델(HMM)](markov_chains/hmm.md) — 필터링·스무딩·최적 상태 경로

**응용**

- [페이지랭크](markov_chains/pagerank_algorithm.md) — 랜덤 서퍼의 정상분포
- [날씨 모델](markov_chains/weather_model.md) — 관측 도수로 1차 사슬을 추정한다
- [텍스트 생성](markov_chains/text_generation.md) — 앞 $k$ 단어를 상태로. 뉴럴 네트워크가 **아니다**

유한 상태의 정리를 일반 상태에 그대로 확장하지 않고, 정상분포의 존재와 초기분포에서의 수렴을 구분한다.

## 2. 몬테카를로 적분과 중요도 샘플링

중요도 샘플링의 뼈대는 [20.3절](monte_carlo/importance_sampling/fundamentals.md)이 따로 다룬다 — [자기정규화 IS](monte_carlo/importance_sampling/self_normalized.md)와 [제안분포 설계](monte_carlo/importance_sampling/proposal_design.md)가 거기 있다. 이 절은 그것을 **쓰고 진단하는** 자리다.

**몬테카를로란**

- [몬테카를로 적분](monte_carlo/integration.md) — 표본평균으로 기댓값을. 유한 분산이 표준오차를 떠받친다

**웨이트가 얼마나 쏠렸나 — ESS**

- [ESS의 두 의미](monte_carlo/ess.md) — 웨이트 집중도와 자기상관 진단은 **다른 것**이다
- [웨이트 ESS](monte_carlo/importance_sampling/ess.md) — 집중도 지표이지 독립 표본 수의 보장이 아니다
- [유효 표본 크기 비교](monte_carlo/04_effective_sample_size.md) — ESS 와 실제 추정 오차의 관계를 반복 실험으로

**사후분포를 근사하기**

- [자기정규화 중요도 샘플링](monte_carlo/02_self_normalized_is.md) — 켤레 사후분포와 맞춰 본다
- [간단한 베이즈 예제](monte_carlo/03_bayesian_simple_examples.md) — 베타-이항·정규-정규·감마-푸아송
- [로지스틱 회귀 중요도 샘플링](monte_carlo/06_logistic_regression_is.md) — 사전 제안과 라플라스 제안

**제안분포를 고르고 진단하기**

- [제안분포 진단](monte_carlo/05_proposal_diagnostics.md) — 집중도·모멘트·꼬리와 다봉 방문
- [혼합 제안분포](monte_carlo/07_mixture_proposals.md) · [방어적 중요도 샘플링](monte_carlo/10_defensive_importance_sampling.md) — 안전 성분을 섞는다
- [적응적 중요도 샘플링](monte_carlo/08_adaptive_importance_sampling.md) — 앞 단계의 가중 표본으로 다음 제안을

**순차와 희귀 사건**

- [순차 중요도 샘플링](monte_carlo/09_sequential_importance_sampling.md) — 재표집은 집중을 줄이고 **다양성도 줄인다**
- [희귀 사건 시뮬레이션](monte_carlo/11_rare_event_simulation.md) — 꼬리에 표본을 몰아준다

**연습**

- [연습 문제](monte_carlo/exercises.md) — 표준 IS 와 SNIS 의 차이를 순서대로

표준 IS의 불편성과 SNIS의 일반적인 유한 표본 바이어스를 구분한다. 관측한 ESS가 높아도 미탐색 영역이나 특정 함수의 정확도가 보장되는 것은 아니다.

## 3. MCMC와 HMC

- [메트로폴리스-헤이스팅스](mcmc/metropolis_hastings.md): 정·역 흐름비와 미지 정규화 상수
- [깁스 샘플링](mcmc/gibbs_sampling.md): 완전조건부 생성·블록·스캔과 상관
- [MCMC 진단](mcmc/diagnostics.md): 여러 체인·R-hat·함수별 ESS·MCSE
- [HMC 개요](mcmc/hmc/overview.md): 확장 목표와 에너지 차이 보정
- [해밀턴 동역학](mcmc/hmc/hamiltonian_dynamics.md): 정확 흐름의 보존 구조
- [위상 공간](mcmc/hmc/phase_space.md): 위치·운동량·에너지 면과 측도
- [HMC 알고리즘](mcmc/hmc/algorithm.md): 실제 갱신과 구현 범위
- [Leapfrog](mcmc/hmc/leapfrog_integrator.md): 대칭 분할·안정성과 수치 오차
- [질량 행렬](mcmc/hmc/mass_matrix.md): 스케일·주파수와 행렬 관례
- [기하학적 해석](mcmc/hmc/geometric_interpretation.md): 심플렉틱 구조·계량·곡률의 구분
- [NUTS](mcmc/nuts.md): 유턴과 목표를 보존하는 트리·후보 선택

수락 규칙의 목표 불변성은 유한 실행의 수렴 증명과 다르다. 스텝·질량·워밍업·모드 탐색과 관심 요약의 수치 오차를 함께 확인한다.

## 4. 랑주뱅과 스코어 기반 방법

- [랑주뱅 기초](langevin/fundamentals.md): 스코어·확산과 연속 목표의 정상성
- [ULA와 SGLD](langevin/ula.md): 이산 바이어스·확률 경사·로스 배율
- [MALA](langevin/mala.md): 비대칭 가우스 제안의 MH 보정
- [스코어 매칭과 확산](langevin/score_and_diffusion.md): 노이즈 조건부 회귀·역시간 과정

온도 거듭제곱·가우스 오염·역확산은 관련된 아이디어지만 일반적으로 같은 목표군이나 전이가 아니다.

## 5. 시뮬레이티드 어닐링과 온도

- [SA 기초](mcmc/simulated_annealing/fundamentals.md): 볼츠만 목표와 최선 방문 해
- [시간 비균질 MCMC](mcmc/simulated_annealing/sa_as_mcmc.md): 순간 불변성과 실제 분포
- [온도 스케줄](mcmc/simulated_annealing/schedules.md): 냉각·재가열과 유한 예산
- [수렴 이론](mcmc/simulated_annealing/convergence.md): 장벽·로그족·수렴 개념
- [결정적 어닐링 EM](mcmc/simulated_annealing/annealed_em.md): 엔트로피를 조절한 잠재 배정
- [온도의 통합적 해석](mcmc/simulated_annealing/temperature_unifying.md): 템퍼링·소프트맥스·엔트로피 목적

SA는 최적화를 위한 과정이며 냉각 궤적 전체를 원래 사후의 고정 목표 표본으로 평균하지 않는다. 점근 전역 집중과 실제 유한 예산의 해 품질도 구분한다.

## 6. 근사 베이즈 계산과 시뮬레이션 기반 추론

- [우도 계산 없는 추론](abc/likelihood_free.md): 시뮬레이터·요약·근사의 대상
- [ABC 거절](abc/rejection_sampling.md): 고정 ABC 목표의 직접 채택 표본
- [ABC-MCMC](abc/abc_mcmc.md): 보조 시뮬레이션을 포함한 MH
- [ABC-SMC](abc/abc_smc.md): 입자 제안·허용오차·중요도 가중

ABC의 허용오차를 줄이는 것과 요약의 정보 로스를 없애는 것은 다른 문제다. 작은 허용오차의 한계는 일반적으로 요약을 조건으로 한 사후이며 충분성 등의 조건이 있어야 원자료 사후와 연결된다.

## 7. 방법 비교와 선택

- [비교 개요](mcmc_comparison/overview.md): 필요한 구조와 비용·진단
- [이론적 비교](mcmc_comparison/theoretical.md): 거리·스펙트럼·점근 분산의 구분
- [차원 스케일링](mcmc_comparison/scaling.md): 반복·평가·적분시간의 단위
- [실전 알고리즘 선택](mcmc_comparison/method_selection.md): 모형·코드·예산에 맞는 후보

## 정리하며

직접 생성·적분 가능성을 먼저 확인하고 필요한 구조를 활용해 방법을 고른다. 목표 근사, 커널 불변성·수렴 조건과 유한 시뮬레이션 오차를 나눠 평가하며, 코드와 출력은 실제 계산의 정의·가정에 맞춰 해석한다.
