# 20장: 샘플링과 추론

이 장은 해석적 적분·직접 생성이 어려운 분포의 기댓값과 사후 요약을 계산하는 방법을 다룬다. 마르코프 사슬의 불변성과 수렴 조건에서 출발해 몬테카를로 적분·중요도 샘플링·MCMC와 시뮬레이션 기반 추론을 연결한다. 최적화·고정 목표 샘플링·학습한 생성모형의 목적을 구분하고 실제 코드의 계산 범위와 유한 실행 오차를 함께 살핀다.

## 1. 마르코프 사슬

- [기초](markov_chains/fundamentals.md): 전이 커널·여러 단계 분포와 시간 평균
- [정상분포](markov_chains/stationary.md): 불변성·유일성과 수렴의 차이
- [에르고드성](markov_chains/ergodicity.md): 접근성·재귀성·주기와 혼합 진단
- [은닉 마르코프 모형](markov_chains/hmm.md): 필터링·스무딩·최적 상태 경로

유한 상태의 정리를 일반 상태에 그대로 확장하지 않고, 정상분포의 존재와 초기분포에서의 수렴을 구분한다.

## 2. 몬테카를로 적분과 중요도 샘플링

- [몬테카를로 적분](monte_carlo/integration.md): 적률 조건·표준오차·분산 감소
- [ESS의 두 의미](monte_carlo/ess.md): 웨이트 집중도와 함수별 자기상관 진단
- [중요도 샘플링 기초](monte_carlo/importance_sampling/fundamentals.md): 표준 IS와 지지집합·분산 조건
- [자기정규화 IS](monte_carlo/importance_sampling/self_normalized.md): 미지 상수를 소거하는 비율 추정
- [웨이트 ESS](monte_carlo/importance_sampling/ess.md): 집중도 지표와 함수별 오차의 한계
- [제안분포 설계](monte_carlo/importance_sampling/proposal_design.md): 관심 적분·모드·꼬리와 적응
- [연습 예제](monte_carlo/exercises.md): 추정량·사전 변경·희귀 사건 비교

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
