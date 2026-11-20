# 39장: 폴리시 기반 방법

폴리시 기반 방법은 파라미터화된 폴리시를 곧바로 최적화하여 기대 리턴을 크게 만들며, 연속 액션 공간과 확률적 폴리시 표현을 학습할 수 있게 한다. 이 장은 폴리시 경사 정리와 REINFORCE, Actor-Critic 구조, 신뢰 영역과 근접 최적화 방법, 오프폴리시 연속 제어 알고리즘, 그리고 최대 엔트로피 강화학습, 계층적 강화학습, 멀티에이전트 설정을 아우르는 고급 주제를 다룬다. 포트폴리오 배분, 연속 액션 기반 거래, 위험에 민감한 목표에서의 금융 응용을 곳곳에서 펼친다.

---

## 1. 차례

### 39.1 폴리시 경사 기초

- [폴리시 파라미터화](foundations/parameterization.md) -- 소프트맥스(이산)와 가우스(연속) 파라미터화로 관찰을 액션 분포에 대응시키기
- [폴리시 경사 정리](foundations/pg_theorem.md) -- 둘레 동역학을 미분하지 않고도 기대 리턴의 그래디언트를 해석적으로 주는 기본 결과
- [REINFORCE](foundations/reinforce.md) -- 샘플링한 에피소드 리턴으로 그래디언트를 추정하는 몬테카를로 폴리시 경사 알고리즘
- [베이스라인 방법](foundations/baselines.md) -- 그래디언트의 불편성을 지키면서 상태 의존 베이스라인으로 분산 줄이기

### 39.2 Actor-Critic

- [Actor-Critic 기초](actor_critic/fundamentals.md) -- 폴리시 기반 액터와 밸류 기반 크리틱을 엮어 분산이 작은 그래디언트 추정 얻기
- [어드밴티지 Actor-Critic(A2C)](actor_critic/a2c.md) -- 동기화된 병렬 둘레와 어드밴티지 기반 폴리시 갱신, 공유 Actor-Critic 네트워크
- [비동기 어드밴티지 Actor-Critic(A3C)](actor_critic/a3c.md) -- 워커 스레드 여럿과 공유 전역 네트워크로 하는 분산 비동기 학습
- [일반화 어드밴티지 추정(GAE)](actor_critic/gae.md) -- 람다로 편향-분산 트레이드오프를 조절할 수 있는, 지수로 웨이트를 준 여러 스텝 어드밴티지 추정

### 39.3 신뢰 영역 방법

- [신뢰 영역 폴리시 최적화(TRPO)](trust_region/trpo.md) -- 단조로운 개선을 보장하는 쿨백-라이블러 발산 신뢰 영역을 둔 제약 폴리시 최적화
- [자연 폴리시 경사](trust_region/npg.md) -- 폴리시 파라미터화에 불변인, 피셔 정보 기반 그래디언트 스텝
- [근접 폴리시 최적화(PPO)](trust_region/ppo.md) -- 단순한 일차 최적화로 TRPO 같은 강건성을 이루는 잘라 낸 대리 목표
- [PPO 구현](trust_region/ppo_implementation.md) -- 벡터 둘레, 어드밴티지 정규화, 구현 모범 사례를 갖춘 완전한 프로덕션 품질의 PPO

### 39.4 오프폴리시 Actor-Critic

- [심층 결정론적 폴리시 경사(DDPG)](off_policy/ddpg.md) -- 연속 제어를 위해 결정론적 액터를 둔 DQN 형태 오프폴리시 학습
- [Twin Delayed DDPG(TD3)](off_policy/td3.md) -- 트윈 크리틱, 지연된 폴리시 갱신, 타깃 폴리시 스무딩으로 DDPG의 과대추정 다루기
- [Soft Actor-Critic(SAC)](off_policy/sac.md) -- 확률 폴리시와 자동으로 조정되는 온도를 쓰는 최대 엔트로피 오프폴리시 학습
- [SAC 구현](off_policy/sac_implementation.md) -- tanh로 압축한 가우스 폴리시, 트윈 Q 네트워크, 자동 조정 엔트로피 계수를 갖춘 완전한 SAC

### 39.5 고급 주제

- [최대 엔트로피 강화학습](advanced/max_entropy.md) -- 엔트로피를 더한 목표, 소프트 벨먼 식, 탐험과 활용의 균형에 대한 이론적 기초
- [계층적 강화학습](advanced/hierarchical.md) -- 시간적 추상화와 장기 과제를 위한 옵션 프레임워크, 봉건 네트워크, 목표 조건부 폴리시
- [멀티에이전트 강화학습](advanced/multi_agent.md) -- 마르코프 게임, 독립 학습, 중앙집중식 학습과 분산 실행, 경쟁적 시장 조성
- [모델 기반 강화학습](advanced/model_based.md) -- 계획을 위한 둘레 동역학 학습, 다이나 형태 방법, 모델 예측 제어

### 39.6 실전 고려 사항

- [리워드 셰이핑](practical/reward_shaping.md) -- 최적 폴리시를 바꾸지 않으면서 더 빨리 배우게 하는, 퍼텐셜 기반 리워드 수정
- [액션 공간](practical/action_spaces.md) -- 이산 액션과 연속 액션의 설계 선택, 제약된 파라미터화, 다차원 액션 구조
- [관찰 정규화](practical/normalization.md) -- 학습을 안정적으로 하려고 관찰, 리워드, 어드밴티지에 이동 평균-분산 정규화 적용하기
- [강화학습 디버깅](practical/debugging.md) -- 폴리시 붕괴, 값 발산, 리워드 해킹을 비롯한 흔한 실패를 체계적으로 진단하는 전략

### 39.7 금융 응용

- [포트폴리오 최적화](finance/portfolio.md) -- 폴리시 기반 강화학습으로 하는 동적 자산 배분, 연속 자본 웨이트 출력, 위험 조정 리워드 목표
- [연속 액션 기반 거래](finance/continuous_trading.md) -- 연속 액션 폴리시로 하는 일중 단위의 실시간 실행과 포지션 사이징
- [위험 민감 강화학습](finance/risk_sensitive.md) -- CVaR, 샤프 비율, 분산 벌점을 강화학습의 최적화 목표에 직접 넣기

---

## 정리하며

이 장의 일곱 절과 각 절의 쪽들을 차례로 살펴보았다.
