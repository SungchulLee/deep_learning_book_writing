# 21장: 근사 추론

복잡한 확률모델에서는 사후분포의 적분과 기댓값을 정확히 계산하기 어려울 수 있다. 변분 추론은 제한된 분포족에서 사후 근사를 최적화하고, 베이즈 뉴럴 넷트웍은 웨이트 불확실성을 예측에 반영한다. EM은 잠재변수 사후의 기댓값을 이용해 파라메터의 우도를 높이는 방법으로, 그 자체가 파라메터 전체의 베이즈 사후를 계산하는 알고리즘은 아니다.

이 장에서는 세 접근의 목적과 계산 구조를 연결하면서 정확한 우도·변분 하한·표본 추정값을 구분한다. 코드와 출력은 각 페이지에 명시한 가정과 한계를 전제로 읽는다.

## 1. 변분 추론

- [기본 구조](variational_inference/framework.md): 사후 근사, KL 방향과 ELBO의 관계
- [증거 하한](variational_inference/elbo.md): Jensen·KL·중요도 가중 해석과 모델 비교의 한계
- [평균장과 CAVI](variational_inference/mean_field.md): 독립 블록, 최적 인자와 상관 누락
- [리파라메리제이션과 블랙박스 VI](variational_inference/reparameterization.md): 점수함수·경로별 경사, 제어변량과 변환밀도
- [상각 변분 추론](variational_inference/amortized.md): 공유 인코더, VAE와 근사·상각 차이
- [기본 연습](variational_inference/exercise_01_basic_vi.md): 가우스·회귀·혼합·확률적 추론의 유도와 구현 골격

## 2. EM 알고리즘

- [EM의 기초](em/foundations.md): 현재 우도에 일치하는 하한, 단조성과 수렴 조건
- [E 단계와 M 단계](em/e_step_m_step.md): GMM·인자 분석·HMM의 사후 적률과 갱신
- [가우시안 혼합 모델](em/gmm.md): 공분산 구조, 특이점, 모델 선택과 혼합분포 해석
- [EM의 변형](em/variants.md): GEM·ECM·ECME·PX-EM, 표본·변분·온라인 근사

## 3. 베이즈 뉴럴 넷트웍

- [기초](bnn/fundamentals.md): 사후 예측과 회귀·분류의 불확실성 분해
- [웨이트 불확실성](bnn/weight_uncertainty.md): 사전 스케일, 웨이트 대칭성과 추론 방법
- [Bayes by Backprop](bnn/bayes_by_backprop.md): 가우스 변분 학습, 미니배치 로스와 KL 배율
- [사전분포 선택](bnn/prior_selection.md): 계층·수축·희소 사전, ARD와 경험적 베이즈

## 정리하며

추론 근사의 표현력, 목적함수의 최적화 오차와 유한 표본 오차는 서로 다른 문제다. 정확한 EM의 우도 비감소성, 변분 좌표 갱신의 하한 비감소성, 확률적 경사 학습의 유한 실행을 구분하면 각 방법의 결과를 더 정확하게 해석할 수 있다.
