# 41장: 모델 해석

이 장은 딥러닝 모델을 해석하는 방법을 기초적인 그래디언트 기반 기법부터 고급 개념 수준의 설명까지 두루 다룬다. 모델이 왜 그런 예측을 내놓았는지 아는 일은 규제를 준수하는 데, 버그를 잡는 데, 그리고 계량 금융·의료·자율 시스템처럼 위험이 큰 분야에서 신뢰를 쌓는 데 꼭 있어야 한다. 이 장은 학습이 끝난 모델을 나중에 해석하는 사후(post-hoc) 방법과 그 자체로 해석되는 구조를 함께 다루며, 엄밀히 평가하는 틀도 곁들인다.

---

## 1. 기초

해석 가능성 분야를 이해하는 데 쓰는 핵심 개념과 분류.

- [모델 해석 입문](foundations/introduction.md) -- 블랙박스 문제, 규제 압력, 해석 가능성이 중요한 까닭
- [해석 방법 분류](foundations/taxonomy.md) -- 범위(국소/전역), 모델 접근 수준, 설명 유형으로 나누기
- [해석 방법 평가](foundations/evaluation.md) -- 신뢰성, 강건성, 포괄성, 사람 중심 평가의 여러 측면

---

## 2. 그래디언트 방법

입력의 어느 피처가 모델의 예측을 흔드는지 그래디언트로 드러내는 방법들이다.

**바닐라 그래디언트에서 시작**

- [Saliency Map과 기본 그래디언트](gradient_methods/saliency_maps.md) — 그래디언트 기반 샐리언시의 수학적 기초. 뒤의 모든 방법이 **이것의 한계를 고치는** 것이다

**잡음을 줄이고 기준을 두기**

- [SmoothGrad](gradient_methods/smoothgrad.md) — 입력에 잡음을 **더해** 평균 내면 오히려 또렷해진다 · [실습](gradient_methods/03_smoothgrad.md)
- [Integrated Gradients](gradient_methods/integrated_gradients.md) — 기준점에서 입력까지의 경로로 적분한다. 공리로 정당화되는 유일한 축이다 · [실습](gradient_methods/04_integrated_gradients.md)

**역전파를 고치기**

- [Guided Backpropagation](gradient_methods/guided_backprop.md) — ReLU 를 거꾸로 지나는 길을 바꾼다 · [실습](gradient_methods/06_guided_backpropagation.md)

**CAM 계열**

- [Grad-CAM](gradient_methods/gradcam.md) — 구조를 바꾸지 않고 CAM 을 쓴다 · [Grad-CAM++](gradient_methods/gradcam_plusplus.md) — 같은 클래스 객체가 **여럿일 때** 더 낫다
- [실습](gradient_methods/05_gradcam.md) · [예제](gradient_methods/example_gradcam.md) · [Guided Grad-CAM](gradient_methods/07_guided_gradcam.md) — 앞 묶음과 합친 것

**어텐션 쪽**

- [어텐션 시각화](gradient_methods/attention_visualization.md) · [어텐션 예제](gradient_methods/example_attention.md) — 트랜스포머의 어텐션 웨이트를 그린다

**비교와 확장**

- [비교 분석](gradient_methods/08_comparative_analysis.md) — 방법들을 나란히 놓는다 · [고급 기법](gradient_methods/09_advanced_techniques.md)

---

## 3. 어텐션 시각화

트랜스포머의 어텐션 패턴과 정보 흐름을 알아보는 기법.

- 어텐션 시각화 살펴보기 -- 어텐션 시각화 도구와 기법 모듈 살펴보기
- 빠른 시작 -- 사전 학습된 트랜스포머로 어텐션 시각화 예제 돌리기
- 어텐션 기초 -- 시각화를 위한 스케일 내적 어텐션과 멀티헤드 어텐션의 수학 기초
- 어텐션 롤아웃 -- 모든 레이어에 걸쳐 어텐션을 누적해 입력의 전체 영향 계산
- 어텐션 흐름 -- 어텐션 웨이트와 그래디언트를 결합해 기여도를 정확하게 매기기
- [어텐션 패턴 분석](attention_viz/pattern_analysis.md) -- 여러 헤드 분석, 레이어를 따라가는 흐름, 크로스 어텐션 해석하기

---

## 4. 피처 기여도 분석

게임 이론으로 예측을 입력 피처들에 나누어 배분하는, 모델에 무관한 방법.

- SHAP: 섀플리 가법 설명 -- 협력 게임 이론의 섀플리 값에 기초를 둔 통합된 피처 기여도 분석
- LIME: 국소적으로 해석 가능한 모델 무관 설명 -- 어떤 블랙박스 모델이든 개별 예측을 해석하는 국소 대리 모델
- [Kernel SHAP](feature_attribution/kernel_shap.md) -- 가중 선형 회귀로 SHAP 값 근사하기
- [Deep SHAP](feature_attribution/deep_shap.md) -- DeepLIFT 역전파와 SHAP을 결합해 뉴럴 네트워크의 기여도를 효율적으로 계산하기
- [Tree SHAP](feature_attribution/tree_shap.md) -- 트리 기반 모델을 위한 다항 시간의 정확한 섀플리 값
- [피처 간 상호작용](feature_attribution/interactions.md) -- 비선형 피처 상호작용을 탐지하는 SHAP 상호작용 값

---

## 5. 개념 방법

원시 피처 기여도 분석을 넘어 사람이 이해할 수 있는 개념 수준의 설명.

- [개념 활성화 벡터(CAV)](concept_methods/cav.md) -- 뉴럴 네트워크가 사람이 이해할 수 있는 개념을 배웠는지 평가
- [TCAV: 개념 활성화 벡터를 이용한 평가](concept_methods/tcav.md) -- 모델의 예측에서 개념이 얼마나 중요한지 재는 통계 검정
- [개념 병목 모델](concept_methods/concept_bottleneck.md) -- 개념을 명시적으로 예측하는 레이어를 둔, 그 자체로 해석되는 구조
- [프로토타입 네트워크](concept_methods/prototypes.md) -- 학습된 프로토타입으로 하는 예시 기반 설명

---

## 6. 모델에 맞춘 방법

특정 뉴럴 네트워크 구조에 맞춘 설명 기법.

- [CNN 시각화와 분해](model_specific/cnn_visualization.md) -- 컨볼루션 네트워크를 위한 레이어별 관련성 전파와 DeepLIFT
- [GNN 해석 방법](model_specific/gnn_explanation.md) -- 그래프 뉴럴 네트워크를 위한 GNNExplainer, PGExplainer, SubgraphX
- [트랜스포머 프로빙](model_specific/transformer_probing.md) -- 트랜스포머의 잠재 표현을 살피는 프로빙 분류기와 BertViz
- [피처 역변환(Feature Inversion)](model_specific/feature_inversion.md) -- 중간 표현에서 입력을 재구성해 모델이 무엇을 보는지 드러내기

---

## 7. 평가

설명의 품질과 신뢰성을 엄밀히 재는 방법.

- 신뢰성 평가 -- 설명이 실제 모델의 행동을 드러내는지 재는 삽입/삭제 곡선
- [안정성 평가](evaluation/stability.md) -- 설명이 얼마나 강건한지 재는 입력 섭동에 대한 강건성과 최대 민감도
- [포괄성 평가](evaluation/comprehensiveness.md) -- 충분성 점수로 설명이 중요한 피처를 모두 담았는지 평가
- 사람 중심 평가 -- 순방향 시뮬레이션, 신뢰 보정, 사용자 연구 방식

---

## 8. 금융 응용

계량 금융 현장에 설명 방법을 쓰기.

- 규제 준수 -- SR 11-7, GDPR 22조, 그리고 금융 모델을 해석하는 프레임워크
- 거래 신호 분석 -- 매매 신호를 피처 그룹으로 분해해 분석하고 모니터링하기
- [요인 기여도 분석](finance/factor_attribution.md) -- 포트폴리오 수익률을 구조적 팩터의 기여도로 분해하는 SHAP 기반 방법
- 신용 위험 설명 -- 신용 판단에 대한 불리한 처분 사유와 규정에 맞는 설명

---

## 정리하며

이 장은 기초, 그래디언트 방법, 어텐션 시각화, 피처 기여도 분석, 개념 방법, 모델에 맞춘 방법, 평가, 금융 응용의 여덟 묶음으로 이루어진다.
