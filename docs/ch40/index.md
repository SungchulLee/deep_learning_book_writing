# 41장: 모델 해석

이 장은 딥러닝 모델을 풀이하는 방법을 기초가 되는 그래디언트 기반 재주에서 앞선 개념 레이어의 풀이까지 두루 다룬다. 모델이 왜 그런 추론을 내놓았는지 아는 일은 규정을 따르는 데, 버그를 잡는 데, 그리고 계량 금융·의료·스스로 도는 구조처럼 걸린 것이 큰 밭에서 믿음을 쌓는 데 꼭 있어야 한다. 이 장은 일이 끝난 뒤에 풀이하는 방법과 그 자체로 풀이되는 구조를 함께 다루며, 엄밀히 따지는 틀도 곁들인다.

---

## 1. 기초

풀이하기의 밭을 알아보는 데 쓰는 핵심 개념과 분류.

- [모델 해석 입문](foundations/introduction.md) -- 블랙박스 문제, 규정이 미는 힘, 풀이하기가 중요한 까닭
- [해석 방법 분류](foundations/taxonomy.md) -- 미치는 데(그 자리/온 세상), 모델에 닿는 만큼, 풀이의 클래스로 나누기
- [해석 방법 평가](foundations/evaluation.md) -- 미더움, 든든함, 두루 갖춤, 사람에 뿌리내린 따짐의 여러 피처

---

## 2. 그래디언트 방법

입력의 어느 피처가 모델의 추론을 흔드는지 드러내는 그래디언트 기반 그림 그리기 재주.

- 모델 해석 살펴보기 -- Grad-CAM과 어텐션 그림 그리기를 아우르는 연장 두루 살피기
- Saliency Map 살펴보기 -- 그래디언트 기반 Saliency Map 재주 모듈 살펴보기
- 빨리 시작하기 -- 5분 만에 Saliency Map 구현 시작하기
- [Saliency Map과 기본 그래디언트](gradient_methods/saliency_maps.md) -- 뉴럴 넷트웍 추론에 대한 그래디언트 기반 현저성의 수학 기초
- [Integrated Gradients](gradient_methods/integrated_gradients.md) -- 예민함과 짜기에 흔들리지 않음이라는 공리를 채우는 길 따라 쌓은 기여도 분석
- [Grad-CAM](gradient_methods/gradcam.md) -- CNN 시각적 해석을 위한 그래디언트 가중 클래스 활성화 맵
- [Grad-CAM++](gradient_methods/gradcam_plusplus.md) -- 여러 물체와 작은 자리를 더 잘 짚어내는 나아진 것
- [Guided Backpropagation](gradient_methods/guided_backprop.md) -- ReLU 그래디언트 퍼뜨리기를 고쳐 얻는 높은 피처 Saliency Map
- [SmoothGrad](gradient_methods/smoothgrad.md) -- 잡음을 써서 그래디언트를 고르게 해 더 또렷하고 깨끗한 Saliency Map 얻기

---

## 3. 어텐션 시각화

트랜스포머의 어텐션 피처와 소식 흐름을 알아보는 재주.

- 어텐션 그림 그리기 살펴보기 -- 어텐션 그림 그리기 연장과 재주 모듈 살펴보기
- 빨리 시작하기 -- 사전 학습된 트랜스포머로 어텐션 그림 그리기 보기 돌리기
- 어텐션 기초 -- 그림 그리기를 위한 잣대 잰 점곱 어텐션과 여러 머리 어텐션의 수학 기초
- 어텐션 굴리기 -- 온 레이어에 걸쳐 어텐션을 쌓아 입력이 미치는 온 힘 계산
- 어텐션 흐름 -- 어텐션 웨이트와 그래디언트를 아울러 몫을 옳게 매기기
- [어텐션 패턴 분석](attention_viz/pattern_analysis.md) -- 여러 머리 살피기, 레이어를 따라가는 흐름, 크로스 어텐션 풀이하기

---

## 4. 피처 기여도 분석

놀이 이론으로 추론을 입력의 피처에 돌리는, 모델을 가리지 않는 방법.

- SHAP: 더해지는 섀플리 풀이 -- 함께 하는 놀이 이론의 섀플리 값에 기초를 둔 하나로 아우른 피처 기여도 분석
- LIME: 그 자리에서 풀이되는 모델 가리지 않는 풀이 -- 어떤 블랙박스 모델이든 낱낱의 추론을 풀이하는 그 자리 대리 모델
- [Kernel SHAP](feature_attribution/kernel_shap.md) -- 웨이트 실은 선형 회귀로 SHAP 값에 다가가기
- [Deep SHAP](feature_attribution/deep_shap.md) -- DeepLIFT 되짚기와 SHAP을 아울러 뉴럴 넷트웍의 몫을 잘 매기기
- [Tree SHAP](feature_attribution/tree_shap.md) -- 나무 기초 모델을 위한 다항식 때의 정확한 섀플리 값
- [피처 간 상호작용](feature_attribution/interactions.md) -- 선형이 아닌 피처 사이 얽힘을 알아내는 SHAP 서로 미침 값

---

## 5. 개념 방법

날 피처 기여도 분석을 넘어 사람이 알아들을 수 있는 개념 레이어의 풀이.

- [개념 활성화 벡터(CAV)](concept_methods/cav.md) -- 뉴럴 넷트웍이 사람이 알아들을 수 있는 개념을 배웠는지 따지기
- [TCAV: 개념 활성화 벡터를 이용한 평가](concept_methods/tcav.md) -- 모델의 추론에서 개념이 얼마나 중요한지 재는 통계 검정
- [개념 병목 모델](concept_methods/concept_bottleneck.md) -- 개념을 또렷이 예측하는 레이어를 둔, 그 자체로 풀이되는 구조
- [프로토타입 네트워크](concept_methods/prototypes.md) -- 배운 프로토타입 자리로 하는 보기 기초 풀이

---

## 6. 모델에 맞춘 방법

특정 뉴럴 넷트웍 구조에 맞춘 풀이 재주.

- [CNN 그림 그리기와 쪼개기](model_specific/cnn_visualization.md) -- 겹친 네트워크를 위한 레이어마다의 쓸모 퍼뜨리기와 DeepLIFT
- [GNN 해석 방법](model_specific/gnn_explanation.md) -- 그래프 뉴럴 넷트웍을 위한 GNNExplainer, PGExplainer, SubgraphX
- [트랜스포머 프로빙](model_specific/transformer_probing.md) -- 트랜스포머의 잠재 표현을 살피는 더듬는 분류기와 BertViz
- [피처 역변환(Feature Inversion)](model_specific/feature_inversion.md) -- 가운데 나타냄에서 입력을 되세워 모델이 무엇을 보는지 드러내기

---

## 7. 따지기

풀이의 성능과 미더움을 엄밀히 재는 방법.

- 미더움 따지기 -- 풀이가 참 모델의 액션을 드러내는지 재는 넣기/빼기 곡선
- [안정성 평가](evaluation/stability.md) -- 풀이가 얼마나 든든한지 재는 입력에 견준 든든함과 가장 큰 예민함
- [포괄성 평가](evaluation/comprehensiveness.md) -- 넉넉함 점수로 풀이가 중요한 피처를 모두 담았는지 따지기
- 사람을 가운데 둔 따지기 -- 앞으로 흉내내기, 믿음 맞추기, 쓰는 이 연구 피처

---

## 8. 금융 응용

계량 금융의 자리에 풀이 방법을 쓰기.

- 규정 따르기 -- SR 11-7, GDPR 22조, 그리고 금융 모델을 풀이하는 짜기 피처
- 거래 신호 살피기 -- 사고파는 신호를 피처 클래스로 쪼개어 살피고 지켜보기
- [요인 기여도 분석](finance/factor_attribution.md) -- 자본 모듈의 돌아옴을 구조에 매인 인자의 몫으로 쪼개는 SHAP 기초 길
- 신용 위험 풀이 -- 신용 판단에 대한 불리한 처분 까닭과 규정에 맞는 풀이

---

## 정리하며

이 절에서는 기초, 그래디언트 방법, 어텐션 그림 그리기, 피처 기여도 분석을 살펴보았다.
