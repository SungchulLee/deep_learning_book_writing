# 34장: 그래프 뉴럴 넷트웍

그래프 뉴럴 넷트웍(GNN)은 노드 피처와 연결 구조를 함께 이용해 표현을 학습한다. 그래프의 방향·웨이트·중복 엣지, 메시지의 집계 방식, 노드에서 그래프 전체로 정보를 모으는 방법이 계산과 표현력에 영향을 준다. 이 장에서는 수학적 조건과 보존된 코드의 실제 동작을 함께 살펴본다.

## 1. 그래프 기초

[그래프 기초](foundations/graph_basics.md)는 노드·엣지·방향·경로의 정의를 정리한다. [그래프 표현](foundations/representations.md)은 인접 리스트·엣지 목록·희소 행렬·PyG의 데이터 규약을 비교한다. [인접 행렬](foundations/adjacency_matrix.md)에서는 방향별 곱셈과 차수·라플라시안을, [그래프의 성질](foundations/graph_properties.md)에서는 연결성·클러스터링·스펙트럼·구조 구별의 한계를 다룬다.

## 2. 메시지 패싱

[MPNN](message_passing/mpnn.md)은 메시지 생성·순서 불변 집계·노드 갱신·readout의 공통 틀이다. [업데이트 함수](message_passing/update.md)에서는 자기 상태와 이웃 정보를 결합하는 선형·잔차·게이트 구조를 살펴본다. 같은 집계값으로 압축된 정보는 후속 함수가 자동으로 복원할 수 없다.

## 3. 스펙트럼 방법

[스펙트럼 그래프 이론](convolutions/spectral_theory.md)에서 라플라시안의 고유값과 필터를 연결하고, [그래프 푸리에 변환](convolutions/graph_fourier.md)에서 기저와 신호 표현을 살펴본다. [ChebNet](convolutions/chebnet.md)은 다항식 근사로 고유분해 없이 국소 필터를 계산하는 관점과 예제의 고유값 스케일 문제를 다룬다.

## 4. 공간 기반 방법

[GCN](convolutions/gcn.md)은 자기 루프와 차수 정규화를, [GraphSAGE](convolutions/graphsage.md)는 이웃 샘플링과 집계를 사용한다. [GAT](convolutions/gat.md)는 이웃별 어텐션을, [GIN](convolutions/gin.md)은 다중집합 집계의 구별 능력을 다룬다. 표현력 정리는 피처·함수·집계의 가정 아래 해석하며, 모델 이름만으로 실제 데이터 성능을 순위화하지 않는다.

## 5. 깊이와 확장 구조

[심층 GNN](advanced/deep_gnns.md)은 잔차·정규화·DropEdge를, [과도한 평활화](advanced/over_smoothing.md)는 거리·에너지 지표와 선형 전파의 조건을 다룬다. [Jumping Knowledge](advanced/jumping_knowledge.md)는 여러 깊이의 표현을 결합한다.

[그래프 Transformer](advanced/graph_transformers.md)는 구조 인코딩과 국소·전역 어텐션을 비교한다. [이종 그래프](advanced/heterogeneous.md)는 노드·관계 유형을, [시간 그래프](advanced/temporal.md)는 사건 시각·과거 상태를, [하이퍼그래프](advanced/hypergraphs.md)는 여러 노드의 집단 소속을 모델링한다. 각각의 표현에 필요한 정보와 추가 비용을 구분한다.

## 6. 그래프 수준 과제

[그래프 분류](graph_tasks/graph_classification.md)와 [그래프 회귀](graph_tasks/graph_regression.md)는 합성 그래프별 예측을 보여 준다. [풀링](graph_tasks/pooling.md), [계층적 풀링](graph_tasks/hierarchical_pooling.md), [Set2Set](graph_tasks/set2set.md)에서 노드 집합을 그래프 표현으로 모으는 방법을 살펴본다. 평균·합·어텐션이 크기와 구성 정보를 어떻게 보존하는지 목표와 맞추어 판단한다.

## 7. 노드와 링크 과제

[노드 분류](node_link/node_classification.md)는 일부 레이블과 전체 구조·피처를 사용하는 전이적 학습이다. [링크 예측](node_link/link_prediction.md)은 인코더·디코더, 음성 샘플링과 평가 엣지의 누출 방지를 다룬다. [노드 임베딩](node_link/node_embedding.md)은 랜덤 워크 문맥을 학습하고, [커뮤니티 탐지](node_link/community_detection.md)는 임베딩과 K-means로 합성 블록 구조를 복원한다.

## 8. 응용 예제의 범위

[분자 특성 예측](applications/molecular.md)과 [약물–표적 예측](applications/drug_discovery.md)은 합성 피처·목표로 계산 구조를 보여 준다. [소셜 네트워크 분석](applications/social_networks.md)은 합성 계정 그룹 분류다. 출력된 로스·정확도는 실제 화학·계정 데이터의 검증을 대신하지 않는다.

[추천 시스템](applications/recommendation.md)은 LightGCN의 전파와 BPR 학습을, [지식 그래프 완성](applications/knowledge_graphs.md)은 메시지 패싱 없이 TransE 테이블의 점수·순위 계산을 살펴본다. 학습 로스와 홀드아웃 성능, 관측 정보와 미관측 정답을 구분한다.

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 천장과 시험대가 둘 다 바깥에서 왔다

이 장의 쪽마다 붙은 상자들을 해 순서로 늘어놓으면 1968년부터 2019년까지다. 그런데 **가장 오래된 둘이 기계학습이 아니다.**

![왼쪽에서 오른쪽으로 여섯 점이 놓인 시간선. 1968년 바이스파일러와 레만은 그래프의 표준형과 대수, 1977년 재커리는 인류학자가 가라테 동아리의 분열을 관찰한 것이고 이 둘은 점선 왼쪽에 기계학습이 아니라고 묶여 있다. 점선 오른쪽에는 2017년 키프와 벨링의 재정규화, 2017년 길머 등의 여덟 모델 한 틀, 2018년 리·한·우의 평활화가 제 고장에 이름을 붙인 일, 2019년 쉬 등의 천장이 1968년의 것이라는 결과가 놓여 있다. 아래에 천장과 시험대가 둘 다 왼쪽에서 왔다고 적혀 있으며, 해는 고르게 놓았고 실제 간격이 아니라고 적혀 있다.](figures/chapter_years.svg)

**1968년**의 바이스파일러와 레만은 그래프를 표준형으로 줄이는 일과 거기서 나오는 대수를 다루었다. 학습이라는 말이 없다. 그런데 [GIN](convolutions/gin.md) 쪽이 보이듯 2019년에 밝혀진 메시지 패싱의 표현력 **천장이 바로 그 절차**다.

**1977년**의 재커리는 인류학자였고, 어느 대학 가라테 동아리가 갈라지는 과정을 3년 동안 적었다. 그 34명이 오늘 이 장에서 여덟 쪽에 걸쳐 모델을 시험하는 그래프다 — [커뮤니티 탐지](node_link/community_detection.md) 쪽이 그 내력을 다룬다.

그러니 이 분야가 자기 한계를 재는 자와 자기 성능을 재는 자를 **둘 다 밖에서 빌려 왔다.** 2017년부터의 넉 줄은 모두 그 사이를 메우는 일이다 — 전파를 안정시키고([GCN](convolutions/gcn.md)), 흩어진 모델을 한 틀에 넣고([MPNN](message_passing/mpnn.md)), 깊이의 고장에 이름을 붙이고([과도한 평활화](advanced/over_smoothing.md)), 천장이 어디인지 밝힌다.

**빌려 온 자에는 빌려 온 눈금이 따라온다.** 1-WL은 **구분**을 재지 쓸모를 재지 않고, 가라테 클럽의 "정답 둘"은 그래프가 아니라 현장 기록에서 온 레이블이다. 아래 「정리하며」가 구조 정의와 평가 분할을 함께 보라고 적어 둔 까닭이 여기에도 있다 — 이 장의 수를 읽을 때는 **그 잣대가 원래 무엇을 재려고 만들어졌는지**까지 함께 읽어야 한다.

_확인한 곳 — 시간선의 여섯 항목은 모두 이 장의 상자에서 가져왔고, 그림을 그리는 코드는 각 항목에 해당하는 쪽이 실제로 있고 그 쪽에 상자가 있는지를 먼저 확인한 뒤에만 그린다. 해의 간격이 고르지 않아 눈금에 올리지 않고 고르게 놓았으며 그림 안에 그렇게 적었다._

</div>

## 정리하며

그래프 학습의 결과는 구조 정의·정보 접근 범위·모델 연산·평가 분할을 함께 보아야 해석할 수 있다. 보존된 예제의 출력은 실행 기록이며, 이 장의 본문 편집에서 재실행하거나 성능을 재검증하지 않았다.
