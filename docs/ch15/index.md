# 16장: 퓨샷 학습과 제로샷 학습

퓨샷 학습은 새 과제의 소수 라벨 표본을 활용하고, 제로샷 학습은 클래스 속성이나 텍스트 설명으로 라벨 있는 표본이 없는 클래스를 예측한다. 이 장에서는 두 문제의 데이터 구성과 평가 조건을 정의하고 거리 기반 방법, 경사 기반 적응, 시각·의미 정렬을 살펴본다.

## 1. 퓨샷 학습과 에피소드

- [퓨샷 학습의 기초](few_shot/few_shot_overview.md): N-way K-shot, 서포트·쿼리와 평가 클래스 분리
- [데이터 로더](few_shot/data_loader.md): 에피소드 추출과 로컬 라벨 매핑
- [평가](few_shot/evaluation.md): 에피소드 평균, 신뢰구간과 클래스별 집계

## 2. 거리와 어텐션 기반 분류

- [샴 네트워크 예제](few_shot/siamese_network.md): 공유 인코더와 거리 학습
- [프로토타입 네트워크 예제](few_shot/prototypical_networks.md): 서포트 평균과 쿼리 거리
- [매칭 네트워크](few_shot/matching_networks.md): 개별 서포트의 유사도 웨이트와 라벨 합산
- [퓨샷 트랜스포머](few_shot/few_shot_transformer.md): 표본 간 문맥과 집합 요약
- [샴 네트워크 상세](metric_learning/siamese.md): 로스, 쌍 샘플링과 원샷 분류
- [프로토타입 네트워크 상세](metric_learning/prototypical.md): 평균의 수학적 성질과 확장

## 3. 메타 학습과 최적화

- [MAML 예제](few_shot/maml.md): 함수형 내부 갱신과 메타 경사
- [MAML 상세](meta_learning/maml.md): MAML·FOMAML·Reptile과 변형 구조
- [학습 기반 옵티마이저](meta_learning/learned_optimizers.md): 갱신 규칙을 학습하는 외부 목표
- [과제 분포 설계](meta_learning/task_distribution.md): 과제 샘플링과 독립적인 평가

## 4. 제로샷 학습

- [제로샷 학습의 기초](zero_shot/zero_shot_overview.md): 학습·미학습 클래스와 의미 정보
- [속성 기반 방법](zero_shot/attribute_based.md): DAP·IAP의 전이 경로
- [의미 임베딩 방법](zero_shot/embedding_based.md): 텍스트 벡터, 호환성 학습과 ConSE
- [제로샷 분류](zero_shot/classification.md): 시각·의미 투영과 비교 로스
- [일반화된 제로샷 학습](zero_shot/generalized.md): 전체 후보 클래스 예측과 점수 보정

## 정리하며

적은 라벨로 학습한다는 조건만으로 방법들의 평가가 같아지지는 않는다. 클래스와 표본의 분할, 쿼리 집합 활용 여부, 사전 학습 데이터와 적응 예산을 기록한다. 예제에는 개념 설명용 무작위 데이터와 미완성 변형도 포함되므로 각 페이지의 구현 범위를 확인한다.
