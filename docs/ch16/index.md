# 17장: 지속 학습

지속 학습은 순차적으로 바뀌는 데이터에서 새 과제를 배우면서 유지해야 할 이전 능력을 보존하는 문제다. 이 장에서는 파국적 망각과 평가 지표를 정의하고 정규화·리플레이·증류·구조 분리의 접근을 비교한다.

## 1. 문제와 평가

- [파국적 망각](continual_learning/catastrophic_forgetting.md): 순차 갱신의 영향과 경사 간섭
- [평가 지표](continual_learning/evaluation_metrics.md): 정확도 행렬, 후방·전방 전이와 망각

테스트 시 과제 ID를 주는지, 전체 누적 클래스 중에서 예측하는지, 표본을 몇 번 학습하는지에 따라 평가가 달라진다. 아래 Split MNIST 예제는 과제별 숫자 쌍을 로컬 이진 라벨로 다시 매기는 설정이며 전체 10개 클래스 증분 분류와 구별한다.

## 2. 순차 학습 예제

- [망각 관찰](continual_learning/01_catastrophic_forgetting_demo.md): 이전 과제 성능 변화 기록
- [단순 순차 학습](continual_learning/01_naive_sequential_learning.md): 추가 보존 장치가 없는 기준
- [경험 리플레이](continual_learning/01_simple_experience_replay.md): 버퍼와 과거 표본 학습
- [EWC](continual_learning/02_elastic_weight_consolidation.md): 경험적 피셔와 가중 이차 벌점
- [LwF](continual_learning/02_learning_without_forgetting.md): 고정 교사 출력 증류
- [방법 비교](continual_learning/03_comprehensive_comparison.md): 지표·시간·메모리 측정 범위

## 3. 정규화 기반 방법

- [개요](regularization_methods/overview.md): 중요도와 기준 웨이트
- [EWC 상세](regularization_methods/ewc.md): 베이즈 근사와 표본별 피셔
- [온라인 EWC](regularization_methods/online_ewc.md): 과제별 통계의 누적 요약
- [EWC·SI·MAS 비교](regularization_methods/comparison.md): 중요도 정의와 비용

## 4. 리플레이 기반 방법

- [경험 리플레이 상세](replay_methods/experience_replay.md): 버퍼 전략과 로스 웨이트
- [A-GEM](replay_methods/agem.md): 참조 평균 경사와 충돌하는 방향의 투영

## 5. 증류 기반 방법

- [LwF 상세](distillation_methods/lwf.md): 새 입력에서 이전 출력 유지
- [피처 증류](distillation_methods/feature_distillation.md): 중간 표현의 정렬과 로스 연결
- [증류 방법 비교](distillation_methods/comparison.md): 출력·피처·어텐션·관계의 제약

## 6. 구조 기반 방법

- [Expert Gate](architecture_methods/expert_gate.md): 과제별 전문가와 재구성 기반 선택
- [Supermask](architecture_methods/supermask.md): 고정 웨이트에서 과제별 부분 네트워크 선택

## 정리하며

과거 성능 유지와 새 과제 학습을 함께 평가한다. 웨이트·출력·표본·경로 중 무엇을 보존하는지, 어떤 데이터와 과제 정보를 사용하는지 명시한다. 초기화·순서·표본 노출 예산과 추가 저장·계산을 맞추고 검증용 설정 선택과 최종 테스트를 분리한다.
