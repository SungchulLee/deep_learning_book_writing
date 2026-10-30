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

<div class="storybox" markdown>

**그림으로 보는 AI 1.** 문제가 먼저 있었고 답은 한참 뒤에 몰려 왔다

이 장의 여섯 절은 저마다 다른 길을 다루지만, 그 길들이 **언제** 났는지를 한 줄에 놓으면 모양이 하나 보인다. 아래 연표의 날짜는 모두 이 장의 각 쪽 상자에서 출처를 직접 확인한 것이다.

![왼쪽에서 오른쪽으로 이어지는 시간선에 아홉 개의 점이 놓여 있다. 1989년 파국적 간섭에 이름이 붙고, 1995년 리허설과 유사 리허설이 나오며, 그 뒤 2015년 증류와 T 제곱, 2016년 Learning without Forgetting, 2017년 EWC, 2017년 GEM과 BWT 및 FWT 지표, 2018년 노트와 답장, 2019년 A-GEM, 2019년 supermask가 잇따른다. 아래에 삼십 년이고 첫 항목은 방법이 아니라 문제라고 적혀 있다.](figures/chapter_timeline.svg)

**맨 앞에 있는 것은 방법이 아니라 문제다.** 1989년에 붙은 것은 해법의 이름이 아니라 **증상의 이름**이었다. 그리고 1절이 평가 지표를 방법보다 먼저 다루는 까닭도 거기에 있다 — 무엇이 문제인지를 재는 자가 먼저 있어야 한다.

**그 다음으로 눈에 띄는 것은 몰려 있다는 것이다.** 아홉 가운데 일곱이 2015년부터 2019년 사이다. 3절부터 6절까지 — 정규화·리플레이·증류·구조 — 가 **거의 같은 몇 해에 한꺼번에** 나왔다. 그래서 이 네 갈래는 서로를 보고 고친 후계자들이 아니라, 같은 문제를 거의 동시에 다른 쪽에서 민 **형제들**에 가깝다. 어느 하나가 앞선 것을 대체했다고 읽으면 안 되는 까닭이고, 3절부터 6절을 순서가 아니라 **분류**로 늘어놓은 것도 그래서다.

**다만 이 연표는 이 장이 다루는 것만 담는다.** 1995년과 2015년 사이가 비어 보이지만 그 사이에 아무 일도 없었다는 뜻이 아니다 — 1절의 참고 문헌에 적힌 1999년의 개관처럼, 상자를 달지 않은 연구가 그 자리에 있다. 연표가 보여 주는 것은 학문의 역사가 아니라 **이 장의 목차가 어디서 왔는가**다.

_확인한 곳 — 연표의 아홉 항목은 모두 이 장의 각 쪽에 달린 상자의 `확인한 곳` 에서 가져왔다. 그림을 그리는 코드는 각 항목이 가리키는 쪽이 실제로 있고 그 쪽에 상자가 들어 있는지를 먼저 확인한 뒤에만 그리므로, 기억으로 적은 날짜는 들어갈 수 없다. 가로 간격은 해 수에 비례하지 않으며 그림 안에 그렇게 적어 두었다._

</div>

## 정리하며

과거 성능 유지와 새 과제 학습을 함께 평가한다. 웨이트·출력·표본·경로 중 무엇을 보존하는지, 어떤 데이터와 과제 정보를 사용하는지 명시한다. 초기화·순서·표본 노출 예산과 추가 저장·계산을 맞추고 검증용 설정 선택과 최종 테스트를 분리한다.
