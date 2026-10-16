# 19장: 확률 그래프 모델

확률 그래프 모델(PGM)은 조건부 독립과 인수분해로 결합분포를 구조화한다. 희소한 그래프는 표현과 추정을 단순하게 만들 수 있지만 정확 추론의 비용이나 인과 방향 식별을 자동으로 해결하지는 않는다. 이 장은 기본 개념에서 베이즈 네트워크 구성·추론·학습까지 연결한다.

## 1. 기본 개념과 표현

- [확률 그래프 모델의 기초](pgm_foundations/fundamentals.md): 조건부 독립, 방향·무방향 인수분해, 주변·조건부·MAP·MPE 질의
- [분포 연산과 d-분리 예제](pgm_foundations/01_pgm_fundamentals.md): 주변화·조건화와 사슬·포크·충돌점
- [베이즈 네트워크 기초](pgm_foundations/02_bayesian_networks_basics.md): CPT 구성과 순방향 샘플링

## 2. 추론

- [전수 열거와 샘플링](pgm_foundations/04_simple_inference.md): 조건부 확률, MPE, 기각·우도 가중 샘플링
- [변수 제거](pgm_inference/01_variable_elimination.md): 인자 곱·합산, 제거 순서와 유도 폭

## 3. 학습

- [구조 학습](pgm_learning/02_structure_learning.md): 독립성 기반 뼈대 추정, BIC 점수 탐색과 마르코프 동치
- [PGM의 EM 알고리즘](pgm_learning/em_for_pgms.md): 기대 충분통계량, 파라메터 최적화와 단조성의 조건

## 정리하며

그래프가 보장하는 조건부 독립과 특정 CPT의 수치적 독립을 구분한다. 화살표를 인과로 해석하려면 추가 가정이 필요하다. 추론에서는 증거의 확률·상태 공간·인자 크기를, 학습에서는 유한 표본 오류·동치류·목적함수의 범위를 확인한다. 보존된 교육용 구현의 제한을 각 페이지에서 함께 설명한다.
