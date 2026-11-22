# 43장: 적대적 강건성

이 장은 딥러닝의 적대적 강건성을 두루 다룬다. 적대적 샘플이 발견된 경위와 그 이론, 화이트박스에서 현실 세계에 이르는 공격 방법, 적대적 트레인과 강건성 인증을 아우르는 방어, 그리고 제대로 평가하는 관행을 담는다. 적대적 위협이 시장 조작, 사기 탐지 회피, 모델 추출로 드러나는 금융 응용을 특히 자세히 살핀다.

---

## 1. 기초

적대적 강건성의 핵심 이론, 위협 모델, 수학적 틀.

- [적대적 강건성 입문](foundations/introduction.md) -- 적대적 샘플이 발견된 경위, 그것이 존재하는 까닭에 대한 여러 가설, 딥러닝에 주는 함의
- [위협 모델](foundations/threat_models.md) -- 화이트박스, 블랙박스, 현실 세계에 걸쳐 적대자의 지식, 능력, 목표를 형식화하기
- [섭동 유형](foundations/perturbations.md) -- L-무한, L-2, L-1 노름의 제약과 그 형태, 그리고 투영 연산
- [강건성 정의](foundations/definitions.md) -- 점별 강건성, 최소 적대적 섭동, 재사용할 수 있는 PyTorch 공격 기반 클래스

---

## 2. 화이트박스 공격

모델의 파라메터와 구조를 **온전히 아는 채로** 거는 그래디언트 기반 공격이다. 아래 네 묶음이 공격이고, 다섯째 묶음의 두 쪽은 방어라 5·6절과 겹친다.

**한 스텝 — FGSM**

- [FGSM](white_box/fgsm.md) — 선형 근사에서 나온 한 스텝 공격(Goodfellow 외, 2015). 그래디언트를 **한 번만** 셈한다 · [기초 실습](adversarial/01_fgsm_basic.md)

**여러 스텝 — PGD**

- [PGD 공격](white_box/pgd.md) — FGSM 을 여러 스텝으로 넓힌 것(Madry 외, 2018). 강건성을 재는 **사실상의 기준**이 되었다 · [실습](adversarial/02_pgd_attack.md)

**최적화로 — C&W**

- [Carlini-Wagner(C&W) 공격](white_box/cw_attack.md) — 적대적 샘플 만들기를 **제약 없는 최적화**로 다시 적는다(2017) · [실습](adversarial/03_cw_attack.md)

**그 밖의 공격**

- [DeepFool](white_box/deepfool.md) — 가장 가까운 결정 경계를 넘는 **최소 섭동**을 찾는다
- [AutoAttack](white_box/autoattack.md) — 하이퍼파라메터를 맞추지 않는 앙상블. 강건성 보고의 기준으로 쓰인다

**막는 쪽**

- [적대적 트레인](adversarial/04_adversarial_training.md) — 가장 잘 듣는 방어. 본격적인 것은 **5절**이다
- [인증 가능한 방어](adversarial/05_certified_defenses.md) — 증명할 수 있는 보장. 본격적인 것은 **6절**이다

**글에 거는 공격**

- [자연어 모델에 대한 적대적 공격](adversarial/nlp_adversarial.md) — 이 갈래는 이미지에서 시작했고, 텍스트는 이산이라 조건이 다르다

---

## 3. 블랙박스 공격

모델 내부를 직접 보지 못한 채 질의 응답이나 전이만으로 하는 공격.

- [전이 기반 공격](black_box/transfer.md) -- 적대적 섭동이 모델을 넘나들며 전이되는 것을 쓰는 질의 없는 공격
- [쿼리 기반 공격](black_box/query_based.md) -- 점수 기반과 결정 기반으로 나뉘는, 모델에 질의하며 반복하는 공격
- [점수 기반 공격](black_box/score_based.md) -- 모델 내부를 보지 않고 출력 확률 분포로 그래디언트를 근사하기
- [결정 기반 공격](black_box/decision_based.md) -- 하드 레이블만 얻는 가장 제한된 블랙박스 환경에서의 공격

---

## 4. 현실 세계 공격

실제 환경에 놓여도 효과가 유지되는 적대적 섭동.

- [적대적 패치](physical/patches.md) -- 현실 세계에서 분류기를 속이는, 국소 영역에 몰린 인쇄 가능한 섭동
- [3차원 적대적 공격](physical/3d_attacks.md) -- 보는 각도를 바꿔도 효과가 유지되는, 물체의 형태·텍스처·조명에 준 섭동
- [실제 환경에서의 강건성](physical/real_world.md) -- 변환에 대한 기댓값(EOT)으로 디지털과 현실 세계의 간극 메우기

---

## 5. 적대적 트레인

트레인할 때 적대적 샘플을 섞는 방어.

- [적대적 트레인](adversarial_training/standard.md) -- 적대적 공격에 가장 잘 듣는 방어인 PGD 기반의 강건 트레인
- [TRADES](adversarial_training/trades.md) -- 클린 정확도와 적대적 강건성 사이의 트레이드오프를 이론에 근거해 조절하기
- [Free Adversarial Training](adversarial_training/free_at.md) -- 그래디언트를 재사용해 일반 트레인에 가까운 비용으로 맞먹는 강건성 얻기
- [Fast Adversarial Training](adversarial_training/fast_at.md) -- 무작위 초기화로 파국적 과적합을 막는 한 스텝 FGSM 트레인
- [MART](adversarial_training/mart.md) -- 오분류된 샘플을 따로 다뤄 어려운 샘플에 방어를 집중하는 트레인

---

## 6. 인증 가능한 방어

어떤 공격이 와도 일정한 반지름 안에서는 강건성을 증명해 주는 방어.

- [무작위 평활화](certified/randomized_smoothing.md) -- 가우스 잡음에 대해 평균을 내어 어떤 분류기든 강건성 인증을 지니게 하기
- [구간 경계 전파(IBP)](certified/ibp.md) -- 네트워크 레이어를 지나며 구간 바운드를 전파해 L-무한 강건성 인증하기
- [CROWN](certified/crown.md) -- 선형으로 완화해 IBP보다 타이트한 바운드를 주는 볼록 완화 기반의 인증
- [립시츠 제약 뉴럴 네트워크](certified/lipschitz.md) -- 네트워크의 민감도를 립시츠 상수로 제한해 강건성 인증하기

---

## 7. 탐지하는 방법

적대적 입력이 분류기에 닿기 앞서 탐지.

- [통계적 탐지](detection/statistical.md) -- 마할라노비스 거리와 활성화 분포 분석으로 적대적 샘플 탐지
- [Feature Squeezing](detection/feature_squeezing.md) -- 원래 입력과 복잡도를 줄인 입력의 예측을 비교하여 적대적 섭동 탐지하기
- [입력 변환 기반 방어](detection/input_transform.md) -- JPEG 압축, 무작위 크기 변환, 디노이징으로 입력 정화하기

---

## 8. 평가

적대적 강건성을 정직하게 평가하는 관행.

- [적응적 공격과 그래디언트 마스킹](evaluation/adaptive.md) -- 실제 강건성 대신 그래디언트를 가리는 방어를 탐지하고 무력화하기
- [평가와 벤치마크](evaluation/benchmarks.md) -- AutoAttack과 RobustBench을 아우른 표준화된 평가 절차
- [인증 정확도](evaluation/certified_accuracy.md) -- 공격의 세기와 상관없이 증명되는 강건성의 하한

---

## 9. 금융 응용

금융 머신러닝 시스템에서의 적대적 강건성.

- [사기 탐지 모델의 강건성](finance/fraud.md) -- 회피 전략을 바꿔 가며 적응하는 적대자에게도 버티는 사기 탐지기 만들기
- [시장 조작 탐지](finance/manipulation.md) -- 허수 주문(스푸핑), 통정 거래, 신호 조작을 적대적 공격으로 보고 탐지
- [금융 시스템의 모델 보안](finance/security.md) -- 모델 추출, 데이터 포이즈닝, 강건한 배포를 아우르는 엔드투엔드 보안

---

## 10. 더 볼 거리

보충 자료와 코드 예제.

- 적대적 강건성 개관 -- 공격, 방어, 평가 기법을 아우르는 개관
- [자연어 모델에 대한 적대적 공격](adversarial/nlp_adversarial.md) -- 텍스트 기반 금융 모델에 대한 문자, 단어, 문장 수준의 공격

---

## 정리하며

이 장은 기초에서 출발해 화이트박스·블랙박스·현실 세계 공격, 적대적 트레인과 인증 가능한 방어, 탐지, 정직한 평가 관행, 그리고 금융 응용까지를 살펴본다.
