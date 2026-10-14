# 43장: 적대적 강건성

이 장은 딥러닝에서 맞섬에 강건한 일을 두루 다룬다. 적대적 샘플을 찾아낸 이야기와 그 이론, 화이트박스에서 참 세상에 이르는 공격 방법, 적대적 학습과 강건성 인증을 아우른 방어, 그리고 제대로 따지는 좋은 버릇을 담는다. 맞섬의 위협이 시장 조작, 속임수 비껴가기, 모델 빼내기로 드러나는 금융 쓰임에 힘주어 살핀다.

---

## 1. 기초

적대적 강건성의 핵심 이론, 위협 모델, 수학 틀.

- [적대적 강건성 입문](foundations/introduction.md) -- 적대적 샘플을 찾아낸 이야기, 그것이 있는 까닭에 대한 여러 짐작, 딥러닝에 걸리는 바
- [위협 모델](foundations/threat_models.md) -- 화이트박스, 블랙박스, 참 세상에 걸쳐 겨루는 이의 앎, 힘, 목표를 꼴로 적기
- [강건성 정의](foundations/definitions.md) -- 점마다의 강건성, 가장 작은 적대적 섭동, 다시 쓸 수 있는 PyTorch 공격 밑 갈래
- [섭동 유형](foundations/perturbations.md) -- L-무한, L-2, L-1 노름의 제약과 그 꼴, 그리고 되비추는 계산

---

## 2. 화이트박스 공격

모델의 매개변수와 구조를 온전히 아는 채로 하는 기울기 기반의 공격.

- [FGSM](white_box/fgsm.md) -- 선형 짐작에서 비롯한, 한 걸음짜리 기초 기울기 공격
- [되비춘 기울기 내림(PGD)](white_box/pgd.md) -- 강건성을 따지는 사실상의 잣대가 된 여러 걸음 되돌이 공격
- [DeepFool](white_box/deepfool.md) -- 가장 가까운 결정 경계를 넘는 가장 작은 섭동을 찾는 꼴 기초의 공격
- [Carlini-Wagner(C&W) 공격](white_box/cw_attack.md) -- 적대적 샘플 만들기를 제약 없는 최소화로 다시 쓴 공격
- [AutoAttack](white_box/autoattack.md) -- 하이퍼파라미터를 맞추지 않고도 강건성을 미덥게 따지는 매개변수 없는 앙상블 공격

---

## 3. 블랙박스 공격

모델 속을 곧바로 보지 못한 채 물음의 되받음이나 전이만으로 하는 공격.

- [전이 기반 공격](black_box/transfer.md) -- 적대적 섭동이 모델을 넘나들며 전이되는 것을 쓰는 물음 없는 공격
- [쿼리 기반 공격](black_box/query_based.md) -- 스코어 기반과 판단 기초의 갈래를 지닌, 모델에 물어 가며 되돌이하는 공격
- [점수 기반 공격](black_box/score_based.md) -- 모델 속을 보지 않고 출력 낌새 분포로 기울기를 어림하기
- [결정 기반 공격](black_box/decision_based.md) -- 굳은 레이블만 얻는 가장 빡빡한 블랙박스 자리에서의 공격

---

## 4. 참 세상 공격

참 자리에 내놓아도 살아남는 적대적 섭동.

- [적대적 패치](physical/patches.md) -- 참 세상에서 분류기를 속이는, 그 자리에 몰린 찍어 낼 수 있는 섭동
- [3차원 적대적 공격](physical/3d_attacks.md) -- 보는 자리를 바꿔도 맞서는, 물체의 꼴·특징·빛에 준 섭동
- [실제 환경에서의 강건성](physical/real_world.md) -- 바뀜에 걸친 바람(EOT)으로 셈틀과 참 세상의 틈 메우기

---

## 5. 적대적 학습

익힐 때 적대적 샘플을 학습에 섞는 방어.

- [여느 적대적 학습](adversarial_training/standard.md) -- 적대적 공격에 가장 잘 듣는 방어인 PGD 기초의 든든하게 하기
- [TRADES](adversarial_training/trades.md) -- 클린 정확도와 맞섬의 강건성 사이를 이론에 닿게 맞바꾸기
- [Free Adversarial Training](adversarial_training/free_at.md) -- 기울기를 되쓰며 여느 학습에 가까운 값으로 맞먹는 강건성 이루기
- [Fast Adversarial Training](adversarial_training/fast_at.md) -- 아무렇게나 첫자리를 잡아 무너지듯 지나치게 맞추는 것을 막는 한 걸음 FGSM 학습
- [MART](adversarial_training/mart.md) -- 잘못 가른 보기를 헤아려 어려운 보기에 방어의 힘을 모으는 학습

---

## 6. 인증 가능한 방어

어떤 공격이 와도 어느 자리 안에서는 강건성을 증명해 주는 방어.

- [아무렇게나 매끄럽게 하기](certified/randomized_smoothing.md) -- 가우스 잡음을 고르게 하여 어떤 분류기든 강건성 인증을 지니게 하기
- [구간 경계 전파(IBP)](certified/ibp.md) -- 네트워크 켜를 지나며 사이 테두리를 퍼뜨려 L-무한 강건성 밝히기
- [CROWN](certified/crown.md) -- 선형으로 눅여 IBP보다 촘촘한 테두리를 주는 볼록 완화 기초의 밝히기
- [립시츠 제약 신경망](certified/lipschitz.md) -- 네트워크가 얼마나 예민한지를 립시츠 값으로 옭아매어 강건성 밝히기

---

## 7. 알아내는 방법

맞서는 입력이 분류기에 닿기 앞서 짚어내기.

- [자로 알아내기](detection/statistical.md) -- 마할라노비스 거리와 활성화 분포 살피기로 적대적 샘플 짚어내기
- [Feature Squeezing](detection/feature_squeezing.md) -- 본디 입력과 얽힘을 줄인 입력의 추론을 비교하여 적대적 섭동에 표시하기
- [입력 변환 기반 방어](detection/input_transform.md) -- JPEG 눌러 담기, 아무렇게나 크기 바꾸기, 잡음 지우기로 입력 맑히기

---

## 8. 따지기

맞섬의 강건성을 솔직히 따지는 좋은 버릇.

- [적응적 공격과 기울기 마스킹](evaluation/adaptive.md) -- 참 강건성 대신 기울기를 가리는 방어를 짚어내고 넘어서기
- [평가와 벤치마크](evaluation/benchmarks.md) -- 오토어택과 RobustBench을 아우른 잣대 잡힌 따짐 절차
- [인증 정확도](evaluation/certified_accuracy.md) -- 공격의 세기와 상관없이 증명되는 강건성의 아래끝

---

## 9. 금융 응용

금융 머신러닝 구조에서의 적대적 강건성.

- [시장 조작 탐지](finance/manipulation.md) -- 속임 주문, 짜고 하는 거래, 꾀 부린 신호 흔들기를 적대적 공격로 보고 짚어내기
- [사기 탐지 모델의 강건성](finance/fraud.md) -- 비껴가는 꾀를 갈아 가며 맞추어 오는 겨루는 이에게도 버티는 속임수 알아내개 짓기
- [금융 구조의 모델 보안](finance/security.md) -- 모델 빼내기, 데이터에 독 타기, 든든히 내놓기를 아우른 끝에서 끝까지의 지킴

---

## 10. 더 볼 거리

곁들일 거리와 코드 보기.

- 적대적 강건성 두루 보기 -- 공격, 방어, 따지는 재주를 아우른 배치 두루 보기
- [자연어 모델에 대한 적대적 공격](adversarial/nlp_adversarial.md) -- 텍스트 기초 금융 모델에 대한 글자, 단어, 월 켜의 공격

---

## 정리하며

이 마당은 기초, 화이트박스 공격, 블랙박스 공격, 참 세상 공격을 차례로 짚었다.
