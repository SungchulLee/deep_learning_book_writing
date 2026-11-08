# 23장: 자연어 처리

자연어 처리에서는 텍스트의 표현뿐 아니라 무엇을 예측하고 어떤 단위로 평가하는지가 중요하다. 이 장은 언어 모델링과 개체명 인식에서 출발해 감성 분류·정보 추출·번역·질의응답·토픽 모델링을 살펴본다. 각 예제의 전처리·출력·학습 범위와 구현상의 제약을 본문에서 함께 설명한다.

## 1. 언어 모델링

다음 토큰의 확률과 텍스트 생성의 선택 규칙을 구분한다. 어휘·문장 경계·인과성·padding 및 같은 기준의 평가가 핵심이다.

**세 가지 틀**

- [n-그램 언어 모델과 평활화](language_modeling/ngram.md) — 최근 $n-1$ 토큰의 빈도로 세운 고전 기준
- [뉴럴 네트워크 언어 모델](language_modeling/neural_lm.md) — 문맥마다 따로 세는 대신 **함수를 공유한다**
- [텍스트 생성의 디코딩·샘플링 전략](language_modeling/sampling_strategies.md) — 분포를 만드는 일과 토큰을 **고르는** 일은 다르다

**빈도로 — n-그램**

- [n-그램 기초](language_modeling/tutorial_01_ngram_basics.md) — 학습이 아니라 **최대우도 빈도 계산**이다
- [확률 평활화](language_modeling/tutorial_02_ngram_smoothing.md) — 미관측 전이 하나가 문장 전체 확률을 0으로 만든다
- [n-그램으로 텍스트 생성](language_modeling/tutorial_03_text_generation_ngrams.md) — 같은 빈도라도 선택 규칙이 결과를 바꾼다

**배우는 모델로**

- [순방향 뉴럴 언어 모델](language_modeling/tutorial_04_feedforward_lm.md) — Bengio 의 고정 창
- [RNN 언어 모델](language_modeling/tutorial_05_rnn_language_model.md) · [LSTM 언어 모델](language_modeling/tutorial_06_lstm_language_model.md) — 창을 떼고 은닉 상태로
- [트랜스포머 언어 모델](language_modeling/tutorial_07_transformer_lm.md) — 위치 표현과 인과 마스크

**사전 학습·생성·평가**

- [사전 학습 언어 모델의 파인튜닝](language_modeling/tutorial_08_pretrained_lm.md) — GPT-2 를 불러 이어 학습한다
- [GPT 모델을 처음부터 사전 학습하기](language_modeling/pretrain_gpt_from_scratch.md) — BPE 토크나이저부터
- [조건부 텍스트 생성과 디코딩](language_modeling/tutorial_09_conditional_generation.md) — greedy·top-k·top-p·빔 서치
- [언어 모델 평가](language_modeling/tutorial_10_lm_evaluation.md) — 퍼플렉서티와 생성 품질은 **다른 물음**이다

**곁가지 — 음성**

- [wav2vec 2.0](language_modeling/10_wav2vec.md) — 글이 아니라 음성의 자기지도 학습이다

## 2. 시퀀스 레이블링과 개체명 인식

개체의 구간·유형을 찾고 단어와 subword의 레이블을 정렬한다. 토큰 정확도와 개체 구간 F1은 서로 다른 지표다.

**무엇을 푸는 과제인가**

- [개체명 인식 기초](sequence_labeling/ner_fundamentals.md) · [개체 유형과 주석 체계](sequence_labeling/entity_types.md) — 무엇을 개체로 볼지는 **체계가 정한다**
- [정확 span 과 집계 기준](sequence_labeling/ner_evaluation.md) — 토큰 정확도와 개체 구간 F1 은 **세는 대상이 다르다**

**경계를 태그로 적기**

- [BIO와 BIOES](sequence_labeling/bio_tagging.md) — 같은 유형이 붙어 있을 때도 경계를 갈라야 한다
- [기초 예제](sequence_labeling/01_ner_basics.md) — 주석을 자료 구조로 두고 IOB2 를 붙인다(자동 추출이 아니다)
- [IOB2·BIOES 변환](sequence_labeling/02_iob_tagging.md) — 경계 규칙 검사와 span 추출

**학습 없이 — 규칙과 사전**

- [규칙 기반](sequence_labeling/03_rule_based_ner.md) — 정규식·우선순위. 역전파가 없다 · [사전 기반](sequence_labeling/04_dictionary_ner.md) — **문맥을 쓰지 않는다**

**고전 — 피처와 CRF**

- [조건부 랜덤 필드](sequence_labeling/crf.md) — 위치별 점수 + 인접 전이 점수
- [수작업 피처 추출](sequence_labeling/05_feature_extraction.md) — 순수 Python · [CRF 개체명 인식](sequence_labeling/06_crf_ner.md) — `sklearn-crfsuite`

**뉴럴로**

- [BiLSTM을 이용한 개체명 인식](sequence_labeling/bilstm_ner.md) — 좌우 문맥을 합친다
- [BiLSTM-CRF 구성의 시작](sequence_labeling/10_bilstm_crf_ner.md) — 이름과 달리 **전이 모델 부분은 아직 없다**
- [사전 학습 트랜스포머의 토큰 분류 head](sequence_labeling/11_transformer_ner.md) — 언어 사전 학습과 NER head 학습은 다른 일이다

**재고 묶어 내보내기**

- [평가 예제](sequence_labeling/07_evaluation_metrics.md) — 토큰 기준과 span 기준이 **같은 이름**을 쓴다
- [데이터셋의 변환과 분할](sequence_labeling/08_dataset_creation.md) · [종합 데모](sequence_labeling/demo_complete_ner.md) — 모두를 학습·평가하지는 않는다
- [파이프라인의 인터페이스 골격](sequence_labeling/15_production_ner.md) — 뼈대이지 서비스가 아니다

## 3. 감성 분류와 모델 해석

문장 전체 감성과 특정 측면의 감성은 다르다. 분류 성능·모델 설명과 업무적 유효성도 각각 평가해야 한다.

- [측면 기반 감성 분석](text_classification/aspect_sentiment.md): 측면 추출과 측면 조건부 분류
- [FinBERT와 금융 텍스트](text_classification/finbert_sentiment.md): 사전 학습 모델 안내와 합성 자료의 작은 모델 구분
- [LIME·SHAP 모델 해석](text_classification/lime_shap_interpretability.md): 삭제 민감도·회귀 계수·설명 기여도의 차이

## 4. 정보 추출

구조화된 기록에는 개체·관계뿐 아니라 출처·시간·사건 상태와 한정 문맥을 보존해야 한다.

- [정보 추출 개요](information_extraction/ie_overview.md)
- [이벤트 추출](information_extraction/event_extraction.md): 트리거와 사건별 인자·역할
- [상호참조 해결](information_extraction/coreference.md): 언급·선행 후보와 클러스터
- [Open IE](information_extraction/open_ie.md): 열린 관계 표현과 문맥 보존

## 5. 번역·질의응답·토픽 모델링

과제마다 출력 공간과 평가 단위가 다르다. 생성과 구간 선택, 확률 분포와 행렬 성분을 구분한다.

- [트랜스포머 기계 번역](machine_translation/transformer_mt.md): 인과·교차 어텐션과 번역 평가
- [추출형 질의응답](question_answering/extractive_qa.md): 유효 답 구간·답 없음·긴 문맥
- [LSA와 LDA 토픽 모델링](topic_modeling/topic_modeling_lda_lsa.md): 저차원 성분·확률 모형·뉴럴 잠재 표현

## 정리하며

전처리·정답 정의·모델 인터페이스·평가 프로토콜이 맞아야 자연어 처리의 결과를 해석할 수 있다. 구조 선언이나 안내 문자열을 완성된 학습·추론의 증거로 취급하지 않고, 동일 조건의 독립 평가로 실제 성능을 확인한다.
