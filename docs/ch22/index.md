# 23장: 자연어 처리

자연어 처리에서는 텍스트의 표현뿐 아니라 무엇을 예측하고 어떤 단위로 평가하는지가 중요하다. 이 장은 언어 모델링과 개체명 인식에서 출발해 감성 분류·정보 추출·번역·질의응답·토픽 모델링을 살펴본다. 각 예제의 전처리·출력·학습 범위와 구현상의 제약을 본문에서 함께 설명한다.

## 1. 언어 모델링

다음 토큰의 확률과 텍스트 생성의 선택 규칙을 구분한다. 어휘·문장 경계·인과성·padding 및 같은 기준의 평가가 핵심이다.

- [n-그램 언어 모델](language_modeling/ngram.md)과 [빈도 모델 기초](language_modeling/tutorial_01_ngram_basics.md): 문맥·종료 사건과 미관측 조합
- [확률 평활화](language_modeling/tutorial_02_ngram_smoothing.md)와 [n-그램 생성](language_modeling/tutorial_03_text_generation_ngrams.md): 정규화·보간·후보 선택
- [신경 언어 모델](language_modeling/neural_lm.md): [순방향 모델](language_modeling/tutorial_04_feedforward_lm.md), [RNN](language_modeling/tutorial_05_rnn_language_model.md), [LSTM](language_modeling/tutorial_06_lstm_language_model.md), [트랜스포머](language_modeling/tutorial_07_transformer_lm.md)
- [사전 학습 모델 파인튜닝](language_modeling/tutorial_08_pretrained_lm.md)과 [처음부터 GPT 사전 학습](language_modeling/pretrain_gpt_from_scratch.md): 가중치 초기화·자료 구성·학습의 범위
- [샘플링 전략](language_modeling/sampling_strategies.md), [조건부 생성](language_modeling/tutorial_09_conditional_generation.md), [언어 모델 평가](language_modeling/tutorial_10_lm_evaluation.md)
- [wav2vec 2.0](language_modeling/10_wav2vec.md): 음성 표현의 자기지도 목적과 축약 구조

## 2. 시퀀스 레이블링과 개체명 인식

개체의 구간·유형을 찾고 단어와 subword의 레이블을 정렬한다. 토큰 정확도와 개체 구간 F1은 서로 다른 지표다.

- [개체명 인식 기초](sequence_labeling/ner_fundamentals.md), [개체 유형](sequence_labeling/entity_types.md), [BIO 레이블](sequence_labeling/bio_tagging.md)
- [기초 예제](sequence_labeling/01_ner_basics.md), [IOB 검토](sequence_labeling/02_iob_tagging.md), [규칙](sequence_labeling/03_rule_based_ner.md), [사전](sequence_labeling/04_dictionary_ner.md), [특징 추출](sequence_labeling/05_feature_extraction.md)
- [CRF](sequence_labeling/crf.md)와 [CRF 예제](sequence_labeling/06_crf_ner.md), [BiLSTM](sequence_labeling/bilstm_ner.md), [BiLSTM-CRF 구조](sequence_labeling/10_bilstm_crf_ner.md), [트랜스포머 구조](sequence_labeling/11_transformer_ner.md)
- [평가 원칙](sequence_labeling/ner_evaluation.md), [평가 예제](sequence_labeling/07_evaluation_metrics.md), [데이터셋 구성](sequence_labeling/08_dataset_creation.md)
- [파이프라인 골격](sequence_labeling/15_production_ner.md)과 [통합 시연](sequence_labeling/demo_complete_ner.md): 실제 구현과 미구현 부분 구분

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
- [LSA와 LDA 토픽 모델링](topic_modeling/topic_modeling_lda_lsa.md): 저차원 성분·확률 모형·신경 잠재 표현

## 정리하며

전처리·정답 정의·모델 인터페이스·평가 프로토콜이 맞아야 자연어 처리의 결과를 해석할 수 있다. 구조 선언이나 안내 문자열을 완성된 학습·추론의 증거로 취급하지 않고, 동일 조건의 독립 평가로 실제 성능을 확인한다.
