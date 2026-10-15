# 시퀀스 모델

문장과 시계열처럼 순서에 따라 의미가 달라지는 데이터를 시퀀스라고 한다. 이 장에서는 단어를 벡터로 표현하는 임베딩에서 시작해 RNN, LSTM과 GRU, Seq2Seq, 어텐션으로 이어지는 모델 구조를 살펴본다. 각 구조가 정보를 어떻게 전달하고, 어떤 제약을 보완하는지에 초점을 맞춘다.

## 1. 단어 임베딩

임베딩은 단어 ID를 학습 가능한 벡터로 바꾼다. 단어 사이의 관계는 벡터를 만드는 것만으로 생기지 않으며, 학습 데이터와 목적 함수에 따라 형성된다.

- [간단한 임베딩](embeddings/01_simple_embeddings.md): 임베딩 조회와 원핫 벡터의 관계
- [Word2Vec](embeddings/01_word2vec_full.md): 단어와 주변 문맥을 이용한 표현 학습
- [CBOW](embeddings/02_cbow_model.md)와 [Skip-gram](embeddings/03_skipgram_model.md): 문맥과 중심 단어를 예측하는 두 방식
- [네거티브 샘플링](embeddings/02_negative_sampling.md): 일부 음성 예제를 이용하는 이진 분류 목적 함수
- [GloVe](embeddings/glove_embedding.md): 단어 동시 출현 통계를 이용한 임베딩
- [사전 학습 임베딩](embeddings/03_pretrained_embeddings.md), [분석](embeddings/04_embedding_analysis.md), [시각화](embeddings/04_embedding_visualization.md)
- [손실 함수 비교](embeddings/01_loss_comparison.md): 다음 단어 예측에서 교차 엔트로피와 음의 로그 우도의 관계

## 2. 순환 신경망

RNN은 이전 시점의 은닉 상태와 현재 입력을 결합해 다음 은닉 상태를 계산한다. 같은 매개변수를 여러 시점에 공유하며 순차 정보를 처리한다.

- [시퀀스 데이터의 기초](rnn/01_sequence_basics.md), [텍스트 전처리](rnn/02_text_preprocessing.md), [시계열 데이터](rnn/03_time_series_basics.md)
- [RNN의 기초](rnn/rnn_fundamentals.md)와 [기본 RNN](rnn/vanilla_rnn.md)
- [시간 역전파(BPTT)](rnn/bptt.md)와 [기울기 소실](rnn/vanishing_gradients.md): 긴 시퀀스의 학습이 어려운 이유
- [양방향 RNN](rnn/07_bidirectional_rnn.md): 앞뒤 문맥을 함께 활용하는 구조
- [언어 모델](rnn/language_models.md)과 [텍스트 생성](rnn/10_text_generation.md)

## 3. LSTM과 GRU

LSTM과 GRU는 게이트로 정보의 유지와 갱신을 조절한다. 기본 RNN의 장기 의존성 학습을 돕지만, 모든 데이터에서 기울기 소실이나 학습 실패를 없애는 것은 아니다.

- [LSTM 구조](lstm_gru/lstm_architecture.md): 입력 게이트, 망각 게이트, 출력 게이트와 셀 상태
- [셀 상태와 기울기 흐름](lstm_gru/cell_state.md), [Peephole 연결](lstm_gru/peephole.md)
- [GRU 구조](lstm_gru/gru_architecture.md): 업데이트 게이트와 리셋 게이트
- [GRU와 LSTM 비교](lstm_gru/gru_vs_lstm.md): 구조, 매개변수 수, 선택 기준
- [LSTM 감성 분류](rnn/05_lstm_sentiment.md)와 [GRU 시계열 예측](rnn/06_gru_time_series.md)

## 4. Seq2Seq 모델

Seq2Seq는 입력 시퀀스를 인코딩한 뒤 디코더로 출력 시퀀스를 생성하는 구조다. 기본 모델의 고정 크기 문맥 벡터가 갖는 제약과, 학습 때의 입력 방식이 추론에 미치는 영향을 살펴본다.

- [Seq2Seq 모델](seq2seq/seq2seq_model.md), [인코더](seq2seq/encoder.md), [디코더](seq2seq/decoder.md)
- [데이터 전처리](seq2seq/data_preprocessing.md)와 [학습](seq2seq/train.md): 패딩, 특수 토큰, 티처 포싱
- [추론](seq2seq/inference.md)과 [길이 정규화](seq2seq/length_normalization.md): 출력 시퀀스 탐색과 점수 비교
- [번역 예제](seq2seq/example_translation.md)와 [프랑스어에서 영어로 번역하는 코드](seq2seq/seq2seq_code.md)

## 5. 어텐션

어텐션은 쿼리와 키의 관련도로 가중치를 구하고, 그 가중치로 값을 결합한다. 디코더가 입력의 여러 위치를 참조하게 하거나, 같은 시퀀스 안의 위치 사이에서 정보를 교환하게 한다.

- [어텐션의 기초](attention/attention_fundamentals.md)
- [스케일 내적 어텐션](attention/scaled_dot_product.md): 내적 점수의 크기를 조절하는 이유
- [셀프 어텐션](attention/self_attention.md)과 [크로스 어텐션](attention/cross_attention.md): 쿼리·키·값의 출처에 따른 구분
- [멀티헤드 어텐션](attention/multi_head_attention.md): 여러 표현 공간에서 정보를 결합하는 방법
- [예제](attention/examples.md): 어텐션 연산의 구현과 해석

## 정리하며

임베딩은 토큰의 표현을, 순환 구조는 시간에 따른 상태 갱신을, Seq2Seq는 시퀀스 간 변환을 담당한다. 어텐션은 위치별 정보를 선택적으로 결합하는 연산이다. 각 예제에서는 입력과 출력의 형태, 학습 목표, 학습과 추론의 차이를 함께 확인해야 한다.
