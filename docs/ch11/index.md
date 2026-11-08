# 시퀀스 모델

문장과 시계열처럼 순서에 따라 의미가 달라지는 데이터를 시퀀스라고 한다. 이 장에서는 [6장](../embeddings/index.md)이 만든 단어 벡터를 받아 RNN, LSTM과 GRU, Seq2Seq, 어텐션으로 이어지는 모델 구조를 살펴본다. 각 구조가 정보를 어떻게 전달하고, 어떤 제약을 보완하는지에 초점을 맞춘다.

## 1. 단어 임베딩은 6장이 다룬다

단어 ID를 학습 가능한 벡터로 바꾸는 일 자체는 **[6장 단어로 보는 임베딩](../embeddings/index.md)**이 처음부터 다룬다 — 간단한 임베딩, Word2Vec(CBOW·Skip-gram·네거티브 샘플링), N-그램으로 배우는 길, 그리고 GloVe와 사전 학습 임베딩까지다.

이 장은 그 벡터들을 **받아서** 순서를 다루는 구조로 넘어간다.

## 2. 순환 뉴럴 네트워크

RNN은 이전 시점의 은닉 상태와 현재 입력을 결합해 다음 은닉 상태를 계산한다. 같은 파라메터를 여러 시점에 공유하며 순차 정보를 처리한다.

이 절의 실습 쪽 여덟은 **안내 메시지만 출력하는 템플릿**이다. 구조와 입출력 약속을 읽는 자리이지 돌려서 수를 얻는 자리가 아니다.

**순환이란 무엇인가**

- [RNN의 기초](rnn/rnn_fundamentals.md) · [기본 RNN](rnn/vanilla_rnn.md) — 엘만 꼴 $h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$
- [시간 역전파(BPTT)](rnn/bptt.md) · [그래디언트 소실](rnn/vanishing_gradients.md) — 긴 시퀀스 학습이 어려운 까닭
- [언어 모델과 시퀀스 모델](rnn/language_models.md) — 시퀀스에 확률을 매긴다

**입력을 준비하기**

- [시퀀스 기초](rnn/01_sequence_basics.md) — 숫자열·문장·기온으로 순서의 역할을 본다
- [텍스트 전처리](rnn/02_text_preprocessing.md) — 토큰화·정수 ID·패딩·임베딩
- [시계열 데이터 준비](rnn/03_time_series_basics.md) — 입력 구간과 예측 시점을 먼저 정한다

**모델 고르기**

- [기본 RNN 시작하기](rnn/04_simple_rnn.md) · [양방향 RNN](rnn/07_bidirectional_rnn.md) — 앞뒤 문맥을 함께
- [LSTM 감성 분석](rnn/05_lstm_sentiment.md) · [GRU 시계열 예측](rnn/06_gru_time_series.md) — 게이트 구조 자체는 12.2절이 다룬다

**길이가 다른 출력으로**

- [기본 Seq2Seq](rnn/08_seq2seq_basic.md) — 입력과 출력의 길이가 달라도 된다
- [RNN과 어텐션](rnn/09_attention_mechanism.md) — 최종 상태 하나에 담는 제약을 푼다
- [순환 모델의 텍스트 생성](rnn/10_text_generation.md) — 고른 토큰을 다시 입력한다

**공통 도구**

- [RNN 유틸리티](rnn/rnn_utils.md) — 네 모델은 **서로 다른 대안**이지 하나로 이어지지 않는다

## 3. LSTM과 GRU

LSTM과 GRU는 게이트로 정보의 유지와 갱신을 조절한다. 기본 RNN의 장기 의존성 학습을 돕지만, 모든 데이터에서 그래디언트 소실이나 학습 실패를 없애는 것은 아니다.

**게이트가 하는 일**

- [LSTM 구조](lstm_gru/lstm_architecture.md) — 입력·망각·출력 게이트와 셀 상태
- [셀 상태와 그래디언트의 흐름](lstm_gru/cell_state.md) — 직접 경로가 있을 뿐 **보존을 보장하지는 않는다**
- [GRU 구조](lstm_gru/gru_architecture.md) — 셀 상태도 출력 게이트도 없다
- [GRU와 LSTM 비교](lstm_gru/gru_vs_lstm.md) — 구조·파라메터 수·고르는 기준
- [Peephole 연결](lstm_gru/peephole.md) — 게이트가 제가 다스릴 셀 상태를 **못 보는** 문제

**NumPy 로 열어 보기**

- [LSTM 이론](lstm_gru/01_lstm_theory.md) · [GRU 이론](lstm_gru/02_gru_theory.md) — 순전파만 적는다. 로스도 역전파도 없다

**PyTorch 로 세워 보기**

- [LSTM](lstm_gru/40_lstm.md) · [GRU](lstm_gru/41_gru.md) — 모양과 파라메터 수만 본다. **학습하지 않는다**

**Seq2Seq 로 잇기**

- [Seq2Seq](lstm_gru/42_seq2seq.md) — 인코더 최종 상태를 디코더에 넘긴다
- [어텐션 Seq2Seq](lstm_gru/43_attention_seq2seq.md) — 가산 어텐션의 문맥을 GRU 디코더에

## 4. Seq2Seq 모델

Seq2Seq는 입력 시퀀스를 인코딩한 뒤 디코더로 출력 시퀀스를 생성하는 구조다. 기본 모델의 고정 크기 문맥 벡터가 갖는 제약과, 학습 때의 입력 방식이 추론에 미치는 영향을 살펴본다.

- [Seq2Seq 모델](seq2seq/seq2seq_model.md), [인코더](seq2seq/encoder.md), [디코더](seq2seq/decoder.md)
- [데이터 전처리](seq2seq/data_preprocessing.md)와 [학습](seq2seq/train.md): 패딩, 특수 토큰, 티처 포싱
- [추론](seq2seq/inference.md)과 [길이 정규화](seq2seq/length_normalization.md): 출력 시퀀스 탐색과 점수 비교
- [번역 예제](seq2seq/example_translation.md)와 [프랑스어에서 영어로 번역하는 코드](seq2seq/seq2seq_code.md)

## 5. 어텐션

어텐션은 쿼리와 키의 관련도로 웨이트를 구하고, 그 웨이트로 값을 결합한다. 디코더가 입력의 여러 위치를 참조하게 하거나, 같은 시퀀스 안의 위치 사이에서 정보를 교환하게 한다.

- [어텐션의 기초](attention/attention_fundamentals.md)
- [스케일 내적 어텐션](attention/scaled_dot_product.md): 내적 점수의 크기를 조절하는 이유
- [셀프 어텐션](attention/self_attention.md)과 [크로스 어텐션](attention/cross_attention.md): 쿼리·키·값의 출처에 따른 구분
- [멀티헤드 어텐션](attention/multi_head_attention.md): 여러 표현 공간에서 정보를 결합하는 방법
- [예제](attention/examples.md): 어텐션 연산의 구현과 해석

## 정리하며

임베딩은 토큰의 표현을, 순환 구조는 시간에 따른 상태 갱신을, Seq2Seq는 시퀀스 간 변환을 담당한다. 어텐션은 위치별 정보를 선택적으로 결합하는 연산이다. 각 예제에서는 입력과 출력의 형태, 학습 목표, 학습과 추론의 차이를 함께 확인해야 한다.
