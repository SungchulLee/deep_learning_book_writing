# 트랜스포머

트랜스포머는 어텐션으로 위치 사이의 정보를 결합하고, 위치 표현과 피드포워드 레이어로 시퀀스를 처리한다. 이 장에서는 구조의 기본 요소를 정리한 뒤 사전 학습 모델, 학습·추론과 비전 응용으로 이어 간다. 모델 정의와 학습 결과, 무작위 입력 시연과 실제 평가를 구분하며 읽는다.

## 1. 트랜스포머 구조

Q·K·V의 투영, 마스크, 위치 표현과 정규화가 어떻게 연결되는지 살펴본다. 인코더·디코더는 각 입력과 학습 목표에 따라 다른 접근 제약을 사용한다.

**무엇으로 되어 있나**

- [트랜스포머 구조 훑어보기](transformer_architecture/transformer_architecture.md) — 병렬 학습의 이점과 어텐션의 계산 비용
- [인코더-디코더 구조](transformer_architecture/encoder_decoder.md) — 인코더 전용·디코더 전용·둘 다가 읽을 수 있는 문맥이 다르다

**블록의 부품**

- [스케일 내적 어텐션](transformer_architecture/scaled_dot_product.md) — 투영 웨이트는 이 함수 **밖**에 있다
- [셀프 어텐션](transformer_architecture/self_attention.md) — `heads` 인수를 받지만 **실제 헤드 분할은 없다**
- [멀티헤드 어텐션](transformer_architecture/multihead_attention.md) — 쪼개고 이어 붙인다. 셀프·크로스 둘 다에 쓴다
- [마스크 셀프 어텐션](transformer_architecture/masked_attention.md) — 인과 마스크와 패딩 마스크
- [트랜스포머의 LayerNorm](transformer_architecture/layer_norm.md) — 잔차 앞에 두느냐 뒤에 두느냐로 **경사 경로가 달라진다**

**위치를 알려 주기**

- [위치 인코딩](transformer_architecture/positional_encoding.md) — 위치 정보가 없으면 셀프 어텐션은 **순열 동변**이다
- [사인·코사인 위치 인코딩](transformer_architecture/sinusoidal_encoding.md) — 고정 · [학습 가능한 위치 인코딩](transformer_architecture/learned_encoding.md) — 표에서 찾는다

**그 앞 세대 — RNN 어텐션**

- [RNN 어텐션 메커니즘](transformer_architecture/attention_mechanisms.md) — Bahdanau 의 가산 점수와 Luong 의 내적 점수
- [어텐션을 결합한 GRU Seq2Seq](transformer_architecture/seq2seq_with_attention.md) — 견줄 순환 모델이지 셀프 어텐션 구조가 아니다
- [번역 학습](transformer_architecture/train_translation.md) — 이름과 달리 **데이터 로딩·로스·갱신을 하지 않는** 준비 예제다

**돌려 보기**

- [시연](transformer_architecture/demo.md) — 무작위 입력의 출력 모양만 본다

## 2. 사전 학습 모델

BERT, GPT와 T5는 공통 구성 요소를 사용하지만 접근 가능한 문맥과 사전 학습 목표가 다르다. 같은 “언어 모델”이라는 이름으로 모든 출력을 해석하지 않는다.

이 절의 구현 쪽은 **거의 다 무작위 초기화**다. 사전 학습 체크포인트를 읽는 것은 마지막 묶음 둘뿐이고, 나머지는 구조를 보이는 자리다.

**무엇을 어떻게 사전 학습하나**

- [BERT](pretrained_models/bert.md) — 양방향 문맥과 마스크 토큰 예측 · [GPT](pretrained_models/gpt.md) — 인과적 다음 토큰 예측
- [T5](pretrained_models/t5.md) — 모든 과제를 **텍스트에서 텍스트로** · [트랜스포머의 변형](pretrained_models/variants.md) — 희소 어텐션과 혼합 모델

**인코더 쪽을 손으로**

- [트랜스포머 인코더](pretrained_models/transformer_encoder.md) — 토큰·위치 임베딩과 블록. 무작위 초기화다
- [BERT](pretrained_models/08_bert.md) — MLM 과 NSP 의 구성 요소를 단순화한 예제
- [BERT 모델](pretrained_models/bert_model.md) — 첫 토큰 풀링 래퍼. **체크포인트를 읽지 않는다**
- [BERT 분류기](pretrained_models/bert_classifier.md) · [감성 분석 학습](pretrained_models/train_sentiment.md) — 뒤엣것은 파라메터 수만 찍는다

**디코더 쪽을 손으로**

- [트랜스포머 디코더](pretrained_models/transformer_decoder.md) — 인과 마스크를 **자동 생성하지 않고** 축도 어긋난다
- [GPT](pretrained_models/09_gpt.md) — 학습 위치 임베딩 + 인과 셀프 어텐션
- [GPT 모델](pretrained_models/gpt_model.md) — 샘플링 루프를 붙였으나 위 디코더의 문제를 물려받는다
- [GPT 생성기](pretrained_models/gpt_generator.md) — 마스킹 없는 교차 어텐션 메모리로 **미래 토큰이 샌다**
- [샘플링 방법](pretrained_models/sampling_strategies.md) — greedy·top-k·top-p. 분포는 바뀌어도 **사실성은 아니다**

**둘을 합친 블록**

- [트랜스포머](pretrained_models/44_transformer.md) — 이름과 달리 기본 클래스에 **디코더와 교차 어텐션이 없다**

**비전 쪽**

- [비전 트랜스포머 ViT](pretrained_models/18_vision_transformer_vit.md) · [스윈 트랜스포머](pretrained_models/19_swin_transformer.md) — 본격적인 것은 4절이다

**사전 학습 웨이트를 실제로 쓰기**

- [허깅페이스 빠른 시작](pretrained_models/huggingface_quickstart.md) — 여기서 처음으로 **진짜 체크포인트**를 읽는다
- [제로샷 텍스트 분류](pretrained_models/zero_shot_text_classification.md) — 새 분야에서 통한다는 보장은 없다

## 3. 학습과 추론

정답 문맥을 사용하는 학습과 새 토큰을 하나씩 선택하는 생성은 실행 조건이 다르다. 속도·메모리는 모델 크기와 시퀀스 길이, 장치와 측정 방식에 따라 비교한다.

- [학습과 추론](training_and_inference/training_inference.md)
- [학습 최적화](training_and_inference/training_optimization.md): 러닝레이트, 워밍업과 정규화
- [트랜스포머 모델](training_and_inference/transformer_model.md)
- [RNN 기준 모델](training_and_inference/rnn_baseline.md), [CNN 기준 모델](training_and_inference/cnn_baseline.md), [속도 비교](training_and_inference/benchmark_speed.md)

## 4. 비전 트랜스포머

이미지를 패치 토큰으로 표현하면 트랜스포머를 이미지 과제에도 적용할 수 있다. 패치 크기와 위치 표현, 전역·윈도 어텐션이 연산과 정보 전달에 주는 영향을 살펴본다.

**ViT 란 무엇인가**

- [비전 트랜스포머 (ViT)](transformers_vision/vit.md) — 패치 토큰과 전역 어텐션. 패치 투영은 **컨볼루션으로도** 구현된다
- [컨볼루션 뉴럴 네트워크와 비전 트랜스포머 비교](transformers_vision/cnn_vs_vit.md) — 설계의 차이를 보되 이름만으로 우열을 정하지 않는다

**부품**

- [패치 임베딩](transformers_vision/patch_embedding.md) — `Conv2d` 하나 뒤에 공간 축을 편다
- [이미지을 위한 위치 임베딩](transformers_vision/position_embeddings.md) — 없으면 패치 **순열 동변**이다

**모델로 조립하기**

- [비전 트랜스포머](transformers_vision/vision_transformer.md) — 패치 임베딩 + PyTorch 인코더 · [비전 트랜스포머 모델](transformers_vision/vit_model.md) — Pre-LN 블록과 Tiny~Large 팩터리

**변형**

- [스윈 트랜스포머](transformers_vision/swin.md) — 국소 윈도와 시프트, 단계별 패치 병합

**학습시키기**

- [이미지 분류 학습](transformers_vision/train_image_classification.md) — 이름은 `train_vit` 이지만 **순전파 모양만 확인한다**
- [비전 트랜스포머 학습](transformers_vision/train_vit.md) — CIFAR-10 에 실제로 학습한다

**들여다보기**

- [예제](transformers_vision/examples.md) — 대부분 **무작위 모델·입력**이다 · [시각화](transformers_vision/visualizations.md) — 그림이 예측의 근거는 아니다

## 정리하며

어텐션의 계산, 위치·마스크의 규약과 학습 목표를 연결해 트랜스포머를 이해한다. 사전 학습 모델과 새로 만든 모델을 구분하고, 웨이트 시각화를 예측의 완전한 설명으로 해석하지 않는다.
