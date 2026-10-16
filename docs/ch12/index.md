# 트랜스포머

트랜스포머는 어텐션으로 위치 사이의 정보를 결합하고, 위치 표현과 피드포워드 레이어로 시퀀스를 처리한다. 이 장에서는 구조의 기본 요소를 정리한 뒤 사전 학습 모델, 학습·추론과 비전 응용으로 이어 간다. 모델 정의와 학습 결과, 무작위 입력 시연과 실제 평가를 구분하며 읽는다.

## 1. 트랜스포머 구조

Q·K·V의 투영, 마스크, 위치 표현과 정규화가 어떻게 연결되는지 살펴본다. 인코더·디코더는 각 입력과 학습 목표에 따라 다른 접근 제약을 사용한다.

- [트랜스포머 구조](transformer_architecture/transformer_architecture.md)와 [인코더·디코더](transformer_architecture/encoder_decoder.md)
- [셀프 어텐션](transformer_architecture/self_attention.md), [멀티헤드 어텐션](transformer_architecture/multihead_attention.md), [스케일 내적 어텐션](transformer_architecture/scaled_dot_product.md)
- [마스크 어텐션](transformer_architecture/masked_attention.md): 미래 토큰과 패딩 위치의 제외
- [위치 인코딩](transformer_architecture/positional_encoding.md), [사인·코사인 인코딩](transformer_architecture/sinusoidal_encoding.md), [학습 가능한 인코딩](transformer_architecture/learned_encoding.md)
- [LayerNorm](transformer_architecture/layer_norm.md)과 [번역 학습 예제](transformer_architecture/train_translation.md)

## 2. 사전 학습 모델

BERT, GPT와 T5는 공통 구성 요소를 사용하지만 접근 가능한 문맥과 사전 학습 목표가 다르다. 같은 “언어 모델”이라는 이름으로 모든 출력을 해석하지 않는다.

- [BERT](pretrained_models/bert.md): 양방향 문맥과 마스크 토큰 예측
- [GPT](pretrained_models/gpt.md): 인과적 다음 토큰 예측과 생성
- [T5](pretrained_models/t5.md): 텍스트 입력·출력 형식의 인코더·디코더
- [Hugging Face 시작하기](pretrained_models/huggingface_quickstart.md)
- [BERT 분류기](pretrained_models/bert_classifier.md), [GPT 생성기](pretrained_models/gpt_generator.md), [샘플링 전략](pretrained_models/sampling_strategies.md)
- [제로샷 텍스트 분류](pretrained_models/zero_shot_text_classification.md)와 [모델 변형](pretrained_models/variants.md)

## 3. 학습과 추론

정답 문맥을 사용하는 학습과 새 토큰을 하나씩 선택하는 생성은 실행 조건이 다르다. 속도·메모리는 모델 크기와 시퀀스 길이, 장치와 측정 방식에 따라 비교한다.

- [학습과 추론](training_and_inference/training_inference.md)
- [학습 최적화](training_and_inference/training_optimization.md): 러닝레이트, 워밍업과 정규화
- [트랜스포머 모델](training_and_inference/transformer_model.md)
- [RNN 기준 모델](training_and_inference/rnn_baseline.md), [CNN 기준 모델](training_and_inference/cnn_baseline.md), [속도 비교](training_and_inference/benchmark_speed.md)

## 4. 비전 트랜스포머

이미지를 패치 토큰으로 표현하면 트랜스포머를 이미지 과제에도 적용할 수 있다. 패치 크기와 위치 표현, 전역·윈도 어텐션이 연산과 정보 전달에 주는 영향을 살펴본다.

- [ViT](transformers_vision/vit.md)와 [모델 구현](transformers_vision/vit_model.md)
- [패치 임베딩](transformers_vision/patch_embedding.md)과 [위치 임베딩](transformers_vision/position_embeddings.md)
- [Swin Transformer](transformers_vision/swin.md): 윈도와 계층 구조
- [CNN과 ViT 비교](transformers_vision/cnn_vs_vit.md)
- [이미지 분류 학습](transformers_vision/train_image_classification.md)과 [시각화](transformers_vision/visualizations.md)

## 정리하며

어텐션의 계산, 위치·마스크의 규약과 학습 목표를 연결해 트랜스포머를 이해한다. 사전 학습 모델과 새로 만든 모델을 구분하고, 웨이트 시각화를 예측의 완전한 설명으로 해석하지 않는다.
