# 24장: 대규모 언어 모델

이 장에서는 대규모 언어 모델의 학습 목표와 평가, 구조와 스케일링, 프롬프트, 검색, 파인튜닝과 추론 최적화를 살펴본다. 모델의 규모가 커질 때 필요한 계산과 메모리를 구분하고, 개념 설명과 예제가 실제로 구현한 범위를 함께 확인한다.

## 1. 학습 목표와 평가

- [사전 학습 목표](llm_foundations/pretraining_objectives.md): 인과 언어 모델링, 마스킹과 잡음 제거, 목표별 입력·정답 구성
- [평가 지표](llm_foundations/evaluation_metrics.md): 퍼플렉서티, 태스크 평가, 생성 품질과 사람·모델 평가의 한계
- [어려움과 한계](llm_foundations/challenges.md): 긴 문맥, 사실성, 바이어스와 개인정보, 외부 도구를 사용하는 시스템의 신뢰 경계

## 2. 모델 구조와 규모

- [구조의 주요 변화](architectures/architectures.md): RMSNorm, SwiGLU, RoPE, MQA·GQA와 슬라이딩 윈도 어텐션
- [GPT 계열](architectures/gpt_series.md): 공개 논문을 바탕으로 살펴보는 학습 방식과 구조의 변화
- [스케일링 법칙](scaling/scaling_overview.md): 로스, 파라미터 수, 데이터 양과 계산 예산의 경험적 관계
- [모델 크기와 데이터의 트레이드오프](scaling/model_vs_data.md): 학습·추론 비용과 반복 데이터의 효과
- [창발적 능력](scaling/emergent_abilities.md): 규모에 따른 성능 변화와 평가 지표가 만드는 불연속성

## 3. 프롬프트와 검색

- [프롬프트 개요](prompting/prompting_overview.md): 지시, 문맥과 예시를 구성하고 결과를 평가하는 방법
- [Chain-of-Thought(CoT)](prompting/chain_of_thought.md): 중간 풀이를 생성하는 프롬프트와 자기 일관성의 조건
- [밀집 검색](rag/dense_retrieval.md): 쿼리·문서 인코더와 대조 학습, 검색 결과를 생성에 연결하는 과정

검색 증강 생성(RAG)에서 검색 품질과 생성의 사실성은 별도로 평가한다. 검색 결과가 존재하거나 출처를 표시한다는 이유만으로 답이 정확해지는 것은 아니다.

## 4. 파라미터 효율적 파인튜닝

- [LoRA](efficient_llm/lora.md): 저랭크 웨이트 업데이트와 병합, 양자화 기반 파인튜닝
- [어댑터](efficient_llm/adapters.md): 직렬·병렬 병목 모듈과 여러 태스크의 적응 모듈 관리
- [Prefix Tuning](efficient_llm/prefix_tuning.md): 학습 가능한 KV 접두부와 입력 프롬프트 튜닝의 차이
- [PEFT 방법 비교](efficient_llm/peft_comparison.md): 학습 파라미터, 실행 메모리와 추론 비용의 비교

## 5. 추론과 모델 압축

- [KV 캐시](inference/kv_cache.md): 이전 키·값을 재사용하는 디코딩과 캐시 메모리
- [FlashAttention](inference/flash_attention.md): 블록 계산과 온라인 소프트맥스로 메모리 입출력 줄이기
- [Speculative Decoding](inference/speculative_decoding.md): 초안 생성, 병렬 검증과 분포를 보존하는 보정
- [모델 압축 개요](inference/model_compression_overview.md): 양자화·가지치기·증류의 역할과 측정 기준
- [언어 모델 양자화](inference/quantization.md): 정수 표현과 스케일, PTQ·QAT, KV 캐시와 혼합 정밀도
- [양자화 인지 학습](inference/04_quantization_aware_training.md): fake quantization 학습과 정수 모델 변환의 차이
- [구조적 가지치기](inference/05_structured_pruning.md): 필터 선택과 연결된 계층의 재구성
- [압축 방법 결합](inference/06_combined_compression.md): 교사 학습, 마스킹과 학생 증류를 연결하는 예제
- [Hugging Face 증류와 벤치마크](inference/07_hf_distillation_benchmark.md): 맞춤 로스와 품질·저장 공간·지연 시간 측정

## 6. 모델 정렬

- [학습 파이프라인](alignment/training_pipeline.md): 사전 학습, 지도 파인튜닝과 선호도 학습, RLHF·DPO의 관계

## 정리하며

대규모 언어 모델을 이해하려면 학습 목표와 평가 방식뿐 아니라 계산·메모리·서비스 비용을 함께 살펴야 한다. 작은 학습 파라미터 수, 적은 저장 비트 수와 짧은 응답 시간은 각각 다른 지표다. 구현의 입력·출력 조건과 평가의 근거를 확인하고, 제한된 예제의 결과를 모든 모델과 환경으로 일반화하지 않는다.
