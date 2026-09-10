# 완전한 예제

완전한 예제: 영어에서 프랑스어로 번역하기. 데이터 준비부터 추론까지의 전 과정을 보인다.

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 순차열 모델의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 1. 코드

```python
"""
완전한 예제: 영어에서 프랑스어로 번역하기
데이터 준비부터 추론까지의 전 과정을 보인다
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# ========================================================================
# 메인
# ========================================================================

from encoder import BasicEncoder
from decoder import AttentionDecoder
from seq2seq_model import Seq2SeqAttention
from data_preprocessing import Tokenizer, Vocabulary, ParallelDataset, split_data
from train import Seq2SeqTrainer, collate_fn, initialize_weights, count_parameters
from inference import Seq2SeqInference, BLEU


def main():
    """주된 학습과 평가의 흐름"""
    
    print("=" * 60)
    print("Seq2Seq Translation Example: English to French")
    print("=" * 60)
    
    # 재현성을 위한 난수 시드 설정
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ========== 1단계: 데이터 준비 ==========
    print("\n" + "=" * 60)
    print("Step 1: Preparing Data")
    print("=" * 60)
    
    # 예제 영어-프랑스어 병렬 말뭉치
    en_texts = [
        "Hello, how are you?",
        "I am fine, thank you.",
        "What is your name?",
        "My name is John.",
        "Nice to meet you.",
        "Where are you from?",
        "I am from Paris.",
        "Do you speak English?",
        "Yes, I speak English.",
        "What time is it?",
        "It is three o'clock.",
        "I like to read books.",
        "She is a teacher.",
        "He works in a hospital.",
        "We are students.",
        "They are learning French.",
        "I have a cat.",
        "The weather is nice today.",
        "Can you help me?",
        "Of course, I can help you.",
        "Thank you very much.",
        "You are welcome.",
        "Good morning.",
        "Good night.",
        "See you later.",
        "How old are you?",
        "I am twenty years old.",
        "What do you do?",
        "I am a doctor.",
        "Where do you live?",
    ]
    
    fr_texts = [
        "Bonjour, comment allez-vous?",
        "Je vais bien, merci.",
        "Quel est votre nom?",
        "Je m'appelle John.",
        "Enchanté.",
        "D'où venez-vous?",
        "Je viens de Paris.",
        "Parlez-vous anglais?",
        "Oui, je parle anglais.",
        "Quelle heure est-il?",
        "Il est trois heures.",
        "J'aime lire des livres.",
        "Elle est enseignante.",
        "Il travaille dans un hôpital.",
        "Nous sommes étudiants.",
        "Ils apprennent le français.",
        "J'ai un chat.",
        "Il fait beau aujourd'hui.",
        "Pouvez-vous m'aider?",
        "Bien sûr, je peux vous aider.",
        "Merci beaucoup.",
        "De rien.",
        "Bonjour.",
        "Bonne nuit.",
        "À plus tard.",
        "Quel âge avez-vous?",
        "J'ai vingt ans.",
        "Que faites-vous?",
        "Je suis médecin.",
        "Où habitez-vous?",
    ]
    
    print(f"Total samples: {len(en_texts)}")
    
    # 데이터 나누기
    (train_en, train_fr), (val_en, val_fr), (test_en, test_fr) = split_data(
        (en_texts, fr_texts), train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"Training samples: {len(train_en)}")
    print(f"Validation samples: {len(val_en)}")
    print(f"Test samples: {len(test_en)}")
    
    # ========== 2단계: 어휘 만들기 ==========
    print("\n" + "=" * 60)
    print("Step 2: Building Vocabularies")
    print("=" * 60)
    
    # 토큰 나누개 만들기
    tokenizer = Tokenizer(lower=True, remove_punct=False)
    
    # 어휘 만들기
    en_vocab = Vocabulary(max_size=1000, min_freq=1)
    en_vocab.build_vocab(train_en, tokenizer.tokenize)
    
    fr_vocab = Vocabulary(max_size=1000, min_freq=1)
    fr_vocab.build_vocab(train_fr, tokenizer.tokenize)
    
    print(f"English vocabulary size: {len(en_vocab)}")
    print(f"French vocabulary size: {len(fr_vocab)}")
    
    # ========== 3단계: 데이터셋과 데이터로더 만들기 ==========
    print("\n" + "=" * 60)
    print("Step 3: Creating Datasets")
    print("=" * 60)
    
    train_dataset = ParallelDataset(
        train_en, train_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    val_dataset = ParallelDataset(
        val_en, val_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    test_dataset = ParallelDataset(
        test_en, test_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    BATCH_SIZE = 4
    PAD_IDX = en_vocab.pad_idx
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # ========== 4단계: 모델 만들기 ==========
    print("\n" + "=" * 60)
    print("Step 4: Creating Model")
    print("=" * 60)
    
    # 모델의 초매개변수
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    
    # 부호기 만들기
    encoder = BasicEncoder(
        input_size=len(en_vocab),
        embedding_dim=ENC_EMB_DIM,
        hidden_size=HID_DIM,
        num_layers=N_LAYERS,
        dropout=ENC_DROPOUT,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    # 복호기 만들기
    decoder = AttentionDecoder(
        output_size=len(fr_vocab),
        embedding_dim=DEC_EMB_DIM,
        hidden_size=HID_DIM * 2,  # 부호기가 양방향이라 2를 곱한다
        encoder_hidden_size=HID_DIM * 2,
        num_layers=N_LAYERS,
        dropout=DEC_DROPOUT,
        rnn_type='LSTM'
    )
    
    # seq2seq 모델 만들기
    model = Seq2SeqAttention(encoder, decoder, device, PAD_IDX).to(device)
    
    # 가중치 초기화
    initialize_weights(model)
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # ========== 5단계: 모델 학습시키기 ==========
    print("\n" + "=" * 60)
    print("Step 5: Training Model")
    print("=" * 60)
    
    # 학습 설정
    LEARNING_RATE = 0.001
    N_EPOCHS = 20
    CLIP = 1.0
    
    # 최적화기와 손실 함수 만들기
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # 학습기 만들기
    trainer = Seq2SeqTrainer(model, optimizer, criterion, device, PAD_IDX, CLIP)
    
    # 모델을 학습시킨다
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=N_EPOCHS,
        checkpoint_dir='checkpoints',
        teacher_forcing_ratio=0.5,
        save_every=5
    )
    
    # ========== 6단계: 모델 평가하기 ==========
    print("\n" + "=" * 60)
    print("Step 6: Evaluating Model")
    print("=" * 60)
    
    # 추론 객체 만들기
    en_vocab_dict = en_vocab.token2idx
    fr_vocab_dict = fr_vocab.token2idx
    
    inference = Seq2SeqInference(
        model, en_vocab_dict, fr_vocab_dict, device,
        sos_idx=fr_vocab.sos_idx,
        eos_idx=fr_vocab.eos_idx,
        pad_idx=PAD_IDX
    )
    
    # 번역 시험
    print("\nTest Translations:")
    print("-" * 60)
    
    bleu_scores = []
    
    for i, (en_text, fr_text) in enumerate(zip(test_en, test_fr)):
        translation, _ = inference.greedy_decode(en_text, max_len=50)
        
        # BLEU 점수 계산
        bleu = BLEU.compute_bleu(fr_text, translation)
        bleu_scores.append(bleu)
        
        print(f"\nExample {i+1}:")
        print(f"English:    {en_text}")
        print(f"French:     {fr_text}")
        print(f"Predicted:  {translation}")
        print(f"BLEU:       {bleu:.4f}")
    
    avg_bleu = np.mean(bleu_scores)
    print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
    
    # ========== 7단계: 대화식 번역 ==========
    print("\n" + "=" * 60)
    print("Step 7: Interactive Translation")
    print("=" * 60)
    
    # 번역 예제
    test_sentences = [
        "Hello, my friend.",
        "How are you today?",
        "I am learning French.",
        "What is the time?"
    ]
    
    print("\nAdditional Examples:")
    print("-" * 60)
    
    for sent in test_sentences:
        translation, _ = inference.greedy_decode(sent)
        print(f"EN: {sent}")
        print(f"FR: {translation}\n")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## 2. 논의

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 순차열 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 훑으며 핵심 설계 결정을 찾아라. 구체적인 구현 선택 세 가지를 열거하고 각각이 순차열 모델에 알맞은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
완전한 예제 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_complete example():
        model = Complete Example(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 완전한 예제

손실 계산은 모델의 출력을 최적화 목표와 이어 준다.

앞의 연습문제 4개로 직접 확인할 수 있다.
