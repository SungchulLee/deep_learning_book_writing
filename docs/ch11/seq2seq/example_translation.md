# 완전한 예제

영어·프랑스어의 작은 병렬 문장 목록을 사용해 전처리, 모델 구성과 학습·추론 도구를 연결하는 예제다. 이 페이지는 다른 모듈에 의존하므로 전체 파이프라인의 입력·반환 계약을 함께 확인해야 한다.

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
    
    # 모델의 초파라메터
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    
    # 인코더 만들기
    encoder = BasicEncoder(
        input_size=len(en_vocab),
        embedding_dim=ENC_EMB_DIM,
        hidden_size=HID_DIM,
        num_layers=N_LAYERS,
        dropout=ENC_DROPOUT,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    # 디코더 만들기
    decoder = AttentionDecoder(
        output_size=len(fr_vocab),
        embedding_dim=DEC_EMB_DIM,
        hidden_size=HID_DIM * 2,  # 인코더가 양방향이라 2를 곱한다
        encoder_hidden_size=HID_DIM * 2,
        num_layers=N_LAYERS,
        dropout=DEC_DROPOUT,
        rnn_type='LSTM'
    )
    
    # seq2seq 모델 만들기
    model = Seq2SeqAttention(encoder, decoder, device, PAD_IDX).to(device)
    
    # 웨이트 초기화
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
    
    # 최적화기와 로스 함수 만들기
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

30개 문장을 앞에서부터 21개 학습, 4개 검증, 5개 테스트로 나눈다. 어휘는 학습 구간에서만 만들므로 검증·테스트에는 OOV가 생길 수 있다. 매우 작은 자료의 결과를 일반 번역 성능으로 해석할 수 없다. torch와 NumPy 시드는 설정하지만 티처 포싱에서 쓰는 Python random은 설정하지 않는다.

현재 ParallelDataset은 정수 목록을 반환하지만 collate_fn은 pad_sequence에 텐서 목록을 전달하는 것을 기대한다. 따라서 이 두 모듈을 그대로 연결하면 배치 구성 단계에서 맞지 않는다. 데이터셋이나 collate 단계에서 long 텐서로 변환하는 계약이 필요하다.

양방향 인코더의 최종 상태는 2H로 이어 붙여지므로 디코더도 그 피처 크기에 맞춰 설정한다. 학습은 검증 최고 체크포인트를 저장하지만 이 예제는 이를 다시 불러오지 않고 마지막 모델로 테스트한다. 최고 검증 모델의 성능으로 보고해서는 안 된다.

학습 토큰화는 구두점을 분리하지만 추론은 소문자·공백만 사용해 불일치한다. BLEU 도구에는 n-그램 분모 오류가 있고, 참조·출력의 대소문자·구두점 처리도 맞추지 않는다. 문장 BLEU를 평균한 값은 코퍼스 BLEU와 다르다. 따라서 이 코드에 보고되는 점수를 표준 번역 성능으로 사용하면 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
30개 문장을 이 예제의 비율로 나누었을 때 크기와 분할 방식을 설명하라.

</div>

??? success "풀이"
    학습 21개, 검증 4개, 테스트 5개다. 비율에 따른 정수 절삭을 사용하고 데이터를 섞지 않는다. 작은 수와 문장 순서 때문에 분포 차이가 클 수 있으며 학습 사전에 없는 단어도 발생할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
ParallelDataset과 collate_fn 사이의 불일치를 설명하라.

</div>

??? success "풀이"
    데이터셋은 ID의 파이썬 목록을 반환하고 pad_sequence는 텐서를 요구한다. 배치 패딩 전에 long 텐서로 바꾸어야 하며 빈 입력과 PAD ID도 검사한다. 모델 구조가 맞아도 데이터 계약이 다르면 학습까지 도달하지 못한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
최종 번역 평가 전에 점검해야 할 세 항목을 제시하라.

</div>

??? success "풀이"
    평가할 체크포인트가 선택한 상태인지 확인하고, 학습·추론의 토큰화를 맞춘다. 표준 BLEU의 n-그램 발생 수·토큰화와 코퍼스 집계 방식을 사용한다. 현재 작은 예제의 점수로 일반화나 다른 모델 대비 우열을 확정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

번역 파이프라인에서는 데이터 목록·배치 텐서, 토큰화·사전, 체크포인트와 지표의 규약이 모두 맞아야 한다. 이 예제의 연결상 제약을 해소하기 전에는 완성된 평가 결과로 해석할 수 없다.
