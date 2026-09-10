# 익히기 2

글자 수준 말 모델을 위한 익히기 대본. 이 대본은 다음을 보인다.

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 1. 코드

```python
"""
글자 수준 말 모델을 위한 익히기 대본

이 대본은 다음을 보인다.
1. 글 자료를 불러와 다듬는다
2. 자기 되돌이 글자 모델 익히기
3. 새 글 표본 만들어 내기
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========================================================================
# 메인
# ========================================================================

from model import CharRNN, SimpleCharTransformer
from data import CharacterDataset, load_sample_text, train_test_split


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: str) -> float:
    """
    한 세대를 학습한다.
    
    인수:
        model: 말 모델
        dataloader: 익히기 자료용 DataLoader
        criterion: 손실 함수(CrossEntropyLoss)
        optimizer: 최적화기
        device: 익힐 기기('cpu'나 'cuda')
        
    반환값:
        바퀴의 평균 손실
    """
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        # 장치로 옮긴다
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # 순전파
        if isinstance(model, CharRNN):
            output, _ = model(batch_x)
        else:  # 변환기
            output = model(batch_x)
        
        # 손실을 계산한다
        loss = criterion(output, batch_y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        
        # 기울기가 터지지 않도록 자른다
        # 이는 되돌이 신경망에서 특히 중요하다
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 가중치를 갱신한다
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            criterion: nn.Module,
            device: str) -> float:
    """
    시험 데이터로 모델을 평가한다.
    
    인수:
        model: 말 모델
        dataloader: 시험 자료용 DataLoader
        criterion: 손실 함수
        device: 평가할 장치
        
    반환값:
        시험 묶음의 평균 손실
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            if isinstance(model, CharRNN):
                output, _ = model(batch_x)
            else:
                output = model(batch_x)
            
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_text(model: nn.Module,
                 dataset: CharacterDataset,
                 seed_text: str,
                 length: int = 200,
                 temperature: float = 0.8,
                 device: str = 'cpu') -> str:
    """
    씨앗에서 글을 자기 되돌이로 만든다.
    
    인수:
        model: 익힌 말 모델
        dataset: 자료 묶음(부호로 바꾸고 풀기용)
        seed_text: 출발 글
        length: 만들 글자의 수
        temperature: 표집 온도
        device: 돌릴 장치
        
    반환값:
        만든 글자열
    """
    model.eval()
    
    # 씨앗 글을 부호로 바꾼다
    seed_indices = dataset.encode(seed_text)
    seed_tensor = torch.LongTensor(seed_indices).to(device)
    
    # 생성
    if isinstance(model, CharRNN):
        generated = model.generate(seed_tensor, length, temperature)
    else:
        # 변환기에서는 만들어 내기를 다르게 짜야 한다
        # 이는 간략한 판본이다
        generated = seed_tensor.clone()
        
        with torch.no_grad():
            for _ in range(length):
                # 마지막 sequence_length개 글자를 쓴다
                input_seq = generated[-dataset.sequence_length:].unsqueeze(0)
                
                # 다음 글자를 헤아린다
                output = model(input_seq)
                output = output / temperature
                probs = torch.softmax(output, dim=-1)
                next_char = torch.multinomial(probs, 1).squeeze()
                
                # 만들어진 수열에 덧붙인다
                generated = torch.cat([generated, next_char.unsqueeze(0)])
    
    # 글로 푼다
    generated_text = dataset.decode(generated.cpu().tolist())
    
    return generated_text


def main():
    """
    으뜸 익히기 물길
    """
    print("=" * 70)
    print("Character-Level Autoregressive Language Model Training")
    print("=" * 70)
    
    # ==================== 채비 ====================
    # GPU이 있는지 살핀다
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # 초매개변수
    SEQUENCE_LENGTH = 50  # 들임 차례의 길이
    BATCH_SIZE = 64       # 묶음 크기
    EMBEDDING_DIM = 128   # 박아 넣기 차원
    HIDDEN_DIM = 256      # 숨은 차원(되돌이 신경망용)
    N_LAYERS = 2          # 층의 수
    N_EPOCHS = 50         # 익히기 바퀴 수
    LEARNING_RATE = 0.001
    
    print(f"\nHyperparameters:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Embedding Dim: {EMBEDDING_DIM}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ==================== 자료 불러오기 ====================
    print(f"\n{'='*70}")
    print("Step 1: Loading and preparing text data...")
    print(f"{'='*70}")
    
    # 보기 글을 불러온다(셰익스피어)
    text = load_sample_text()
    
    # 파일에서 불러올 수도 있다:
    # text = load_text_file('your_text_file.txt')
    
    print(f"\nSample of text:")
    print(text[:200])
    print()
    
    # 데이터셋 생성
    dataset = CharacterDataset(text, sequence_length=SEQUENCE_LENGTH)
    
    # 차례를 만든다
    X, y = dataset.create_sequences()
    
    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.9)
    
    print(f"\nData prepared:")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    print(f"  Vocabulary size: {dataset.vocab_size}")
    
    # 데이터로더들을 만든다
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # ==================== 모델 첫자리매김 ====================
    print(f"\n{'='*70}")
    print("Step 2: Initializing model...")
    print(f"{'='*70}")
    
    # 모델 갈래를 고른다
    # 고르기 1: 되돌이 신경망(긴 짧은 기억)
    model = CharRNN(
        vocab_size=dataset.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS
    ).to(device)
    model_name = "CharRNN"
    
    # 고르기 2: 변환기(쓰려면 주석을 푼다)
    # model = SimpleCharTransformer(
    #     vocab_size=dataset.vocab_size,
    #     embedding_dim=EMBEDDING_DIM,
    #     n_heads=4,
    #     n_layers=N_LAYERS,
    #     max_seq_length=SEQUENCE_LENGTH
    # ).to(device)
    # model_name = "변환기"
    
    # 매개변수 개수 세기
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name}")
    print(f"Parameters: {n_params:,}")
    
    # 손실과 최적화기
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ==================== 익히기 ====================
    print(f"\n{'='*70}")
    print("Step 3: Training model...")
    print(f"{'='*70}\n")
    
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(N_EPOCHS), desc="Epochs"):
        # 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 평가한다
        test_loss = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # 10바퀴마다 나아감을 찍는다
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            
            # 보기 글을 만든다
            seed = "To be"
            sample = generate_text(model, dataset, seed, length=100, 
                                 temperature=0.8, device=device)
            print(f"\n  Sample generation (seed: '{seed}'):")
            print(f"  {sample}")
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final test loss: {test_losses[-1]:.4f}")
    
    # ==================== 그려 보기 ====================
    print(f"\n{'='*70}")
    print("Step 4: Creating visualizations...")
    print(f"{'='*70}")
    
    # 익히기 곡선을 그린다
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'{model_name}: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("✓ Saved training_history.png")
    
    # ==================== 표본 만들기 ====================
    print(f"\n{'='*70}")
    print("Step 5: Generating text samples...")
    print(f"{'='*70}")
    
    seeds = ["To be", "Whether", "And by"]
    temperatures = [0.5, 0.8, 1.2]
    
    print("\nGenerated samples with different temperatures:\n")
    
    for temp in temperatures:
        print(f"Temperature: {temp}")
        print("-" * 70)
        for seed in seeds:
            sample = generate_text(model, dataset, seed, length=150,
                                 temperature=temp, device=device)
            print(f"Seed: '{seed}'")
            print(sample)
            print()
        print()
    
    # ==================== 간추리기 ====================
    print(f"{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"\nKey Observations:")
    print(f"1. Lower temperature (0.5) → More conservative, repetitive")
    print(f"2. Medium temperature (0.8) → Balanced creativity")
    print(f"3. Higher temperature (1.2) → More random, creative but chaotic")
    print(f"\nThe model learned to generate text autoregressively,")
    print(f"predicting one character at a time based on previous context!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 만들어 내는 모델 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.

## 정리하며

**다룬 것** — 익히기 2

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
