# 문자 언어 모델의 학습과 생성

이 스크립트는 고정 길이 문자 문맥에서 다음 문자 하나를 예측하는 LSTM을 교차엔트로피로 학습한다. 학습 타깃과 모델 출력이 같은 문자 사전·창 정의를 사용해야 하며, 낮은 다음 문자 로스와 자유 생성의 품질은 구분해서 평가한다.

## 1. 학습 코드와 의존성

```python
"""
글자 수준 말 모델을 위한 익히기 대본

이 대본은 다음을 보인다.
1. 글 자료를 불러와 다듬는다
2. 자기 회귀 글자 모델 익히기
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
        criterion: 로스 함수(CrossEntropyLoss)
        optimizer: 최적화기
        device: 익힐 기기('cpu'나 'cuda')
        
    반환값:
        바퀴의 평균 로스
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
        
        # 로스를 계산한다
        loss = criterion(output, batch_y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트가 터지지 않도록 자른다
        # 이는 회귀 뉴럴 넷트웍에서 특히 중요하다
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 웨이트를 갱신한다
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
        criterion: 로스 함수
        device: 평가할 장치
        
    반환값:
        시험 배치의 평균 로스
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
    씨앗에서 글을 자기 회귀로 만든다.
    
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
    
    # 초파라메터
    SEQUENCE_LENGTH = 50  # 들임 차례의 길이
    BATCH_SIZE = 64       # 배치 크기
    EMBEDDING_DIM = 128   # 임베딩 차원
    HIDDEN_DIM = 256      # 숨은 차원(회귀 뉴럴 넷트웍용)
    N_LAYERS = 2          # 레이어의 수
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
    
    # 모델 클래스를 고른다
    # 고르기 1: 회귀 뉴럴 넷트웍(긴 짧은 기억)
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
    
    # 파라메터 개수 세기
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name}")
    print(f"Parameters: {n_params:,}")
    
    # 로스와 최적화기
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

main은 문맥 길이 50, 배치 64, 임베딩 128, LSTM 은닉 256, 2레이어, Adam 0.001과 50에포크를 설정한다. 기본 모델은 CharRNN이며 SimpleCharTransformer는 import하고 헬퍼에서 지원하지만 main에서 실제로 선택하지 않는다. 자료는 문맥당 정수 타깃 하나이고 모델 logits는 `(B,V)`다. LSTM hidden은 배치마다 새로 시작하며 배치 사이로 이어 가지 않는다.

현재 디렉터리에는 model.py는 있지만 data.py는 없다. import하는 CharacterDataset·load_sample_text·train_test_split을 [문자 데이터 절](data_2.md) 등의 실제 모듈로 제공해야 한다. 해당 데이터 예시 자체에도 표본·sample 변수 불일치가 있어 연결을 준비하지 않으면 실행되지 않는다. 이 페이지의 코드를 단독 실행 가능한 완성된 학습 결과로 설명하지 않는다.

## 2. 로스 기록과 평가 범위

CrossEntropyLoss는 배치의 다음 문자 NLL 평균이며 softmax를 미리 적용할 필요가 없다. epoch 로스는 배치 평균들의 단순 평균이므로 마지막 배치가 작으면 전체 타깃당 평균과 다르다. 정확한 문자당 NLL에는 각 배치 타깃 수로 가중해야 한다. 같은 문자 사전·문맥·평가 규칙에서 exp(평균 NLL)를 perplexity로 사용할 수 있다.

학습에서는 dropout과 매 배치 업데이트가 있고 평가는 종료 모델의 eval·no_grad 경로다. 따라서 train·test 곡선의 차이가 과적합 하나만을 뜻하지 않는다. 짧고 겹치는 텍스트 창의 순서 분할은 독립 문서 평가와도 다르다. 설정·중단 기준에는 별도 검증 자료를 사용하고 최종 테스트 자료를 분리해야 한다.

매 배치 그래디언트 노름을 1로 클리핑하지만 이것이 과적합·조건 누출·생성 오류를 해결하는 것은 아니다. 체크포인트 저장이나 최적 모델 복원은 현재 main에 없다. 모델 웨이트 외에 문자 사전·전처리·문맥 길이와 생성 규칙도 보존해야 한다.

## 3. 두 생성 경로의 차이

CharRNN 경로는 model.generate를 호출한다. [모델 절](model.md)에서 설명했듯이 이 메서드는 전체 접두사를 반복 입력하면서 hidden도 재사용해 이전 문맥을 중복 처리한다. 따라서 현재 자유 생성은 학습한 표준 접두사 조건과 다른 상태 경로를 따른다.

Transformer 경로는 마지막 sequence_length개 문자만 잘라 매 스텝 새로 계산한다. 위치 인덱스도 창마다 0에서 다시 시작하며 KV cache는 없다. 이 문맥 길이는 위치 임베딩 최대 길이 이하여야 한다. 두 생성 폴리시의 차이를 두 구조의 고유한 언어 생성 품질 차이로 혼동하지 않는다.

시드 텍스트는 비어 있지 않고 사전에 있어야 하며 temperature는 유한한 양수여야 한다. 반환 텍스트에는 시드와 추가 생성 length개가 함께 포함된다. 낮은 온도가 항상 좋은 글이나 적절한 반복을, 0.8이 균형 잡힌 창의성을 보장하는 것은 아니다. 출력 안내문은 이런 경험적 결과를 입증하지 않는다. 이 편집에서는 학습·생성을 실행하지 않았다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
현재 배치 학습에서 그래디언트 초기화와 클리핑은 각각 무엇을 하는가?

</div>

??? success "풀이"

    zero_grad는 이전 step의 그래디언트를 비우고 backward는 현재 배치의 그래디언트를 계산한다. clip_grad_norm은 그 계산된 전체 그래디언트 노름을 제한한 뒤 Adam step에 전달한다. 초기화를 생략하면 이전 그래디언트까지 누적되어 원래와 다른 업데이트가 되며 단순한 러닝레이트 배수와 같지 않다. 클리핑이 데이터 정렬·생성 상태 계약을 고치는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문에서 Adam으로 바꾸는 실험이 새로운 비교인가?

</div>

??? success "풀이"

    이미 Adam 0.001이므로 같은 설정으로 다시 지정하는 것은 변경이 아니다. 다른 옵티마이저나 러닝레이트를 비교하려면 동일한 자료·사전·초기화·평가 목적과 같은 튜닝 예산을 유지한다. 문자당 가중 평균 검증 NLL과 생성 폴리시를 맞추고 여러 시드·비용도 비교한다. 짧은 코퍼스에서 관찰한 로스를 일반 언어 능력의 우열로 확대하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
PyTorch의 한 레이어 단방향 LSTM과 GRU의 파라미터 수를 비교하라.

</div>

??? success "풀이"

    입력 크기 E, 은닉 H, bias=True이면 LSTM은 4H(E+H+2), GRU는 3H(E+H+2)개다. 입력·순환 바이어스가 각각 있어 +2가 들어간다. 바이어스를 하나로 합친 관례의 식과 구분한다. 같은 크기에서 GRU 순환 레이어는 LSTM의 75%지만 embedding·출력 레이어까지 포함한 전체 모델 비율은 다르다. 셀 상태는 별도의 학습 파라미터가 아니며 성능 우열은 파라미터 개수만으로 정해지지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료 골격을 본문에 적용할 때 어떤 계약을 맞춰야 하는가?

</div>

??? success "풀이"

    본문 evaluate에는 model뿐 아니라 dataloader·criterion·device가 필요하므로 아래의 두 인수 호출과 맞지 않는다. 검증 분할을 만들고 실제 학습 단계에 연결해야 한다. state_dict().copy는 얕은 복사여서 최적 텐서 보존에 충분하지 않으며 독립 복제나 파일 저장이 필요하다. 조기 종료·정상 종료 모두 최적 웨이트를 복원하고 사전·구조·전처리도 함께 기록한다. 최종 테스트 자료는 모델 선택에서 분리한다.

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


## 정리하며

문자 모델 학습은 창당 다음 문자 타깃과 배치 logits를 연결한다. 의존 데이터 모듈, 로스의 표본 가중 평균과 생성 상태 계약을 확인해야 한다. 검증 NLL·perplexity와 자유 생성은 다른 평가이며 사전·문맥·온도 설정을 맞춰 비교한다.
