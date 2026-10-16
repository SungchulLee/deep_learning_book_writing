# 학습

Seq2Seq의 배치 패딩, 로스 계산, 그래디언트 클리핑과 체크포인트 관리를 모은 예제다. 실행부는 무작위 정수 시퀀스를 사용하므로 실제 번역 학습의 품질을 보여 주지 않는다. 모델의 반환값과 평가 목적을 공통 학습 함수에 맞춰야 한다.

## 1. 코드

```python
"""
Seq2Seq 모델을 위한 학습 스크립트
학습 반복문, 평가, 검사점 저장을 담고 있다
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import math
from pathlib import Path

# ========================================================================
# 메인
# ========================================================================


class Seq2SeqDataset(Dataset):
    """
    Seq2Seq 학습을 위한 사용자 정의 데이터셋
    
    인수:
        src_data: 원본 순차열의 목록 (토큰 색인)
        trg_data: 표적 순차열의 목록 (토큰 색인)
        src_vocab: 원본 어휘
        trg_vocab: 표적 어휘
    """
    
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.trg_data[idx])


def collate_fn(batch, pad_idx=0):
    """
    길이가 다른 순차열을 배치로 묶는 함수
    
    인수:
        batch: (src, trg) 쌍의 목록
        pad_idx: 덧댐 토큰의 색인
        
    반환값:
        src_batch: 덧댄 원본 순차열
        trg_batch: 덧댄 표적 순차열
        src_lengths: 원본 순차열의 실제 길이
        trg_lengths: 표적 순차열의 실제 길이
    """
    src_batch, trg_batch = zip(*batch)
    
    # 길이 얻기
    src_lengths = torch.tensor([len(s) for s in src_batch])
    trg_lengths = torch.tensor([len(t) for t in trg_batch])
    
    # 순차열 덧대기
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
    
    return src_batch, trg_batch, src_lengths, trg_lengths


class Seq2SeqTrainer:
    """
    Seq2Seq 모델을 위한 학습기 클래스
    
    인수:
        model: Seq2Seq 모델
        optimizer: 최적화기
        criterion: 로스 함수
        device: 학습에 쓸 장치
        pad_idx: 덧댐 토큰의 색인
        clip: 그래디언트를 자를 값
    """
    
    def __init__(self, model, optimizer, criterion, device, pad_idx=0, clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.pad_idx = pad_idx
        self.clip = clip
        
    def train_epoch(self, dataloader, teacher_forcing_ratio=0.5):
        """
        한 세대 학습시킨다
        
        인수:
            dataloader: 학습 데이터로더
            teacher_forcing_ratio: 교사 강요를 쓸 확률
            
        반환값:
            epoch_loss: 그 세대의 평균 로스
        """
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (src, trg, src_lengths, trg_lengths) in enumerate(dataloader):
            src = src.to(self.device)
            trg = trg.to(self.device)
            src_lengths = src_lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 순전파
            if hasattr(self.model, 'encoder'):
                # Seq2SeqAttention 모델
                output, _ = self.model(src, trg, teacher_forcing_ratio, src_lengths)
            else:
                output = self.model(src, trg, teacher_forcing_ratio, src_lengths)
            
            # 로스 계산을 위해 출력과 표적의 모양 바꾸기
            # output: (배치 크기, trg_len, output_dim)
            # trg: (배치 크기, trg_len)
            output_dim = output.shape[-1]
            
            # 표적의 첫 토큰(<sos>) 건너뛰기
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # 로스 계산
            loss = self.criterion(output, trg)
            
            # 역전파
            loss.backward()
            
            # 경사를 자른다
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            # 파라메터 갱신
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        모델을 평가한다
        
        인수:
            dataloader: 검증 데이터로더
            
        반환값:
            epoch_loss: 그 세대의 평균 로스
        """
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch_idx, (src, trg, src_lengths, trg_lengths) in enumerate(dataloader):
                src = src.to(self.device)
                trg = trg.to(self.device)
                src_lengths = src_lengths.to(self.device)
                
                # 순전파 (평가 중에는 교사 강요 없음)
                if hasattr(self.model, 'encoder'):
                    output, _ = self.model(src, trg, teacher_forcing_ratio=0, src_lengths=src_lengths)
                else:
                    output = self.model(src, trg, teacher_forcing_ratio=0, src_lengths=src_lengths)
                
                # 로스 계산을 위해 모양 바꾸기
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # 로스 계산
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir='checkpoints', 
              teacher_forcing_ratio=0.5, save_every=1):
        """
        여러 세대에 걸쳐 모델을 학습시킨다
        
        인수:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            num_epochs: 학습할 세대 수
            checkpoint_dir: 검사점을 저장할 디렉터리
            teacher_forcing_ratio: 처음의 교사 강요 비율
            save_every: N세대마다 검사점 저장
            
        반환값:
            train_losses: 학습 로스의 목록
            val_losses: 검증 로스의 목록
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 학습
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio)
            
            # 평가한다
            val_loss = self.evaluate(val_loader)
            
            # 로스 담기
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            # 진행 상황 출력
            print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')
            
            # 검사점 저장
            if (epoch + 1) % save_every == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
            
            # 최고 성능 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_dir / 'best_model.pt'
                self.save_checkpoint(best_checkpoint_path, epoch, train_loss, val_loss)
                print(f'\t[Saved Best Model]')
            
            # 교사 강요 비율 줄이기 (선택)
            teacher_forcing_ratio = max(0.5 * teacher_forcing_ratio, 0.1)
        
        return train_losses, val_losses
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        """모델 검사점을 저장한다"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, path)
    
    def load_checkpoint(self, path):
        """모델 검사점을 불러온다"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']


def initialize_weights(model):
    """모델의 웨이트를 초기화한다"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    """학습 가능한 파라메터를 센다"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 사용 예
    from encoder import BasicEncoder
    from decoder import AttentionDecoder
    from seq2seq_model import Seq2SeqAttention
    
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 초파라메터
    INPUT_DIM = 10000
    OUTPUT_DIM = 10000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    PAD_IDX = 0
    
    # 모델 생성
    encoder = BasicEncoder(
        input_size=INPUT_DIM,
        embedding_dim=ENC_EMB_DIM,
        hidden_size=HID_DIM,
        num_layers=N_LAYERS,
        dropout=ENC_DROPOUT,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    decoder = AttentionDecoder(
        output_size=OUTPUT_DIM,
        embedding_dim=DEC_EMB_DIM,
        hidden_size=HID_DIM * 2,
        encoder_hidden_size=HID_DIM * 2,
        num_layers=N_LAYERS,
        dropout=DEC_DROPOUT,
        rnn_type='LSTM'
    )
    
    model = Seq2SeqAttention(encoder, decoder, device, PAD_IDX).to(device)
    
    # 웨이트 초기화
    initialize_weights(model)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # 최적화기와 로스 함수 만들기
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # 학습기 만들기
    trainer = Seq2SeqTrainer(model, optimizer, criterion, device, PAD_IDX, clip=1.0)
    
    # 임시 데이터 만들기
    print("\nCreating dummy dataset...")
    num_samples = 1000
    src_data = [np.random.randint(3, INPUT_DIM, size=np.random.randint(10, 30)).tolist() 
                for _ in range(num_samples)]
    trg_data = [np.random.randint(3, OUTPUT_DIM, size=np.random.randint(10, 30)).tolist() 
                for _ in range(num_samples)]
    
    train_dataset = Seq2SeqDataset(src_data[:800], trg_data[:800])
    val_dataset = Seq2SeqDataset(src_data[800:], trg_data[800:])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    # 모델을 학습시킨다
    print("\nStarting training...")
    train_losses, val_losses = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=5,
        checkpoint_dir='checkpoints',
        teacher_forcing_ratio=0.5
    )
    
    print("\nTraining completed!")
```

## 2. 논의

collate_fn은 원문·타깃을 각각 배치 최대 길이로 패딩하고 길이를 반환한다. 로스는 타깃 첫 위치를 제외한 출력과 정답을 평탄화해 계산한다. PAD 제외는 trainer의 pad_idx 자체가 아니라 전달된 criterion의 ignore_index가 수행하므로 두 설정을 맞춰야 한다.

모델에 encoder 속성이 있으면 반환값을 `(output, _)`로 분해하지만, 기본 Seq2Seq도 encoder를 가지면서 텐서만 반환한다. 따라서 이 분기는 두 모델을 올바르게 구별하지 못한다. 반환 계약을 명시적으로 맞추어야 하며 배치 크기에 따라 텐서가 잘못 분해되는 경우도 주의한다.

검증은 teacher_forcing_ratio=0으로 이전 예측을 입력하면서 정답 위치별 교차 엔트로피를 계산한다. 이는 정답 문맥에 조건을 두는 일반적인 언어 모델 음의 로그 우도와 다르다. 따라서 표시된 exp(val_loss)를 표준 퍼플렉시티로 해석하면 안 된다. 학습 역시 티처 포싱 비율을 바꾸므로 에포크 간 로스의 입력 조건이 달라진다.

기록 로스는 배치 로스의 단순 평균이며 배치 크기와 유효 토큰 수가 다르면 전체 토큰 평균이 아니다. 데이터는 원문·타깃이 독립적인 무작위 ID이고 SOS·EOS도 넣지 않으므로 형식과 동작을 살펴보는 시연으로 읽는다.

체크포인트에는 모델·옵티마이저와 에포크·로스를 저장하지만 난수 상태, 사전·전처리와 티처 포싱 비율은 없다. 최고 검증 모델을 저장해도 마지막에 자동 복원하지는 않는다. load_checkpoint는 저장된 위치를 반환할 뿐 나머지 학습 설정까지 복원하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
매 배치의 그래디언트 초기화를 제거하면 어떤 변화가 생기는가? 의도적인 그래디언트 누적과 구분하라.

</div>

??? success "연습문제 1 풀이"
    backward는 기존 grad에 값을 더하므로 매번 optimizer.step을 하면서 지우지 않으면 서로 다른 웨이트에서 계산한 과거 그래디언트까지 계속 남는다. 단순히 러닝레이트가 단계 수만큼 커진 것과 같지는 않다. 의도적인 누적은 정해진 배치 구간 동안 웨이트를 고정하고 로스 배율을 맞춘 뒤 한 번 갱신하고 초기화한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 검증 함수의 exp(loss)를 일반적인 퍼플렉시티라고 부르기 어려운 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    검증 입력이 이전 정답이 아니라 모델 예측이어서 정답 문맥의 로그 우도와 다르다. 또한 배치 평균을 다시 단순 평균해 유효 토큰별 평균이 아니다. 표준 퍼플렉시티에는 같은 토큰화·마스크에서 정답 문맥의 토큰 음의 로그 우도를 합산하고 유효 토큰 수로 나눈 뒤 지수화해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
최고 체크포인트를 저장한 것과 학습을 정확히 재개할 수 있는 것은 어떻게 다른가?

</div>

??? success "연습문제 3 풀이"
    선택한 모델의 웨이트를 복원하는 것과 같은 데이터 순서·난수 상태로 이후 업데이트를 재현하는 것은 다르다. 정확한 재개에는 옵티마이저 외에도 난수·로더 상태와 스케줄, 티처 포싱 비율 등 현재 학습 상태가 필요하다. 사전과 구조 설정도 함께 관리해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 코드 조각에서 현재 Seq2SeqTrainer에 연결하기 위해 필요한 수정과 최고 웨이트 복원 단계를 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 조기 종료의 제어 흐름 예시다. 실제 trainer의 검증 호출과 체크포인트 형식에 맞춰 연결해야 한다.
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
    `train`의 기존 루프에 최저 검증 로스와 연속 미개선 횟수를 유지하고, 개선할 때 실제 모델 상태를 별도로 저장한다. 중단 후 그 웨이트를 명시적으로 불러와야 한다. 검증을 반복해 선택한 결과는 독립 테스트 평가와 구분한다. 유효하지 않은 로스나 저장 전에 중단되는 경우도 처리한다.

## 정리하며

Seq2Seq 학습에서는 반환 형식, 타깃 이동과 PAD 로스 제외를 맞춘다. 이전 예측으로 입력을 만드는 검증과 정답 문맥의 우도 평가를 구분하고, 배치 로스 평균과 체크포인트 복원 범위를 명시해야 한다.
