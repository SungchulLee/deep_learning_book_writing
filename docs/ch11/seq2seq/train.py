"""train — train 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch09/seq2seq/train.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

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
        criterion: 손실 함수
        device: 학습에 쓸 장치
        pad_idx: 덧댐 토큰의 색인
        clip: 기울기를 자를 값
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
            epoch_loss: 그 세대의 평균 손실
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
            
            # 손실 계산을 위해 출력과 표적의 모양 바꾸기
            # output: (배치 크기, trg_len, output_dim)
            # trg: (배치 크기, trg_len)
            output_dim = output.shape[-1]
            
            # 표적의 첫 토큰(<sos>) 건너뛰기
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # 손실 계산
            loss = self.criterion(output, trg)
            
            # 역전파
            loss.backward()
            
            # 경사를 자른다
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            # 매개변수 갱신
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        모델을 평가한다
        
        인수:
            dataloader: 검증 데이터로더
            
        반환값:
            epoch_loss: 그 세대의 평균 손실
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
                
                # 손실 계산을 위해 모양 바꾸기
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # 손실 계산
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
            train_losses: 학습 손실의 목록
            val_losses: 검증 손실의 목록
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
            
            # 손실 담기
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
    """모델의 가중치를 초기화한다"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    """학습 가능한 매개변수를 센다"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 사용 예
    from encoder import BasicEncoder
    from decoder import AttentionDecoder
    from seq2seq_model import Seq2SeqAttention
    
    # 장치 지정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 초매개변수
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
    
    # 가중치 초기화
    initialize_weights(model)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    # 최적화기와 손실 함수 만들기
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
