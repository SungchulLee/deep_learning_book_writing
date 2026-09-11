"""data_preprocessing — data_preprocessing 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch09/seq2seq/data_preprocessing.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
Seq2Seq 모델을 위한 데이터 전처리 도구
토큰화, 어휘 만들기, 데이터 적재를 담고 있다
"""

import re
from collections import Counter
import pickle
from pathlib import Path
import unicodedata

# ========================================================================
# 메인
# ========================================================================


class Tokenizer:
    """
    텍스트 전처리를 위한 간단한 토큰 나누개
    """
    
    def __init__(self, lower=True, remove_punct=False):
        self.lower = lower
        self.remove_punct = remove_punct
    
    def tokenize(self, text):
        """텍스트를 토큰으로 나눈다"""
        if self.lower:
            text = text.lower()
        
        if self.remove_punct:
            # 문장 부호 없애기
            text = re.sub(r'[^\w\s]', '', text)
        else:
            # 문장 부호를 공백으로 떼기
            text = re.sub(r'([.,!?;:])', r' \1 ', text)
        
        # 공백으로 쪼개고 빈 문자열 걸러 내기
        tokens = text.split()
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def detokenize(self, tokens):
        """토큰을 다시 텍스트로 바꾼다"""
        text = ' '.join(tokens)
        # 문장 부호 앞의 공백 없애기
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text


class Vocabulary:
    """
    토큰과 색인의 대응을 관리하는 어휘 클래스
    
    인수:
        max_size: 어휘의 최대 크기 (None이면 제한 없음)
        min_freq: 토큰이 들어가려면 넘어야 할 최소 빈도
        special_tokens: 특수 토큰의 목록
    """
    
    def __init__(self, max_size=None, min_freq=1, 
                 special_tokens=['<pad>', '<sos>', '<eos>', '<unk>']):
        self.max_size = max_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens
        
        # 토큰을 색인으로 잇기
        self.token2idx = {}
        self.idx2token = {}
        
        # 특수 토큰 더하기
        for idx, token in enumerate(special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        self.pad_idx = self.token2idx.get('<pad>', 0)
        self.sos_idx = self.token2idx.get('<sos>', 1)
        self.eos_idx = self.token2idx.get('<eos>', 2)
        self.unk_idx = self.token2idx.get('<unk>', 3)
    
    def build_vocab(self, texts, tokenizer=None):
        """
        텍스트에서 어휘를 만든다
        
        인수:
            texts: 텍스트 문자열의 목록 또는 토큰 목록의 목록
            tokenizer: 토큰 나누개 함수 (선택)
        """
        # 토큰의 빈도 세기
        counter = Counter()
        
        for text in texts:
            if tokenizer is not None:
                tokens = tokenizer(text)
            elif isinstance(text, str):
                tokens = text.split()
            else:
                tokens = text
            
            counter.update(tokens)
        
        # 빈도로 걸러 내기
        tokens = [token for token, freq in counter.items() if freq >= self.min_freq]
        
        # 빈도로 정렬 (가장 흔한 것 먼저)
        tokens = sorted(tokens, key=lambda t: counter[t], reverse=True)
        
        # 어휘 크기 제한
        if self.max_size is not None:
            tokens = tokens[:self.max_size - len(self.special_tokens)]
        
        # 어휘에 토큰 더하기
        for token in tokens:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
    
    def encode(self, tokens):
        """토큰을 색인으로 바꾼다"""
        if isinstance(tokens, str):
            tokens = tokens.split()
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices, skip_special=True):
        """색인을 토큰으로 바꾼다"""
        tokens = []
        for idx in indices:
            if skip_special and idx in [self.pad_idx, self.sos_idx, self.eos_idx]:
                if idx == self.eos_idx:
                    break
                continue
            tokens.append(self.idx2token.get(idx, '<unk>'))
        return tokens
    
    def __len__(self):
        return len(self.token2idx)
    
    def save(self, path):
        """어휘를 파일에 저장한다"""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'max_size': self.max_size,
                'min_freq': self.min_freq,
                'special_tokens': self.special_tokens
            }, f)
    
    @classmethod
    def load(cls, path):
        """파일에서 어휘를 불러온다"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(
            max_size=data['max_size'],
            min_freq=data['min_freq'],
            special_tokens=data['special_tokens']
        )
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        
        return vocab


class ParallelDataset:
    """
    순차열 대 순차열 과제를 위한 병렬 데이터셋
    
    인수:
        src_texts: 원본 텍스트의 목록
        trg_texts: 표적 텍스트의 목록
        src_vocab: 원본 어휘
        trg_vocab: 표적 어휘
        src_tokenizer: 원본 토큰 나누개
        trg_tokenizer: 표적 토큰 나누개
        max_len: 순차열의 최대 길이
    """
    
    def __init__(self, src_texts, trg_texts, src_vocab, trg_vocab,
                 src_tokenizer=None, trg_tokenizer=None, max_len=None):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer or Tokenizer()
        self.trg_tokenizer = trg_tokenizer or Tokenizer()
        self.max_len = max_len
    
    def process_pair(self, src_text, trg_text):
        """원본-표적 쌍을 처리한다"""
        # 토큰으로 나누기
        src_tokens = self.src_tokenizer.tokenize(src_text)
        trg_tokens = self.trg_tokenizer.tokenize(trg_text)
        
        # 필요하면 잘라 내기
        if self.max_len is not None:
            src_tokens = src_tokens[:self.max_len]
            trg_tokens = trg_tokens[:self.max_len]
        
        # 부호화
        src_indices = self.src_vocab.encode(src_tokens)
        trg_indices = [self.trg_vocab.sos_idx] + self.trg_vocab.encode(trg_tokens) + [self.trg_vocab.eos_idx]
        
        return src_indices, trg_indices
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        return self.process_pair(self.src_texts[idx], self.trg_texts[idx])


def load_parallel_data(src_path, trg_path, max_samples=None):
    """
    파일에서 병렬 데이터를 불러온다
    
    인수:
        src_path: 원본 파일의 경로
        trg_path: 표적 파일의 경로
        max_samples: 불러올 표본의 최대 수
        
    반환값:
        src_texts: 원본 텍스트의 목록
        trg_texts: 표적 텍스트의 목록
    """
    with open(src_path, 'r', encoding='utf-8') as f:
        src_texts = [line.strip() for line in f]
    
    with open(trg_path, 'r', encoding='utf-8') as f:
        trg_texts = [line.strip() for line in f]
    
    # 길이를 같게 맞추기
    assert len(src_texts) == len(trg_texts), "Source and target files must have same length"
    
    # 지정되었으면 표본 수 제한
    if max_samples is not None:
        src_texts = src_texts[:max_samples]
        trg_texts = trg_texts[:max_samples]
    
    return src_texts, trg_texts


def normalize_text(text):
    """
    텍스트를 정규화한다 (유니코드 정규화 등)
    
    인수:
        text: 입력 텍스트
        
    반환값:
        normalized_text: 정규화된 텍스트
    """
    # 유니코드 정규화
    text = unicodedata.normalize('NFD', text)
    
    # 강세 부호 없애기
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    return text


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """
    데이터를 학습·검증·시험 집합으로 나눈다
    
    인수:
        data: 데이터의 목록이나 쌍
        train_ratio: 학습 데이터의 비율
        val_ratio: 검증 데이터의 비율
        
    반환값:
        train_data, val_data, test_data: 나뉜 데이터셋
    """
    if isinstance(data, tuple):
        # 여러 데이터셋 (원본과 표적 따위)
        total_len = len(data[0])
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        train_data = tuple(d[:train_len] for d in data)
        val_data = tuple(d[train_len:train_len + val_len] for d in data)
        test_data = tuple(d[train_len + val_len:] for d in data)
    else:
        # 데이터셋 하나
        total_len = len(data)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        train_data = data[:train_len]
        val_data = data[train_len:train_len + val_len]
        test_data = data[train_len + val_len:]
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # 사용 예
    print("Data Preprocessing Example")
    print("-" * 50)
    
    # 예제 데이터
    src_texts = [
        "Hello, how are you?",
        "I am doing well, thank you.",
        "What is your name?",
        "My name is Claude.",
        "Nice to meet you!"
    ]
    
    trg_texts = [
        "Bonjour, comment allez-vous?",
        "Je vais bien, merci.",
        "Quel est votre nom?",
        "Je m'appelle Claude.",
        "Enchanté de vous rencontrer!"
    ]
    
    # 토큰 나누개 만들기
    tokenizer = Tokenizer(lower=True, remove_punct=False)
    
    # 어휘 만들기
    print("\nBuilding vocabularies...")
    src_vocab = Vocabulary(max_size=1000, min_freq=1)
    src_vocab.build_vocab(src_texts, tokenizer.tokenize)
    
    trg_vocab = Vocabulary(max_size=1000, min_freq=1)
    trg_vocab.build_vocab(trg_texts, tokenizer.tokenize)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(trg_vocab)}")
    
    # 부호화와 복호 시험
    print("\nTesting encoding/decoding...")
    test_text = "Hello, how are you?"
    tokens = tokenizer.tokenize(test_text)
    indices = src_vocab.encode(tokens)
    decoded = src_vocab.decode(indices, skip_special=False)
    
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Indices: {indices}")
    print(f"Decoded: {decoded}")
    
    # 데이터셋 생성
    print("\nCreating dataset...")
    dataset = ParallelDataset(
        src_texts, trg_texts, src_vocab, trg_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 시험 데이터셋
    src_indices, trg_indices = dataset[0]
    print(f"\nSample data:")
    print(f"Source: {src_texts[0]}")
    print(f"Source indices: {src_indices}")
    print(f"Target: {trg_texts[0]}")
    print(f"Target indices: {trg_indices}")
    
    # 어휘 저장
    print("\nSaving vocabularies...")
    src_vocab.save('src_vocab.pkl')
    trg_vocab.save('trg_vocab.pkl')
    print("Vocabularies saved!")
    
    # 어휘 불러오기
    print("\nLoading vocabularies...")
    loaded_src_vocab = Vocabulary.load('src_vocab.pkl')
    loaded_trg_vocab = Vocabulary.load('trg_vocab.pkl')
    print(f"Loaded source vocabulary size: {len(loaded_src_vocab)}")
    print(f"Loaded target vocabulary size: {len(loaded_trg_vocab)}")
