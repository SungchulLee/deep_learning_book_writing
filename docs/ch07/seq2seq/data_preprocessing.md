# 데이터 전처리

Seq2Seq 모델을 위한 데이터 전처리 도구. 토큰화, 어휘 만들기, 데이터 적재를 담고 있다.

순차열 모형은 시간적이고 순서가 있는 데이터를 다루는 데 바탕이 된다. 이 구현은 순차열 모델의 핵심 착상을 다루며, 순환 계산과 학습된 표현이 시각 사이의 의존을 어떻게 붙잡는지 보인다.

## 코드

```python
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
    print(f"Loaded target vocabulary size: {len(loaded_trg_vocab)}")```

## 논의

이 구현은 깔끔하고 읽기 좋은 PyTorch 코드로 순차열 모델의 핵심 개념을 보인다. 모듈식 짜임 덕분에 부품 하나하나를 살펴보고 다른 과제나 데이터셋에 맞추어 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 순차열 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 훑으며 핵심 설계 결정을 찾아라. 구체적인 구현 선택 세 가지를 열거하고 각각이 순차열 모델에 알맞은 까닭을 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
데이터 전처리 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_tokenizer():
        model = Tokenizer(...)
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
