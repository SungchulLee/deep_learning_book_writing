# 데이터 전처리

번역 등의 병렬 문장을 토큰화하고 입력·출력 사전과 정수 시퀀스로 바꾸는 도구다. 학습 모델은 포함하지 않으며, 전처리와 특수 토큰의 규약을 일관되게 유지하는 데 초점을 맞춘다.

## 1. 코드

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
    print(f"Loaded target vocabulary size: {len(loaded_trg_vocab)}")
```

**출력:**

```
Data Preprocessing Example
--------------------------------------------------

Building vocabularies...
Source vocabulary size: 26
Target vocabulary size: 25

Testing encoding/decoding...
Original: Hello, how are you?
Tokens: ['hello', ',', 'how', 'are', 'you', '?']
Indices: [10, 5, 11, 12, 4, 6]
Decoded: ['hello', ',', 'how', 'are', 'you', '?']

Creating dataset...
Dataset size: 5

Sample data:
Source: Hello, how are you?
Source indices: [10, 5, 11, 12, 4, 6]
Target: Bonjour, comment allez-vous?
Target indices: [1, 8, 4, 9, 10, 5, 2]

Saving vocabularies...
Vocabularies saved!

Loading vocabularies...
Loaded source vocabulary size: 26
Loaded target vocabulary size: 25
```

## 2. 논의

Tokenizer는 기본적으로 소문자로 바꾸고 일부 구두점 앞뒤에 공백을 넣는다. 구두점 제거 설정은 문장 부호를 삭제하므로 원문을 완전히 복원할 수 없다. Vocabulary.build_vocab에는 호출 가능한 토큰화 함수를 넘겨야 하며 Tokenizer 객체 자체는 __call__이 없어 직접 전달할 수 없다.

Vocabulary는 특수 토큰을 먼저 등록하고 빈도 내림차순으로 단어를 추가한다. 동률은 입력에서 관측된 순서에 영향을 받는다. max_size는 특수 토큰을 포함한 한도이므로 그보다 작게 설정하면 음수 슬라이싱이 생길 수 있다. 입력 검증이 필요하며 사전은 학습 데이터로 만든 뒤 검증·테스트에 고정한다.

ParallelDataset은 원문에는 특수 경계 토큰을 붙이지 않고, 타깃에는 SOS와 EOS를 추가한다. max_len은 특수 토큰을 추가하기 전의 단어 수 제한이라 타깃의 최종 길이는 최대 max_len+2다. 빈 원문은 길이 0이 되어 패킹에 부적합할 수 있다. 데이터셋 생성자 자체는 원문·타깃 목록 길이를 검사하지 않는다.

split_data는 섞지 않고 앞에서부터 구간을 나눈다. 문서·주제·중복 문장의 분포를 고려해 분할을 설계한다. normalize_text는 NFD 분해 뒤 결합 문자를 제거하므로 언어의 구별 정보를 잃거나 한글이 자모로 분해된 채 남을 수 있다. 이 정규화는 전처리 전체에 자동 적용되지 않는다.

사전 저장은 pickle 기반이다. 사전과 토큰화 설정을 학습 모델과 함께 관리해야 하며, decode는 기본적으로 첫 EOS에서 멈추고 PAD·SOS를 제외한다. encode만 호출한다고 소문자 변환이나 구두점 분리가 자동 수행되는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
토큰화, 사전 인코딩과 타깃 경계 토큰 추가의 책임을 구분하라.

</div>

??? success "연습문제 1 풀이"
    Tokenizer가 텍스트를 토큰으로 나누고 Vocabulary.encode는 이미 준비된 토큰을 ID로 바꾼다. ParallelDataset.process_pair는 타깃 앞뒤에 SOS·EOS를 추가한다. encode에 문자열을 넣으면 단순 공백 분리만 하므로 학습 토큰화 규칙을 먼저 적용해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
max_len=10일 때 원문·타깃의 최대 길이와 배치 구성에서 확인할 점을 설명하라.

</div>

??? success "연습문제 2 풀이"
    일반 토큰을 최대 10개로 자르므로 원문 최대 길이는 10, 타깃은 SOS·EOS를 더해 12다. 배치에서는 각각의 실제 길이와 패딩을 따로 계산한다. 빈 원문, 양수 길이와 유효 인덱스 범위도 검사한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
언어 정보가 전처리에서 로스될 수 있는 경우와 학습·추론 불일치를 설명하라.

</div>

??? success "연습문제 3 풀이"
    소문자 변환은 대소문자 정보를, 구두점 제거와 결합 문자 제거는 문장 구조나 발음 구별을 없앨 수 있다. 학습에서는 구두점을 분리했는데 추론에서는 공백만 나누면 같은 문장도 다른 토큰이 된다. 규칙을 함께 저장하고 동일하게 적용해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

전처리는 토큰화·사전·경계 토큰·길이 제한의 규약을 함께 정하는 과정이다. 데이터 분할 후 사전을 고정하고 추론에서도 같은 변환을 적용한다.
