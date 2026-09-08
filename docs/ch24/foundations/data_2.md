# 자료 2

글자 수준 말 모델을 위한 자료 도구. 이 단원은 자기 되돌이 말 모델을 익히기 위한 글 다루기를 맡는다

자기 되돌이 모델은 앞선 모든 낱개를 조건으로 삼아 낱개마다 미리 헤아려 자료를 만든다. 이 단원은 자기 되돌이 모델 부품의 짜기를 보이며 차례대로 만들어 내는 과정과 그 얼개의 요구를 그려 보인다.

## 1. 코드

```python
"""
글자 수준 말 모델을 위한 자료 도구

이 단원은 자기 되돌이 말 모델을 익히기 위한 글 다루기를 맡는다
익히기 대본.
"""

import torch
import numpy as np
from typing import Tuple, List

# ========================================================================
# 메인
# ========================================================================


class CharacterDataset:
    """
    글자 수준 말 나타내기를 위한 자료 묶음.
    
    이 갈래는:
    1. 글을 수 어깨수로 바꾼다(토큰 나누기)
    2. 익히기를 위한 들임-내놓기 짝을 만든다
    3. 부호로 바꾸고 푸는 도구를 준다
    """
    
    def __init__(self, text: str, sequence_length: int = 50):
        """
        글자 자료 묶음을 첫자리매김한다.
        
        인수:
            text: 배울 들임 글자열
            sequence_length: 익히기용 글자 차례의 길이
        """
        self.text = text
        self.sequence_length = sequence_length
        
        # 서로 다른 글자(낱말)를 얻는다
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # 옮김을 만든다: 글자 <-> 정수
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Text length: {len(text)} characters")
        print(f"Vocabulary size: {self.vocab_size} unique characters")
        print(f"Characters: {self.chars}")
        
    def encode(self, text: str) -> List[int]:
        """
        글을 정수 목록으로 바꾼다.
        
        인수:
            text: 부호로 바꿀 글자열
            
        반환값:
            정수 어깨수 목록
        """
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices: List[int]) -> str:
        """
        정수 목록을 다시 글로 바꾼다.
        
        인수:
            indices: 정수 어깨수 목록
            
        반환값:
            푼 글자열
        """
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        글에서 익히기 차례를 만든다.
        
        자기 되돌이 말 나타내기에서는:
        - 들임: 글자의 차례 [c1, c2, ..., cn]
        - 목표: 자리마다 다음 글자 [c2, c3, ..., cn+1]
        
        반환값:
            텐서로 된 (들임, 목표) 짝
        """
        # 온 글을 부호로 바꾼다
        encoded_text = self.encode(self.text)
        
        sequences = []
        targets = []
        
        # 글 위로 창을 미끄러뜨린다
        for i in range(len(encoded_text) - self.sequence_length):
            # 들임: i에서 i+sequence_length까지의 글자
            seq = encoded_text[i:i + self.sequence_length]
            
            # 목표: 차례 다음에 오는 글자
            target = encoded_text[i + self.sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        # 텐서로 바꾼다
        X = torch.LongTensor(sequences)
        y = torch.LongTensor(targets)
        
        return X, y


def load_sample_text() -> str:
    """
    보여 주기용 보기 글을 불러온다.
    
    반환값:
        보기 글자열
    """
    # 보기 글: 셰익스피어풍 글월
    표본 = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    """
    
    return sample.strip()


def load_text_file(filepath: str) -> str:
    """
    파일에서 글을 불러온다.
    
    인수:
        filepath: 글 파일의 길
        
    반환값:
        글자열로 된 글 내용
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def train_test_split(X: torch.Tensor, 
                     y: torch.Tensor, 
                     train_ratio: float = 0.9) -> Tuple[torch.Tensor, ...]:
    """
    자료를 익히기 묶음과 시험 묶음으로 가른다.
    
    글에서는 흔히 익히기 비율을 더 높게 쓴다(예컨대 90/10)
    익히기 자료를 되도록 많이 얻으려 하기 때문이다.
    
    인수:
        X: 들임 차례
        y: 목표 글자
        train_ratio: 익히기에 쓸 몫
        
    반환값:
        (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    split_idx = int(n * train_ratio)
    
    # 글에서는 앞뒤가 이어지도록 섞지 않는다
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """
    보여 주기: 보기 글을 다룬다
    """
    
    # 보기 글을 불러온다
    text = load_sample_text()
    print("Sample text:")
    print(text[:200] + "...")
    print()
    
    # 데이터셋 생성
    dataset = CharacterDataset(text, sequence_length=20)
    
    # 부호화와 복호 시험
    sample_text = "Hello, World!"
    encoded = dataset.encode(sample_text)
    decoded = dataset.decode(encoded)
    
    print(f"\nOriginal: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {sample_text == decoded}")
    
    # 차례를 만든다
    X, y = dataset.create_sequences()
    print(f"\nCreated {len(X)} training sequences")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # 보기를 보인다
    print(f"\nExample sequence:")
    print(f"Input (encoded): {X[0]}")
    print(f"Input (decoded): '{dataset.decode(X[0].tolist())}'")
    print(f"Target (encoded): {y[0]}")
    print(f"Target (decoded): '{dataset.decode([y[0].item()])}'")
    
    # 데이터 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.9)
    print(f"\nData split:")
    print(f"Train: {len(X_train)} sequences")
    print(f"Test: {len(X_test)} sequences")```

## 2. 논의

이 짜기는 깔끔하고 읽기 좋은 PyTorch 코드로 자기 되돌이 모델의 핵심 개념을 보인다. 조각으로 나뉜 짜임 덕분에 부품을 하나씩 익히고 다른 일이나 자료 묶음에 맞추기 쉽다.

여기서 보인 결은 더 복잡한 경우로 자연스럽게 넓혀진다. 웃매개변수와 얼개 변형, 여러 자료 묶음을 실험하면 만들어 내는 모델 일에 대한 이해가 깊어지고 실제 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 설계 결정을 가려내라. 구체적인 짜기 고르기 셋을 적고 저마다 자기 되돌이 모델에 어울리는 까닭을 설명하라.

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
Data 2 짜기를 확인하는 두루 갖춘 시험 함수를 적어라. 빈 들임, 낱개 하나짜리 들임, 아주 큰 들임, 끝값(0이나 아주 큰 수)을 담은 들임 같은 가장자리 경우를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_characterdataset():
        model = CharacterDataset(...)
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

**다룬 것** — 자료 2

이 짜기는 깔끔하고 읽기 좋은 PyTorch 코드로 자기 되돌이 모델의 핵심 개념을 보인다.

고갱이 갈래는 `CharacterDataset`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
