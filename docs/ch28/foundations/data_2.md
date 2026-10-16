# 문자 언어 모델의 데이터 구성

문자 언어 모델은 앞선 문자에서 다음 문자의 분포를 예측한다. 이 도구는 텍스트에 나온 문자를 정수 ID로 매핑하고, 고정 길이 문맥과 그 다음 문자 하나를 학습 쌍으로 만든다. 텍스트의 공백·줄바꿈·문장부호도 관측 문자로 취급한다.

## 1. 문자 사전과 창 구성

```python
"""
글자 수준 말 모델을 위한 자료 도구

이 단원은 자기 회귀 말 모델을 익히기 위한 글 다루기를 맡는다
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
    
    이 클래스는:
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
        
        자기 회귀 말 나타내기에서는:
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
    자료를 익히기 배치와 시험 배치로 가른다.
    
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
    print(f"Test: {len(X_test)} sequences")
```

CharacterDataset은 이름과 달리 __len__·__getitem__이 있는 PyTorch Dataset이 아니다. create_sequences로 X·y 텐서를 만든 뒤 TensorDataset 등에 넣는 보조 객체다. 학습 파라미터나 forward도 없다.

chars는 텍스트의 고유 문자를 정렬한 목록이며 두 사전이 문자↔ID를 연결한다. Python 문자열 순회의 단위를 사용하므로 한글 조합 문자나 결합 기호가 사용자가 보는 글자 하나와 항상 일치하지는 않는다. encode는 사전에 없는 문자에 KeyError를 내고 decode도 범위를 벗어난 ID를 처리하지 않는다. UNK·BOS·EOS·PAD는 별도로 정의되어 있지 않다.

길이 N의 텍스트와 문맥 길이 L에서 N−L개의 창을 만든다. i번째 입력은 위치 i부터 i+L−1까지이고 타깃은 위치 i+L의 문자 하나다. 출력은 CPU int64 X `(N−L,L)`와 y `(N−L,)`이다. 모든 위치의 다음 문자 타깃을 동시에 반환하는 shifted sequence 방식과는 다르다.

## 2. 제공된 실행 예의 문제

load_sample_text는 본문 문자열을 변수 표본에 저장하지만 return에서는 정의되지 않은 sample을 참조한다. 따라서 표시된 main은 이 단계에서 NameError가 나며 이후 인코딩·창 출력이 그대로 실행된 결과라고 설명할 수 없다. 코드는 보존하고 이 계약 차이를 본문에 명시한다.

또한 예시 문자열 Hello, World!에는 제공 코퍼스에 없는 느낌표 등이 있다. 로더의 변수 문제를 고쳐도 이 문자열 전체가 현재 사전에 들어 있다고 가정하면 encode가 실패할 수 있다. 코퍼스에서 뽑은 문자만으로 왕복 인코딩을 검사하거나 미등록 문자 폴리시를 먼저 정해야 한다.

load_text_file은 UTF-8 내용을 그대로 읽는다. 텍스트 정규화·중복 제거·문서 경계·개인별 문서 분할 등을 자동 수행하지 않으므로 실제 말뭉치의 목적에 맞춰 별도 설계한다. 문자열 출력은 데이터 구조 확인이며 언어 모델 학습 결과가 아니다.

## 3. 경계 조건과 평가 분할

1≤L<N을 기본 사용 조건으로 삼는다. L≥N이면 sequences가 비어 X가 `(0,L)`이 아닌 1차원 빈 텐서가 되고, L=0이나 음수도 의도와 다른 창을 만들 수 있다. 빈 텍스트는 어휘도 비므로 다음 모델의 embedding·출력 차원까지 유효하지 않을 수 있다.

train_test_split은 창의 앞부분·뒷부분을 순서대로 나눈다. 인접 창은 문맥을 대부분 공유한다. 단일 텍스트에서 과거 문자를 계속 관측하는 다음 문자 평가라면 경계 이전 문맥을 사용하는 것은 가능하지만, 독립 문서의 일반화를 평가하려면 문서나 원문 구간을 먼저 분리하고 창을 만들어야 한다. 검증·테스트 문자로 어휘를 구성할지의 폴리시도 명시한다. 엄격한 학습 범위의 어휘를 사용하면 미등록 문자 처리와 평가 대상 토큰 집합을 정해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 데이터 도구의 실제 설계 결정 세 가지를 설명하라.

</div>

??? success "연습문제 1 풀이"

    고유 문자를 정렬해 일관된 ID를 주는 점, 문맥 길이 L의 겹치는 창에서 다음 문자 하나를 타깃으로 삼는 점, 정수 int64 텐서를 반환하는 점이다. 각각 사전 복원·문맥 길이·Embedding 및 교차엔트로피의 계약과 연결된다. 공백·줄바꿈도 포함한다는 선택과 미등록 문자 처리의 부재를 함께 설명한다. 활성화·정규화·잔차 연결은 이 도구에 없는 뉴럴 넷트웍 설계다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
인코딩·창 생성·분할에서 어떤 입력 검증이 필요한가?

</div>

??? success "연습문제 2 풀이"

    텍스트 타입·길이, 유효한 문맥 길이, 사전에 있는 문자와 decode ID 범위를 확인한다. 창 생성 후 X·y의 표본 수와 정수 자료형·어휘 범위를 점검한다. 분할 비율과 양쪽 자료 수, X·y 대응도 유지해야 한다. 필요한 특수 토큰·유니코드 정규화·미등록 문자 폴리시를 명시한다. 이 객체에 이미지 채널 검사나 float32 입력 검사를 붙이는 것은 맞지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
표시된 main이 실패할 수 있는 두 가지 원인을 설명하라.

</div>

??? success "연습문제 3 풀이"

    load_sample_text의 저장 변수 표본과 반환 변수 sample이 일치하지 않아 먼저 NameError가 난다. 그 문제를 해결해도 사전에 없는 문자를 encode하면 KeyError가 난다. 둘 다 학습 발산·과적합과 다른 데이터 계약 문제다. 실행 순서와 예시 문자열의 사전 포함 여부를 정적으로 확인하고, 오류 원인을 뉴럴 넷트웍 하이퍼파라미터로 해결하려 하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 객체에 적합한 검증 성질을 설계하라.

</div>

??? success "연습문제 4 풀이"

    아래 골격은 CharacterDataset을 nn.Module처럼 호출하고 parameters·backward를 사용해 맞지 않는다. 대신 코퍼스 문자에 대한 decode(encode(text))의 왕복, ID 사전의 일대일 대응, 작은 텍스트의 창·타깃 정렬을 확인한다. 예를 들어 ababa,L=2이면 입력 ab,ba,ab와 타깃 a,b,a가 되어야 한다. 빈 자료·짧은 자료·미등록 문자에 어떤 폴리시를 적용할지도 정한다. 경사 흐름은 데이터를 받은 실제 학습 모듈의 별도 검증이다.

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


## 정리하며

문자 데이터 구성은 사전·문맥 창·타깃 정렬의 계약을 정한다. 제공 예시에는 로더 변수와 사전 외 문자 문제가 있으며 학습을 검증하는 코드가 아니다. 분할 목적에 따라 문서 경계·어휘 범위·미등록 문자 폴리시를 명확히 해야 한다.
