# 자료 뭉치 만들기

이름 알아보기 자료 뭉치 만들기와 꼴 갖추기. 이름 알아보기 자료 뭉치를 만들고 꼴을 갖추는 도구.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""
이름 알아보기 자료 뭉치 만들기와 꼴 갖추기
====================================

이름 알아보기 자료 뭉치를 만들고 꼴을 갖추는 도구.

받치는 것:
- IOB 이름표 붙이기
- 자료 나누기
- 꼴 바꾸기

지은이: 배움 목적
날짜: 2025
"""

from typing import List, Tuple, Dict
import random

# ========================================================================
# 메인
# ========================================================================


class NERDatasetBuilder:
    """제대로 꼴을 갖춘 이름 알아보기 자료 뭉치 세우기."""
    
    def __init__(self):
        """자료 뭉치 세우개 첫자리매김."""
        self.samples = []
    
    def add_sample(self, text: str, entities: List[Dict]):
        """
        자료 뭉치에 표본 하나 더하기.
        
        인수:
            text: 글월 문자열
            entities: 'text', 'type', 'start', 'end'를 담은 것 사전의 목록
        """
        self.samples.append({
            'text': text,
            'entities': entities
        })
    
    def to_iob_format(self) -> List[Tuple[List[str], List[str]]]:
        """
        자료 뭉치를 IOB 꼴로 바꾸기.
        
        반환값:
            (토막들, 이름표들) 튜플의 목록
        """
        iob_data = []
        
        for sample in self.samples:
            text = sample['text']
            entities = sample['entities']
            
            # 단순 토막내기
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # 이름표 매기기(간추림 — 토막이 맞는다고 가정)
            for entity in entities:
                entity_tokens = entity['text'].split()
                # 토막 목록에서 것의 자리 찾기
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        labels[i] = f"B-{entity['type']}"
                        for j in range(1, len(entity_tokens)):
                            labels[i+j] = f"I-{entity['type']}"
                        break
            
            iob_data.append((tokens, labels))
        
        return iob_data
    
    def train_test_split(self, test_size: float = 0.2, 
                        random_seed: int = 42) -> Tuple[List, List]:
        """
        자료 뭉치를 익힘 뭉치와 시험 뭉치로 나누기.
        
        인수:
            test_size: 시험 뭉치로 쓸 자료의 몫
            random_seed: 같은 결과를 얻기 위한 마구잡이 씨앗
            
        반환값:
            (익힘 표본, 시험 표본) 튜플
        """
        random.seed(random_seed)
        samples_copy = self.samples.copy()
        random.shuffle(samples_copy)
        
        split_idx = int(len(samples_copy) * (1 - test_size))
        train = samples_copy[:split_idx]
        test = samples_copy[split_idx:]
        
        return train, test


if __name__ == "__main__":
    # 예
    builder = NERDatasetBuilder()
    
    builder.add_sample(
        "Steve Jobs founded Apple Inc.",
        [
            {'text': 'Steve Jobs', 'type': 'PER', 'start': 0, 'end': 10},
            {'text': 'Apple Inc.', 'type': 'ORG', 'start': 19, 'end': 29}
        ]
    )
    
    iob_data = builder.to_iob_format()
    print(f"IOB format: {iob_data}")```

## 2. 논의

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다. 단원별로 나뉜 짜임 덕분에 낱낱의 조각을 익히고 다른 일이나 자료 뭉치에 맞게 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 차례 이름표 붙이기에 알맞은지 설명하여라.

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
자료 뭉치 만들기의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_nerdatasetbuilder():
        model = NERDatasetBuilder(...)
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

**다룬 것** — 자료 뭉치 만들기

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다.

고갱이 갈래는 `NERDatasetBuilder`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
