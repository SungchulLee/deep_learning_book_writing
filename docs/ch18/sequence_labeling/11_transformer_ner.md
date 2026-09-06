# 변환기 이름 알아보기

미리 익힌 모델을 쓰는 변환기 바탕 이름 알아보기. 이름 알아보기에 BERT/RoBERTa 쓰기.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 코드

```python
"""
미리 익힌 모델을 쓰는 변환기 바탕 이름 알아보기
===============================================

이름 알아보기에 BERT/RoBERTa 쓰기.

핵심 개념:
- 미리 익힌 변환기 곱게 다듬기
- 아래낱말 토막내기 다루기
- 토막 갈래 매기기

지은이: 배움 목적
날짜: 2025
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict

# ========================================================================
# 메인
# ========================================================================


class TransformerNER(nn.Module):
    """
    미리 익힌 모델을 쓰는 변환기 바탕 이름 알아보기.
    
    구조:
    들임 → BERT/RoBERTa → 선형 → 소프트맥스 → 이름표
    """
    
    def __init__(self, model_name: str = 'bert-base-cased', num_labels: int = 9):
        """
        변환기 이름 알아보기 첫자리매김.
        
        인수:
            model_name: Hugging Face 모델 이름
            num_labels: 것 이름표의 개수
        """
        super(TransformerNER, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # 미리 익힌 변환기 읽어 들이기
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 정칙화를 위한 드롭아웃
        self.dropout = nn.Dropout(0.1)
        
        # 분류 머리
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None):
        """
        앞먹임.
        
        인수:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        반환값:
            logits: [batch_size, seq_len, num_labels]
        """
        # 변환기의 내놓음 얻기
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 숨은 상태 얻기
        sequence_output = outputs[0]  # [배치, seq_len, hidden_size]
        
        # 떨구기 쓰기
        sequence_output = self.dropout(sequence_output)
        
        # 분류
        logits = self.classifier(sequence_output)  # [batch, seq_len, num_labels]
        
        return logits
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        글 속의 것 어림하기.
        
        인수:
            text: Input text string
            
        반환값:
            (토막, 이름표) 짝의 목록
        """
        # 토큰으로 나누기
        encoding = self.tokenizer(text, return_tensors='pt', 
                                  padding=True, truncation=True)
        
        # 순전파
        with torch.no_grad():
            logits = self.forward(encoding['input_ids'], 
                                 encoding['attention_mask'])
        
        # 예측을 얻는다
        predictions = torch.argmax(logits, dim=-1)[0]
        
        # 토막으로 되돌려 대응시키기
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        return list(zip(tokens, predictions.tolist()))


if __name__ == "__main__":
    print("Transformer NER model template")
    print("Note: Requires transformers library and pre-trained models")
    print("Example: BERT, RoBERTa, DistilBERT for token classification")```

## 논의

`TransformerNER` 클래스는 PyTorch의 `nn.Module` 사이를 써서 모델 얼개를 감싼다. `forward` 메서드가 셈 그래프를 정하므로 익히는 동안 PyTorch의 자동 미분 체계가 기울기 셈을 알아서 다룬다. 이 단원별 꾸밈 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 끼워 넣기가 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
붙박이 첫자리매김일 때 `TransformerNER`의 배울 수 있는 매개변수 전체 개수를 셈하여라. 무게와 치우침을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `TransformerNER`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = TransformerNER(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
