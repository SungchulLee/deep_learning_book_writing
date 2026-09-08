# 실습 08

길잡이 08: 미리 익힌 말 모델 곱게 다듬기. GPT-2 같은 미리 익힌 모델을 뒤따르는 일에 써먹는 법을 배운다.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 말 모델 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""
길잡이 08: 미리 익힌 말 모델 곱게 다듬기
====================================================

GPT-2 같은 미리 익힌 모델을 뒤따르는 일에 써먹는 법을 배운다.
옮겨 배우기 덕분에 큰 규모 미리 익히기의 앎을 쓸 수 있다.

핵심 개념:
- 미리 익힌 모델(GPT-2, BERT, RoBERTa)
- 곱게 다듬기와 특징 뽑기
- 분야 맞추기
- 몇 개만으로 배우기

미리 익힌 모델의 이점:
1. 배운 말의 앎
2. 익히는 시간과 자료가 줄어든다
3. 두루 통함이 더 낫다
4. 가장 앞선 성능
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

# ========================================================================
# 메인
# ========================================================================


class TextDataset(Dataset):
    """미리 익힌 모델을 곱게 다듬기 위한 자료 뭉치."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = []
        
        for text in texts:
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.inputs.append({
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()
            })
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]


def finetune_gpt2(train_texts, model_name='gpt2', epochs=3):
    """
    맞춤 글 자료로 GPT-2 곱게 다듬기.
    
    인수:
        train_texts: 익힘 글의 목록
        model_name: 미리 익힌 모델 이름('gpt2', 'gpt2-medium' 등)
        epochs: 학습 에포크 수
    """
    print(f"Fine-tuning {model_name}")
    print("=" * 60)
    
    # 미리 익힌 모델과 토막내개 읽어 들이기
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # 자료 뭉치 갖추기
    train_dataset = TextDataset(train_texts, tokenizer)
    
    # 익히기 인자
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        warmup_steps=100,
    )
    
    # 익히개
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # 곱게 다듬기
    trainer.train()
    
    return model, tokenizer


def generate_from_pretrained(model, tokenizer, prompt, max_length=50,
                             temperature=1.0, top_k=50, top_p=0.95):
    """
    곱게 다듬은 모델로 글 만들어 내기.
    
    인수:
        model: 곱게 다듬은 모델
        tokenizer: 토막내개
        prompt: 시작 글
        max_length: 만들어 낼 최대 길이
        temperature: 표집 온도
        top_k: 상위 k 표집 매개변수
        top_p: 알갱이 표집 매개변수
    """
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def demonstrate_pretrained():
    """미리 익힌 GPT-2 쓰기 보이기."""
    
    print("Loading pretrained GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence is",
        "In a distant galaxy"
    ]
    
    print("\nGenerating from pretrained model:\n")
    for prompt in prompts:
        text = generate_from_pretrained(model, tokenizer, prompt, 
                                       max_length=30)
        print(f"Prompt: {prompt}")
        print(f"Generated: {text}\n")


if __name__ == "__main__":
    print("""
미리 익힌 말 모델
==========================

널리 쓰이는 모델:
1. GPT(지어내는 미리 익힌 변환기)
   - 자기되돌리기, 왼쪽에서 오른쪽
   - 만들어 내기 일에 좋다
   
2. BERT(두 방향 부호기)
   - 가린 말 나타내기
   - 이해하는 일에 좋다

3. T5(글에서 글로)
   - 하나로 묶은 얼거리
   - 모든 일을 글 만들어 내기로

곱게 다듬기 전략:
1. 온전한 곱게 다듬기: 모든 매개변수를 고친다
2. 어댑터 단원: 작은 익힐 수 있는 층 더하기
3. 시킴말 다듬기: 이어진 시킴말을 가장 좋게 한다
4. LoRA: 무게의 낮은 계수 맞추기

좋은 버릇:
- 미리 익힐 때보다 작은 배움 비율 쓰기
- 자료 뭉치가 작으면 앞쪽 층 얼리기
- 큰 모델을 위한 기울기 쌓기 짜기
- 섞인 정밀도 익히기 쓰기
- 큰 잊음 지켜보기

익힘 문제:
1. 분야별 글로 GPT-2 곱게 다듬기
2. 모델 크기 견주기(기본, 중형, 대형)
3. 어댑터 바탕 곱게 다듬기 짜기
4. 시킴말 바탕 몇 개만으로 배우기 해 보기
5. 헷갈림도와 만든 글의 좋음으로 값매김하기
6. 큰 모델에서 작은 모델로 앎 내리기 짜기
    """)
    
    # demonstrate_pretrained()
    print("\nNote: Requires 'transformers' library: pip install transformers")```

## 2. 논의

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 말 모델의 핵심 개념을 보여 준다. 단원별로 나뉜 짜임 덕분에 낱낱의 조각을 익히고 다른 일이나 자료 뭉치에 맞게 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 말 모델에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

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
실습 08 구현을 검증하는 종합 시험 함수를 작성하라. 빈 입력, 원소가 하나뿐인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 경계 상황을 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_textdataset():
        model = TextDataset(...)
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

**다룬 것** — 실습 08

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 말 모델의 핵심 개념을 보여 준다.

고갱이 갈래는 `TextDataset`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
