# 사전 학습 언어 모델의 파인튜닝

사전 학습한 언어 모델은 기존 가중치에서 출발해 새 자료의 다음 토큰 예측을 학습할 수 있다. 이 페이지는 GPT-2와 토크나이저를 불러오고 Trainer로 전체 파인튜닝하는 함수, 프롬프트에서 이어 쓰는 함수를 소개한다. 전체 파인튜닝과 LoRA·어댑터·프롬프트 튜닝은 서로 다른 갱신 방식이며 이 코드가 모두 구현하는 것은 아니다.

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
   
2. BERT(두 방향 인코더)
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
    print("\nNote: Requires 'transformers' library: pip install transformers")
```

## 2. 토큰화와 다음 토큰 목표

TextDataset은 각 문서를 max_length까지 자르고 같은 길이로 padding하여 input_ids·attention_mask·labels를 저장한다. 기본 max_length는 512다. 토큰 ID는 정수이며 attention_mask는 실제 토큰과 padding 위치를 구분한다. 길이가 1일 때도 길이 축을 유지해야 하므로 모든 크기 1인 축을 제거하는 squeeze 사용에는 주의가 필요하다.

[GPT-2의 언어 모델 head](https://huggingface.co/docs/transformers/model_doc/gpt2)는 labels를 내부에서 한 위치 이동한다. 따라서 labels에 input_ids를 넣는 방식 자체는 맞다. 다만 손실에서 제외할 위치는 −100이어야 한다. 현재 코드는 padding의 labels도 EOS ID로 남겨 두므로 attention_mask만으로 padding 목표의 손실이 제외되지 않는다.

PAD와 EOS를 같은 ID로 쓰더라도 실제 문장 종료와 padding의 역할은 구분해야 한다. padding은 attention_mask를 기준으로 목표에서 제외하고, 학습할 실제 EOS는 남기는 정책이 필요하다. 현재 토큰화는 문서 끝에 실제 EOS를 명시적으로 추가하지 않으므로 padding에서 생긴 EOS와 문서 경계 학습을 혼동하지 않는다.

## 3. 파인튜닝과 검증

finetune_gpt2는 사전 학습 가중치를 불러오고 동결이나 어댑터 설정 없이 Trainer에 넘긴다. 학습률·배치 크기·warmup은 예시 설정이며 모든 자료에 적합한 값은 아니다. 검증 데이터셋이나 검증 지표·최적 체크포인트 선택은 구성하지 않는다. 작은 자료에서도 무조건 앞쪽 층을 동결해야 한다는 규칙은 없다.

자료는 문서 단위로 학습·검증·시험을 분리하고 중복·누출을 확인한다. 검증 손실과 생성 결과, 기존 분야 성능의 변화를 함께 보아 과적합과 망각을 판단한다. 토큰화한 텐서를 모두 미리 저장하는 방식은 자료 규모에 따라 메모리 부담도 달라진다. 이 편집에서는 다운로드·파인튜닝·생성을 실행하지 않는다.

## 4. 프롬프트에서 생성하기

generate_from_pretrained는 eval·no_grad를 사용하고 temperature·top-k·top-p로 샘플링한다. 프롬프트도 반환 문자열에 포함된다. [생성 길이 설정](https://huggingface.co/docs/transformers/main_classes/text_generation)에서 디코더 전용 모델의 max_length는 프롬프트를 포함한 전체 길이 제한이며 max_new_tokens는 새 토큰 수 제한이다. 두 의미를 혼동하지 않는다.

현재 함수는 입력을 CPU에 만들고 모델 장치로 옮기지 않으며 attention_mask도 generate에 전달하지 않는다. GPU 모델이나 padding한 배치에 그대로 일반화할 수 없다. 모델·입력 장치, 문맥 한도, 빈 프롬프트와 종료 규칙을 함께 맞춰야 한다. main은 설명 문구만 출력하고 demonstrate_pretrained나 finetune_gpt2를 호출하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 예제에서 실제로 구현한 파인튜닝 방식과 데이터 처리상의 핵심 선택 세 가지를 설명하라.

</div>

??? success "연습문제 1 풀이"

    사전 학습 가중치를 전체 갱신하며 LoRA·어댑터를 추가하지 않는다. 문서별 truncation·고정 길이 padding을 사용하고 모든 텐서를 미리 저장한다. labels는 입력 ID를 복사하지만 padding을 −100으로 바꾸지 않아 손실 제외가 누락되어 있다. 이 선택들은 각각 갱신 범위·문맥/메모리·학습 목표에 영향을 준다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
attention_mask와 labels의 −100이 각각 무엇을 제어하며 PAD=EOS일 때 어떤 구분이 필요한지 설명하라.

</div>

??? success "연습문제 2 풀이"

    attention_mask는 어텐션에서 무시할 입력 위치를 지정하고 labels의 −100은 손실에서 제외할 목표 위치를 지정한다. 같은 기능이 아니다. PAD와 EOS가 같은 ID여도 padding 위치만 attention_mask로 식별해 제외하고 실제 종료 토큰의 목표는 남겨야 한다. 현재 코드에는 별도 어텐션 드롭아웃을 추가하는 것이 아니라 이 데이터·손실 계약을 먼저 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
검증 퍼플렉서티를 비교하기 위해 필요한 자료 분리와 토큰 집계를 설명하라.

</div>

??? success "연습문제 3 풀이"

    학습에 쓰지 않은 문서를 동일한 토크나이저·문맥 길이·경계 규칙으로 평가한다. padding과 제외 목표를 뺀 음의 로그우도 합을 실제 예측 토큰 수로 나눈 뒤 지수화한다. 배치별 평균을 단순 평균하면 길이가 다른 배치의 가중치가 달라질 수 있다. 생성의 유창성·관련성·사실성은 별도 기준으로 평가한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 test_textdataset이 왜 Dataset의 유효한 검증 함수가 아닌지 설명하고 정적 검토할 계약을 제시하라.

</div>

??? success "연습문제 4 풀이"

    TextDataset은 nn.Module이 아니므로 model(x)·parameters·backward를 적용하는 대상이 아니다. 아래 함수의 입력과 expected_shape도 자리표시자라 완성된 테스트가 아니다. 확인할 계약은 len과 인덱스 접근, 정수 ID와 어휘 범위, 각 항목의 길이, attention_mask와 padding 목표의 대응, truncation·빈 문서·길이 1 처리다. 실제 학습·경사 검증은 별도의 언어 모델과 손실에 대해 설계해야 한다.

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


## 정리하며

사전 학습 가중치를 사용해도 토큰화·목표 이동·padding 손실 제외·자료 분리의 책임은 남는다. 이 예제는 전체 파인튜닝과 생성 함수를 정의하지만 main에서 실행하지 않으며, padding 목표와 장치·검증 구성을 보완해야 실제 성능을 해석할 수 있다.
