# GPT 계열: 자기회귀 언어 모델의 흐름

GPT 계열의 연구는 생성형 사전 학습과 과제별 파인튜닝에서 프롬프트 기반 평가·맥락 내 학습·사람 선호에 따른 후속 학습으로 이어졌다. 이 절은 2018~2023년 주요 논문의 역사적 맥락을 다루며 현재 제품 목록·가격·성능 순위를 제공하지 않는다. 공개된 연구 결과와 비공개 구조에 대한 추측을 구분한다.

## 1. GPT-1과 생성형 사전 학습

[GPT-1 연구](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)는 언어 모델 사전 학습 뒤 분류 등 과제의 지도 파인튜닝을 결합해 여러 자연어 처리 과제에서 전이 효과를 보였다. 사전 학습·전이 학습 자체를 처음 발명한 연구라는 뜻은 아니다.

```python
# GPT-1 자리매김
config = {
    'n_layers': 12,
    'n_heads': 12,
    'd_model': 768,
    'vocab_size': 40000,  # 바이트 짝 부호
    'context_length': 512,
    'parameters': '117M'
}
```

이 설정 사전은 12층·768차원·12head와 512 문맥의 개념 요약이며 완성된 모델 생성 코드는 아니다. 실제 토크나이저 어휘·특수 토큰·구현과 매개변수 집계 기준을 맞춰야 한다.

```python
def gpt1_finetune_objective(lm_logits, task_logits, labels, lm_weight=0.5):
    """
    GPT-1은 말 모델 손실과 일에 맞춘 손실을 아우른다.
    
    전체 손실 = 일 손실 + λ * 말 모델 손실
    """
    task_loss = cross_entropy(task_logits, labels)
    lm_loss = cross_entropy(lm_logits[:-1], lm_logits[1:])
    
    return task_loss + lm_weight * lm_loss
```

보조 언어 모델 목적을 파인튜닝에 함께 사용하는 생각을 보여 주지만 gpt1_finetune_objective는 올바른 완성 손실이 아니다. cross_entropy가 정의되어 있지 않고 lm_logits[1:]를 목표 토큰 ID 대신 넣는다. B×T×V 로짓에서 첫 축을 자르면 배치 축을 이동하는 문제도 있다. 언어 모델 목표는 입력·정답 ID의 시간 축 정렬과 손실 포함 위치로 구성해야 한다.

## 2. GPT-2와 프롬프트 기반 전이

[GPT-2 보고서](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)는 WebText와 더 큰 모델에서 과제별 추가 파인튜닝 없이 평가하는 설정을 연구했다. 보고서 표의 규모와 공개 구현의 정확한 매개변수 집계는 차이가 있을 수 있으므로 수치의 출처를 명시해야 한다. 모델 크기만으로 모든 과제의 zero-shot 성공을 보장하지 않는다.

```python
# WebText 고르기 물길
def webtext_filter(url, content):
    """
    레딧 바탕 품질 거르기:
    - 레딧에서 카르마 3 이상인 이음
    - 겹침 없앰
    - 위키백과 뺌(값매김과 겹침)
    """
    return reddit_karma(url) >= 3 and not is_wikipedia(url)

# 결과: 문서 800만 개, 글 40GB
```

webtext_filter는 수집 아이디어를 요약한 의사코드다. reddit_karma·is_wikipedia가 정의되어 있지 않으며 중복 제거·본문 추출·품질·권리·개인정보 처리까지 구현하는 수집기가 아니다.

```python
def gpt2_zero_shot_translation():
    """
    GPT-2는 자료를 시킴말 꼴로 꾸미면 맨몸으로도 일을 할 수 있음을 알아냈다.
    """
    prompt = """
    영어: Hello, how are you?
    프랑스어: Bonjour, comment allez-vous?
    
    영어: The weather is nice today.
    French:"""
    
    # 모델이 이어 쓴다: " Le temps est beau aujourd'hui."
    return generate(prompt)
```

gpt2_zero_shot_translation이라는 이름과 달리 프롬프트에는 번역 예시가 이미 하나 들어 있다. 평가 대상 과제의 정답 시범이 없는 zero-shot과 구분해야 한다. generate도 정의되지 않아 실행 결과가 없으며 단지 프롬프트를 보여 주는 예다.

GPT-2는 각 sub-block 입력으로 LayerNorm을 옮긴 pre-norm과 마지막 LayerNorm을 사용했다. 따라서 GPT 계열을 전부 원래 트랜스포머의 post-norm으로 설명하는 것은 맞지 않는다.

## 3. GPT-3와 맥락 내 학습

[GPT-3 논문](https://arxiv.org/abs/2005.14165)은 최대 1750억 매개변수 모델을 연구하며 과제 설명과 시범을 입력에 제공하고 평가 시 가중치를 갱신하지 않는 few-shot 설정을 비교했다. 맥락 내 학습(ICL)은 추론 문맥에 조건을 거는 방식이며 영구적인 파인튜닝과 다르다.

```python
gpt3_config = {
    'n_layers': 96,
    'n_heads': 96,
    'd_model': 12288,
    'head_dim': 128,  # d_model / n_heads
    'd_ff': 49152,     # 4 * d_model
    'vocab_size': 50257,
    'context_length': 2048,
    'parameters': '175B'
}

# 빽빽한 어텐션과 성긴 어텐션을 번갈아 쓰는 결
# 큰 규모에서 안정되도록 고친 첫자리매김
```

위 사전은 큰 GPT-3 변형의 구조 요약이다. 학습 시간·비용은 하드웨어·병렬화·이용률·데이터·계약에 따라 달라지므로 특정 GPU 수와 달러 추정치를 논문이 보장한 고정 사실로 제시하지 않는다. Ada·Babbage·Curie·Davinci 같은 과거 API 이름을 논문의 특정 크기에 일대일 대응시켜 단정하지도 않는다.

```python
def in_context_learning_demo():
    """GPT-3의 세 가지 틀."""
    
    # 맨몸: 일 설명만
    zero_shot = """
    영어를 프랑스어로 옮겨라:
    cheese =>"""
    
    # 한 보기: 보기 하나
    one_shot = """
    영어를 프랑스어로 옮겨라:
    sea otter => loutre de mer
    cheese =>"""
    
    # 몇 보기: 보기 여럿
    few_shot = """
    영어를 프랑스어로 옮겨라:
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush giraffe => girafe en peluche
    cheese =>"""
    
    # 몇 보기 성능이 곱게 다듬은 모델에 다가간다
```

이 함수는 zero-shot·one-shot·few-shot 프롬프트를 지역 변수로 만들 뿐 모델 호출이나 반환·출력을 하지 않는다. 예시 수·순서·레이블·형식·길이는 ICL 성능에 영향을 줄 수 있다. 번역 답이 맞을 것이라는 보장도 없고, 사전 학습 중 번역 관련 자료를 전혀 보지 않았다는 결론도 낼 수 없다.

## 4. InstructGPT와 후속 학습

[InstructGPT 연구](https://arxiv.org/abs/2203.02155)는 사람의 시범을 이용한 SFT, 출력 비교를 이용한 보상 모델, PPO 기반 최적화를 연구했다. 이는 다음 토큰 우도와 사람의 선호가 같은 목표가 아니라는 점을 보여 준다.

```python
def instructgpt_training_stages():
    """
    결 맞추기를 위한 세 단계 익히기 과정.
    """
    
    # 1단계: 이끌린 곱게 다듬기(SFT)
    # 사람이 쓴 본보기
    sft_data = [
        {"prompt": "Explain quantum computing", 
         "response": "[Human-written explanation]"},
        # ... 보기 1만 3천 개
    ]
    
    # 2단계: 갚음 모델 익히기
    # 사람의 선호: A > B
    comparison_data = [
        {"prompt": "...", 
         "response_a": "...", 
         "response_b": "...",
         "preference": "a"},  # 사람이 A를 골랐다
        # ... 견줌 3만 3천 개
    ]
    
    # 3단계: PPO(가까운 방침 가장 좋게 하기)
    # 갚음 모델에 맞춰 방침을 가장 좋게 한다
    # SFT 모델에 가깝게 머물도록 KL 벌주기를 곁들여
```

함수는 자료 형식의 예만 선언하며 SFT·보상 학습·PPO를 실행하지 않는다. 이 연구의 자료 수와 모델 크기를 GPT-3.5나 ChatGPT 전체의 공개되지 않은 학습 규모로 바꾸어 인용해서는 안 된다. 선호 학습이 사실성·안전성·불확실성 보정을 완전히 해결하는 것도 아니다.

## 5. GPT-4 보고서의 공개 범위

[GPT-4 기술 보고서](https://arxiv.org/html/2303.08774v6)는 이미지·텍스트 입력과 텍스트 출력을 처리하는 모델 및 여러 평가 결과를 설명한다. 성능 수치는 해당 보고서의 과제·프롬프트·채점 조건에 따른 역사적 결과다. 모든 시험·업무의 사람 수준 능력이나 현재 최고 성능으로 일반화하지 않는다.

```python
# GPT-4 잣대 성능(어림)
benchmarks = {
    'MMLU': 86.4,      # GPT-3.5 대비: 70.0
    'HellaSwag': 95.3, # GPT-3.5 대비: 85.5
    'HumanEval': 67.0, # GPT-3.5 대비: 48.1
    'GSM8K': 92.0,     # GPT-3.5 대비: 57.1
    'Bar Exam': '90th percentile'
}
```

위 benchmarks는 고정 숫자를 담은 사전이며 이 파일에서 시험을 실행해 얻은 결과가 아니다. 정확한 재인용에는 원 보고서의 평가 조건과 모델 스냅샷을 확인해야 한다.

보고서는 모델 크기와 상세 구조·학습 계산 등을 공개하지 않는다고 명시한다. 따라서 다음 기존 코드에 남은 전문가 수·총/활성 매개변수 등의 숫자는 **확인되지 않은 추측**이며 구조의 사실 자료로 사용하지 않는다. 여러 시점의 GPT-4 제품 문맥 길이도 원 모델의 단일 설정으로 합치지 않는다.

```python
# 짐작한 GPT-4 구조(OpenAI가 확인하지 않음)
gpt4_rumored = {
    'architecture': 'Mixture of Experts (MoE)',
    'total_parameters': '~1.8T',
    'active_parameters': '~220B per forward pass',
    'num_experts': 16,
    'experts_per_token': 2,
    'context_lengths': [8192, 32768, 128000]
}
```

```python
def gpt4_vision_example():
    """
    GPT-4V는 글과 그림이 엇갈려 놓인 것을 처리한다.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]
        }
    ]
    
    # 모델이 할 수 있는 일:
    # - 그림 내용을 적기
    # - 그림에 대한 물음에 답하기
    # - 글자 알아보기
    # - 도표와 그래프 살피기
```

gpt4_vision_example은 메시지 형식의 지역 변수를 만들 뿐 모델을 호출하거나 답을 반환하지 않는다. base64도 자리표시자다. 이 절의 목적은 역사적 멀티모달 능력의 설명이며 현재 API의 완성된 사용 예가 아니다.

## 8. 기존 흐름 도식 읽기

```
GPT-1 (2018)                GPT-2 (2019)
117M params                 1.5B params
Fine-tuning required   →    Zero-shot possible
Single task focus           Multi-task potential
         ↓                        ↓
         
GPT-3 (2020)                GPT-3.5/ChatGPT (2022)
175B params                 ~175B params + RLHF
In-context learning    →    Aligned to preferences
Few-shot master             Conversational AI
         ↓                        ↓
         
GPT-4 (2023)                GPT-4o (2024)
~1T+ params (MoE?)          Multimodal I/O
Multimodal input       →    Native audio/video
Strongest reasoning         Real-time interaction
```

이 도식은 기존 내용을 보존한 것으로 GPT-3.5의 매개변수 추정, GPT-4의 MoE·조 단위 추정과 ‘가장 강한 추론’ 같은 표현을 검증된 사실로 받아들이면 안 된다. 계열의 발전은 규모 하나만이 아니라 자료·최적화·평가·후속 학습과 시스템 구성의 변화도 포함한다. 모델별 품질은 같은 과제·시점·설정에서 비교해야 한다.

## 9. GPT 형태의 축약 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTBlock(nn.Module):
    """여느 GPT 변환기 덩이."""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['d_model'])
        self.ln2 = nn.LayerNorm(config['d_model'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['d_model'], 4 * config['d_model']),
            nn.GELU(),
            nn.Linear(4 * config['d_model'], config['d_model']),
            nn.Dropout(config['dropout'])
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """GPT 언어 모형."""
    
    def __init__(self, config):
        super().__init__()
        
        self.tok_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_emb = nn.Embedding(config['context_length'], config['d_model'])
        self.drop = nn.Dropout(config['dropout'])
        
        self.blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config['n_layers'])
        ])
        
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # 무게 묶기
        self.tok_emb.weight = self.head.weight
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
```

GPTBlock은 pre-norm 어텐션과 GELU FFN의 두 잔차 경로를 표현한다. 그러나 CausalSelfAttention은 정의되어 있지 않고 앞 설정 사전들에는 dropout도 없어 그대로 완성된 실행 예가 아니다. 실제 인과·padding 마스크와 학습·생성 루프도 더 필요하다.

GPT.forward는 B×T ID를 토큰·위치 임베딩으로 바꾸고 로짓을 반환한다. targets를 내부에서 이동하지 않으므로 호출부가 이미 다음 토큰 ID를 같은 모양으로 준비해야 한다. T는 위치 임베딩 한도를 넘을 수 없고, 입력·목표의 장치·정수 범위·padding 제외를 맞춰야 한다. 입력 임베딩과 출력 head는 같은 가중치를 공유한다.

이 축약 모델은 학습·체크포인트 로딩·평가를 수행하지 않으며 실제 GPT-4 구조를 구현하는 코드가 아니다. 공개 가중치 모델과 호스팅 모델은 재현성·운영 자원·접근 방식이 다르지만 이 차이만으로 성능·가격·파인튜닝 가능성을 일률적으로 순위화하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
GPT-1·GPT-2·GPT-3 논문의 연구 설정 변화와 평가 시 가중치 갱신 여부를 구분하라.

</div>

??? success "연습문제 1 풀이"

    GPT-1은 언어 모델 사전 학습 후 지도 파인튜닝의 전이를 연구했다. GPT-2는 추가 과제별 학습 없이 프롬프트 기반 전이를, GPT-3는 가중치 갱신 없이 시범을 문맥에 주는 few-shot 평가를 폭넓게 연구했다. 이런 구분이 작은 모델에서는 ICL이 절대 불가능하거나 모델이 해당 과제 관련 자료를 전혀 보지 않았다는 뜻은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
GPT-4의 매개변수·MoE 구조를 이 페이지의 사전에서 확정할 수 없는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"

    기술 보고서는 상세 구조·모델 크기를 공개하지 않는다고 밝힌다. gpt4_rumored와 기존 도식의 숫자는 비공식 추측이며 측정이나 재현 근거가 아니다. 확인된 입력·출력 능력과 평가 결과를 비공개 내부 구조의 증거로 바꾸지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
GPT.forward의 목표 정렬과 앞 페이지 CausalLMHead의 내부 이동을 비교하라.

</div>

??? success "연습문제 3 풀이"

    GPT.forward는 logits와 targets를 같은 위치에서 곧바로 교차 엔트로피에 넣으므로 호출부에서 다음 토큰 목표를 준비해야 한다. 내부에서 한 칸 이동하는 head에 이미 이동한 목표를 주면 두 번 이동한다. 각 함수의 계약을 확인하고 인과성이 은닉 표현의 모든 정보 경로에서 유지되는지도 검토해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
ICL·파인튜닝·사전 학습 데이터에 관한 주장들을 구분하라.

</div>

??? success "연습문제 4 풀이"

    ICL은 현재 입력의 예시에 조건을 걸어 예측하며 그 호출에서 가중치를 갱신하지 않는다. 파인튜닝은 학습 자료의 손실로 매개변수를 갱신한다. 평가 때 파인튜닝하지 않았다는 사실만으로 사전 학습에서 유사 문제·번역·설명을 보지 않았다고 결론 내릴 수 없다. 예시 순서·길이·분할과 오염을 함께 평가한다.



## 정리하며

GPT 계열의 흐름은 사전 학습·프롬프트·ICL·후속 선호 학습의 변화를 보여 준다. 논문 결과·의사코드·현재 제품·비공개 구조 추측을 구분하고, 목표 정렬과 실제 구현 범위를 확인해야 한다.

**참고 문헌**

1. Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training.
2. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
3. Brown, T., et al. (2020). Language Models are Few-Shot Learners.
4. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.
5. OpenAI. (2023). GPT-4 기술 보고서.
