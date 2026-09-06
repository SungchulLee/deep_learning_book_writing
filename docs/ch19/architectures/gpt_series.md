# GPT 계열: 자기되돌리기 말 모델의 흐름
## 학습 목표

- GPT-1에서 GPT-4까지의 흐름을 좇는다
- 판마다의 얼개와 익히기 바뀜을 이해한다
- 규모에서 떠오르는 능력을 가능하게 한 것을 살핀다
- GPT 변종과 열린 소스 대안을 견준다

## GPT-1: 미리 익히기로 말 이해하기(2018)

### 핵심 새로움

지어내는 미리 익히기와 가르는 곱게 다듬기가 여러 자연어 일에서 센 결과를 냄을 처음 보였다.

### 구조

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

### 학습

- **자료**: BooksCorpus(책 약 7000권, 낱말 8억 개)
- **목표**: 인과 말 나타내기
- **새로움**: 자연어 다루기의 옮겨 배우기

### 곱게 다듬기 방식

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

## GPT-2: 영 발 일 옮기기(2019)

### 핵심 새로움

더 큰 모델이 시킴말의 꼴만으로 곱게 다듬지 않고도 일을 영 발로 해냄을 보였다.

### 규모의 흐름

| 변종 | 매개변수 | 층 | 숨은 크기 | 머리 |
|---------|------------|--------|--------|-------|
| 소형 | 1억 1700만 | 12 | 768 | 12 |
| 중형 | 3억 4500만 | 24 | 1024 | 16 |
| 대형 | 7억 6200만 | 36 | 1280 | 20 |
| 초대형 | 15억 | 48 | 1600 | 25 |

### 익힘 자료: WebText

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

### 영 발 시킴말의 발견

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

## GPT-3: 맥락 안에서 배우기(2020)

### 핵심 새로움

매개변수 1750억까지 키우자 **맥락 안에서 배우기**가 드러났다. 곧 매개변수를 고치지 않고 시킴말의 보기만으로 일을 해낸다.

### 얼개 자세히

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

# 빽빽한 눈길과 성긴 눈길을 번갈아 쓰는 결
# 큰 규모에서 안정되도록 고친 첫자리매김
```

### 익히기 규모

| 자원 | 양 |
|----------|--------|
| 익힘 토막 | 3000억 |
| 셈 | 약 3.14 × 10²³ FLOPs |
| 익히는 시간 | V100 1024대로 약 34일 |
| Cost estimate | \$4.6M |

### 맥락 안에서 배우기의 틀

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

### GPT-3 변종

| 모델 | 매개변수 | 쓰임새 |
|-------|------------|----------|
| Ada | 3억 5000만 | 단순한 일, 빠름 |
| Babbage | 13억 | 알맞은 복잡도 |
| Curie | 67억 | 좋은 균형 |
| Davinci | 1750억 | 가장 좋은 품질 |

## InstructGPT / GPT-3.5: 결 맞추기(2022)

### 핵심 새로움

모델의 내놓음을 사람의 뜻에 맞추는 사람 되먹임 북돋움 배움(RLHF).

### 익히기 물길

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

### ChatGPT 익힘 자료 규모

| 단계 | 자료 크기 |
|-------|-----------|
| 미리 익히기 | 글 약 570GB |
| 살펴 배우는 곱게 다듬기 | 시범 약 1만 3천 |
| 갚음 모델 | 견줌 약 3만 3천 |
| PPO | 시킴말 약 3만 1천 |

## GPT-4: 여러 갈래 능력(2023)

### 핵심 새로움

1. **여러 갈래 들임**: 글과 그림
2. **더 긴 맥락**: 토막 8K/32K/128K
3. **나아진 따짐**: 큰 규모의 생각의 사슬
4. **더 나은 눈금 맞추기**: 더 믿음직한 믿음도

### 알려진 능력

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

### 얼개(소문/비공식)

```python
# 짐작한 GPT-4 얼개(OpenAI가 확인하지 않음)
gpt4_rumored = {
    'architecture': 'Mixture of Experts (MoE)',
    'total_parameters': '~1.8T',
    'active_parameters': '~220B per forward pass',
    'num_experts': 16,
    'experts_per_token': 2,
    'context_lengths': [8192, 32768, 128000]
}
```

### 보기 능력

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

## 흐름 간추림

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

## GPT 얼개 코드

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

## 열린 모델과의 견줌

| 갈래 | GPT-4 | LLaMA-2 70B | Mistral 7B |
|--------|-------|-------------|------------|
| 매개변수 | 약 1조(전문가 섞기) | 700억 | 70억 |
| 열린 무게 | ✗ | ✓ | ✓ |
| 따짐 | 가장 좋음 | 셈 | 좋음 |
| 물음당 값 | $$$ | 직접 띄우기 | 직접 띄우기 |
| 곱게 다듬기 | API만 | 온전히 열림 | 온전히 열림 |

## 핵심 정리

1. **규모가 능력을 연다**: GPT 판마다 새 능력이 열렸다
2. **시킴말이 나아왔다**: 곱게 다듬기 → 영 발 → 몇 발 → 맥락 안에서
3. **결 맞추기가 중요하다**: 사람 되먹임 북돋움 배움이 모델을 일반 쓰는 이에게 쓸모 있게 만들었다
4. **얼개의 새로움**: 전문가 섞기가 규모를 더 키우게 할 수 있다
5. **여러 갈래가 넓어진다**: 보기, 소리, 영상 아우르기

## 참고 문헌

1. Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training.
2. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
3. Brown, T., et al. (2020). Language Models are Few-Shot Learners.
4. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.
5. OpenAI. (2023). GPT-4 Technical Report.

## 연습문제

**연습문제 1.**
GPT, BERT, T5의 얼개 차이를 견주어라. 미리 익히기 목표는 어떻게 다른가?

??? success "연습문제 1 풀이"
    | 모델 | 얼개 | 미리 익히기 목표 | 방향 |
    |-------|-------------|----------------------|----------------|
    | **GPT** | 풀개만의 변환기 | 인과 말 나타내기(다음 토막 어림) | 왼쪽에서 오른쪽만 |
    | **BERT** | 부호기만의 변환기 | 가린 말 나타내기 + 다음 월 어림 | 두 방향 |
    | **T5** | 부호기-풀개 변환기 | 구간 망가뜨리기(잡음 없애기) | 부호기: 두 방향, 풀개: 왼쪽에서 오른쪽 |

    GPT는 만들어 내기에, BERT는 이해와 갈래 매기기에 뛰어나며, T5는 모든 일을 글에서 글로 세워 둘 다 잘한다.

---

**연습문제 2.**
GPT-1에서 GPT-4까지의 흐름을 좇아라. 걸음마다의 핵심 규모 눈썰미는 무엇인가?

??? success "연습문제 2 풀이"
    **GPT-1**(매개변수 1억 1700만): 살펴보지 않는 미리 익히기와 살펴 배우는 곱게 다듬기가 여러 일에 통함을 보였다. **GPT-2**(15억): 규모만으로 영 발 성능이 나옴을 보였고 "말 모델은 살펴보지 않는 여러 일 배우개"라는 눈썰미를 들여왔다. **GPT-3**(1750억): 기울기 고침 없이 몇 발 맥락 안에서 배우기를 보였고 규모 법칙을 세웠다. **GPT-4**(크기 미공개): 여러 갈래(글 + 그림)이며 사람 되먹임 북돋움 배움으로 따짐, 시킴 따르기, 안전을 크게 낫게 했다. 핵심 눈썰미: 세대마다 매개변수, 자료, 셈을 키우면 작은 규모에는 없던 떠오르는 능력이 나옴을 보였다.

---

**연습문제 3.**
자기되돌리기 말 나타내기와 가린 말 나타내기의 차이는 무엇인가? 요즘 큰 말 모델은 왜 대부분 자기되돌리기인가?

??? success "연습문제 3 풀이"
    **Autoregressive** models predict the next token given all previous tokens: $p(x_t | x_{<t})$. **Masked** models predict randomly masked tokens given all unmasked tokens: $p(x_t | x_{\setminus t})$. Autoregressive models dominate because: (1) they naturally generate text left-to-right, matching how humans read and write, (2) the causal structure enables efficient KV-caching during inference, (3) scaling laws favor autoregressive objectives (better sample efficiency at scale), and (4) in-context learning and instruction following emerge more naturally from next-token prediction.

---

**연습문제 4.**
맥락 안에서 배우기라는 생각을 밝혀라. 왜 놀라우며 한계는 무엇인가?

??? success "연습문제 4 풀이"
    맥락 안에서 배우기(ICL)는 큰 말 모델이 기울기를 조금도 고치지 않고 시킴말에 든 보기에 조건을 걸어 일을 해내는 힘이다. "Translate English to French: sea otter => loutre de mer, cheese => " 같은 시킴말을 주면 모델이 "fromage"를 올바로 내놓는다. 모델을 옮김에 대놓고 익힌 적이 없는데도 그렇다는 점이 놀랍다. 갖가지 글로 미리 익히는 동안 보기에서 일의 무늬를 미루는 법을 배운 것이다. **한계**: (1) 여러 걸음 따짐이 필요한 복잡한 일에서는 성능이 떨어진다. (2) 보기의 차례와 꼴에 민감하다. (3) 맥락 창의 길이에 매인다. (4) 특화된 일에서는 곱게 다듬은 성능에 못 미친다. (5) 그 얼개가 이론으로 온전히 밝혀지지 않았다.
