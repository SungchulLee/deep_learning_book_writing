# 허깅페이스 빠른 시작

허깅페이스의 `transformers` 라이브러리는 사전 학습된 트랜스포머 모형을 쓰기 위한 높은 수준의 API를 준다. 맨바닥부터 모형을 세워 보는 것이 이해에 값지지만, 실제 시스템은 추론과 미세 조정과 생성에 대개 허깅페이스에 기댄다. 이 안내에서는 파이프라인 API, 손수 모형 싣기, 토큰 나누기의 세부, 미세 조정, 글 생성을 다룬다.

## 1. 코드

```python
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import pipeline

# 1. 파이프라인 API - 코드 없는 추론
classifier = pipeline("sentiment-analysis")
result = classifier("The actors were very convincing.")
print(f"Sentiment: {result}")

results = classifier([
    "I loved every minute of this film.",
    "The plot was confusing and the ending was disappointing.",
])
for r in results:
    print(f"  {r['label']:>8s}  (confidence: {r['score']:.3f})")

# 2. 특정 모형으로 하는 자연어 추론
model_name = "cross-encoder/nli-distilroberta-base"
nli_pipe = pipeline("text-classification", model=model_name)

pairs = [
    "A man is eating pizza. [SEP] A man is eating food.",
    "A woman is playing guitar. [SEP] Nobody is playing music.",
    "It is raining outside. [SEP] The grass is wet.",
]
for text in pairs:
    result = nli_pipe(text)
    print(f"  {result[0]['label']:>14s} ({result[0]['score']:.3f})")

# 3. 손수 싣기: AutoTokenizer와 AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

text = "HuggingFace makes NLP incredibly easy!"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)

labels = model.config.id2label
for i, (label_id, label_name) in enumerate(labels.items()):
    print(f"  {label_name}: {probs[0, i]:.4f}")

# 4. 토큰 나누기의 세부
sentences = [
    "Short text.",
    "This is a much longer sentence that demonstrates padding behaviour.",
]
encoded = tokenizer(
    sentences, padding=True, truncation=True,
    max_length=32, return_tensors="pt",
)
print(f"input_ids shape: {encoded['input_ids'].shape}")
print(f"attention_mask shape: {encoded['attention_mask'].shape}")

# 5. 미세 조정
from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, TensorDataset

texts = [
    "The stock market rallied today on strong earnings.",
    "GDP growth exceeded expectations this quarter.",
    "The company filed for bankruptcy after years of losses.",
    "Inflation continues to erode consumer purchasing power.",
    "New regulations aim to stabilize the financial sector.",
    "Unemployment claims hit a record high this month.",
]
labels = [1, 1, 0, 0, 1, 0]

encoded = tokenizer(
    texts, padding=True, truncation=True,
    max_length=64, return_tensors="pt"
)
dataset = TensorDataset(
    encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels),
)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

ft_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
optimizer = torch.optim.AdamW(ft_model.parameters(), lr=2e-5, weight_decay=0.01)

ft_model.train()
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        input_ids, attention_mask, labs = batch
        outputs = ft_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labs,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"  Epoch {epoch + 1}: loss = {total_loss / len(loader):.4f}")

# 6. GPT-2로 하는 글 생성
generator = pipeline("text-generation", model="gpt2")
prompt = "Deep learning has revolutionized"
results = generator(
    prompt, max_new_tokens=50, num_return_sequences=2,
    temperature=0.8, do_sample=True,
)
for i, r in enumerate(results):
    print(f"  Generation {i + 1}: {r['generated_text'][:200]}")
```

## 2. 논의

파이프라인 API는 사전 학습된 모형을 쓰는 가장 빠른 길이다. 코드 한 줄로 감성 분석, 개체명 인식, 질의응답, 글 생성을 할 수 있다. 속에서는 파이프라인이 토큰 나누기, 모형 싣기, 추론, 뒤처리를 다룬다. 실제로 쓸 때는 되풀이 가능하도록 특정 모형 체크포인트를 지정할 수 있다.

`AutoTokenizer`와 `AutoModel`로 손수 실으면 추론 파이프라인을 온전히 다스릴 수 있다. 토큰 나누개는 글을 입력 번호로 바꾸고, 배치 입력의 채움과 자르기를 다루며, `[CLS]`와 `[SEP]` 같은 특별 토큰을 관리한다. 주의 가림은 어느 토큰이 실제 내용이고 어느 것이 채움인지 알려 주어 모형이 뜻있는 자리에만 주의하게 한다.

사전 학습된 모형의 미세 조정은 어렵지 않다. 이름표 수에 맞는 분류 머리를 갖춘 모형을 싣고, 데이터를 토큰으로 나누고, 표준 파이토치 최적화기로 학습시키면 된다. 핵심 초매개변수는 학습률(대개 $2 \times 10^{-5}$에서 $5 \times 10^{-5}$), 가중치 감쇠, 세대 수이다. 데이터셋이 아주 작아도 미세 조정은 사전 학습에서 배운 넉넉한 표현의 덕을 본다.

## 연습문제

**연습문제 1.**
허깅페이스 파이프라인으로 "The new electric car has impressive range and acceleration"이라는 글을 후보 이름표 `["automotive", "finance", "sports", "technology"]`에 대해 영 예시 분류하라. 어느 이름표가 가장 높은 점수를 받으며 그 까닭은 무엇인가?

??? success "연습문제 1 풀이"
    ```python
    from transformers import pipeline
    zs = pipeline("zero-shot-classification")
    result = zs(
        "The new electric car has impressive range and acceleration",
        candidate_labels=["automotive", "finance", "sports", "technology"]
    )
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label}: {score:.3f}")
    ```
    글이 두 분야 모두와 관련된 전기 자동차를 다루므로 대체로 "automotive"와 "technology"가 가장 높은 점수를 받는다. 자연어 추론에 바탕한 영 예시 모형은 글과 "This text is about automotive." 같은 가설 사이의 함의를 따진다.

---

**연습문제 2.**
토큰 나누개의 `padding=True`와 `padding="max_length"`의 차이를 설명하라. 각각 언제 더 좋으며 기억 면에서는 어떤 뜻이 있는가?

??? success "연습문제 2 풀이"
    `padding=True`는 모든 수열을 배치에서 가장 긴 수열의 길이에 맞추어 채우므로 길이가 비슷한 입력의 배치에서 기억을 아낀다. `padding="max_length"`는 실제 길이와 무관하게 모두 `max_length`까지 채우므로 수열이 짧으면 기억을 버리지만 배치마다 텐서 꼴이 한결같게 된다. 학습 중에는 효율을 위해 `padding=True`를 쓴다. ONNX로 내보내거나 컴파일된 모형을 쓸 때처럼 크기가 고정된 텐서가 필요하면 `padding="max_length"`를 쓴다.

---

**연습문제 3.**
학습 단계의 처음 10% 동안 선형으로 예열한 뒤 선형으로 잦아드는 학습률 조정기를 넣도록 미세 조정 고리를 고쳐라. 허깅페이스의 `get_scheduler`를 쓰라.

??? success "연습문제 3 풀이"
    ```python
    from transformers import get_scheduler

    num_epochs = 3
    num_training_steps = num_epochs * len(loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=2e-5)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    ft_model.train()
    for epoch in range(num_epochs):
        for batch in loader:
            input_ids, attention_mask, labs = batch
            outputs = ft_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labs
            )
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    ```
    선형 예열은 미세 조정을 흔들 수 있는 초기의 큰 갱신을 막고, 선형 감쇠는 모형이 수렴함에 따라 학습률을 차츰 줄여 마지막 성능을 높인다.

## 정리하며

**다룬 것** — 허깅페이스 빠른 시작

파이프라인 API는 사전 학습된 모형을 쓰는 가장 빠른 길이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
