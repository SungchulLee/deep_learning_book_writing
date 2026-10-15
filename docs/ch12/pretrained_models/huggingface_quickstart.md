# 허깅페이스 빠른 시작

Hugging Face Transformers로 사전 학습 체크포인트를 불러와 추론하고 작은 예제에서 미세 조정 흐름을 살펴본다. 파이프라인, 명시적인 토큰화·모델 호출과 생성의 역할을 구분한다. 기본 모델과 API 동작은 버전에 따라 달라질 수 있어 사용한 환경과 모델 revision을 기록한다.

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

pipeline은 전처리·모델 호출·후처리를 묶은 추론 인터페이스다. 모델을 생략한 예제는 라이브러리의 기본값에 의존하므로 재현하려면 모델·토크나이저·revision을 명시한다. 지원 인수와 장치 설정은 [파이프라인 문서](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial)를 확인한다. 점수는 모델의 출력이며 언제나 보정된 신뢰도를 뜻하지는 않는다.

**NLI 입력의 한계:** 예제는 두 문장을 문자열의 [SEP]로 이어 text-classification에 넘긴다. RoBERTa 토크나이저에 이 문자열이 문장 쌍을 올바르게 구분한다는 보장은 없다. 실제 전제·가설은 해당 파이프라인이나 토크나이저의 text/text_pair 인터페이스로 전달해야 한다. “비가 오면 잔디가 젖었다” 역시 전제에서 반드시 따라오는 논리적 함의는 아니다.

명시적인 추론은 eval과 no_grad를 함께 사용한다. 라벨 이름은 각 로짓의 실제 ID를 model.config.id2label에서 찾아 연결해야 하며, 사전의 순회 순서가 클래스 번호 순서라고 일반화하면 안 된다. attention_mask는 일반적으로 1이 유효, 0이 패딩이며 패딩 쿼리 출력이나 학습 정답까지 자동으로 제거하지는 않는다.

미세 조정은 기본 DistilBERT 인코더에 새 분류 헤드를 붙여 여섯 문장으로 세 에포크 학습한다. 이 데이터에는 검증·테스트 분할이 없고 라벨도 예제용이므로 성능을 판단할 수 없다. 앞의 SST-2 토크나이저를 재사용하지만 두 체크포인트의 어휘·특수 토큰이 같은지 확인해야 한다. 실제 과제에서는 모델과 맞는 토크나이저를 함께 선택한다.

GPT-2 생성은 확률 샘플링이라 결과가 달라질 수 있다. max_new_tokens는 새로 추가할 최대 토큰 수이며 출력 문자열에 프롬프트가 포함되는지 옵션도 확인한다. 영어 예제의 결과가 한국어 과제에도 그대로 적용되는 것은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 제로샷 분류 예제에서 후보 레이블과 가설 템플릿이 결과에 주는 영향을 설명하라. 실행 결과 없이 레이블 순위를 확정할 수 있는가?

</div>

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
    전기 자동차라는 내용에서 automotive와 technology가 관련 있을 것으로 예상할 수 있지만 실제 모델의 순위를 확정할 수는 없다. NLI 기반 파이프라인은 글을 전제로, 후보를 넣은 가설을 구성해 점수를 계산한다. 후보 집합·표현과 multi_label 설정에 따라 점수의 의미가 달라지며 기본 체크포인트도 명시해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
padding=True와 padding="max_length"의 길이·절단 규칙을 구분하라.

</div>

??? success "연습문제 2 풀이"
    True는 보통 현재 배치의 가장 긴 입력까지 패딩한다. max_length는 지정한 길이나 모델 최대 길이를 사용한다. 패딩만으로 긴 입력을 자르지는 않으므로 truncation을 별도로 지정한다. 동적 패딩은 짧은 배치의 계산을 줄일 수 있고 고정 크기는 일부 컴파일·배치 구성에 편리하다. 어느 쪽이 빠른지는 실제 길이 분포와 커널에 따라 다르다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 선형 워밍업·감쇠 코드의 전체 갱신 수와 기존 예제를 이어 사용할 때의 조건을 설명하라.

</div>

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
    loader의 배치가 에포크당 3개이므로 총 9번의 갱신에 int(0.1×9)=0번의 워밍업이 설정된다. “처음 10% 워밍업”이라는 설명과 실제 정수 값이 다르다. 또한 앞 코드를 이어 사용하면 이미 학습한 ft_model에 새 옵티마이저로 추가 학습한다. 원래 조건과 비교하려면 초기 가중치와 감쇠 등 설정을 맞춰야 한다. 워밍업·감쇠가 항상 성능을 높이는 것은 아니다.

## 정리하며

파이프라인의 편의성과 모델·토크나이저를 직접 다루는 방식을 연결해 읽는다. 문장 쌍 입력, 패딩·절단, 라벨 ID와 스케줄의 실제 값을 확인하며 작은 학습 예제나 생성 결과를 일반화 성능으로 해석하지 않는다.
