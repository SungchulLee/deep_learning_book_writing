# 감성 분석 학습

감성 분석을 위해 BERT 기반 모형을 학습시키는 일은 표준 전이 학습 파이프라인, 곧 사전 학습된 인코더를 싣고 분류 머리를 붙여 이름표 달린 데이터로 미세 조정하는 흐름을 보여 준다. 감성 분석은 가장 흔한 자연어 처리 잣대 가운데 하나이며, 트랜스포머 모형이 아래쪽 응용에 어떻게 맞추어 가는지 이해하기에 아주 좋은 첫 미세 조정 과제이다.

## 코드

```python
import torch
import torch.nn as nn
from bert_classifier import BERTClassifier


def train_sentiment():
    vocab_size = 10000
    num_classes = 2
    model = BERTClassifier(vocab_size, num_classes)

    print(f"BERT Classifier created!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Ready for training on sentiment analysis tasks")


if __name__ == '__main__':
    train_sentiment()
```

## 논의

이 학습 스크립트는 감성 분석(긍정·부정)을 위한 이진 분류 머리를 갖춘 `BERTClassifier`를 만든다. 어휘가 토큰 1만 개이고 768차원 임베딩의 트랜스포머 인코더 층 6개라는 기본 구조에서 모형의 매개변수는 수백만 개이다. 온전한 학습 파이프라인이라면 큰 규모의 비지도 사전 학습에서 얻은 지식을 쓰려고 이것들을 사전 학습된 체크포인트에서 초기화할 것이다.

이진 감성 분석은 입력 글마다 두 이름표 가운데 하나로 잇댄다. 모형은 `[CLS]` 토큰 표현을 분류 머리의 입력으로 써서 부류마다 로짓을 낸다. 학습 중에는 이 로짓을 교차 엔트로피 손실로 참 이름표와 견주고, 모형 전체(인코더와 분류기)를 작은 학습률로 처음부터 끝까지 미세 조정한다.

실제로 이를 온전한 학습 고리로 넓히려면 날글을 토큰 번호로 바꾸는 토큰 나누개, 배치를 묶는 DataLoader, (대개 가중치 감쇠를 곁들인 AdamW인) 최적화기, 예열을 갖춘 학습률 조정기가 필요하다. 미세 조정은 대개 $2 \times 10^{-5}$에서 $5 \times 10^{-5}$ 범위의 학습률로 3~5 세대 돈다.

## 연습문제

**연습문제 1.**
이름표(0 또는 1)를 붙인 무작위 토큰 수열로 인공 학습 데이터를 만든 뒤, `BERTClassifier`를 5 세대 학습시키는 온전한 학습 고리를 작성하라. 세대마다 학습 손실을 알려라.

??? success "연습문제 1 풀이"
    ```python
    from torch.utils.data import DataLoader, TensorDataset

    vocab_size = 10000
    model = BERTClassifier(vocab_size, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 인공 데이터
    X = torch.randint(0, vocab_size, (100, 32))
    y = torch.randint(0, 2, (100,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")
    ```

---

**연습문제 2.**
사전 학습된 트랜스포머 모형을 미세 조정할 때 작은 학습률(이를테면 $2 \times 10^{-5}$)을 쓰고 맨바닥부터 학습할 때는 대체로 더 큰 학습률(이를테면 $10^{-3}$)을 쓰는 까닭을 설명하라.

??? success "연습문제 2 풀이"
    사전 학습된 모형은 이미 큰 데이터셋에서 쓸모 있는 표현을 배웠다. 학습률이 크면 초기 학습 단계에서 가중치가 크게 흔들려 그 배운 특징이 부서진다. 작은 학습률은 사전 학습에서 얻은 지식을 지키면서 표현을 아래쪽 과제에 부드럽게 맞추어 가게 한다. 잘 맞추어진 악기를 처음부터 다시 만드는 대신 미세하게 조율하는 것과 같다. 무작위로 초기화된 분류 머리에는 층별로 다른 학습률을 써서 조금 더 높게 잡을 수 있다.

---

**연습문제 3.**
세 부류 감성(부정, 중립, 긍정)을 받치도록 모형을 넓히고 검증 집합에서 정확도, 정밀도, 재현율, F1 점수를 셈하는 평가 함수를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    model = BERTClassifier(vocab_size=10000, num_classes=3)

    def evaluate(model, val_loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.tolist())

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1: {f1:.4f}")
    ```
    거시 평균은 부류마다 따로 지표를 셈해 가중치 없이 평균 내므로, 데이터셋에서의 빈도와 무관하게 모든 부류를 똑같이 다룬다.
