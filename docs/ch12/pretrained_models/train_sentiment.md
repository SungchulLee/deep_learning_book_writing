# 감성 분석 학습

무작위로 초기화한 BERTClassifier를 생성하고 파라메터 수를 출력한다. 기본 함수는 데이터나 학습 루프를 포함하지 않으며, 사전 학습 모델을 로드하지도 않는다. 연습문제의 무작위 데이터는 학습 절차를 설명하기 위한 것이다.

## 1. 코드

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

## 2. 논의

어휘 10,000개, 피처 768차원, 인코더 6레이어와 두 클래스 헤드로 구성되며 파라메터는 50,799,362개다. 함수는 모델을 만든 뒤 준비 메시지만 출력한다. “감성 분석에 바로 사용할 수 있는 사전 학습 모델”이라는 뜻은 아니다.

실제 학습에는 텍스트·레이블의 분리, 어휘와 [CLS]·PAD 구성, 배치 마스크가 필요하다. BERTClassifier는 첫 위치를 읽으며 True가 차단인 패딩 마스크를 받는다. 교차 엔트로피의 클래스 인덱스와 로짓 축도 맞춰야 한다.

연습문제 1은 독립적인 무작위 토큰과 무작위 레이블을 만들기 때문에 의미 있는 감성 규칙을 학습하는 실험이 아니다. 훈련 로스를 줄여도 감성 분석 일반화의 증거가 되지 않는다. 실제 사전 학습 체크포인트를 쓸 때는 그 모델의 토크나이저와 구조를 함께 불러와야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
아래 무작위 데이터 학습 루프가 검증하는 것과 감성 분석 성능을 검증하지 못하는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    로짓·로스·역전파·갱신의 연결을 살피는 조각이다. 독립 무작위 레이블에는 일반화할 감성 관계가 없고 검증 분할도 없다. 100개 표본의 마지막 배치가 작으므로 배치 평균 로스의 단순 평균은 전체 표본 평균과 다르다.

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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
사전 학습 모델과 무작위 초기화 모델의 러닝레이트를 같은 규칙으로 정해도 되는가?

</div>

??? success "연습문제 2 풀이"
    사전 학습 웨이트의 급격한 변화를 줄이려고 낮은 러닝레이트를 출발점으로 삼을 수 있지만 최적값은 모델·데이터·배치·스케줄에 따라 다르다. 이 페이지 모델은 사전 학습되지 않았으므로 미세 조정용 작은 러닝레이트가 자동으로 적절하지는 않다. 검증셋에서 선택하고 새 헤드에 별도 러닝레이트를 줄 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 세 클래스 평가 함수의 지표와 모델 선택에서 보완할 점을 설명하라.

</div>

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
    매크로 평균은 클래스별 지표를 동일 웨이트로 평균한다. 특정 클래스가 없거나 예측되지 않을 때의 zero_division·labels 규칙을 정해야 한다. 이 조각은 새 세 클래스 모델을 만들지만 학습·val_loader 구성은 제공하지 않는다. 기존 이진 모델의 웨이트가 자동 이전되지도 않으며, 패딩 배치에는 평가에서도 마스크 전달이 필요하다.

## 정리하며

모델 생성과 실제 감성 학습을 구분한다. 무작위 데이터는 루프의 연결을 살필 뿐이며, 의미 있는 성능 평가에는 실제 레이블, 독립 검증·테스트와 일관된 토큰화·패딩 처리가 필요하다.
