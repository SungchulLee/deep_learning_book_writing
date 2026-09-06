# nn.Module과 최적화기

`nn.Module` 기반 클래스는 PyTorch의 모든 신경망 모델의 토대이다. 자동 매개변수 관리, 모델 직렬화, 깔끔한 코드 구성을 제공한다. 여기에 `torch.optim`의 내장 최적화기를 결합하면, 손수 하던 경사 초기화와 매개변수 갱신을 `optimizer.zero_grad()`와 `optimizer.step()`으로 대체하는 실무급 학습 루프가 된다.

## 코드

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_samples = 200

def generate_circle_data(n_samples):
    angles = torch.rand(n_samples) * 2 * np.pi
    radii = torch.zeros(n_samples)
    radii[:n_samples//2] = torch.rand(n_samples//2) * 2
    radii[n_samples//2:] = torch.rand(n_samples//2) * 2 + 3
    X = torch.stack([radii * torch.cos(angles),
                     radii * torch.sin(angles)], dim=1)
    y = torch.zeros(n_samples, 1)
    y[n_samples//2:] = 1
    X += torch.randn_like(X) * 0.3
    return X, y

X, y = generate_circle_data(n_samples)
X, y = X.to(device), y.to(device)

model = SimpleNet(2, 16, 1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    y_pred = model(X)
    predictions = (y_pred > 0.5).float()
    accuracy = (predictions == y).float().mean().item() * 100
print(f"Final Accuracy: {accuracy:.2f}%")
```

## 논의

`nn.Module`을 상속하려면 메서드 두 개가 필요하다. `__init__`은 층들을 인스턴스 속성으로 정의하고, `forward`는 계산을 정의한다. PyTorch가 모든 `nn.Module`과 `nn.Parameter` 속성을 자동으로 찾아내어, 최적화기에 등록할 수 있도록 `model.parameters()`로, 직렬화할 수 있도록 `model.state_dict()`로 노출한다. 층이 단순히 일렬로 이어질 때는 `nn.Sequential` 컨테이너가 더 간결한 대안이 된다.

최적화기는 매개변수 갱신 전략을 감싼다. `Adam`은 경사의 1차·2차 적률 추정값을 써서 매개변수마다 학습률을 조절하며, 학습률을 정성껏 조율하지 않아도 대체로 SGD보다 나은 성능을 낸다. `zero_grad`, 순전파, 손실, 역전파, step으로 이어지는 다섯 단계 학습 루프가 PyTorch 전반에서 쓰이는 표준 패턴이다.

`model.train()`과 `model.eval()`로 학습 모드와 평가 모드를 오가는 것은 드롭아웃이나 배치 정규화 같은 층에 영향을 준다. 학습 중에 드롭아웃은 뉴런을 무작위로 0으로 만들고 배치 정규화는 배치 통계량을 쓴다. 평가 중에는 드롭아웃이 꺼지고 배치 정규화는 누적된 이동 통계량을 쓴다. 이런 층이 없더라도 추론 전에 `model.eval()`을 호출하는 것이 좋은 습관이다.

## 연습문제

**연습문제 1.**
`torch.save(model.state_dict(), 'model.pth')`로 학습된 모델을 저장한 뒤 새 `SimpleNet` 인스턴스에 불러오라. 불러온 모델이 동일한 예측을 내는지 확인하라.

??? success "연습문제 1 풀이"
    ```python
    torch.save(model.state_dict(), 'model.pth')
    loaded_model = SimpleNet(2, 16, 1).to(device)
    loaded_model.load_state_dict(torch.load('model.pth', weights_only=True))
    loaded_model.eval()
    with torch.no_grad():
        loaded_pred = loaded_model(X)
        match = torch.allclose(y_pred, loaded_pred)
    print(f"Predictions match: {match}")  # True
    ```

---

**연습문제 2.**
Adam을 SGD(학습률 0.1, 운동량 0.9)로 바꿔라. 1000 에폭에 걸친 두 최적화기의 손실 곡선을 그려 수렴 속도를 비교하라.

??? success "연습문제 2 풀이"
    ```python
    for opt_name, opt_cls, kwargs in [('Adam', optim.Adam, {'lr': 0.01}),
                                       ('SGD', optim.SGD, {'lr': 0.1, 'momentum': 0.9})]:
        model = SimpleNet(2, 16, 1).to(device)
        optimizer = opt_cls(model.parameters(), **kwargs)
        losses = []
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = nn.BCELoss()(model(X), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        plt.plot(losses, label=opt_name)
    plt.legend()
    plt.show()
    ```
    Adam은 대체로 초반에 더 빠르게 수렴하고, 운동량을 넣은 SGD는 잘 조율하면 비슷하거나 더 나은 최종 정확도에 이를 수 있다.

---

**연습문제 3.**
모델의 전체 매개변수 개수와 각 매개변수 텐서의 모양을 출력하라. 입력 2, 은닉 16, 출력 1인 신경망의 매개변수 개수가 왜 그 값이 되는지 손으로 계산하라.

??? success "연습문제 3 풀이"
    ```python
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, {param.numel()} params")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total: {total}")
    ```
    1층: 가중치 $(16, 2) = 32$개 + 편향 $(16,) = 16$개 = 48개. 2층: 가중치 $(1, 16) = 16$개 + 편향 $(1,) = 1$개 = 17개. 합계: $48 + 17 = 65$개.
