# 신경 상미분 방정식

신경 상미분 방정식은 2018년 논문 "Neural Ordinary Differential Equations"에서 나왔다. 이어진 깊이 모델과 기억을 아끼는 뒷걸음 퍼뜨리기이다.

신경 상미분 방정식은 띄엄띄엄한 층을 이어진 움직임으로 바꾸어 신경망의 깊이를 이어진 양으로 뜻매김한다. 이 짜기는 상미분 방정식 풀개를 깊은 배움 안에서 어떻게 쓸 수 있는지 보이며 맞추어 가는 셈과 기억을 아끼는 익히기를 가능하게 한다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
신경 상미분 방정식 - Neural Ordinary Differential Equations
논문: "Neural Ordinary Differential Equations" (2018)
2018년 NeurIPS 최우수 논문상을 받았다
핵심: 이어진 깊이 모델, 기억을 아끼는 뒷걸음 퍼뜨리기
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim),
        )
    
    def forward(self, t, y):
        return self.net(y)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, method='euler', step_size=0.1):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.step_size = step_size
    
    def forward(self, x):
        # 단순한 오일러 적분(보여 주기용)
        t = 0
        t_end = 1
        
        while t < t_end:
            dx = self.odefunc(t, x)
            x = x + self.step_size * dx
            t += self.step_size
        
        return x

class NeuralODE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, num_classes=10):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.odeblock = ODEBlock(ODEFunc(hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.odeblock(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = NeuralODE()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(32, 784)
    print(f"Input: {x.shape}, Output: {model(x).shape}")
```

**출력:**

```
Parameters: 59,210
Input: torch.Size([32, 784]), Output: torch.Size([32, 10])
```

## 2. 논의

이 짜기는 함께 어울려 온전한 깊은 배움 얼개를 이루는 갈래 3개(`ODEFunc`, `ODEBlock`, `NeuralODE`)를 뜻매김한다. 갈래마다 뚜렷한 조각을 감싸 코드를 조각내고 넓히기 쉽게 만든다. `forward` 메서드가 파이토치의 자동 미분에 쓰이는 셈 그래프를 뜻매김한다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 기계 배움 일에 대한 실전 직관이 선다.

## 연습문제

**연습문제 1.**
기본 첫자리매김으로 `ODEFunc`의 배울 수 있는 매개변수의 온 개수를 셈하라. 무게와 치우침을 모두 담아 층마다 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `ODEFunc`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이가 달라지는 얼개를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`으로 되풀이한다. (여느 파이썬 목록이 아니라) `nn.ModuleList`을 쓰면 파이토치가 모든 매개변수를 가장 좋게 하기에 등록한다. `for n in [2, 4, 8]: model = ODEFunc(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험하라.

## 정리하며

**다룬 것** — 신경 상미분 방정식

이 짜기는 함께 어울려 온전한 깊은 배움 얼개를 이루는 갈래 3개(`ODEFunc`, `ODEBlock`, `NeuralODE`)를 뜻매김한다.

고갱이 갈래는 `ODEFunc`, `ODEBlock`, `NeuralODE`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
