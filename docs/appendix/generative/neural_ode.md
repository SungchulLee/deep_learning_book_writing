# Neural ODEs

Neural ODEs은 2018년 글 "Neural Ordinary Differential Equations"에서 나왔다. 이어진 깊이의 모형이며 기억을 아끼는 되돌려 퍼뜨리기를 쓴다.

여기 짜보기는 Neural ODEs을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
Neural ODEs - 신경 상미분 방정식
논문: "Neural Ordinary Differential Equations" (2018)
NeurIPS 2018 최우수 논문상을 받았다
고갱이: 깊이가 이어지는 모형과 기억을 아끼는 뒤로 퍼뜨리기
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
        # 단순한 오일러 쌓기(보이기 위함)
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

이 짜보기는 갈래 3개(`ODEFunc`, `ODEBlock`, `NeuralODE`)를 매기고, 이들이 어울려 온전한 만들개 모형 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch가 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 연습문제

**연습문제 1.**
기본 첫자리로 잡은 `ODEFunc`의 배울 수 있는 매개변수를 모두 세어라. 짐과 치우침을 아울러 켜마다 나누어 적어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개이고 치우침이 `out_features`개다(`bias=False`가 아니면). `nn.Conv2d(in_c, out_c, k)`마다 짐이 `in_c * out_c * k * k`개, 치우침이 `out_c`개다. `nn.Embedding(num, dim)`은 매개변수가 `num * dim`개다. 켜를 모두 더한다. `sum(p.numel() for p in model.parameters())`으로 따져 볼 수 있다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 갈래에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차수를 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫자리 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 잣대 잡기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 따짐 잃음이 오르면 짚어낸다. 다독임(드롭아웃, 짐 줄이기, 자료 불리기)이나 모형 크기 줄이기로 고친다. 익힘과 따짐 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
`ODEFunc`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = ODEFunc(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.

## 정리하며

**다룬 것** — Neural ODEs

이 짜보기는 갈래 3개(`ODEFunc`, `ODEBlock`, `NeuralODE`)를 매기고, 이들이 어울려 온전한 만들개 모형 얼개를 이룬다.

고갱이 갈래는 `ODEFunc`, `ODEBlock`, `NeuralODE`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
