# Capsule Net

캡슐 그물은 2017년 글 "캡슐 사이의 움직이는 길잡이"에서 나왔으며, 보고 알아보는 일에 밑바탕부터 다른 길을 낸다. 홑값 살림을 쓰는 여느 CNN과 달리 캡슐은 알아본 것의 낌새와 자세(자리, 방향, 크기)를 함께 담은 벡터를 낸다. 이 꾸밈은 여느 CNN이 조각과 온몸의 사이나 자리의 층을 그리지 못하는 탈을 다룬다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
CapsNet - 꼬투리 그물
논문: "Dynamic Routing Between Capsules" (2017)
고갱이: 벡터를 내놓는 꼬투리와 움직이는 길 잡기
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        super().__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=2, padding=0)
            for _ in range(num_capsules)
        ])
    
    def forward(self, x):
        outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
        outputs = torch.cat(outputs, dim=-1)
        return self.squash(outputs)
    
    def squash(self, tensor):
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        
        # 길잡이
        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1)
        if x.is_cuda:
            b_ij = b_ij.cuda()
        
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij
        
        return v_j.squeeze(1)
    
    def squash(self, tensor):
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

class CapsNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps(num_capsules=num_classes)
        
        # 풀개
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x, y=None):
        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = torch.eye(classes.size(1)).cuda().index_select(dim=0, index=max_length_indices)
        
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        
        return classes, reconstructions

if __name__ == "__main__":
    model = CapsNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 8,215,568
```

## 2. 논의

캡슐 그물의 밑바탕 새로움은 캡슐이라는 깨침이다. 캡슐은 어떤 갈래의 것이 어떻게 놓였는지를 나타내는 움직임 벡터를 지닌 신경 낱자리의 무리다. 캡슐이 내는 벡터의 길이는 그것이 있을 낌새를 나타내고, 방향은 자세, 일그러짐, 살결 같은 결을 담는다. 홑값 살림이 모으기를 거치며 자리 사이의 소식을 잃는 여느 CNN과는 크게 다르다.

쥐어짜기 함수는 캡슐의 날임에 거는 곧지 않은 함수다. 짧은 벡터는 거의 0으로, 긴 벡터는 길이 1에 조금 못 미치게 줄여, 벡터의 길이를 낌새로 볼 수 있게 한다. 식 $v = \frac{\|s\|^2}{1 + \|s\|^2} \cdot \frac{s}{\|s\|}$은 들임의 방향은 지키면서 크기만 맞춘다.

움직이는 길잡이는 아래 켜 캡슐이 제 날임을 어느 위 켜 캡슐에 보낼지 정하는 얼개다. 여러 번 되돌며 이음 값 $c_{ij}$을 다듬어, 아래 켜 캡슐이 제 미루어 봄과 가장 잘 맞물리는 날임을 지닌 위 켜 캡슐로 날임을 보낸다. 풀개 그물은 숫자 캡슐의 날임에서 들임 그림을 되살려 다독임 노릇을 하며, 캡슐이 뜻있는 놓임 매개변수를 담도록 이끈다.

## 연습문제

**연습문제 1.**
들임 텐서의 꼴이 `(batch=4, channels=256, height=20, width=20)`이고 `num_capsules=8`, `out_channels=32`일 때 `PrimaryCaps` 켜의 날임 꼴을 셈하여라.

??? success "연습문제 1 풀이"
    엮음 캡슐 8개가 저마다 $20 \times 20$ 결 그림에 걸음 2, 덧대기 없이 $9 \times 9$ 엮음을 건다. 날임의 자리 크기는 $\lfloor (20 - 9) / 2 \rfloor + 1 = 6$이다. 캡슐마다 $(4, 32, 6, 6)$ 꼴을 내고 이를 $(4, 1152, 1)$으로 바꾼다. 캡슐 8개를 마지막 차수로 이어 붙이면 $(4, 1152, 8)$이다. 쥐어짜기 함수를 거쳐도 날임 꼴은 $(4, 1152, 8)$ 그대로다.

---

**연습문제 2.**
캡슐의 날임에 단순한 시그모이드나 소프트맥스 대신 쥐어짜기 함수를 쓰는 까닭을 밝혀라. 다른 것들이 지키지 못하는 어떤 결을 지키는가?

??? success "연습문제 2 풀이"
    쥐어짜기 함수는 들임 벡터의 방향은 지키면서 크기만 0과 1 사이로 옭아맨다. 시그모이드를 낱낱이 걸면 성분마다 따로 잣대가 바뀌어 자세를 담은 방향 소식이 무너진다. 소프트맥스는 성분을 더해 1이 되게 맞추는데, 성분이 낌새 분포가 아니라 자리나 돌림 같은 서로 다른 놓임 매개변수이므로 이 또한 알맞지 않다. 쥐어짜기 함수만이 벡터의 방향은 그것이 "어떻게" 생겼는지를, 길이는 그것이 "있는지"를 담게 지켜 준다.

---

**연습문제 3.**
`CapsNet` 얼개를 잿빛 $28 \times 28$ 대신 크기 $32 \times 32$의 RGB 그림(CIFAR-10 따위)을 받도록 고쳐라. 켜의 차수와 풀개 날임 크기를 어떻게 바꿔야 하는지 밝혀라.

??? success "연습문제 3 풀이"
    고쳐야 할 것은 이렇다. (1) `self.conv1`의 들임 갈래를 1에서 3으로 바꾼다: `nn.Conv2d(3, 256, kernel_size=9, stride=1)`. conv1 뒤의 자리 크기는 $(32 - 9 + 1) = 24$이 된다. (2) $9 \times 9$ 걸음 2 엮음을 쓰는 `PrimaryCaps` 뒤에는 자리 크기가 $\lfloor (24 - 9) / 2 \rfloor + 1 = 8$이 된다. `DigitCaps`도 그에 맞춘다: `num_routes = 32 * 8 * 8 = 2048`. (3) 풀개의 날임을 784에서 $3 \times 32 \times 32 = 3072$으로 바꾼다. `nn.Linear(1024, 784)`을 `nn.Linear(1024, 3072)`으로 갈음한다.

## 정리하며

**다룬 것** — Capsule Net

캡슐 그물의 밑바탕 새로움은 캡슐이라는 깨침이다.

고갱이 갈래는 `PrimaryCaps`, `DigitCaps`, `CapsNet`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
