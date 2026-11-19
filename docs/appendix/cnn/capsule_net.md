# Capsule Net

캡슐 네트워크(capsule network)는 2017년 논문 "Dynamic Routing Between Capsules"에서 나왔으며, 시각 인식에 근본적으로 다른 접근을 제시한다. 스칼라 활성화 값을 쓰는 일반적인 CNN과 달리, 캡슐은 대상이 존재할 확률과 그 자세(위치, 방향, 크기)를 함께 담은 벡터를 출력한다. 이 설계는 일반적인 CNN이 부분과 전체의 관계나 공간적 계층 구조를 표현하지 못하는 문제를 다룬다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
CapsNet - 캡슐 네트워크
논문: "Dynamic Routing Between Capsules" (2017)
핵심: 벡터를 출력하는 캡슐과 동적 라우팅
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
        
        # 동적 라우팅
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
        
        # 디코더
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

캡슐 네트워크의 근본적인 혁신은 캡슐이라는 개념이다. 캡슐은 특정 종류의 대상이 어떤 상태로 놓여 있는지를 나타내는 활성화 벡터를 출력하는 뉴런의 묶음이다. 캡슐이 내는 벡터의 길이는 그 대상이 존재할 확률을 나타내고, 방향은 자세, 변형, 질감 같은 속성을 담는다. 스칼라 활성화 값이 풀링을 거치며 공간적 관계 정보를 잃는 일반적인 CNN과는 크게 다르다.

스쿼시(squash) 함수는 캡슐의 출력에 적용하는 비선형 함수다. 짧은 벡터는 거의 0으로, 긴 벡터는 길이 1에 조금 못 미치게 줄여, 벡터의 길이를 확률로 해석할 수 있게 한다. 식 $v = \frac{\|s\|^2}{1 + \|s\|^2} \cdot \frac{s}{\|s\|}$은 입력의 방향은 유지하면서 크기만 조정한다.

동적 라우팅(dynamic routing)은 아래 레이어의 캡슐이 제 출력을 어느 위 레이어 캡슐에 보낼지 정하는 메커니즘이다. 여러 번 반복하며 결합 계수 $c_{ij}$을 갱신하여, 아래 레이어의 캡슐이 제 예측과 가장 잘 일치하는 출력을 가진 위 레이어 캡슐로 출력을 보낸다. 디코더 네트워크는 숫자 캡슐의 출력에서 입력 이미지를 복원하여 정칙화 역할을 하며, 캡슐이 의미 있는 자세 파라메터를 담도록 유도한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
입력 텐서의 형태가 `(batch=4, channels=256, height=20, width=20)`이고 `num_capsules=8`, `out_channels=32`일 때 `PrimaryCaps` 레이어의 출력 형태를 계산하여라.

</div>

??? success "풀이"
    컨볼루션 캡슐 8개가 저마다 $20 \times 20$ 피처 맵에 스트라이드 2, 패딩 없이 $9 \times 9$ 컨볼루션을 적용한다. 출력의 공간 크기는 $\lfloor (20 - 9) / 2 \rfloor + 1 = 6$이다. 캡슐마다 $(4, 32, 6, 6)$ 형태를 내고 이를 $(4, 1152, 1)$으로 바꾼다. 캡슐 8개를 마지막 차원으로 이어 붙이면 $(4, 1152, 8)$이다. 스쿼시 함수를 거쳐도 출력 형태는 $(4, 1152, 8)$ 그대로다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
캡슐의 출력에 단순한 시그모이드나 소프트맥스 대신 스쿼시 함수를 쓰는 까닭을 설명하여라. 다른 함수들이 보존하지 못하는 어떤 성질을 보존하는가?

</div>

??? success "풀이"
    스쿼시 함수는 입력 벡터의 방향은 유지하면서 크기만 0과 1 사이로 제한한다. 시그모이드를 성분마다 적용하면 성분마다 따로 스케일이 바뀌어 자세를 담은 방향 정보가 무너진다. 소프트맥스는 성분의 합이 1이 되게 정규화하는데, 캡슐 벡터의 성분은 확률 분포가 아니라 위치나 회전 같은 서로 다른 자세 파라메터이므로 이 또한 알맞지 않다. 스쿼시 함수만이 벡터의 방향은 대상이 "어떻게" 생겼는지를, 길이는 대상이 "있는지"를 담도록 보존한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`CapsNet` 구조를 그레이스케일 $28 \times 28$ 대신 크기 $32 \times 32$의 RGB 이미지(CIFAR-10 등)를 받도록 고쳐라. 레이어의 차원과 디코더 출력 크기를 어떻게 바꿔야 하는지 설명하여라.

</div>

??? success "풀이"
    고쳐야 할 것은 이렇다. (1) `self.conv1`의 입력 채널을 1에서 3으로 바꾼다: `nn.Conv2d(3, 256, kernel_size=9, stride=1)`. conv1 뒤의 공간 크기는 $(32 - 9 + 1) = 24$가 된다. (2) $9 \times 9$ 스트라이드 2 컨볼루션을 쓰는 `PrimaryCaps` 뒤에는 공간 크기가 $\lfloor (24 - 9) / 2 \rfloor + 1 = 8$이 된다. `DigitCaps`도 그에 맞춘다: `num_routes = 32 * 8 * 8 = 2048`. (3) 디코더의 출력을 784에서 $3 \times 32 \times 32 = 3072$로 바꾼다. `nn.Linear(1024, 784)`을 `nn.Linear(1024, 3072)`으로 대체한다.

## 정리하며

**다룬 것** — Capsule Net

캡슐 네트워크의 근본적인 혁신은 캡슐이라는 개념이다.

핵심 클래스는 `PrimaryCaps`, `DigitCaps`, `CapsNet`이며 앞의 연습문제 3개로 스스로 확인해 볼 수 있다.
