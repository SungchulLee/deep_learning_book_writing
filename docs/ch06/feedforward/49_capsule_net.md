# 캡슐 신경망 (CapsNet)

CapsNet은 2017년 논문 "Dynamic Routing Between Capsules"에서 소개되었다. 벡터 출력을 갖는 캡슐과 동적 라우팅을 쓴다.

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

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
    
    # 꼬투리 그물의 활성화 함수다. ReLU가 스칼라 하나를 다루는 것과
    # 달리, 이쪽은 벡터의 "방향"은 그대로 두고 "길이"만 0과 1 사이로
    # 누른다. 방향이 무엇을 보았는지를, 길이가 그것이 있을 확률을
    # 나타내도록 설계했기 때문이다.
    # scale = |v|^2 / (1 + |v|^2)이라 길이가 짧으면 더 짧아지고 길면
    # 1에 다가간다. 뒤의 나눗셈이 벡터를 단위 길이로 만들므로 최종
    # 길이가 정확히 scale이 된다. 1e-8은 0으로 나누기를 막는다
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
        
        # 라우팅
        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1)
        if x.is_cuda:
            b_ij = b_ij.cuda()
        
        # 움직이는 길 잡기(동적 라우팅).
        # 보통의 신경망에서 층과 층을 잇는 가중치는 학습으로 정해지지만,
        # 여기 결합 계수 c_ij는 순전파 도중에 정해진다. 아래 세 번의
        # 되풀이가 곧 그 과정이며, 학습되는 것이 아니라 입력마다
        # 새로 계산된다는 점이 핵심이다
        num_iterations = 3
        for iteration in range(num_iterations):
            # dim=2는 상위 꼬투리 축이다. 즉 하위 꼬투리 하나가 자기
            # 출력을 어느 상위 꼬투리로 보낼지, 그 몫의 합이 1이 되도록
            # 나눈다. 처음에는 b_ij가 0이라 모두에게 고르게 보낸다
            c_ij = F.softmax(b_ij, dim=2)
            # 하위 꼬투리들의 예측을 가중합해 상위 꼬투리의 입력을 만든다
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            # 마지막 회에는 갱신하지 않는다. 어차피 다시 쓰이지 않기
            # 때문이다
            if iteration < num_iterations - 1:
                # 하위 꼬투리의 예측 u_hat과 상위 꼬투리의 결과 v_j를
                # 내적한다. 둘의 방향이 맞을수록 값이 커지고, 그만큼
                # b_ij가 늘어 다음 회에 그 경로의 몫이 커진다.
                # 곧 "여럿이 같은 답을 가리키면 그 길을 강화한다"는
                # 합의의 원리다
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
        
        # 복호기
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

이 구현은 3개의 클래스(`PrimaryCaps`, `DigitCaps`, `CapsNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`PrimaryCaps`의 순전파를 따라가며 텐서의 모양을 추적하라. 기본 매개변수로 입력 표본 4개의 배치를 넣었을 때, 주요 연산(합성곱, 풀링, 선형 층)마다 그 뒤의 모양을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 번째 합성곱 층의 `in_channels`를 현재 값에서 3으로 바꾼다. 공식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$을 써서 합성곱 층과 풀링 층마다 공간 차원을 다시 계산한다. 마지막 합성곱/풀링 층의 평탄화된 출력에 맞도록 첫 번째 선형 층의 `in_features`를 고친다. 다음으로 확인한다. `model = PrimaryCaps(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `PrimaryCaps`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = PrimaryCaps(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 캡슐 신경망 (CapsNet)

이 구현은 3개의 클래스(`PrimaryCaps`, `DigitCaps`, `CapsNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다.

핵심 클래스는 `PrimaryCaps`, `DigitCaps`, `CapsNet`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
