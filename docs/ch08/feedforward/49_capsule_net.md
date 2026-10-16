# 캡슐 뉴럴 넷트웍 (CapsNet)

CapsNet은 여러 성분을 가진 캡슐의 출력과 동적 라우팅을 사용한다. 이 페이지는 구조 정의를 보여 주며, 기록된 출력은 파라메터 수를 인쇄한 결과이다. 전체 순전파나 학습이 검증되었다는 의미는 아니다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
CapsNet - 꼬투리 넷트웍
논문: "Dynamic Routing Between Capsules" (2017)
핵심: 벡터를 내놓는 꼬투리와 움직이는 길 잡기
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
    
    # 꼬투리 넷트웍의 엑티베이션 함수다. ReLU가 스칼라 하나를 다루는 것과
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
        # 보통의 뉴럴 넷트웍에서 레이어와 레이어를 잇는 웨이트는 학습으로 정해지지만,
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

PrimaryCaps는 컨볼루션 결과를 캡슐 벡터로 묶고 squash 함수를 적용한다. DigitCaps는 각 하위 캡슐에서 상위 캡슐로의 예측을 가중합하고 라우팅 계수를 반복 갱신하는 구조를 의도한다.

정적 차원 추적에서 수정이 필요한 부분이 보인다. 28×28 입력은 첫 컨볼루션 뒤 `(B, 256, 20, 20)`, PrimaryCaps 뒤 `(B, 1152, 8)`이다. DigitCaps의 `transpose(1, 2)` 이후 마지막 벡터 차원이 1152가 되지만, W는 마지막 입력 차원 8을 기대하므로 이어지는 행렬 곱의 차원이 맞지 않는다.

예측 레이블 마스크를 만드는 경로는 `.cuda()`를 직접 호출하므로 CPU나 다른 장치에 있는 모델과 호환되지 않을 수 있다. 또한 본문은 학습 루프나 로스를 제공하지 않는다. 원 논문의 캡슐 길이와 마진 로스를 그대로 구현한 학습 결과로 해석하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
배치 4개의 28×28 단일 채널 입력이 PrimaryCaps를 통과할 때까지 모양을 추적하라.

</div>

??? success "연습문제 1 풀이"
    첫 Conv2d는 9×9, stride 1이므로 `(4, 256, 20, 20)`이다. 각 PrimaryCaps 컨볼루션은 9×9, stride 2이므로 `(4, 32, 6, 6)`이다. 이를 `(4, 1152, 1)`로 바꿔 8개를 연결하면 `(4, 1152, 8)`이다. 이 부분에 풀링이나 선형 레이어는 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력으로 바꾸면 첫 채널 수 외에 무엇을 수정해야 하는가?

</div>

??? success "연습문제 2 풀이"
    첫 컨볼루션의 입력 채널을 3으로 바꾸면 공간 크기는 56×56, PrimaryCaps 뒤에는 24×24가 된다. 라우트 수는 32×24×24=18432로 바뀐다. DigitCaps의 축 처리, W의 라우트 차원, 복원 디코더의 출력 차원 3×64×64를 함께 맞춰야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
파라메터 수 인쇄가 순전파의 정확성을 검증하지 못하는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    파라메터 수는 모듈을 생성해 텐서 원소 수를 합한 값이다. forward를 호출하지 않아 축 순서와 행렬 곱의 호환성, CPU/GPU 장치 오류, 출력과 로스의 규약을 확인하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

캡슐 구조는 벡터 차원과 라우트 수의 구분이 핵심이다. 이 페이지의 보존된 코드는 차원과 장치 처리에 수정이 필요한 구조 예이며, 파라메터 수 출력만으로 실행이나 학습의 타당성을 판단해서는 안 된다.
