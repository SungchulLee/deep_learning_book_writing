# ConvNeXt V2

ConvNeXt V2은 2023년 글 "ConvNeXt V2: 가린 제 부호기와 함께 엮음 그물을 꾸미고 키우기"에서 나왔으며, 본디 ConvNeXt에 두 가지 고갱이 새로움을 더한다. 가린 제 부호기(FCMAE)로 스스로 이끌며 미리 익히는 꾀와, 두루 되받음 잣대 잡기(GRN)이라는 새 잣대 잡기 켜다. 이 둘 덕에 ConvNeXt V2은 더 잘 커지고 여러 크기에서 더 나은 됨됨이를 보인다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ConvNeXt V2 - 설계를 다듬은 요즘 엮음 그물
논문: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (2023)
고갱이: 두루 반응 고르기(GRN)와 다듬은 익히기 방책
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(96) for _ in range(3)])
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean([2, 3])
        return self.head(x)

if __name__ == "__main__":
    model = ConvNeXtV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 341,608
```

## 2. 논의

두루 되받음 잣대 잡기(GRN) 켜가 ConvNeXt V2 얼개의 고갱이다. 가린 제 부호기로 엮음 그물을 미리 익힐 때 생기는 결 주저앉음을 다룬다. GRN은 먼저 갈래마다 자리 되받음의 $L^2$ 노름을 셈하고, 그 노름을 갈래에 걸친 평균에 견주어 맞춘 뒤, 맞춘 값으로 본디 결의 잣대를 다시 잡는다. 배울 수 있는 매개변수 $\gamma$과 $\beta$이 너그러움을 주고, 나머지 이음 덕에 처음에는 제 자리 함수처럼 움직일 수 있다.

GRN은 보기 변환기에는 잘 듣는 가린 제 부호기 미리 익히기가 엮음 그물에서는 결의 군더더기를 낳는다는 살핌에서 비롯했다. GRN이 없으면 많은 갈래가 비슷한 결을 배워 모형이 참으로 담는 힘이 줄어든다. 두루 잣대를 맞추어 갈래끼리 겨루게 하면 GRN이 결의 다양함을 북돋우고 갈래마다의 몫을 더 뚜렷하게 한다.

본디 ConvNeXt과 견주면 V2 덩이는 켜 잣대 얼개를 없애고 뒤집힌 목의 GELU 살림 뒤에 GRN을 둔다. 이 고침과 온통 엮음으로 된 가린 제 부호기(FCMAE) 미리 익히기를 함께 쓰면, ConvNeXt V2은 아주 작은 것(매개변수 400만)에서 아주 큰 것(6억 넘음)에 이르는 여러 크기에서 변환기 바탕 모형에 맞먹거나 앞선다.

## 연습문제

**연습문제 1.**
GRN 켜에 꼴이 $(B, H, W, D)$인 들임 텐서가 들어올 때 걸음마다 $G_x$과 $N_x$의 꼴을 밝혀라.

??? success "연습문제 1 풀이"
    꼴이 $(B, H, W, D)$인 들임 $x$에서 비롯한다. (1) 차수 $(1, 2)$(높이와 너비)에 걸쳐 keepdim으로 셈한 $G_x = \|x\|_2$의 꼴은 $(B, 1, 1, D)$이다. 이는 갈래마다 자리 되받음의 $L^2$ 노름이다. (2) $G_x.\text{mean}(\text{dim}=-1, \text{keepdim}=\text{True})$은 $D$ 차수에 걸쳐 고르게 하여 꼴이 $(B, 1, 1, 1)$이다. (3) $N_x = G_x / (\text{mean} + \epsilon)$의 꼴은 $(B, 1, 1, D)$이며, 갈래마다의 되받음을 갈래 평균에 견준 값이다. 마지막 날임 $\gamma \cdot (x \cdot N_x) + \beta + x$의 꼴은 $(B, H, W, D)$이다.

---

**연습문제 2.**
GRN과 쥐어짜 북돋우기(SE) 덩이를 견주어라. 갈래 되받음을 다스리는 방식에서 무엇이 닮고 무엇이 다른가?

??? success "연습문제 2 풀이"
    GRN과 SE 덩이는 둘 다 두루 걸친 자리 소식으로 갈래마다 결의 눈금을 다시 잡는다. SE 덩이는 두루 고르게 모으기로 자리 차수를 쥐어짜고 시그모이드를 쓰는 목 MLP으로 갈래 짐을 배워 $[0, 1]$의 짐을 낸다. GRN은 그 대신 자리 차수에 걸쳐 $L^2$ 노름을 셈하고 갈래를 넘나드는 평균으로 맞추어 마디 없는 잣대 값을 낸다. 고갱이 다름은 이렇다. (1) SE은 배운 곧지 않은 바꿈(MLP)을 쓰고 GRN은 붙박인 잣대 맞추기 식에 배운 아핀 매개변수를 곁들인다. (2) SE의 짐은 시그모이드로 마디 지어지고 GRN의 잣대는 마디가 없다. (3) GRN에는 나머지 이음이 처음부터 들어 있다. (4) GRN의 매개변수는 $2D$개뿐이라 SE($r$이 줄임 견줌일 때 $2D^2/r$개)보다 훨씬 적다.

---

**연습문제 3.**
NHWC으로 옮기지 않고 NCHW 꼴에서 움직여 여느 엮음 켜와 맞물리는 GRN 갈래를 짜라.

??? success "연습문제 3 풀이"
    ```python
    class GRN_NCHW(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

        def forward(self, x):
            # x: (B, C, H, W)
            Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)    # (B, C, 1, 1)
            return self.gamma * (x * Nx) + self.beta + x
    ```
    고갱이 고침은 노름의 차수를 $(1, 2)$에서 $(2, 3)$으로, 평균의 차수를 $-1$에서 $1$으로 바꾸고, 배울 수 있는 매개변수의 꼴을 $(1, D, 1, 1)$으로 바꾸어 NCHW 텐서와 펴 맞추기가 되게 하는 것이다.

## 정리하며

**다룬 것** — ConvNeXt V2

두루 되받음 잣대 잡기(GRN) 켜가 ConvNeXt V2 얼개의 고갱이다.

고갱이 갈래는 `GRN`, `ConvNeXtV2Block`, `ConvNeXtV2`, `GRN_NCHW`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
