# ConvNeXt

ConvNeXt은 2022년 글 "2020년대를 위한 엮음 그물"에서 나왔으며, 보기 변환기에서 빌려 온 꾸밈 원칙을 넣어 옛 ResNet 얼개를 짜임새 있게 요즘 것으로 바꾼다. 그 결과는 순수한 엮음 얼개이면서도 여러 보기 잣대에서 스윈 변환기에 맞먹거나 앞선다. 요즘의 익힘 손질과 얼개 고름을 곁들이면 엮음의 타고난 치우침이 여전히 아주 쓸 만함을 보여 준다.

## 1. 코드

```python
#!/usr/bin/env python3
'''
ConvNeXt - 2020년대를 위한 엮음 그물
논문: "A ConvNet for the 2020s" (2022)
고갱이: 보기 변환기의 설계를 들여와 요즘에 맞게 고친 ResNet
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 4, stride=4),
            nn.LayerNorm([96, 56, 56])
        )
        self.stages = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = x.mean([2, 3])
        x = self.norm(x)
        return self.head(x)

if __name__ == "__main__":
    model = ConvNeXt()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 2. 논의

ConvNeXt 덩이는 변환기에서 여러 꾸밈을 가져온다. $7 \times 7$ 깊이별 엮음은 스스로 눈길의 너른 받는 밭을 본떴고, `groups=dim`으로 셈을 잘 들게 한다. 묶음 잣대 잡기 대신 켜 잣대 잡기를 써서 변환기의 버릇과 맞추고, 묶음의 자가 못 미더울 수 있는 옮겨 배우기 자리에서 됨됨이를 올린다. 뒤집힌 목 꾸밈(차수를 $4 \times$으로 넓혔다가 되돌리기)은 변환기 덩이의 앞먹임 그물과 그대로 짝을 이룬다.

아주 작은 값($10^{-6}$)으로 첫자리를 잡은 배울 수 있는 매개변수 `gamma`이 다스리는 켜 잣대도 종요로운 몫이다. 이는 그물이 제 자리에 가까운 나머지 이음으로 익힘을 비롯하게 하여 깊은 모형의 다듬기를 든든하게 한다. ReLU 대신 GELU 살림을 쓰고, 여러 켜로 세게 성기게 하던 옛 밑동 대신 조각내는 밑동($4 \times 4$ 걸음 4 엮음 하나)을 쓰는 것도 요즘 것으로 바꾼 대목이며, 이들이 함께 변환기 얼개와의 틈을 메운다.

ConvNeXt이 주는 더 큰 가르침은, 보기 변환기가 낫다고 여겨진 까닭이 스스로 눈길 그 자체가 아니라 곁들여진 꾸밈과 익힘의 나아짐에 있다는 것이다. 그 나아짐을 엮음 얼개로 되돌려 놓으면 순수한 엮음 그물도 아주 쓸 만하며, 옮겨도 함께 움직임과 가까운 데끼리 잇는 데서 오는 잘 듦도 그대로 지닌다.

## 연습문제

**연습문제 1.**
`dim=96`인 `ConvNeXtBlock` 하나의 매개변수 수를 셈하여라.

??? success "연습문제 1 풀이"
    깊이별 엮음은 $96 \times 7 \times 7 + 96 = 4800$개(짐 + 치우침)다. 켜 잣대 잡기는 $96 + 96 = 192$개(짐과 치우침)다. 첫 점별 선형 켜는 $96 \times 384 + 384 = 37,248$개, 둘째 점별 선형 켜는 $384 \times 96 + 96 = 36,960$개다. 켜 잣대 gamma은 $96$개다. 모두 $4800 + 192 + 37,248 + 36,960 + 96 = 79,296$개다.

---

**연습문제 2.**
`ConvNeXtBlock`의 앞으로 걸음에 있는 `permute` 셈이 무엇을 하는지 밝혀라. NCHW 텐서에 `nn.LayerNorm`을 곧바로 걸지 않는 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    PyTorch의 `nn.LayerNorm`은 들임의 마지막 차수를 따라 잣대를 맞춘다. NCHW 꼴 텐서에 걸면 W(너비)만 맞추거나 `[C, H, W]`을 밝혀야 하는데, 그러면 잣대 맞추기가 어떤 자리 결에 매인다. NHWC 꼴로 옮기면 켜 잣대 잡기가 자리마다 갈래 차수 C을 따라 자연스레 맞추는데, 이것이 바라는 결이다(변환기에서 결 차수를 따라 켜 잣대 잡기를 하는 것과 같다). 선형 켜도 마지막 차수에서 움직이므로 NHWC 자리에서 갈래를 곧바로 다룰 수 있다. 마지막 permute은 깊이별 엮음과 맞도록 NCHW으로 되돌린다.

---

**연습문제 3.**
`ConvNeXt` 모형을 도막 사이에 성기게 하기를 두는 여러 도막 얼개로 고쳐라. `dim=192`인 둘째 도막과 두 도막 사이의 성기게 하는 켜를 더하여라.

??? success "연습문제 3 풀이"
    ```python
    class ConvNeXtMultiStage(nn.Module):
        def __init__(self, num_classes=1000):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 96, 4, stride=4),
                nn.LayerNorm([96, 56, 56])
            )
            self.stage1 = nn.Sequential(*[ConvNeXtBlock(96) for _ in range(3)])
            self.downsample = nn.Sequential(
                nn.LayerNorm([96, 56, 56]),
                nn.Conv2d(96, 192, 2, stride=2),
            )
            self.stage2 = nn.Sequential(*[ConvNeXtBlock(192) for _ in range(3)])
            self.norm = nn.LayerNorm(192)
            self.head = nn.Linear(192, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage1(x)
            x = self.downsample(x)
            x = self.stage2(x)
            x = x.mean([2, 3])
            x = self.norm(x)
            return self.head(x)
    ```
    성기게 하는 켜는 켜 잣대 잡기 다음에 $2 \times 2$ 걸음 2 엮음을 쓰며, 자리 결을 반으로 줄이면서 갈래 수를 96에서 192으로 곱절 늘린다.

## 정리하며

**다룬 것** — ConvNeXt

ConvNeXt 덩이는 변환기에서 여러 꾸밈을 가져온다.

고갱이 갈래는 `ConvNeXtBlock`, `ConvNeXt`, `ConvNeXtMultiStage`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
