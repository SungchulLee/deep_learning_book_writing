# ShuffleNet

2017년 논문 "ShuffleNet: An Extremely Efficient CNN for Mobile Devices"에서 나온 ShuffleNet은 묶음 누비기의 채널 사이로 앎이 흐르게 하는 채널 섞기 연산을 들여왔다. 덕분에 채널 묶음이 서로 떨어져 생기는 나타냄의 병목 없이 그물 전체에서 묶음 누비기를 쓸 수 있다.

## 코드

```python
import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1,
                     groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, 2, 1)
    
    def forward(self, x):
        out = self.bottleneck(x)
        out = channel_shuffle(out, 2)
        out = self.depthwise(out)
        out = self.expand(out)
        if self.stride == 2:
            out = torch.cat([out, self.shortcut(x)], 1)
        else:
            out += x
        return nn.functional.relu(out)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 2, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = ShuffleNetV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

묶음 누비기는 채널을 서로 얽히지 않은 묶음으로 나누어 셈을 줄이지만 묶음끼리 앎을 주고받지 못하게 한다. 채널 섞기 연산은 텐서를 (묶음, 묶음당 채널) 꼴로 바꾸고 뒤바꾼 뒤 다시 펴서 이를 푼다. 이 단순한 자리바꿈으로 뒤따르는 묶음 누비기마다 앞 층의 모든 묶음에서 들임을 받는다.

ShuffleNet은 손전화에 펼치기 알맞은, 정확도와 효율의 좋은 맞바꿈을 이룬다. ShuffleNetV2는 이론상의 FLOPs가 아니라 실제 미룸 빠르기에 바탕한 실전 지침으로 그 꾸밈을 더 다듬는다.

## 연습문제

**연습문제 1.**
채널 섞기 연산을 한 단계씩 짜고, 서로 다른 묶음의 채널이 올바로 엇갈리는지 확인하여라.

??? success "연습문제 1 풀이"
    `groups=2`, `channels=6`일 때:

    - 들임 채널: `[g0_c0, g0_c1, g0_c2, g1_c0, g1_c1, g1_c2]`
    - `(2, 3)`으로 꼴 바꾸기: `[[g0_c0, g0_c1, g0_c2], [g1_c0, g1_c1, g1_c2]]`
    - `(3, 2)`으로 뒤바꾸기: `[[g0_c0, g1_c0], [g0_c1, g1_c1], [g0_c2, g1_c2]]`
    - 펴기: `[g0_c0, g1_c0, g0_c1, g1_c1, g0_c2, g1_c2]`

    이제 서로 다른 묶음의 채널이 엇갈려 있다.

---

**연습문제 2.**
성큼이 2일 때 ShuffleNet은 왜 지름길을 더하지 않고 이어 붙이는가?

??? success "연습문제 2 풀이"
    성큼이 2이면 자리 차원이 반으로 준다. 바라는 내놓는 채널 수를 지키면서 (단계 사이에서 흔히 그렇듯) 채널을 두 배로 늘리려고, (자리 차원을 맞추도록 평균 모으기를 한) 지름길을 주된 가지의 내놓음에 이어 붙인다. 이러면 지름길에 내리쬐기 누비기를 둘 필요가 없어 셈이 가장 적게 든다.

---

**연습문제 3.**
들임과 내놓음 채널이 같을 때 묶음이 $g$개인 묶음 누비기와 보통 누비기의 셈 값을 견주어라.

??? success "연습문제 3 풀이"
    For $C_{\text{in}}$ input channels, $C_{\text{out}}$ output channels, kernel $K \times K$, and $g$ groups:

    - Standard: $C_{\text{in}} \times C_{\text{out}} \times K^2$ parameters
    - Group: $(C_{\text{in}}/g) \times (C_{\text{out}}/g) \times K^2 \times g = C_{\text{in}} \times C_{\text{out}} \times K^2 / g$ parameters

    묶음 누비기는 매개변수를 $g$분의 1로 줄인다.
