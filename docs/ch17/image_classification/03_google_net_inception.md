# GoogLeNet 인셉션

GoogLeNet(인셉션 v1)은 구글의 Christian Szegedy 외가 쓴 2014년 논문 "Going Deeper with Convolutions"에서 나왔다. 매개변수를 약 680만 개, 곧 AlexNet의 12분의 1만 쓰면서 어긋남 6.67%로 ILSVRC 2014에서 우승했다. 핵심 새로움은 인셉션 단원인데, 거르개 크기가 다른 누비기를 나란히 하고 그 결과를 이어 붙여 그물이 여러 잣수 특징을 효율적으로 배우게 한다.

## 1. 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    인셉션 단원(차원 줄이기를 곁들인 소박한 판)
    
    1x1, 3x3, 5x5 누비기와 3x3 최대 모으기를 나란히 하고,
    그다음 내놓음을 이어 붙인다.
    
    차원을 줄이려 3x3과 5x5 앞에 1x1 누비기를 쓴다
    셈 값을 줄인다.
    """
    
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, 
                 ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1번 가지: 1x1 누비기
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 2번 가지: 1x1 누비기 -> 3x3 누비기
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3번 가지: 1x1 누비기 -> 5x5 누비기
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 4번 가지: 3x3 최대 모으기 -> 1x1 누비기
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class InceptionAux(nn.Module):
    """깊은 그물의 기울기 사라짐을 이겨내는 곁딸린 갈래 매개."""
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = GoogLeNet(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
```

**출력:**

```
Total parameters: 13,378,280
Input shape: torch.Size([2, 3, 224, 224])
Output shape: torch.Size([2, 1000])
```

## 2. 논의

인셉션 모듈은 GoogLeNet의 고갱이 설계 이바지다. 켜마다 거르개 크기를 하나만 고르는 대신, 인셉션은 $1 \times 1$, $3 \times 3$, $5 \times 5$ 합성곱과 $3 \times 3$ 최대 풀링을 나란히 걸고 그 날임을 채널 차원으로 이어 붙인다. 그래서 설계자가 하나를 골라 두지 않아도 그물이 켜마다 어떤 공간 잣대가 가장 알려 주는 바가 큰지 배운다. 더 큰 거르개 앞에 둔 $1 \times 1$ 합성곱은 차원을 줄이는 병목 노릇을 하여 들임 채널 수를 줄이고 셈 값을 크게 낮춘다.

익히는 동안 가운데 층에 곁딸린 갈래 매개를 붙여 앞쪽 층에 더 가까이 기울기 신호를 넣어 주며, 22층 깊이 그물에 딸린 기울기 사라짐 문제를 이겨낸다. 이 갈래 매개는 미룸 때에 버린다. 큰 온전히 이은 층 대신 전체 평균 모으기를 쓴 것과 어우러져, GoogLeNet은 매개변수를 약 12분의 1만 쓰고도 AlexNet을 넘어서는 정확도를 내며, 얼개를 잘 꾸미는 것이 막무가내로 키우는 것보다 나을 수 있음을 보여 준다.

나란한 가지로 여러 잣대의 특징을 살피고 $1 \times 1$ 합성곱으로 차원을 줄인다는 인셉션의 생각은 뒤이은 구조에 오래 영향을 미쳤다. 뒤의 인셉션 갈래(v2, v3, v4)는 나눈 합성곱, 레이블 스무딩, 잔차 연결로 이 생각을 다듬었지만 고갱이 원칙은 그대로다. 켜마다 알맞은 특징 잣대를 그물이 스스로 고르게 하라는 것이다.

## 연습문제

**연습문제 1.**
자리매김이 `(in=192, ch1x1=64, ch3x3_reduce=96, ch3x3=128, ch5x5_reduce=16, ch5x5=32, pool_proj=32)`인 인셉션 단원 하나의 매개변수 전체 개수를 셈하고 PyTorch 모델과 견주어 확인하여라.

??? success "연습문제 1 풀이"
    가지마다 매개변수를 세어라(무게 + 치우침):

    - Branch 1 ($1 \times 1$): $192 \times 64 + 64 = 12{,}352$
    - Branch 2 ($1 \times 1$ then $3 \times 3$): $(192 \times 96 + 96) + (96 \times 128 \times 9 + 128) = 18{,}528 + 110{,}720 = 129{,}248$
    - Branch 3 ($1 \times 1$ then $5 \times 5$): $(192 \times 16 + 16) + (16 \times 32 \times 25 + 32) = 3{,}088 + 12{,}832 = 15{,}920$
    - 가지 4(풀링 뒤 $1 \times 1$): $192 \times 32 + 32 = 6{,}176$

    모두: $12{,}352 + 129{,}248 + 15{,}920 + 6{,}176 = 163{,}696$개의 매개변수.

    코드로 확인하기:

    ```python
    module = InceptionModule(192, 64, 96, 128, 16, 32, 32)
    print(sum(p.numel() for p in module.parameters()))
    ```

---

**연습문제 2.**
곁딸린 갈래 매개가 왜 익힐 때만 살아 있고 미룸 때에는 그렇지 않은지 설명하여라. 시험 때에도 살려 두면 어떻게 되겠는가?

??? success "연습문제 2 풀이"
    곁딸린 갈래 매개는 가운데 층에 기울기 신호를 더 넣어 주는 벌주기 노릇을 하며 더 깊은 그물을 익히는 데 도움을 준다. 미룸 때에는 다음 까닭으로 필요 없다:

    1. 그물 꼭대기의 주된 갈래 매개가 가장 잘 다듬어진 특징을 쓰며 가장 좋은 어림을 내놓는다.
    2. 곁딸린 내놓음은 해상도가 낮고 덜 다듬어진 특징에서 나오므로 본디 덜 정확하다.
    3. 살려 두면 내놓음 여럿을 아울러야(보기로 무게를 준 고루내기) 하는데, 한결같은 이득 없이 복잡함만 는다.

    살려 두면 세 내놓음을 모아 쓸 수도 있지만, 실험을 보면 나아짐이 미미하거나 없다. 처음 논문은 익히는 동안 곁딸린 손실에 무게 0.3을 주어, 그것이 기울기 흐름을 돕기 위한 약한 신호일 뿐임을 인정했다.

---

**연습문제 3.**
`InceptionModule`에서 $5 \times 5$ 합성곱을 $3 \times 3$ 합성곱 둘을 쌓은 것으로 갈음하도록 고쳐라(인셉션 v2가 그렇게 한다). 두 갈래의 매개변수 수를 견주어라.

??? success "연습문제 3 풀이"
    인셉션 단원의 3번 가지를 갈음하여라:

    ```python
    # 원래 3번 가지: 1x1 -> 5x5
    self.branch3 = nn.Sequential(
        nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        nn.ReLU(inplace=True)
    )

    # 고친 3번 가지: 1x1 -> 3x3 -> 3x3
    self.branch3 = nn.Sequential(
        nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    ```

    `ch5x5_reduce=16`, `ch5x5=32`인 가지의 매개변수 견줌:

    - Original $5 \times 5$: $16 \times 32 \times 25 + 32 = 12{,}832$
    - Two $3 \times 3$: $(16 \times 32 \times 9 + 32) + (32 \times 32 \times 9 + 32) = 4{,}640 + 9{,}248 = 13{,}888$

    쌓은 $3 \times 3$ 합성곱 둘은 매개변수 수가 비슷하지만 사이에 비선형이 하나 더 들어가면서 실효 수용 영역이 더 넓어져 표현력이 좋아진다.

## 정리하며

**다룬 것** — GoogLeNet 인셉션

인셉션 모듈은 GoogLeNet의 고갱이 설계 이바지다.

고갱이 갈래는 `InceptionModule`, `InceptionAux`, `GoogLeNet`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
