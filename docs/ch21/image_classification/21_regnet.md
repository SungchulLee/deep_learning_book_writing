# RegNet

RegNet은 개별 모델 하나보다 좋은 모델들을 생성하는 설계 공간의 규칙을 연구한 CNN 계열이다. Radosavovic 외의 CVPR 2020 논문은 블록별 폭의 선형 추세를 양자화해 단계별 폭과 깊이를 구성한다. 아래 예제는 잔차 블록과 한 단계의 동작만 보여 주며 이 폭 생성 규칙을 구현하지 않는다.

## 1. 코드

```python
import torch
import torch.nn as nn


class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1,
                              groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


class RegNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.layer1 = self._make_layer(32, 64, 2)
        self.head = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            layers.append(RegNetBlock(
                in_channels if i == 0 else out_channels, out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


if __name__ == "__main__":
    model = RegNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 156,872
```

## 2. 논의

### 블록별 연속 폭에서 단계로

깊이 d의 블록 인덱스 j=0,…,d−1에서 연속 폭 uⱼ=w₀+w_a j를 만든다. w₀>0, w_a≥0이고 폭 배수 w_m>1이라 할 때 대표적인 양자화는 다음과 같다.

$$k_j=\operatorname{round}\left(\frac{\log(u_j/w_0)}{\log w_m}\right),\qquad\widetilde w_j=w_0w_m^{k_j}.$$

이를 채널 정렬 단위 q의 배수로 반올림하고, 연속해서 같은 폭인 블록들을 묶어 단계를 만든다. 각 폭이 단계의 폭이고 반복 횟수가 단계의 깊이다. 단순히 단계 인덱스 j에 w₀+w_a j를 적용해 그룹 폭의 배수로 자르는 규칙과 다르다. 총 깊이·폭 배수·채널 정렬·그룹 호환성도 필요하다.

### 그룹 폭과 그룹 수

그룹 폭은 한 그룹에 들어가는 채널 수이고, PyTorch `groups`는 그룹의 개수다. 병목 폭 B가 그룹 폭 G의 배수이면 그룹 수는 B/G다. 이 둘을 같은 인수로 대입하면 다른 연산이 된다. 코드의 `RegNetBlock`은 `groups`를 직접 받고 기본값 1을 사용한다.

### 설계 공간의 평가

RegNet의 방법론은 후보 모델을 표집해 실제로 학습·평가한 결과의 분포를 비교하고 설계 공간을 정제한다. ‘탐색 중 학습할 필요가 없어 비용이 없다’는 설명은 틀리다. NAS도 무작위·진화·미분·강화학습 등 여러 탐색법을 포함하므로 특정 두 알고리즘만으로 엄격히 나눌 수 없다. 설계 공간과 평가 예산, 최종 학습·추론 비용을 함께 명시한다.

### 예제의 범위

아래 본체는 stem 뒤 64채널 블록 두 개만 사용하며 한 단계의 유효한 분류 골격이다. 전체 RegNet의 다단계 폭 생성·병목 비율·그룹 폭 조정이나 RegNetY의 SE를 구현하지 않는다. 연습 코드는 임의 단계 폭을 출력할 뿐 모델을 구성하지도 학습하지도 않는다. 코드와 출력의 규모를 공개 RegNet 모델의 결과로 해석하지 않는다.

정식 매개변수화와 설계 공간의 실험은 [원 논문](https://openaccess.thecvf.com/content_CVPR_2020/html/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.html)을 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
RegNet의 블록별 연속 폭과 양자화 단계가 어떻게 단계 깊이를 결정하는지 설명하라.

</div>

??? success "연습문제 1 풀이"
    uⱼ=w₀+w_a j를 만든 뒤 log(uⱼ/w₀)/log w_m을 반올림해 기하급수적 폭 수준을 선택하고 채널 정렬을 적용한다. 연속한 같은 폭을 한 단계로 묶고 그 개수를 단계 깊이로 삼는다. 단계별 선형 폭을 직접 정하는 단순 규칙과 다르다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
RegNet의 설계 공간 평가에서 모델 학습이 필요한가?

</div>

??? success "연습문제 2 풀이"
    필요하다. 표집한 네트워크의 학습·평가 결과로 공간의 품질과 규칙을 비교한다. 이를 비용 없는 이론적 생성으로 설명할 수 없다. NAS와의 차이는 개별 구조의 선택뿐 아니라 해석 가능한 모델 집단의 공간을 정제한다는 관점이며, 평가 예산과 방법을 명시해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 [24,48,72,96] 폭과 [2,4,6,2] 깊이가 정식 RegNet 생성 규칙을 구현하는지 설명하라.

</div>

??? success "연습문제 3 풀이"

    이 코드는 미리 정한 단계별 선형 폭을 8의 배수로 내림해 출력한다. 값들이 이미 8의 배수라 폭은 그대로다. 블록별 선형 폭의 로그 양자화나 반복 횟수에서 단계를 생성하는 과정은 없고, RegNet 본체의 단계를 바꾸지도 않는다. 교육용 임의 구성과 정식 생성 절차를 구분한다.

    ```python
widths = [24 + 24 * j for j in range(4)]  # [24, 48, 72, 96]
# 가장 가까운 8의 배수로 양자화:
widths = [(w // 8) * 8 for w in widths]  # [24, 48, 72, 96]
depths = [2, 4, 6, 2]

# 단계를 세운다
for i, (w, d) in enumerate(zip(widths, depths)):
    print(f"Stage {i}: width={w}, depth={d}")
# 단계 0: 너비=24, 깊이=2
# 단계 1: 너비=48, 깊이=4
# 단계 2: 너비=72, 깊이=6
# 단계 3: 너비=96, 깊이=2
```


## 정리하며

RegNet은 블록별 폭 추세의 양자화로 단계 폭과 깊이를 만드는 설계 공간이다. 그룹 폭과 그룹 수를 구분하고, 표집 모델의 학습 비용과 한 단계 예제의 범위를 명확히 한다.
