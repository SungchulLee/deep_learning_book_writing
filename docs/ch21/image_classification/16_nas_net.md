# NASNet

NASNet은 작은 과제에서 탐색한 셀 구조를 더 큰 이미지 인식 모델로 옮기는 뉴럴 넷트웍 아키텍처 탐색 방법이다. Zoph 외의 CVPR 2018 논문은 normal cell과 reduction cell을 설계 공간으로 삼는다. 아래 클래스에는 셀이나 탐색 과정이 구현되어 있지 않아 NASNet을 재현하는 완성 모델은 아니다.

## 1. 코드

```python
import torch
import torch.nn as nn


class NASNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(1056, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


if __name__ == "__main__":
    model = NASNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**출력:**

```
Parameters: 1,059,784
```

## 2. 논의

### 셀 구조의 탐색

제어기 RNN은 셀의 입력 연결과 연산을 선택하는 시퀀스를 생성하고, 후보 모델의 검증 성능을 리워드로 학습할 수 있다. 최적이라는 표현은 정해진 탐색 공간·계산 예산·평가 추정 안에서 해석해야 한다. 전 가능한 구조의 전역 최적해를 보장하거나 사람의 설계 바이어스를 없애는 방법은 아니다. 탐색 공간·평가 방식·리워드 자체가 사람이 정한 가정이다.

normal cell은 공간 해상도를 유지하고 reduction cell은 해상도를 줄인다. 두 종류는 별도로 탐색한 구조를 가질 수 있어 같은 연결에 stride만 바꾼다고 설명하면 안 된다. 여러 normal cell을 반복하고 선택한 경계에 reduction cell을 배치한다. 모든 셀이 1대1로 번갈아 나타나는 것도 아니다.

### 구조 전이와 웨이트 학습

작은 데이터에서 얻은 셀의 **연결 구조**를 더 많이 반복하거나 폭을 늘려 큰 모델을 구성한다. 셀 복제본이 같은 웨이트를 공유한다는 뜻은 아니다. 새로운 과제에서 각 복제본의 파라메터를 학습한다. 이런 전이는 실험적으로 유용할 수 있지만 과제·해상도·장치 비용이 달라지면 검색한 구조의 상대 성능도 달라질 수 있다.

탐색 중 많은 후보와 하이퍼파라미터를 검증셋으로 고르면 검증 결과에 선택 바이어스가 생길 수 있다. 최종 평가는 별도 데이터와 같은 학습 예산에서 수행해야 한다. 탐색 비용·최종 학습비용·추론비용을 나누어 보고하고, 대리 과제의 정확도가 목표 장치의 속도까지 최적화한다고 가정하지 않는다.

### 코드의 범위

`NASNet`은 96채널 stem 뒤 전역 평균과 1056입력 선형 레이어만 정의한다. 피처는 96차원이므로 분류기와 맞지 않아 순전파가 실패한다. normal·reduction cell, 제어기, 후보 학습과 리워드 계산이 없으며 기존 출력은 파라미터 수만 센다. 코드와 출력을 그대로 유지했으므로 완성된 탐색 또는 평가 결과로 해석하지 않는다.

방법의 정의와 전이 실험은 [원 논문](https://doi.org/10.1109/CVPR.2018.00907)을 참고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
normal cell과 reduction cell을 같은 구조의 stride 차이로 설명할 수 있는가?

</div>

??? success "풀이"
    일반적으로 아니다. 해상도 유지와 축소라는 역할은 다르며 NASNet에서는 두 종류의 내부 연산·연결을 별도로 탐색할 수 있다. 여러 normal cell을 쌓고 특정 단계에서 reduction cell을 사용하는 구조다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
NAS가 없애지 못하는 설계 선택과 평가 바이어스를 설명하라.

</div>

??? success "풀이"
    연산 후보·연결 제약·검색 알고리즘·리워드·데이터와 예산은 사람이 선택한다. 많은 후보를 같은 검증셋으로 고르면 검증 선택 바이어스가 생길 수 있다. 최종 구조의 성능을 별도 평가하고 탐색·훈련·추론의 비용을 구분해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
블록 5개가 입력 2개와 연산 2개씩 선택하는 단순 셀 검색을 설명하라.

</div>

??? success "풀이"
    각 블록이 입력 선택 두 번과 연산 선택 두 번을 한다는 정의라면 총 20개의 범주 선택이다. 후보 입력 수는 앞선 블록에 따라 달라질 수 있으므로 20이라는 수가 검색 공간 크기는 아니다. 선택한 연산의 출력 형태를 맞춘 뒤 결합하고, 검증 리워드와 비용으로 제어기를 학습할 수 있다. 이 탐색 과정은 아래 NASNet 클래스에 구현되어 있지 않다.

## 정리하며

NASNet은 탐색한 셀 구조를 다른 규모의 모델로 전이하는 방법이다. 구조와 웨이트의 전이, 별도로 탐색한 셀 종류, 탐색·최종 평가 비용을 구분한다. 이 예제에는 셀 탐색과 중간 네트워크가 누락되어 있다.
