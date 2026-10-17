# 컨볼루션 뉴럴 넷트웍 기준 모델

실수 시퀀스를 1차원 컨볼루션과 풀링으로 처리하는 분류 기준 모델이다. 국소 피처의 리셉티브 필드와 마지막 전역 평균 풀링을 구분해 읽는다. 입력 길이·채널을 맞추고 실제 품질·속도는 실험 조건별로 판단한다.

## 1. 코드

```python
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    pass
```

## 2. 논의

입력은 $(B,L,C)$이며 Conv1d를 위해 $(B,C,L)$로 전치한다. 세 커널 크기 3의 컨볼루션은 패딩 1로 길이를 유지하지만 첫 두 블록의 MaxPool1d(2)는 길이를 각각 내림 절반으로 줄인다. 세 번째 블록 뒤에는 전역 AdaptiveAvgPool1d(1)이 있다.

두 최대 풀링을 통과하려면 원래 길이가 적어도 4여야 한다. 마지막 컨볼루션 위치 하나의 이론적 리셉티브 필드는 18개 입력 위치이고, 전역 평균 풀링은 모든 마지막 위치를 합친다. 따라서 최종 분류기가 원래 입력의 한 국소 구간만 본다고 해석하면 안 된다.

기본 input_channels=1이며 비교 벤치마크는 64로 맞춘다. 가변 길이 패딩의 영향을 제외하지 않으므로 컨볼루션·최대 풀링과 평균의 유효 길이를 함께 다뤄야 한다. 양쪽 패딩 컨볼루션과 전역 풀링은 온라인 인과 예측용 구성이 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
풀링을 생략한 세 컨볼루션과 실제 모델의 마지막 컨볼루션 리셉티브 필드를 각각 계산하라.

</div>

??? success "풀이"
    풀링 없이 stride=1인 커널 3 세 개면 3→5→7이다. 실제 순서 conv3→pool2→conv3→pool2→conv3에서는 리셉티브 필드 r과 간격 j가 (3,1)→(4,2)→(8,2)→(10,4)→(18,4)가 된다. 단순히 7×4=28로 계산할 수 없다. 마지막 전역 평균은 이 국소 피처들을 전체 길이에 걸쳐 집계한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아래 BatchNorm 변형의 통계 축과 평가·패딩 조건을 설명하라.

</div>

??? success "풀이"
    ```python
    self.conv_layers = nn.Sequential(
        nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )
    ```
    BatchNorm1d는 (B,C,L)에서 채널별로 B·L 축 통계를 사용한다. 기본 평가 모드는 저장된 이동 통계를 사용하며 PAD도 학습 통계에 섞일 수 있다. 내부 공변량 이동 감소만으로 효과를 확정하거나 항상 수렴·일반화가 좋아진다고 보장해서는 안 된다. 같은 분할·러닝레이트 탐색 예산에서 비교한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 stride=2 컨볼루션 교체가 바꾸는 계수·길이와 정보 처리의 차이를 설명하라.

</div>

??? success "풀이"
    ```python
    self.conv_layers = nn.Sequential(
        nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),  # 학습되는 줄이기
        nn.ReLU(),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )
    ```
    채널 64·128의 stride 컨볼루션 두 개로 12,352+49,280=61,632개 계수가 추가된다. 커널 3·패딩 1·stride 2는 길이를 올림 절반으로 줄여 기본 MaxPool의 내림 절반과 홀수 길이에서 다르다. 학습 가능한 다운샘플이 항상 정보를 더 보존하지 않으며 최대 풀링도 완전한 이동 불변성을 보장하지 않는다.

## 정리하며

컨볼루션의 국소 리셉티브 필드와 최종 전역 집계, 채널·최소 길이를 구분한다. 정규화·다운샘플 변경은 계수와 출력 길이도 바꾸므로 패딩 처리와 같은 학습 조건에서 평가한다.
