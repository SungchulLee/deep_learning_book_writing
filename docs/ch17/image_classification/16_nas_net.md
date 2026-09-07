# NASNet

2018년 논문 "Learning Transferable Architectures for Scalable Image Recognition"에서 나온 NASNet(신경 얼개 찾기 그물)은 북돋움 배움으로 가장 좋은 칸 얼개를 저절로 찾아낸다. 찾아낸 칸을 쌓아 깊이와 너비가 다른 그물을 만든다.

## 코드

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

## 논의

NASNet은 북돋움 배움으로 익힌 다스림 되돌이 그물로 가장 좋은 칸 얼개를 찾는다. 칸은 두 갈래로 나오는데, 보통 칸(자리 차원을 지킴)과 줄임 칸(자리 차원을 반으로 줄임)이다. 이 칸을 쌓아 온전한 그물을 이룬다.

찾아낸 칸을 옮겨 쓸 수 있다는 것이 핵심 발견이다. 곧 작은 대리 일(보기로 CIFAR-10)에서 찾은 칸이 키웠을 때 더 큰 자료 뭉치(ImageNet)에도 잘 옮겨 간다. 작은 자료 뭉치에서 한 번만 찾으면 되므로 값비싼 찾기 과정이 쓸 만해진다.

## 연습문제

**연습문제 1.**
NASNet의 보통 칸과 줄임 칸의 차이를 설명하여라.

??? success "연습문제 1 풀이"
    보통 칸은 들임의 자리 차원을 지키고(성큼 1), 줄임 칸은 자리 차원을 반으로 줄인다(성큼 2). 둘 다 얼개 찾기로 찾아낸 같은 속 이음 무늬를 갖지만 성큼 자리매김이 다르다. 흔한 NASNet은 보통 칸 묶음과 줄임 칸을 번갈아 놓는다.

---

**연습문제 2.**
사람이 손수 꾸미는 것과 견주어 신경 얼개 찾기의 좋은 점과 나쁜 점은 무엇인가?

??? success "연습문제 2 풀이"
    좋은 점: 손수 꾸민 것을 앞서는, 직관에 어긋나는 얼개를 찾아낼 수 있고, 사람의 치우침을 없애며, 특정 하드웨어 제약에 맞게 다듬을 수 있다. 나쁜 점: 셈 값이 엄청나게 비싸고(GPU 수천 시간), 찾을 자리는 여전히 사람이 꾸며야 하며, 찾아낸 얼개는 읽어 내기 어려울 수 있고, 찾기가 대리 일에 지나치게 맞춰질 수 있다.

---

**연습문제 3.**
연산 5가지를 갖는 간추린 얼개 찾기 공간을 꾸미고, 다스림개가 거기서 얼개를 어떻게 뽑는지 설명하여라.

??? success "연습문제 3 풀이"
    연산: $\{3 \times 3$ 합성곱, $5 \times 5$ 합성곱, $3 \times 3$ 깊이별, $3 \times 3$ 최대 풀링, 지름길 연결$\}$. 제어기 RNN은 (1) 어느 앞선 켜에서 이을지, (2) 어떤 연산을 걸지를 밝히는 토큰 이음을 내놓는다. 마디가 5개인 셀에서는 제어기가 $5 \times 2 \times 2 = 20$ 번 판단한다(마디마다 들임 2개와 연산 2개). 뽑은 구조를 학습하고 검증 정확도가 REINFORCE의 보상 신호가 된다.
