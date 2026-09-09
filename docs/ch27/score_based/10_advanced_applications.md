# 나아간 쓰임새

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 앞선 쓰임새을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""
FILE: 10_advanced_applications.py
어려움: 나아간 단계
걸리는 시간: 3~4시간
미리 알 것: 07-09

학습 목표:
    1. 점수 모델로 조건 만들어 내기를 짠다
    2. 그림 안 그리기를 이해한다
    3. 점수로 역문제 푸는 법을 배운다
    4. 다스릴 수 있는 만들어 내기를 살핀다

수학 바탕:
    조건 있는 만들어 내기:
    y가 조건(보기: 갈래 이름표)일 때 s_θ(x, y) = ∇log p(x|y)을 배운다.
    
    가름개 이끎:
    s(x|y) = s(x) + ∇log p(y|x)
    
    여기서 둘째 항이 바라는 갈래 쪽으로 이끈다.
    
    INPAINTING:
    덮개 M과 본 화소 x_obs가 주어질 때 다음을 풀어라.
    argmax_x log p(x) s.t. x_M = x_obs
    
    뽑는 동안 쏘아서 할 수 있다.
"""

import torch
import torch.nn as nn
import numpy as np

# ========================================================================
# 메인
# ========================================================================


class ConditionalScoreNetwork(nn.Module):
    """갈래 이름표를 조건으로 삼는 점수 신경망."""
    
    def __init__(self, data_dim=2, n_classes=10, hidden_dim=128):
        super().__init__()
        
        # 갈래 박아 넣기
        self.class_embed = nn.Embedding(n_classes, hidden_dim)
        
        # 점수 신경망
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x, y):
        """
        조건 있는 점수 s(x|y)을 셈한다.
        
        인수:
            x: 자료 점, 꼴 (N, D)
            y: 갈래 이름표, 꼴 (N,)
        """
        y_emb = self.class_embed(y)
        inp = torch.cat([x, y_emb], dim=-1)
        return self.net(inp)


def inpaint_with_scores(model, x_obs, mask, n_steps=1000, step_size=0.01):
    """
    점수 바탕 방법을 쓴 그림 안 그리기.
    
    전략:
    1. p(x)에서 표본을 뽑으려고 랑주뱅 움직임을 돌린다
    2. 걸음마다 본 화소에 맞도록 쏜다
    
    인수:
        model: 점수 모델
        x_obs: 본 화소, 꼴 (H, W, C)
        mask: 두 값 덮개, 1이면 본 것, 0이면 모르는 것, 꼴 (H, W, C)
        n_steps: 뽑기 걸음 수
        step_size: 랑주뱅 걸음 크기
    
    반환값:
        x: 안을 그린 그림
    """
    x = torch.randn_like(x_obs)
    
    with torch.no_grad():
        for step in range(n_steps):
            # 점수 바탕 고침
            score = model(x.flatten(), torch.tensor([0.0]))
            score = score.reshape(x.shape)
            
            noise = torch.randn_like(x)
            x = x + (step_size / 2) * score + np.sqrt(step_size) * noise
            
            # 쏘기: 본 화소를 바꾼다
            x = mask * x_obs + (1 - mask) * x
    
    return x


def demo_conditional_generation():
    """조건 점수 나타내기를 보여 준다."""
    print("Conditional Score-Based Generation")
    print("=" * 80)
    
    # 인공 조건 자료: 갈래에 따라 다른 정규 분포
    n_classes = 3
    samples_per_class = 300
    
    data = []
    labels = []
    
    for c in range(n_classes):
        # 갈래마다 평균이 다르다
        mean = torch.tensor([c * 2.0, 0.0])
        samples = torch.randn(samples_per_class, 2) * 0.5 + mean
        data.append(samples)
        labels.append(torch.full((samples_per_class,), c, dtype=torch.long))
    
    data = torch.cat(data)
    labels = torch.cat(labels)
    
    print(f"\nDataset: {len(data)} samples, {n_classes} classes")
    
    # 조건 모델을 익힌다
    model = ConditionalScoreNetwork(data_dim=2, n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining conditional score model...")
    for epoch in range(2000):
        # 단순한 잡음 없애는 점수 맞추기 손실
        noise = torch.randn_like(data) * 0.5
        noisy_data = data + noise
        
        pred_score = model(noisy_data, labels)
        target_score = -noise / (0.5 ** 2)
        
        loss = torch.mean(torch.sum((pred_score - target_score) ** 2, dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    # 조건 표본을 만든다
    print("\nGenerating conditional samples...")
    with torch.no_grad():
        samples_0 = torch.randn(100, 2) * 3
        samples_1 = torch.randn(100, 2) * 3
        samples_2 = torch.randn(100, 2) * 3
        
        y_0 = torch.zeros(100, dtype=torch.long)
        y_1 = torch.ones(100, dtype=torch.long)
        y_2 = torch.full((100,), 2, dtype=torch.long)
        
        # 랑주뱅 뽑기(단순하게 만든 것)
        for _ in range(100):
            score_0 = model(samples_0, y_0)
            score_1 = model(samples_1, y_1)
            score_2 = model(samples_2, y_2)
            
            samples_0 = samples_0 + 0.01 * score_0 + 0.1 * torch.randn_like(samples_0)
            samples_1 = samples_1 + 0.01 * score_1 + 0.1 * torch.randn_like(samples_1)
            samples_2 = samples_2 + 0.01 * score_2 + 0.1 * torch.randn_like(samples_2)
    
    print("✓ Conditional generation successful!")
    print(f"\nGenerated samples - Class 0: {samples_0.shape}")
    print(f"Generated samples - Class 1: {samples_1.shape}")
    print(f"Generated samples - Class 2: {samples_2.shape}")


if __name__ == "__main__":
    demo_conditional_generation()
```

## 2. 논의

앞선 쓰임새의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.

## 정리하며

**다룬 것** — 나아간 쓰임새

앞선 쓰임새의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `ConditionalScoreNetwork`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
