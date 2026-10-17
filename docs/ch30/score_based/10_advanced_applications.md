# 클래스 조건 생성과 인페인팅의 기초

스코어 모델에 클래스 레이블을 전달하면 조건별 분포를 학습할 수 있다. 일부 좌표를 관측값으로 고정하고 나머지를 갱신하는 방식은 인페인팅의 출발점이 될 수 있다. 이 페이지는 2차원 클래스 조건 DSM 예제와 별도의 미완성 인페인팅 함수를 구분해 살펴본다.

## 1. 코드와 기존 출력

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
    y가 조건(보기: 클래스 이름표)일 때 s_θ(x, y) = ∇log p(x|y)을 배운다.
    
    분류기 이끎:
    s(x|y) = s(x) + ∇log p(y|x)
    
    여기서 둘째 항이 바라는 클래스 쪽으로 이끈다.
    
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
    """클래스 이름표를 조건으로 삼는 점수 뉴럴 넷트웍."""
    
    def __init__(self, data_dim=2, n_classes=10, hidden_dim=128):
        super().__init__()
        
        # 클래스 임베딩
        self.class_embed = nn.Embedding(n_classes, hidden_dim)
        
        # 점수 뉴럴 넷트웍
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
            y: 클래스 이름표, 꼴 (N,)
        """
        y_emb = self.class_embed(y)
        inp = torch.cat([x, y_emb], dim=-1)
        return self.net(inp)


def inpaint_with_scores(model, x_obs, mask, n_steps=1000, step_size=0.01):
    """
    점수 바탕 방법을 쓴 그림 안 그리기.
    
    전략:
    1. p(x)에서 표본을 뽑으려고 랑주뱅 액션을 돌린다
    2. 스텝마다 본 화소에 맞도록 쏜다
    
    인수:
        model: 점수 모델
        x_obs: 본 화소, 꼴 (H, W, C)
        mask: 두 값 덮개, 1이면 본 것, 0이면 모르는 것, 꼴 (H, W, C)
        n_steps: 뽑기 스텝 수
        step_size: 랑주뱅 스텝 크기
    
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
    
    # 인공 조건 자료: 클래스에 따라 다른 정규 분포
    n_classes = 3
    samples_per_class = 300
    
    data = []
    labels = []
    
    for c in range(n_classes):
        # 클래스마다 평균이 다르다
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
        # 단순한 잡음 없애는 점수 맞추기 로스
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

```
Conditional Score-Based Generation
================================================================================

Dataset: 900 samples, 3 classes

Training conditional score model...
Epoch    0 | Loss: 7.838658
Epoch  500 | Loss: 4.003914
Epoch 1000 | Loss: 3.961377
Epoch 1500 | Loss: 3.786815

Generating conditional samples...
✓ Conditional generation successful!

Generated samples - Class 0: torch.Size([100, 2])
Generated samples - Class 1: torch.Size([100, 2])
Generated samples - Class 2: torch.Size([100, 2])
```

## 2. 클래스 조건 네트워크

`ConditionalScoreNetwork`는 정수 레이블 y를 128차원 임베딩으로 바꾸고 2차원 데이터와 연결한다. 입력 경로는 B×2와 B×128→B×130→B×128→B×128→B×2다. 레이블은 길이 B의 정수 인덱스이며 모델의 클래스 범위 안에 있어야 한다. 클래스 조건만 받고 노이즈 수준이나 시간 조건은 받지 않는다.

데모는 평균 (0,0), (2,0), (4,0), 공분산 0.25I인 가우시안에서 클래스당 300개씩 총 900개 표본을 만든다. 클래스 수는 3으로 지정한다. 학습은 표준편차 0.5의 추가 노이즈 η를 더하고 −η/0.5²를 회귀한다. 로스는 두 좌표 제곱오차의 합을 표본 사이에서 평균 내며 Adam 러닝레이트 0.001로 전체 배치를 2,000번 갱신한다.

모집단의 조건별 목표는 원본 p(x|y)가 아니라 **노이즈 주변 pσ(x̃|y)**의 스코어다. 이 가우시안 예에서는 평활화 공분산이 (0.25+0.25)I=0.5I이므로 참 주변 스코어는 $s(x,y)=-2(x-\mu_y)$다. 유한 데이터·학습에서는 그 근사 오차가 남는다.

## 3. 생성 반복의 드리프트·잡음 균형

각 클래스의 100개 상태를 표준편차 3의 원점 가우시안에서 시작해 다음 식으로 100번 갱신한다.

$$x\leftarrow x+0.01s_\theta(x,y)+0.1\xi$$

드리프트 계수 h=0.01인 단위 온도 랑주뱅 식의 잡음 크기는 √(2h)≈0.1414다. 현재 잡음은 0.1이므로 그 식과 일치하지 않는다. 연속시간 관점에서 정확한 스코어를 쓰면 온도 1/2, 즉 pσ²에 비례하는 다른 평형을 겨냥한다. 단순히 원래 학습 분포를 정확히 샘플링한다고 할 수 없다.

위의 이상적인 가우시안 스코어를 대입하면 AR 계수는 0.98이고 정상분산은 0.1²/(1−0.98²)≈0.2525다. 원본 분산 0.25와 우연히 가깝지만 모든 분포의 평활화를 되돌리는 일반적인 방법은 아니다. 100회에서는 초기 상태 영향도 남는다.

출력은 클래스별 반환 모양을 보여 줄 뿐 실제 평균·공분산, 조건 정확도나 분포 포괄성을 검사하지 않는다. 성공 메시지를 모델 일반화나 정확한 생성의 증거로 해석하지 않는다. 난수 시드와 검증·체크포인트도 지정되어 있지 않다.

## 4. 인페인팅 함수의 인터페이스 불일치

`inpaint_with_scores`는 관측된 위치를 mask=1, 미관측 위치를 0으로 가정해 매 단계 뒤 관측값을 덮어쓴다. 이 함수는 **데모에서 호출하지 않는다**. mask가 0과 1이 아니면 관측 고정 대신 두 값을 섞는 다른 연산이 된다.

함수는 `model(x.flatten(), torch.tensor([0.0]))`를 호출한다. 그러나 위 클래스는 B×data_dim 데이터와 정수 클래스 인덱스를 기대한다. 모든 좌표를 하나로 평탄화한 벡터와 float 레이블은 이 인터페이스와 맞지 않으며 CUDA 장치도 일치하지 않을 수 있다. 따라서 위 모델에 바로 연결되는 완성된 이미지 인페인팅 구현이 아니다.

이론적으로 정해진 관측 좌표 아래 정확한 결합 스코어의 미관측 성분을 사용하면 조건부 랑주뱅 접근을 구성할 수 있다. 실제 이미지 모델에서는 노이즈 시간, 관측 잡음 모형과 제약 처리를 맞춰야 하고 고정 스트라이드·스코어 오차·혼합 문제도 남는다. 이 짧은 함수가 그 조건을 모두 구현한다고 보아서는 안 된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
클래스 레이블이 네트워크에 들어가는 경로와 노이즈 수준 조건의 유무를 설명하라.

</div>

??? success "풀이"

    길이 B의 정수 레이블을 B×128 임베딩으로 바꾸고 B×2 데이터와 연결해 B×130을 만든다. 최종 출력은 B×2다. 이 모델에는 σ나 시간 입력이 없으므로 고정된 학습 노이즈 수준에만 대응한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
원본 클래스 공분산이 0.25I이고 추가 노이즈 공분산도 0.25I이면 DSM 목표의 모집단 스코어는 무엇인가?

</div>

??? success "풀이"

    노이즈 주변 공분산은 합인 0.5I다. 따라서 조건별 스코어는 −(0.5I)⁻¹(x−μᵧ)=−2(x−μᵧ)다. 개별 회귀 목표 −η/0.25의 관측 조건부 평균이 이 스코어이며 원본 스코어 −4(x−μᵧ)와 다르다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
생성 반복과 인페인팅 함수에서 각각 어떤 전제를 확인해야 하는가?

</div>

??? success "풀이"

    생성은 드리프트와 잡음의 비율이 목표 온도와 맞는지, 반복 수가 충분한지 확인한다. 현재 0.01 드리프트와 0.1 잡음은 단위 온도와 다르다. 인페인팅은 배치 모양, 정수 조건, 장치와 0/1 mask가 필요하며 현재 모델 호출은 그 계약과 맞지 않는다. 이 코드에는 선형·코사인 노이즈 스케줄이 없다.



## 정리하며

클래스 조건 DSM은 레이블별 노이즈 주변 스코어를 학습한다. 생성 반복의 드리프트·잡음 비율을 학습 대상과 맞추고 조건부 생성과 시간 조건을 구분한다. 별도의 인페인팅 함수는 현재 모델과 입력 계약이 맞지 않아 완성된 실행 예제로 볼 수 없다.
