# 디노이징 스코어 매칭의 학습 구성

DSM 학습에 미니배치, LayerNorm, AdamW, 기울기 클리핑과 학습률 스케줄을 결합한 예제다. 두 개의 반달 모양 데이터를 고정 σ로 평활화한 스코어를 학습하고 벡터장을 그린다. 각 장치는 최적화를 조절하는 선택이며 특정 데이터에서의 안정성이나 샘플 품질을 자동으로 보장하지는 않는다.

## 1. 코드와 기존 출력

```python
"""
실제의 잡음 없애는 점수 맞추기
============================================

어려움: 중간
시간: 3~4시간
미리 알 것: 단원 01-03

학습 목표:
- 실제로 쓸 만한 잡음 없애는 점수 맞추기 익히기를 짠다
- 여러 잡음 차례표를 다룬다
- 흔한 익히기 문제를 잡는다
- 점수 신경망의 품질을 따진다

핵심 식:
L_DSM(θ) = 𝔼_x 𝔼_ε ||s_θ(x + σε) + ε/σ||²

지은이: 이성철 @ 연세대학교
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================

print("MODULE 04: Practical DSM Implementation")
print("="*80)

class MLPScore(nn.Module):
    """건너뛰기 이음을 갖춘 여러 층 신경망 점수 신경망"""
    def __init__(self, input_dim=2, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU()
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def train_dsm(data, sigma=0.5, epochs=2000, batch_size=256, lr=1e-3):
    """가장 좋은 방식을 담은 온전한 잡음 없애는 점수 맞추기 익히기 되풀이"""
    model = MLPScore(input_dim=data.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch,) in loader:
            # 잡음 더하기
            noise = torch.randn_like(batch)
            noisy = batch + sigma * noise
            
            # 점수 미리보기
            pred_score = model(noisy)
            target_score = -noise / sigma
            
            # 손실
            loss = torch.mean((pred_score - target_score) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        losses.append(epoch_loss / len(loader))
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]:.6f}, LR = {scheduler.get_last_lr()[0]:.2e}")
    
    return model, losses

# 장난감 자료를 만든다
from sklearn.datasets import make_moons
data, _ = make_moons(n_samples=2000, noise=0.05)

print("\nTraining DSM on moons dataset...")
model, losses = train_dsm(data, sigma=0.1, epochs=2000)

# 시각화한다
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# 점수 마당
x = np.linspace(-2, 3, 20)
y = np.linspace(-1.5, 2, 20)
X, Y = np.meshgrid(x, y)
pos = torch.FloatTensor(np.stack([X.ravel(), Y.ravel()], axis=1))
with torch.no_grad():
    scores = model(pos).numpy().reshape(X.shape + (2,))

ax2.scatter(data[:, 0], data[:, 1], s=1, alpha=0.3, c='blue')
ax2.quiver(X, Y, scores[:,:,0], scores[:,:,1], alpha=0.6, color='red', scale=50)
ax2.set_title('Learned Score Field')
ax2.set_aspect('equal')
plt.tight_layout()
plt.savefig('04_dsm_practical.png', dpi=150)
plt.close()
print("✓ Saved: 04_dsm_practical.png")

print("""
핵심 짜기 자세히:
1. 안정된 익히기를 위한 층 고르게 맞추기
2. 규칙 세우기를 위한 무게 줄임을 갖춘 AdamW
3. 터짐을 막는 기울기 자르기
4. 코사인 식힘 배움 빠르기 차례표
5. 효율을 위한 배치 다루기

벌레잡이 살핌표:
□ 손실이 꾸준히 줄어드는가?
□ 점수가 자료 쪽을 가리키는가?
□ 기울기 노름이 알맞은가(10 미만)?
□ 배움 빠르기가 알맞은가?
□ 잡음 수준 σ이 자료의 잣수와 맞는가?
""")

print("\n✓ Module 04 complete!")


if __name__ == "__main__":
    pass
```

```
MODULE 04: Practical DSM Implementation
================================================================================

Training DSM on moons dataset...
Epoch 0: Loss = 98.927470, LR = 1.00e-03
Epoch 400: Loss = 64.795087, LR = 9.04e-04
Epoch 800: Loss = 60.604453, LR = 6.54e-04
Epoch 1200: Loss = 60.307520, LR = 3.45e-04
Epoch 1600: Loss = 59.040576, LR = 9.50e-05
✓ Saved: 04_dsm_practical.png

핵심 짜기 자세히:
1. 안정된 익히기를 위한 층 고르게 맞추기
2. 규칙 세우기를 위한 무게 줄임을 갖춘 AdamW
3. 터짐을 막는 기울기 자르기
4. 코사인 식힘 배움 빠르기 차례표
5. 효율을 위한 배치 다루기

벌레잡이 살핌표:
□ 손실이 꾸준히 줄어드는가?
□ 점수가 자료 쪽을 가리키는가?
□ 기울기 노름이 알맞은가(10 미만)?
□ 배움 빠르기가 알맞은가?
□ 잡음 수준 σ이 자료의 잣수와 맞는가?


✓ Module 04 complete!
```

## 2. 네트워크와 데이터 모양

`MLPScore`는 데이터의 마지막 특징 차원을 입력·출력 차원으로 사용한다. 기본 은닉층은 128차원 두 개이며 각 선형층 뒤에 LayerNorm과 SiLU를 둔다. 2차원 데이터에서는 2→128→128→2이고 LayerNorm의 학습 가능한 스케일·편향을 포함한 파라미터는 17,666개다.

LayerNorm은 각 표본의 은닉 특징을 정규화한다. 배치 정규화처럼 배치 전체의 평균·분산을 누적하는 계층은 아니다. 정규화가 입력 노이즈를 제거하거나 목표 스코어를 정확히 만든다는 뜻도 아니다. 이 구조에는 노이즈 수준을 전달하는 입력이 없어 한 σ에 대응한다.

main은 make_moons로 생성한 2,000×2 데이터에 σ=0.1을 사용한다. make_moons의 noise=0.05는 원본 데이터 생성 단계의 노이즈이며 DSM이 추가하는 노이즈와 별개다. 모델·데이터·시각화는 CPU float32 기준이고 난수 시드는 고정하지 않는다.

## 3. 손실과 최적화 단계

배치 x에 표준 정규 노이즈 ξ를 뽑아 x̃=x+σξ를 만들고 −ξ/σ를 목표로 한다. 이 함수의 손실은 배치와 좌표를 모두 평균 낸다.

$$L(\theta)=\frac1{Bd}\sum_{i=1}^B\left\|s_\theta(\widetilde x_i)+\frac{\xi_i}{\sigma}\right\|^2$$

제곱오차의 모집단 최적점은 노이즈 주변 밀도의 스코어다. 개별 노이즈 목표를 완벽하게 복원해야 한다는 뜻이 아니며 최적 손실에 조건부 분산이 남을 수 있다. 서로 다른 σ나 좌표 평균·합산 관례의 손실 크기를 바로 비교하지 않는다.

AdamW는 학습률 0.001, weight_decay=0.0001을 사용한다. 파라미터의 기울기를 계산한 뒤 전체 기울기 노름을 1로 클리핑한다. 이는 역전파 후의 기울기를 제한하는 것이며 비유한 계산이나 학습 불안정을 항상 막지는 않는다. weight decay도 입력에 대한 스코어의 적합성·보존성을 직접 보장하지 않는다.

배치 크기 256이면 에포크마다 8번 갱신하고 마지막 배치는 208개다. 따라서 2,000에포크에서 최적화 갱신은 16,000번이다. epoch_loss는 배치 평균 손실의 **동일 가중 평균**이므로 표본 전체의 정확한 가중 평균과 약간 다르다. 각 배치마다 노이즈도 새로 뽑으므로 손실은 단조롭게 줄 필요가 없다.

## 4. 코사인 학습률 스케줄

`CosineAnnealingLR`는 에포크마다 한 번 갱신하며 T_max=epochs다. ηmin 기본값이 0일 때 학습률은 다음 형태다.

$$\eta_t=\frac{\eta_0}{2}\left(1+\cos\frac{\pi t}{T_{\max}}\right)$$

이는 **학습률 스케줄**이며 확산 과정의 βt나 노이즈 표준편차 σ를 바꾸지 않는다. 출력에 표시되는 LR은 해당 에포크의 optimizer 갱신을 마치고 scheduler.step을 호출한 뒤 값이다. 손실 계산에 실제로 사용한 그 에포크의 시작 학습률과 구분한다.

학습 함수에는 검증 집합, 체크포인트와 조기 종료가 없으며 고정된 전체 에포크를 수행한다. LayerNorm·AdamW·클리핑·스케줄의 개별 효과를 비교하는 제거 실험도 없다. 따라서 이 조합이 다른 모든 구성보다 우수하다고 설명하지 않는다.

## 5. 스코어 벡터장 읽기

코드는 20×20 격자의 400×2 점에서 스코어를 계산해 20×20×2로 바꾸고 화살표를 그린다. 화살표는 학습한 벡터장의 방향·크기를 보여 주지만 진짜 주변 스코어와의 오차를 수치적으로 평가하지는 않는다. 데이터에서 먼 영역은 학습 관측이 드물어 외삽에 의존할 수 있다.

시각화는 no_grad를 사용하고 모델을 eval로 바꾸지는 않는다. 현재 LayerNorm·SiLU 구조는 학습·평가 모드에 따라 동작이 바뀌지 않지만 다른 계층을 추가하면 확인해야 한다. 이 예제에는 랑주뱅 샘플링이나 생성 이미지 평가가 없으므로 벡터장과 학습 손실만으로 생성 성능을 주장할 수 없다. 출력의 '기울기 노름 10 미만'도 모든 모델의 보편적 정상 범위가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
기본 2차원 모델의 입력·출력과 LayerNorm의 파라미터 수를 설명하라.

</div>

??? success "연습문제 1 풀이"

    배치 입력은 B×2, 두 은닉층은 B×128, 최종 출력은 B×2다. LayerNorm 하나는 128개 스케일과 128개 편향을 가져 두 개 합계 512개다. 선형층 17,154개와 합쳐 전체 17,666개다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
원소 평균 DSM 손실의 기울기와 기울기 클리핑 적용 순서를 설명하라.

</div>

??? success "연습문제 2 풀이"

    오차 eᵢ=sθ(x̃ᵢ)+ξᵢ/σ에 대해 기울기는 (2/(Bd))ΣJθsθ(x̃ᵢ)ᵀeᵢ다. loss.backward 뒤 clip_grad_norm을 호출하고 optimizer.step을 수행한다. 클리핑은 파라미터 기울기의 노름을 제한하며 입력 스코어 자체의 노름이나 업데이트 크기를 같은 값으로 고정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
코사인 학습률 스케줄을 사용하면 학습 데이터의 노이즈 수준도 바뀌는가?

</div>

??? success "연습문제 3 풀이"

    아니다. scheduler는 optimizer의 학습률만 바꾸며 train_dsm의 σ는 전 에포크에서 고정된다. 여러 노이즈 수준의 스코어를 학습하려면 그 수준의 데이터·목표와 모델 조건 입력을 별도로 설계해야 한다. 두 스케줄을 이름만으로 혼동하지 않는다.



## 정리하며

실용적인 DSM 학습 구성에서는 데이터 노이즈와 학습률 스케줄, 손실 집계와 기울기 클리핑의 역할을 구분한다. 이 예제는 고정 σ의 벡터장 학습이며 생성 샘플 품질이나 각 최적화 장치의 우열을 검증한 실험은 아니다.
