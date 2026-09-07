# 잡음 없애는 점수 맞추기 실제 짜기

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 잡음 없애는 점수 맞추기 실제 짜기을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
단원 04: 실제의 잡음 없애는 점수 맞추기
============================================

어려움: 중간
시간: 3~4시간
미리 알 것: 단원 01-03

학습 목표:
- 실제로 쓸 만한 잡음 없애는 점수 맞추기 익히기를 짠다
- 여러 잡음 차례표를 다룬다
- 흔한 익히기 문제를 잡는다
- 점수 신경망의 품질을 따진다

고갱이 식:
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
고갱이 짜기 자세히:
1. 안정된 익히기를 위한 층 고르게 맞추기
2. 규칙 세우기를 위한 무게 줄임을 갖춘 AdamW
3. 터짐을 막는 기울기 자르기
4. 코사인 식힘 배움 빠르기 차례표
5. 효율을 위한 묶음 다루기

벌레잡이 살핌표:
□ 손실이 꾸준히 줄어드는가?
□ 점수가 자료 쪽을 가리키는가?
□ 기울기 노름이 알맞은가(10 미만)?
□ 배움 빠르기가 알맞은가?
□ 잡음 수준 σ이 자료의 잣수와 맞는가?
""")

print("\n✓ Module 04 complete!")


if __name__ == "__main__":
    pass```

## 논의

잡음 없애는 점수 맞추기 실제 짜기의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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
