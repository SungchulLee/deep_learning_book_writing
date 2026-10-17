# DDIM: 시간 부분열을 이용한 확산 샘플링

확산 모델의 학습에서는 특정 시간의 노이즈 입력을 직접 만들지만, DDPM 생성은 많은 역전이를 순서대로 수행한다. [DDIM](https://arxiv.org/abs/2010.02502)은 같은 전방 주변분포를 이용하는 노이즈 예측기를 재사용하면서 생성 단계 수와 확률성을 조절한다. 이 페이지는 건너뛴 시간 간격에 맞는 업데이트를 유도한다.

## 1. 재사용할 수 있는 학습 정보

누적 신호 계수 ᾱₜ에 대해 전방 주변분포는 다음과 같다.

$$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$$

균등한 t와 독립 ε를 뽑아 노이즈 예측 MSE를 계산할 때 필요한 것은 이 주변분포다. 시간들 사이의 결합분포 전체를 샘플링하지 않는다. DDIM은 같은 주변분포를 갖는 비마르코프 과정의 계열을 구성해 이 학습 정보를 활용한다. 모든 변분 목적의 시간 웨이트까지 자동으로 동일하다는 뜻은 아니다.

## 2. 건너뛴 시간으로 이동하는 식

현재 노이즈 시간 t에서 더 깨끗한 시간 s<t로 이동한다고 하자. 먼저 예측 노이즈로 원본을 추정한다.

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$$

원본 성분과 노이즈 방향을 새 시간의 계수로 다시 결합하고 필요한 분산을 별도로 더한다.

$$x_s=\sqrt{\bar\alpha_s}\hat x_0+
\sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\epsilon_\theta(x_t,t)+\sigma_{t\to s}z,
\qquad z\sim\mathcal N(0,I)$$

여기서 확률성 조절 계수 η를 사용해 다음과 같이 정한다.

$$\sigma_{t\to s}=\eta\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}
\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right)}$$

0≤η≤1, 0<ᾱₜ≤ᾱₛ≤1에서 분산과 방향 계수를 실수 범위로 구성할 수 있다. 원본 x₀와 실제 ε를 넣으면 평균 성분과 추가 노이즈의 분산 합이 1−ᾱₛ가 되어 원하는 주변분포를 유지한다. 학습된 예측기를 대입한 유한 단계 생성이 원래 분포를 정확히 복원한다는 보장은 별도다.

## 3. η=0과 DDPM의 관계

η=0이면 단계별 새 노이즈가 없어 초기값과 모델이 고정된 결정적 경로가 된다. η=1이고 모든 인접 시간을 사용하면 σ²=β̃ₜ가 되어 **후방분산을 사용하는 DDPM 전이**와 연결된다. 단순히 σ=√βₜ로 두는 것이 위 DDIM 계열의 해당 선택은 아니다. 시간들을 건너뛴 η=1 경로도 기존 T단계 DDPM과 같은 경로는 아니다.

결정적이라는 말은 생성에서 무작위성이 전혀 없다는 뜻이 아니다. 초기 x_T는 무작위로 뽑는다. 또한 함수 x₀=fθ(x_T)가 정의된다는 사실만으로 일대일 대응이나 정확한 역함수가 보장되지는 않는다. 유한 단계 근사와 x̂₀ 자르기는 역변환을 더 어렵게 만들 수 있다.

## 4. 알고리즘의 시간 표기

```
알고리즘: DDIM 뽑기
────────────────────────
Input: Trained ε_θ, subsequence τ = [t_S, ..., t_1], η (stochasticity)

x_T ~ N(0, I)

for i = S, S-1, ..., 1:
    t = τ[i]
    t_prev = τ[i-1] if i > 1 else 0
    
    # x_0을 헤아린다
    x̂_0 = (x_t - sqrt(1-ᾱ_t) * ε_θ(x_t, t)) / sqrt(ᾱ_t)
    
    # 분산 계산
    σ_t = η * sqrt((1-ᾱ_{t_prev})/(1-ᾱ_t)) * sqrt(1-ᾱ_t/ᾱ_{t_prev})
    
    # x_t을 가리키는 방향
    dir_xt = sqrt(1 - ᾱ_{t_prev} - σ_t²) * ε_θ(x_t, t)
    
    # x_{t_prev}을 뽑는다
    noise = N(0, I) if t > 1 else 0
    x_{t_prev} = sqrt(ᾱ_{t_prev}) * x̂_0 + dir_xt + σ_t * noise

return x_0
```

이 의사코드는 깨끗한 끝점의 누적 계수를 1로 두는 수학적 관례를 사용한다. 구현의 인덱스 0은 첫 노이즈 단계이므로 마지막에 alpha_bar_prev=1을 별도로 지정해야 한다. 부분열의 저장 방향과 반복 인덱스도 일치시켜야 한다. 위 의사코드의 역순 목록 설명과 τ[i] 표기를 그대로 Python의 0 기반 인덱스로 옮기면 안 된다.

## 5. 구현과 실제 단계 수

```python
import torch

class DDIMSampler:
    def __init__(self, model, alphas_bar, T=1000):
        self.model = model
        self.alphas_bar = alphas_bar
        self.T = T
    
    def get_timestep_sequence(self, num_steps):
        """고르게 벌린 때 스텝 부분 차례를 만든다."""
        step_size = self.T // num_steps
        return list(range(0, self.T, step_size))[::-1]  # 내림 차례
    
    @torch.no_grad()
    def sample(self, shape, device, num_steps=50, eta=0.0):
        """
        DDIM 뽑기.
        
        인수:
            shape: 내놓기 꼴(배치, 채널, 높이, 너비)
            device: 토치 장치
            num_steps: 뽑기 스텝 수
            eta: 마구잡이 정도(0이면 정해진 대로, 1이면 DDPM처럼)
        
        반환값:
            만든 표본
        """
        # 때 스텝 부분 차례를 얻는다
        timesteps = self.get_timestep_sequence(num_steps)
        
        # 잡음에서 시작한다
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # alpha_bar 값을 얻는다
            alpha_bar_t = self.alphas_bar[t]
            alpha_bar_prev = self.alphas_bar[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)
            
            # 잡음을 헤아린다
            eps = self.model(x, t_batch)
            
            # x_0을 헤아린다
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            
            # 필요하면 x0 헤아림을 자른다
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # 시그마를 셈한다
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            
            # x_t을 가리키는 방향
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
            
            # x_{t-1}을 뽑는다
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
            else:
                x = x0_pred
        
        return x
```

구현은 마지막에 alpha_bar_prev=1을 사용하고 x0_pred를 반환해 깨끗한 끝점으로 이동한다. 이는 [방법 비교 예제](../ddpm/compare_methods.md)의 마지막 이전 인덱스 0 처리와 다르다. 매 단계 x̂₀를 [−1,1]로 자르므로 이 범위로 정규화한 데이터에 맞추어 해석한다.

시간 목록은 간격 T//num_steps의 range로 만든다. T=1000, num_steps=50이면 실제 목록은 980,960,…,0으로 50개이며 999는 빠진다. num_steps=60이면 간격 16으로 실제 63개다. num_steps는 1 이상이어야 하고 T를 넘으면 간격이 0이 된다. η의 유효 범위도 함수가 검사하지 않는다.

모델·alphas_bar·입력의 장치를 호출자가 맞춰야 하며 이 클래스가 자동으로 이동하지 않는다. no_grad는 켜지만 model.eval()은 호출하지 않는다. Dropout 같은 계층이 훈련 모드라면 η=0이어도 모델 내부의 난수가 결과에 영향을 줄 수 있다. 따라서 결정성을 주장하려면 모델 모드와 연산 조건까지 확인해야 한다.

## 6. 속도와 품질을 비교하는 기준

T회 대신 S회 호출하면 순차 호출 수를 줄일 수 있다. 그러나 실제 호출 수는 시간 목록 길이이고, 벽시계 시간은 배치·해상도·장치·가이던스와 부가 비용에 달렸다. 50단계면 항상 20배 빠르거나 모든 데이터에서 품질이 같다는 고정 결론은 없다.

시간 간격은 균등 인덱스 외에 노이즈 크기나 log-SNR에 맞추어 선택할 수 있다. “이차 간격”이라는 이름만으로 노이즈가 큰 쪽에 단계가 더 많다고 판단하지 않는다. 실제로 선택한 t와 SNR을 나열하고 시작 분포·끝점·품질을 함께 비교해야 한다.

DDIM의 결정적 경로는 연속시간의 ODE 관점과 연결된다. 정확한 스코어와 적절한 조건 아래 확률 흐름 ODE는 대응 SDE와 같은 시간별 주변분포를 갖는다. 하지만 여기의 유한 단계·자르기 포함 업데이트가 임의의 ODE 솔버와 같거나 정확히 가역이라는 뜻은 아니다. 고차 솔버와 적응 간격은 별도의 수치해석 및 평가가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
DDIM이 DDPM의 노이즈 예측기를 재사용할 수 있는 이유를 설명하라.

</div>

??? success "풀이"

    단순 노이즈 MSE는 q(xₜ|x₀)에서 직접 만든 입력과 ε를 사용하므로 전방과정의 전체 결합 경로가 필요하지 않다. DDIM은 이 주변분포를 유지하는 과정 계열을 구성한다. 실제 학습 목적의 웨이트나 유한 단계 생성의 정확도까지 모두 같다는 뜻은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
η=1과 모든 인접 시간을 사용할 때 σ²가 무엇인지 구하라.

</div>

??? success "풀이"

    s=t−1이면 ᾱₜ/ᾱₛ=αₜ=1−βₜ이다. 따라서 σ²=[(1−ᾱₜ₋₁)/(1−ᾱₜ)]βₜ=β̃ₜ가 된다. 자르기 조건도 같다면 후방분산 DDPM의 평균·분산과 연결된다. σ²=βₜ라는 답과 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
결정적 DDIM이 일대일 대응과 정확한 이미지 역변환을 자동으로 보장하는가?

</div>

??? success "풀이"

    아니다. 초기값이 주어졌을 때 출력이 정해진다는 성질은 함수의 단사성을 보장하지 않는다. 유한 단계 근사와 자르기 때문에 다른 초기점이 같은 결과로 갈 수 있다. 모델의 Dropout이 훈련 모드이면 η=0만으로 결정성도 충분하지 않다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
T=1000, num_steps=60의 실제 호출 수와 마지막 단계의 처리 방식을 설명하라.

</div>

??? success "풀이"

    간격은 16이고 목록은 0부터 992까지 63개이므로 모델을 63번 호출한다. 마지막에는 alpha_bar_prev를 1로 두고 자른 x0_pred를 직접 반환한다. 시작 단계 999를 포함하지 않는다는 점과 단계 수 인수의 의미를 함께 확인해야 한다.



## 정리하며

DDIM은 현재 상태의 원본 추정과 노이즈 방향을 선택한 이전 시간의 계수로 다시 결합한다. η=0의 결정적 경로와 초기 무작위성, η=1의 후방분산 DDPM 연결을 구분한다. 구현에서는 실제 시간 목록, 깨끗한 끝점과 평가 모드가 중요하다.
