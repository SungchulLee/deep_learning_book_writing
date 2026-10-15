# 자기회귀 인수분해

자기회귀 모델은 결합 분포를 순서에 따른 조건부 분포의 곱으로 표현한다. 이 방식은 앞선 값을 바탕으로 다음 값의 분포를 예측하므로 텍스트·이미지·시계열에 적용할 수 있다. 사슬 규칙은 정확한 확률 항등식이지만 유한한 신경망이 참 조건부 분포를 정확히 학습한다는 보장은 아니다.

## 1. 사슬 규칙과 변수 순서

이산 변수 x₁,…,x_n에 대해 다음과 같다. 연속 변수에서는 적절히 정의한 확률밀도로 같은 형태를 쓴다.

$$p(x_1,\ldots,x_n)=\prod_{i=1}^{n}p(x_i|x_{<i}).$$

첫 항은 p(x₁)다. 조건이 확률 0인 이력에서 조건부의 정의는 주의가 필요하지만 분포를 표현하는 사슬 규칙에는 독립성 가정이 필요하지 않다. 예를 들어 p(x₁)p(x₂|x₁)p(x₃|x₁,x₂)와 p(x₃)p(x₂|x₃)p(x₁|x₂,x₃)는 같은 결합 분포를 다른 순서로 표현한다.

실제 모델 p_θ에서는 순서가 조건부 함수를 근사하는 난이도·귀납 편향·계산비용에 영향을 준다. 시간 순서에 따른 예측도 관측된 통계 의존성의 모델링이지 개입에 관한 인과관계의 증명은 아니다. 이미지의 래스터 순서처럼 시간과 무관한 순서도 사용할 수 있다. [NADE](https://proceedings.mlr.press/v15/larochelle11a.html)는 계산 가능한 신경 조건부 분포를 결합하는 대표적인 접근이다.

## 2. 이산 조건부 분포 구현

각 단계의 정규화된 범주형 분포를 logits와 softmax로 정의한다. 그러면 시퀀스 로그확률은 조건부 로그확률의 합으로 계산한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteARModel(nn.Module):
    """
    띄엄띄엄한 차례용 자기 되돌이 모델.
    
    자리마다 낱말에 대한 범주형 분포를 헤아린다.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        
        # 임베딩 층: 띄엄띄엄한 토큰을 이어진 벡터로 바꾼다
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 맥락 인코더: 앞 토큰을 다룬다
        self.encoder = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            batch_first=True
        )
        
        # 내놓기 쏘기: 숨은 상태 -> 낱말 로짓
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        자리마다 다음 토큰 헤아리기의 로짓을 셈한다.
        
        인수:
            x: 들임 차례 [배치 크기, 차례 길이]
            
        반환값:
            로짓 [배치 크기, 차례 길이, 낱말 수]
        """
        # 들임 토큰을 박아 넣는다
        embedded = self.embedding(x)  # [배치, 차례 길이, 임베딩 차원]
        
        # 맥락 부호화
        hidden_states, _ = self.encoder(embedded)  # [배치, 차례 길이, 숨은 차원]
        
        # 어휘로 사영한다
        logits = self.output_proj(hidden_states)  # [배치, 차례 길이, 낱말 수]
        
        return logits
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        음의 로그 가능도 손실 셈하기.
        
        스승 밀어 넣기를 쓴다. 곧 x[0:t]으로 x[t]을 헤아린다.
        """
        # 다음 토큰 맞히기를 위해 민다
        inputs = x[:, :-1]   # 마지막 토큰만 뺀 모두
        targets = x[:, 1:]   # 첫 토큰만 뺀 모두
        
        # 예측을 얻는다
        logits = self.forward(inputs)
        
        # 어긋 엔트로피 손실 = 음의 로그 가능도
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        return loss
```

입력은 정수 토큰 `(B,T)`이고 logits는 `(B,T,V)`다. compute_loss는 x[:, :-1]에서 다음 토큰 x[:, 1:]을 예측해 배치·시간 평균 교차엔트로피를 반환한다. 즉 첫 토큰이 주어진 조건부 시퀀스 목적이며 첫 토큰의 분포를 별도로 학습하지 않는다. 전체 결합 분포를 학습하려면 BOS와 적절한 시작·종료 규칙 또는 첫 토큰 분포가 필요하다. 패딩 마스크도 이 구현에는 없다.

logits의 같은 시간 위치는 해당 입력 토큰까지 본 뒤 **다음** 토큰을 예측한다. 타깃과 한 칸 이동을 맞추지 않으면 현재 토큰을 그대로 보는 정보 누출이 생길 수 있다. 정규화된 각 조건부는 주어진 모델의 확률을 계산 가능하게 하지만 미관측 데이터에 대한 적합성은 별도로 평가한다.

## 3. 연속 조건부 분포

연속 변수에는 가우시안 등 정규화된 밀도를 사용할 수 있다. 단일 가우시안의 경우 평균과 양의 분산을 예측하며, 다변량 출력에서 대각 분산을 쓰면 과거가 주어진 상태의 성분별 조건부 독립을 가정한다. 일반 로지스틱 분포는 실수 전체를 지지하므로 그 자체가 유계 픽셀 분포는 아니다. 양자화 관측에는 구간 적분을 사용한 이산화 분포나 범주형 모형을 정의해야 한다.

```python
class ContinuousARModel(nn.Module):
    """
    정규 분포 내놓기를 갖춘 이어진 차례용 자기 되돌이 모델.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # 평균과 로그 흩어짐을 내놓는다
        self.mean_proj = nn.Linear(hidden_dim, input_dim)
        self.logvar_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor):
        """
        자리마다 정규 분포 매개변수를 셈한다.
        
        인수:
            x: 들임 차례 [배치 크기, 차례 길이, 들임 차원]
            
        반환값:
            mean: [배치 크기, 차례 길이, 들임 차원]
            logvar: [배치 크기, 차례 길이, 들임 차원]
        """
        hidden_states, _ = self.encoder(x)
        
        mean = self.mean_proj(hidden_states)
        logvar = self.logvar_proj(hidden_states)
        
        return mean, logvar
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        정규 분포의 음의 로그 가능도를 셈한다.
        """
        inputs = x[:, :-1, :]
        targets = x[:, 1:, :]
        
        mean, logvar = self.forward(inputs)
        
        # 정규 분포 음의 로그 가능도: 0.5 * (log(2π) + logvar + (x-μ)²/var)
        nll = 0.5 * (
            logvar + 
            (targets - mean).pow(2) / logvar.exp() +
            torch.log(torch.tensor(2 * torch.pi))
        )
        
        return nll.mean()
```

위 연속 모델은 `(B,T,input_dim)`을 받고 다음 시점 벡터의 평균·로그분산을 예측한다. NLL 식에는 정규화 상수가 포함되어 있지만 mean은 배치·시간·특징 전체 평균이므로 벡터당 NLL은 특징 합산 기준으로 환산해야 한다. 첫 벡터 분포는 없고 초기 이력에 조건부로 학습한다. logvar.exp의 극단값에 대한 별도 안정화도 없다. 함수 안에서 만든 정규화 상수 텐서의 장치·자료형 정책은 실제 실행 환경에서 확인해야 한다.

## 4. 최대우도 학습과 teacher forcing

고정된 관측 시퀀스의 음의 로그확률은 다음과 같다.

$$-\log p_\theta(x)=-\sum_i\log p_\theta(x_i|x_{<i}).$$

합으로 나뉜다는 것이 각 항의 독립성을 뜻하지는 않는다. 대부분 파라미터를 공유하고 같은 관측의 겹치는 이력을 사용한다. Teacher forcing은 학습 때 관측된 앞 토큰을 조건으로 사용하는 방식이며 이 자기회귀 우도를 직접 계산하는 표준 절차다.

```python
def train_step(model, optimizer, batch):
    """
    스승 밀어 넣기로 하는 익히기 걸음 하나.
    
    모델은 자리마다 참 앞 토큰을 본다.
    """
    optimizer.zero_grad()
    
    # 스승 밀어 넣기: 참 들임을 쓴다
    loss = model.compute_loss(batch)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

마스크 CNN·인과적 Transformer는 주어진 전체 입력에서 여러 위치를 병렬 평가할 수 있다. 반면 위 GRU는 하나의 호출로 시퀀스를 처리해도 시간축 은닉 재귀는 남는다. “학습은 순방향 한 번”을 길이에 무관한 O(1) 연산이나 완전한 시간축 병렬화로 읽으면 안 된다.

생성 때는 모델이 만든 이력을 사용하므로 모델 오차가 이후 조건을 바꿀 수 있다. 이를 exposure bias 관점에서 분석하기도 한다. Scheduled sampling은 학습 이력을 혼합하지만 원래 최대우도와 다른 목적이며 일관성 문제가 있을 수 있다([이론적 비판](https://arxiv.org/abs/1511.05101)). 빔 서치는 탐색 방법으로 조건 분포 자체의 학습 오류를 고치는 기법은 아니다. 시퀀스 수준 목적도 어떤 지표·추정량을 최적화하는지 구분해야 한다.

## 5. 조상 샘플링과 디코딩 정책

조건부를 순서대로 정확히 샘플링하면 정의한 자기회귀 모델의 분포에서 표본을 얻는다. 첫 토큰을 고정했다면 그 토큰에 조건부인 분포이며 무조건부 전체 분포는 아니다.

```python
@torch.no_grad()
def sample(model, start_token, max_length, temperature=1.0):
    """
    조상 뽑기로 차례를 만든다.
    
    인수:
        model: 익힌 자기 되돌이 모델
        start_token: 첫 토큰
        max_length: 최대 차례 길이
        temperature: 표본 추출의 온도 (높을수록 더 무작위)
    
    반환값:
        만든 차례
    """
    model.eval()
    
    # 시작 토큰으로 첫자리매김한다
    generated = [start_token]
    hidden = None
    
    for _ in range(max_length - 1):
        # 지금 들임을 얻는다
        x = torch.tensor([[generated[-1]]])
        
        # 다음 토큰의 로짓을 셈한다
        embedded = model.embedding(x)
        output, hidden = model.encoder(embedded, hidden)
        logits = model.output_proj(output[:, -1, :])
        
        # 온도를 적용한다
        logits = logits / temperature
        
        # 분포에서 뽑는다
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated.append(next_token)
    
    return generated
```

이 코드는 시작 토큰을 결과에 포함하고 max_length까지 생성하며 EOS 처리는 없다. GRU hidden을 재사용해 이전 토큰 전체를 매번 다시 계산하지 않는다. 새 토큰 입력은 CPU에서 생성하므로 GPU 모델에는 장치 조정이 필요하다. temperature는 유한한 양수여야 하며 원래 모델 표본은 temperature=1일 때다.

온도 τ는 p_τ(v|h)∝p(v|h)^(1/τ)로 조건부 분포를 바꾼다. 낮은 온도는 더 큰 확률에 집중하고 높은 온도는 더 평평하게 하지만 의미적 품질·다양성 개선을 보장하지는 않는다.

```python
def top_k_sampling(logits, k):
    """가장 그럴듯한 웃 k개 토큰에서 뽑는다."""
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    return indices.gather(-1, sampled_idx)
```

Top-k는 확률이 큰 k개만 남겨 재정규화한다. 위 구현은 마지막 축이 어휘이며 torch.multinomial이 받는 1·2차원 logits를 전제로 한다. 1≤k≤V와 적어도 하나의 유한 후보가 필요하다.

```python
def top_p_sampling(logits, p):
    """쌓인 확률이 p 이상이 되는 가장 작은 토큰 모임에서 뽑는다."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 자를 자리를 찾는다
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 거르고 뽑는다
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    
    return sorted_indices.gather(-1, sampled_idx)
```

Top-p는 확률 내림차순에서 누적 질량이 문턱에 도달할 때까지 후보를 남긴다. 위 구현은 문턱 초과 위치를 한 칸 이동해 경계를 넘기는 첫 토큰도 포함하며 적어도 첫 후보는 남긴다. 통상 0<p≤1을 사용하고 정렬 동점·경계 관례를 명시한다. 두 방법 모두 샘플링 분포를 바꾸므로 기본 모델의 조상 샘플링과 구분한다.

## 6. 계산과 생성 모델 비교

일반적인 조상 샘플링은 n개의 조건부 결정을 순서대로 수행하지만 배치의 여러 시퀀스는 병렬 처리할 수 있다. 은닉 상태·KV cache 등 재사용과 모델 구조에 따라 결정 한 번의 비용도 달라진다. 병렬 검증을 이용한 생성 가속도 가능하지만 단순히 순차 의존성이 없어지는 것은 아니다.

자기회귀의 장점은 계산 가능한 정규화 조건부로 로그확률·밀도를 직접 구성한다는 점이다. VAE의 통상적 ELBO, GAN의 통상적 암묵적 생성, 흐름의 가역 밀도 계산과 비교할 수 있지만 학습 속도·샘플 품질·모드 포괄도의 보편적인 순위를 매길 수는 없다. 비자기회귀 모델도 구조에 따라 계산 가능한 우도를 가질 수 있다. perplexity·bits/dim 비교에는 같은 토큰화·단위·전처리와 시작·종료·패딩 규칙이 필요하다.

## 7. 시계열 조건부 밀도 예시

시계열에서는 과거 관측으로 다음 시점 분포의 파라미터를 예측할 수 있다. 아래는 Student-t 출력 파라미터를 만드는 네트워크의 골격이다.

```python
class FinancialARModel(nn.Module):
    """
    돈살림 시계열용 자기 되돌이 모델.
    
    돈살림 수익률에 흔한 두꺼운 꼬리를 담으려
    스튜던트 t 분포의 매개변수를 내놓는다.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # 스튜던트 t 매개변수: 자리, 잣수, 자유도
        self.loc_proj = nn.Linear(hidden_dim, 1)
        self.scale_proj = nn.Linear(hidden_dim, 1)
        self.df_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        hidden, _ = self.lstm(x)
        
        loc = self.loc_proj(hidden)
        scale = F.softplus(self.scale_proj(hidden)) + 1e-6
        df = F.softplus(self.df_proj(hidden)) + 2  # 흩어짐이 유한하려면 df > 2
        
        return loc, scale, df
```

출력은 위치 loc, 양의 scale, 2보다 큰 자유도 df다. scale은 표준편차 자체가 아니며 분산은 scale²×df/(df−2)다. 현재 벡터를 읽은 시간 위치의 출력이 다음 시점 타깃을 예측하도록 정렬하고, 분포의 log_prob·훈련 목적·초기 이력·롤아웃을 별도로 정의해야 한다. 이 클래스만으로 다자산 결합 시나리오나 예측 성능이 완성되지는 않는다. 시계열 검증은 미래 정보가 전처리·분할·입력에 새지 않도록 설계한다.

## 연습문제

1. 양의 확률을 갖는 세 이산 변수의 결합표를 정하고 두 변수 순서의 사슬 인수분해가 같은 결합확률을 주는지 계산하라. 두 순서에서 조건부 표가 왜 달라지는지도 설명하라.
2. 제공된 이산·연속 compute_loss에서 첫 관측의 확률, 시간 이동과 평균 축을 구분하라. 전체 시퀀스 NLL을 보고하려면 어떤 계약이 더 필요한가?
3. Teacher forcing 검증 손실과 자유 생성 품질을 함께 평가하는 실험을 설계하라. 같은 자료·길이·디코딩 정책을 유지하고 scheduled sampling을 사용할 때 원래 최대우도와의 차이를 설명하라.
4. 연속 시계열 예시의 Student-t scale과 표준편차를 구분하고 다음 시점 타깃 정렬·시간 분할·정규화 통계의 학습 범위를 설계하라. 실제 학습 결과를 가정하지 말고 필요한 검증 항목을 설명하라.

## 정리하며

자기회귀 인수분해는 사슬 규칙으로 정규화된 조건부를 결합한다. 구현에서는 첫 관측의 분포, 한 칸 이동, 손실 합산과 패딩을 명확히 해야 한다. Teacher forcing의 계산 효율은 구조에 달려 있고 온도·top-k·top-p는 기본 샘플링 분포를 바꾼다. 계산 가능한 우도와 실제 생성·예측 품질은 별도로 검증한다.

**참고 문헌**

1. Bengio, Y., & Bengio, S. (2000). Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks. *NeurIPS*.
2. Larochelle, H., & Murray, I. (2011). The Neural Autoregressive Distribution Estimator. *AISTATS*.
3. Uria, B., Côté, M. A., Gregor, K., Murray, I., & Larochelle, H. (2016). Neural Autoregressive Distribution Estimation. *JMLR*.
4. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.

---
