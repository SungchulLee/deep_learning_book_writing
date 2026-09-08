# 잡음 주입

잡음 주입은 학습 중에 입력, 가중치, 또는 활성화에 무작위 섭동을 더하는 정칙화 기법이다. 잡음이 섞인 데이터를 모델에게 보여 주면 모델은 더 견고한 표현을 배우고, 자연스러운 변이나 섭동이 있을 수 있는 시험 데이터에 더 잘 일반화한다.

---

## 1. 잡음 주입의 종류

### 입력 잡음

입력 특징에 직접 잡음을 더한다.

$$
\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

### 가중치 잡음

순전파 중에 모델의 가중치에 잡음을 더한다.

$$
\tilde{w} = w + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

### 경사 잡음

최적화 중에 기울기에 잡음을 더한다.

$$
\tilde{g} = g + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_t^2 I)
$$

### 활성화 잡음

은닉층의 활성화에 잡음을 더한다.

$$
\tilde{h} = h + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

---

## 2. 이론적 바탕

### 정칙화 효과

입력에 잡음을 더하는 선형 회귀에서, 입력에 정규 잡음 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$을 더하는 것은 L2 정칙화와 동등하다.

$$
\mathbb{E}_\epsilon[\|y - (x + \epsilon)^T w\|^2] = \|y - x^T w\|^2 + \sigma^2 \|w\|^2
$$

이는 입력 잡음이 큰 가중치에 암묵적으로 벌점을 준다는 것을 보여 준다.

### 견고성의 관점에서의 해석

잡음 주입은 매끄럽게 다듬어진 손실 지형을 만든다.

$$
\mathcal{L}_{\text{smooth}}(w) = \mathbb{E}_\epsilon[\mathcal{L}(w + \epsilon)]
$$

모델은 한 점에서만이 아니라 그 근방 전체에서 손실을 최소화하도록 배우며, 그 결과 더 평평하고 일반화가 잘 되는 극소점에 이른다.

---

## 3. PyTorch 구현

### 입력 잡음

```python
import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    """학습 중에 입력에 정규 잡음을 더한다."""
    
    def __init__(self, std: float = 0.1, relative: bool = False):
        """
        인수:
            std: 잡음의 표준편차
            relative: True이면 std가 입력의 크기에 상대적이다
        """
        super().__init__()
        self.std = std
        self.relative = relative
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.training이 문지기다. nn.Module이 model.train()과
        # model.eval()에 맞추어 이 값을 뒤집어 준다. 평가할 때까지
        # 잡음을 더하면 같은 입력에 매번 다른 답이 나온다
        if self.training and self.std > 0:
            if self.relative:
                # 절댓값이 큰 자리에 큰 잡음을 준다. 특징마다 눈금이
                # 제각각인 데이터에서 유용하다. 고정된 std를 쓰면
                # 눈금이 작은 특징만 잡음에 파묻힌다
                noise_std = self.std * torch.abs(x)
            else:
                noise_std = self.std
            noise = torch.randn_like(x) * noise_std
            # 제자리 연산(x += noise)이 아니라 새 텐서를 돌려준다.
            # 제자리로 바꾸면 호출한 쪽의 입력까지 함께 오염된다
            return x + noise
        return x

class UniformNoise(nn.Module):
    """학습 중에 입력에 균등 잡음을 더한다."""
    
    def __init__(self, low: float = -0.1, high: float = 0.1):
        super().__init__()
        self.low = low
        self.high = high
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 정규 잡음과 달리 균등 잡음은 [low, high] 밖으로 나가지
            # 않는다. 아주 드물게라도 큰 값이 튀는 것을 막고 싶을 때 쓴다
            noise = torch.empty_like(x).uniform_(self.low, self.high)
            return x + noise
        return x

class SaltAndPepperNoise(nn.Module):
    """이미지를 위한 소금-후추 잡음."""
    
    def __init__(self, prob: float = 0.05):
        super().__init__()
        self.prob = prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.prob > 0:
            # 앞의 둘은 값을 흔들지만 이쪽은 값을 통째로 갈아 끼운다.
            # 균등 난수 하나를 뽑아 양 끝에서 prob/2씩 잘라 내므로,
            # 소금과 후추를 합쳐 대략 prob 비율의 화소가 바뀐다.
            # 하나의 mask를 양쪽에 쓰는 덕에 두 조건은 겹칠 수 없다
            mask = torch.rand_like(x)
            salt = mask < self.prob / 2
            pepper = mask > (1 - self.prob / 2)
            # 값을 0과 1로 못박으므로 입력이 [0, 1]로 정규화된 이미지여야
            # 뜻이 맞는다. ImageNet 평균/표준편차로 표준화한 텐서에
            # 그대로 걸면 1.0은 흰색이 아니다
            
            # clone이 필수다. 이 뒤가 제자리 대입이라, 복제하지 않으면
            # 호출한 쪽이 들고 있는 원본 이미지가 함께 망가진다
            x = x.clone()
            x[salt] = 1.0
            x[pepper] = 0.0
        return x
```

### 가중치 잡음

```python
class NoisyLinear(nn.Module):
    """가중치에 잡음을 주입하는 선형층."""
    
    def __init__(self, in_features: int, out_features: int, 
                 noise_std: float = 0.1, bias: bool = True):
        super().__init__()
        self.noise_std = noise_std

        # 1/sqrt(fan_in)으로 나누어 초기화한다. 입력이 많을수록 합이
        # 커지므로 그만큼 미리 줄여 두어야 출력의 분산이 일정하게 유지된다
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / 
                                   (in_features ** 0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            # None을 그냥 대입하지 않고 register_parameter로 등록한다.
            # 이래야 self.bias가 존재하면서 state_dict에도 자리를 남겨
            # 나중에 저장하고 불러올 때 모양이 어긋나지 않는다
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 드롭아웃이 활성을 끄는 데 견주어, 여기서는 가중치 자체를 흔든다.
        # 매 걸음 조금씩 다른 가중치로 학습하는 셈이라, 손실 지형에서
        # 좁고 뾰족한 골짜기 대신 넓고 평평한 골짜기를 찾게 된다.
        # 평평한 최소점이 일반화가 낫다는 것이 이 기법의 근거다
        if self.training and self.noise_std > 0:
            # 주의: 잡음을 self.weight에 제자리로 더하지 않고 새 텐서를
            # 만든다. 제자리로 더하면 학습된 가중치가 영구히 오염된다
            weight_noise = torch.randn_like(self.weight) * self.noise_std
            noisy_weight = self.weight + weight_noise

            if self.bias is not None:
                bias_noise = torch.randn_like(self.bias) * self.noise_std
                noisy_bias = self.bias + bias_noise
            else:
                noisy_bias = None

            # nn.Linear 대신 functional.linear를 쓰는 까닭은, 매 걸음
            # 달라지는 가중치를 인자로 곧바로 넘겨야 하기 때문이다
            return nn.functional.linear(x, noisy_weight, noisy_bias)

        # 평가할 때는 잡음 없이 배운 가중치를 그대로 쓴다
        return nn.functional.linear(x, self.weight, self.bias)
```

### 경사 잡음

```python
class GradientNoiseCallback:
    """
    학습 중에 기울기에 잡음을 더한다.
    
    잡음의 일정은 보통 sigma_t^2 = eta / (1 + t)^gamma 으로 줄어든다
    """
    
    def __init__(self, eta: float = 0.01, gamma: float = 0.55):
        self.eta = eta
        self.gamma = gamma
        self.step = 0
    
    def get_noise_std(self) -> float:
        """현재 잡음의 표준편차를 계산한다."""
        # 일정이 정하는 것은 분산이고 잡음에 곱할 값은 표준편차이므로,
        # 마지막에 제곱근을 취해야 한다. 이 줄을 빠뜨리면 초반 잡음이
        # eta 그대로여서 훨씬 세진다.
        # gamma는 줄어드는 속도다. 0.55는 원 논문의 값으로, 처음에는
        # 극소점을 벗어날 만큼 흔들다가 뒤로 갈수록 잦아들게 한다
        variance = self.eta / ((1 + self.step) ** self.gamma)
        return variance ** 0.5
    
    def add_gradient_noise(self, model: nn.Module):
        """모든 기울기에 잡음을 더한다."""
        std = self.get_noise_std()
        
        # 부르는 자리가 정해져 있다. loss.backward() "뒤", optimizer.step()
        # "앞"이다. 앞서 부르면 아직 기울기가 없어 아무 일도 하지 않고,
        # 뒤에 부르면 이미 갱신이 끝난 뒤라 잡음이 다음 걸음으로 밀린다
        with torch.no_grad():
            for param in model.parameters():
                # 학습에 참여하지 않은 파라미터는 grad가 None이다.
                # 걸러 내지 않으면 AttributeError가 난다
                if param.grad is not None:
                    # 여기서는 제자리 add_가 맞다. 최적화기가 읽는 것은
                    # param.grad 그 텐서이므로, 새 텐서를 만들면
                    # 잡음이 반영되지 않는다
                    noise = torch.randn_like(param.grad) * std
                    param.grad.add_(noise)
        
        self.step += 1
```

### 활성화 잡음

```python
class ActivationNoise(nn.Module):
    """층의 활성화에 잡음을 더한다."""
    
    def __init__(self, std: float = 0.1, additive: bool = True):
        """
        인수:
            std: 잡음의 표준편차
            additive: True이면 잡음을 더하고, False이면 곱한다
        """
        super().__init__()
        self.std = std
        self.additive = additive
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            if self.additive:
                # 더하기 잡음: 활성의 크기와 상관없이 같은 세기로 흔든다.
                # 값이 작은 활성일수록 상대적으로 크게 흔들리는 셈이다
                noise = torch.randn_like(x) * self.std
                return x + noise
            else:
                # 곱하기 잡음: 평균이 1인 값을 곱한다. 활성에 비례해
                # 흔들리므로 큰 값은 크게, 작은 값은 작게 바뀐다.
                # 드롭아웃도 사실 곱하기 잡음의 한 갈래다(0 또는 1/(1-p)를 곱한다).
                # 다만 여기서는 기댓값이 1이라 따로 눈금을 맞출 필요가 없다
                noise = 1 + torch.randn_like(x) * self.std
                return x * noise
        # 평가할 때는 그대로 통과시킨다. 두 방식 모두 잡음의 기댓값이
        # 각각 0과 1이라 평균으로 보면 항등이기 때문이다
        return x

class NetworkWithActivationNoise(nn.Module):
    """활성화 잡음을 쓰는 신경망 예제."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, noise_std=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                # 잡음을 ReLU "뒤"에 둔다. 앞에 두면 잡음이 ReLU를 거치며
                # 음수 쪽이 잘려 나가 평균이 0이 아니게 되고, 그만큼
                # 활성 전체가 위로 밀린다
                ActivationNoise(std=noise_std)
            ])
            prev_dim = hidden_dim

        # 출력층 뒤에는 잡음을 두지 않는다. 예측 자체를 흔들면
        # 정칙화가 아니라 그냥 성능이 나빠진다
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

---

## 4. 심화 기법

### 일정에 따른 잡음

```python
class ScheduledNoise(nn.Module):
    """학습이 진행되며 크기가 바뀌는 잡음."""
    
    def __init__(self, initial_std: float = 0.2, final_std: float = 0.01,
                 decay_steps: int = 10000):
        super().__init__()
        # 학습 초기에는 잡음을 세게 넣어 넓게 탐색하게 하고, 뒤로 갈수록
        # 줄여 세밀하게 수렴시킨다. 학습률 스케줄과 같은 발상이다
        self.initial_std = initial_std   # 시작할 때의 잡음 크기
        self.final_std = final_std       # 끝에서의 잡음 크기
        self.decay_steps = decay_steps   # 몇 걸음에 걸쳐 줄일 것인가
        self.current_step = 0            # 지금까지 지난 걸음 수

    @property
    def current_std(self) -> float:
        # 다 줄이고 나면 final_std로 붙박는다. 이 가드가 없으면
        # progress가 1을 넘어 잡음이 음수 쪽으로 넘어간다
        if self.current_step >= self.decay_steps:
            return self.final_std

        # 0에서 1로 가는 진행률에 맞추어 두 값 사이를 선형으로 잇는다.
        # progress=0이면 initial_std, progress=1이면 final_std가 된다
        progress = self.current_step / self.decay_steps
        return self.initial_std + (self.final_std - self.initial_std) * progress

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.training은 model.train()/model.eval()이 세워 주는 깃발이다.
        # 평가할 때 잡음을 넣으면 같은 입력에 다른 답이 나오므로 반드시 꺼야 한다
        if self.training:
            # randn_like: x와 모양·자료형·장치가 같은 표준정규 잡음을 만든다
            noise = torch.randn_like(x) * self.current_std

            # 걸음 세기를 forward 안에서 한다. 옵티마이저와 따로 놀지 않게
            # 하려면 학습 루프에서 명시적으로 올려 주는 편이 더 안전하지만,
            # 이 방식은 층을 꽂기만 하면 되어 손이 덜 간다
            self.current_step += 1
            return x + noise

        return x   # 평가할 때는 아무것도 하지 않는다
```

### 변분 층 (학습 가능한 잡음)

```python
class VariationalLayer(nn.Module):
    """알맞은 잡음 수준을 배우는 층."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # 앞의 NoisyLinear가 잡음 크기를 사람이 정해 주었다면, 여기서는
        # 그 크기까지 학습한다. 가중치 하나하나를 값이 아니라 "분포"로
        # 들고 있는 셈이며, 이것이 베이즈 신경망의 가장 단순한 꼴이다.

        # 분산이 아니라 로그 분산을 매개변수로 둔다. 분산은 반드시
        # 양수여야 하는데, 로그로 두면 어떤 실수를 넣어도 exp가 양수를
        # 돌려주므로 제약 없이 최적화할 수 있다
        self.w_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        # -10에서 시작한다. exp(-10/2) 는 거의 0이라 처음에는 잡음이 없는
        # 보통의 층처럼 굴다가, 학습이 진행되며 필요한 만큼만 잡음을 키운다
        self.w_log_var = nn.Parameter(torch.full((out_features, in_features), -10.0))

        self.b_mean = nn.Parameter(torch.zeros(out_features))
        self.b_log_var = nn.Parameter(torch.full((out_features,), -10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 재매개변수화 요령(reparameterization trick).
            # w ~ N(mean, std^2) 에서 바로 뽑으면 그 뽑기를 미분할 수 없다.
            # 대신 표준정규에서 뽑아 mean + std * eps 로 만들면, 무작위성이
            # eps에만 남고 mean과 std로는 기울기가 흘러 학습할 수 있다.
            # 로그 분산에 0.5를 곱해 exp하면 표준편차가 된다
            w_std = torch.exp(0.5 * self.w_log_var)
            w = self.w_mean + w_std * torch.randn_like(self.w_mean)

            b_std = torch.exp(0.5 * self.b_log_var)
            b = self.b_mean + b_std * torch.randn_like(self.b_mean)
        else:
            # 평가할 때는 분포의 평균만 쓴다. 여러 번 뽑아 평균 내면
            # 불확실성까지 얻을 수 있으나(MC 드롭아웃과 같은 발상)
            # 그만큼 느려진다
            w = self.w_mean
            b = self.b_mean

        return nn.functional.linear(x, w, b)

    def kl_divergence(self) -> torch.Tensor:
        """사전분포(표준정규분포)로부터의 KL 발산."""
        # 이 항을 손실에 더해야 학습이 성립한다. 없으면 모델이 잡음을
        # 0으로 줄이는 것이 언제나 이득이라 보통의 층으로 되돌아간다.
        # KL이 "표준정규에서 너무 멀어지지 말라"는 벌점 노릇을 한다.
        #
        # 두 가우스 사이의 KL을 닫힌 꼴로 적은 것이며,
        # -0.5 * sum(1 + log(s^2) - m^2 - s^2) 가 그 식이다
        kl_w = -0.5 * torch.sum(1 + self.w_log_var - self.w_mean.pow(2) - 
                                 self.w_log_var.exp())
        kl_b = -0.5 * torch.sum(1 + self.b_log_var - self.b_mean.pow(2) - 
                                 self.b_log_var.exp())
        return kl_w + kl_b
```

---

## 5. 완전한 학습 예제

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_noise_injection(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_noise_std: float = 0.1,
    gradient_noise_eta: float = 0.01,
    epochs: int = 100
) -> dict:
    """여러 종류의 잡음 주입과 함께 모델을 학습시킨다."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    input_noise = GaussianNoise(std=input_noise_std)
    gradient_noise = GradientNoiseCallback(eta=gradient_noise_eta)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 입력 잡음 더하기
            X_noisy = input_noise(X_batch)
            
            outputs = model(X_noisy)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 기울기 잡음 더하기
            gradient_noise.add_gradient_noise(model)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 검증 (잡음 없음)
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
    
    return history
```

---

## 6. 잡음 종류의 비교

| 종류 | 위치 | 효과 | 쓰임새 |
|------|----------|--------|----------|
| 입력 잡음 | 데이터 | 데이터 증강, 견고성 | 데이터가 적을 때, 입력에 잡음이 있을 때 |
| 가중치 잡음 | 매개변수 | 근사 베이즈 추론 | 불확실성 추정 |
| 기울기 잡음 | 최적화 | 국소 극소점 탈출 | 깊은 신경망, 비볼록 손실 |
| 활성화 잡음 | 은닉층 | 드롭아웃과 비슷 | 일반적인 정칙화 |

---

## 7. 실무 지침

### 잡음의 종류 고르기

1. **입력 잡음**: 시험 데이터에 자연스러운 변이가 있을 수 있을 때
2. **가중치 잡음**: 불확실성 정량화가 필요할 때
3. **기울기 잡음**: 아주 깊은 신경망이나 어려운 최적화일 때
4. **활성화 잡음**: 범용 정칙화

### 잡음 크기의 선택

- **너무 작으면**: 정칙화 효과가 거의 없다
- **너무 크면**: 학습을 막고 신호를 없앤다
- **지침**:
  - 입력 잡음: 입력 표준편차의 1~10%
  - 가중치 잡음: 가중치 크기에 견주어 0.01~0.1
  - 기울기 잡음: eta=0.01에서 시작하여 학습이 진행되며 줄인다

### 다른 기법과 결합하기

잡음 주입은 다음을 보완할 수 있다.

- **드롭아웃**: 작동 원리가 달라 함께 쓰면 상승효과가 나는 일이 많다
- **L2 정칙화**: 잡음이 암묵적으로 L2와 비슷한 효과를 준다
- **데이터 증강**: 잡음은 연속적인 증강의 한 형태이다

---

## 연습문제

**연습문제 1.**
잡음 주입의 세 가지 종류인 입력 잡음, 가중치 잡음, 기울기 잡음을 설명하라.

??? success "연습문제 1 풀이"
    입력 잡음은 입력에 $\epsilon \sim \mathcal{N}(0, \sigma^2)$을 더해 손실 곡면을 매끄럽게 한다. 가중치 잡음은 순전파마다 매개변수에 잡음을 더해 L2 정칙화처럼 작동한다. 기울기 잡음은 기울기에 잡음을 더해 뾰족한 극소점을 벗어나게 돕고 일반화를 개선한다.

---

**연습문제 2.**
가중치에 정규 잡음을 더하는 것이 L2 정칙화와 근사적으로 동등함을 보여라.

??? success "연습문제 2 풀이"
    $\epsilon \sim \mathcal{N}(0, \sigma^2)$인 잡음 섞인 가중치 $\tilde{w} = w + \epsilon$에 대해 테일러 전개로 $\mathbb{E}[L(\tilde{w})] \approx L(w) + \frac{\sigma^2}{2}\text{tr}(H)$을 얻는다. 헤세 행렬의 대각합은 곡률이 큰 방향에 벌점을 주며, 이는 선형화한 모델에서의 L2 정칙화와 비슷하다.

---

**연습문제 3.**
Neelakantan 등(2015)의 일정 $\sigma_t^2 = \eta/(1+t)^\gamma$을 쓰는 기울기 잡음 주입을 구현하라.

??? success "연습문제 3 풀이"
    ```python
    for t, (x, y) in enumerate(dataloader):
        loss = criterion(model(x), y)
        loss.backward()
        sigma = (eta / (1 + t)**gamma)**0.5
        for p in model.parameters():
            p.grad += sigma * torch.randn_like(p.grad)
        optimizer.step()
    ```

---

**연습문제 4.**
드롭아웃이나 가중치 감쇠에 견주어 잡음 주입이 가장 이로운 때는 언제인가?

??? success "연습문제 4 풀이"
    손실 지형에 뾰족한 국소 극소점이 많을 때(잡음이 그것을 벗어나도록 돕는다), 학습 데이터에 레이블 잡음이 있을 때(입력 잡음이 그 영향을 누그러뜨린다), 그리고 지속 학습에서(기울기 잡음이 탐색을 유지하여 파국적 망각을 막는다) 잡음 주입이 특히 뛰어나다.

## 정리하며

이 마당은 잡음 주입의 종류、이론적 바탕、PyTorch 구현、심화 기법을 차례로 짚었다.

**참고 문헌**

1. Bishop, C. M. (1995). Training with Noise is Equivalent to Tikhonov Regularization. *Neural Computation*, 7(1), 108-116.
2. Neelakantan, A., et al. (2015). Adding Gradient Noise Improves Learning for Very Deep Networks. *arXiv*.
3. Fortunato, M., et al. (2018). Noisy Networks for Exploration. *ICLR*.
4. An, G. (1996). The Effects of Adding Noise During Backpropagation Training. *Neural Computation*, 8(3), 643-674.
