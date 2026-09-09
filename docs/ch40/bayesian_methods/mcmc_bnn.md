# 베이즈 신경 그물의 MCMC 방법

---

## 1. 두루 보기

마르코프 사슬 몬테카를로(MCMC) 방법은 베이즈 신경 그물에서 뒷분포를 미루어 보는 가장 이치에 닿는 길로, 끝에 가면 참 뒷분포로 모이는 표본을 낳는다. 셈이 비싸긴 하나 아리송함 재기의 으뜸 잣대 노릇을 하며, 어림 방법을 따질 때 견주는 밑금이 된다.

---

## 2. 해밀턴 몬테카를로(HMC)

### 알고리즘

HMC는 매개변수 밭에 밀어 나감 변수 $\mathbf{r}$을 덧대어 해밀턴 움직임을 흉내 낸다.

$$H(\theta, \mathbf{r}) = U(\theta) + K(\mathbf{r})$$

여기서 $U(\theta) = -\log p(\theta | \mathcal{D})$은 감춘 힘(음수 로그 뒷분포)이고 $K(\mathbf{r}) = \frac{1}{2}\mathbf{r}^T M^{-1} \mathbf{r}$은 움직임의 힘이다.

개구리뜀 적분기는 반걸음을 번갈아 밟는다.

$$\mathbf{r}_{t+\epsilon/2} = \mathbf{r}_t - \frac{\epsilon}{2} \nabla_\theta U(\theta_t)$$

$$\theta_{t+\epsilon} = \theta_t + \epsilon M^{-1} \mathbf{r}_{t+\epsilon/2}$$

$$\mathbf{r}_{t+\epsilon} = \mathbf{r}_{t+\epsilon/2} - \frac{\epsilon}{2} \nabla_\theta U(\theta_{t+\epsilon})$$

### PyTorch로 짜기

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Tuple

class HamiltonianMonteCarlo:
    """
    신경 그물 뒷분포를 위한 HMC 표본 뽑개.
    
    붙임말: 온전한 HMC는 자료 꾸러미 온통에 걸쳐 기울기를 셈해야 하므로
    큰 문제에는 쓰기 어렵다. 크게 늘릴 수 있는 갈음으로는
    SGLD/SGHMC을 쓴다.
    """
    
    def __init__(
        self,
        model: nn.Module,
        log_posterior_fn: Callable,
        step_size: float = 0.001,
        n_leapfrog: int = 10,
        mass_matrix: str = 'identity'
    ):
        self.model = model
        self.log_posterior_fn = log_posterior_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        
        # 표본을 뽑으려고 매개변수를 펴 놓는다
        self.param_shapes = [p.shape for p in model.parameters()]
        self.n_params = sum(p.numel() for p in model.parameters())
    
    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.model.parameters()])
    
    def _unflatten_params(self, flat: torch.Tensor):
        idx = 0
        for p, shape in zip(self.model.parameters(), self.param_shapes):
            n = p.numel()
            p.data = flat[idx:idx+n].reshape(shape)
            idx += n
    
    def _compute_potential_energy(self) -> torch.Tensor:
        """U(θ) = -log p(θ|D)"""
        return -self.log_posterior_fn(self.model)
    
    def _compute_gradient(self) -> torch.Tensor:
        """∇U(θ)"""
        self.model.zero_grad()
        U = self._compute_potential_energy()
        U.backward()
        return torch.cat([p.grad.flatten() for p in self.model.parameters()])
    
    def step(self) -> Tuple[bool, float]:
        """
        HMC 한 걸음: 개구리뜀 적분 + 메트로폴리스 받기/물리기.
        
        Returns:
            accepted: 내놓은 값을 받았는지
            log_prob: 이제 자리의 로그 낌새
        """
        # 이제 상태를 담아 둔다
        current_params = self._flatten_params().clone()
        
        # 밀어 나감을 뽑는다
        momentum = torch.randn(self.n_params)
        current_momentum = momentum.clone()
        
        # 이제의 해밀턴 값
        current_U = self._compute_potential_energy().item()
        current_K = 0.5 * torch.sum(current_momentum ** 2).item()
        
        # 개구리뜀 적분
        grad = self._compute_gradient()
        momentum = momentum - 0.5 * self.step_size * grad
        
        for i in range(self.n_leapfrog - 1):
            params = self._flatten_params() + self.step_size * momentum
            self._unflatten_params(params)
            
            grad = self._compute_gradient()
            momentum = momentum - self.step_size * grad
        
        # 마지막 반걸음
        params = self._flatten_params() + self.step_size * momentum
        self._unflatten_params(params)
        grad = self._compute_gradient()
        momentum = momentum - 0.5 * self.step_size * grad
        
        # 내놓은 자리의 해밀턴 값
        proposed_U = self._compute_potential_energy().item()
        proposed_K = 0.5 * torch.sum(momentum ** 2).item()
        
        # 메트로폴리스 받기/물리기
        log_accept = (current_U + current_K) - (proposed_U + proposed_K)
        
        if np.log(np.random.uniform()) < log_accept:
            return True, -proposed_U
        else:
            self._unflatten_params(current_params)
            return False, -current_U
    
    def sample(
        self, n_samples: int, burn_in: int = 100, thin: int = 1
    ) -> List[torch.Tensor]:
        """뒷분포 표본을 모은다."""
        samples = []
        n_accepted = 0
        
        for i in range(burn_in + n_samples * thin):
            accepted, log_prob = self.step()
            n_accepted += int(accepted)
            
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(self._flatten_params().clone())
        
        total_steps = burn_in + n_samples * thin
        accept_rate = n_accepted / total_steps
        print(f"받은 비율: {accept_rate:.3f}")
        
        return samples
```

---

## 3. 확률 기울기 랑주뱅 움직임(SGLD)

SGLD는 잔 묶음 기울기에 잡음을 섞어 크게 늘릴 수 있는 베이즈 미루어 봄을 이룬다.

$$\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2}\left(\nabla \log p(\theta_t) + \frac{N}{n}\sum_{i \in S_t} \nabla \log p(y_i | x_i, \theta_t)\right) + \eta_t$$

여기서 $\eta_t \sim \mathcal{N}(0, \epsilon_t I)$이고 $\epsilon_t$은 줄어드는 배움 비율이다.

```python
class SGLDOptimizer(torch.optim.Optimizer):
    """
    확률 기울기 랑주뱅 움직임 가장 좋게 하는 개.
    
    뒷분포 표본을 뽑으려고 SGD에 가우스 잡음 섞기를 더한다.
    배움 비율이 줄면 표본은 뒷분포로 모인다.
    """
    
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 noise_scale=1.0, temperature=1.0):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                       noise_scale=noise_scale, temperature=temperature)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # 짐 줄이기(앞선 분포)
                if group['weight_decay'] != 0:
                    d_p = d_p + group['weight_decay'] * p.data
                
                # SGD 고침
                lr = group['lr']
                p.data.add_(d_p, alpha=-lr)
                
                # 랑주뱅 잡음 섞기
                noise = torch.randn_like(p.data)
                noise_std = (2.0 * lr * group['temperature']) ** 0.5
                p.data.add_(noise, alpha=noise_std * group['noise_scale'])

def train_with_sgld(
    model: nn.Module,
    train_loader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    collect_every: int = 10,
    burn_in_epochs: int = 50,
    weight_decay: float = 1e-4,
    dataset_size: int = None
) -> List[dict]:
    """
    SGLD로 익히며 뒷분포 표본을 모은다.
    
    매개변수 찰칵(뒷분포 표본)의 목록을 돌려준다.
    """
    optimizer = SGLDOptimizer(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    samples = []
    
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            
            # 잔 묶음에 맞게 기울기 잣대를 맞춘다
            if dataset_size is not None:
                loss = loss * dataset_size / len(y)
            
            loss.backward()
            optimizer.step()
        
        # 몸풀기가 끝난 뒤 표본을 모은다
        if epoch >= burn_in_epochs and epoch % collect_every == 0:
            snapshot = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
            samples.append(snapshot)
    
    print(f"뒷분포 표본 {len(samples)}개를 모았다")
    return samples
```

---

## 4. SGHMC: 확률 기울기 해밀턴 몬테카를로

SGHMC은 더 잘 둘러보도록 SGLD에 밀어 나감을 더한다.

$$\theta_{t+1} = \theta_t + \epsilon_t \mathbf{v}_t$$

$$\mathbf{v}_{t+1} = (1 - \alpha)\mathbf{v}_t + \epsilon_t \hat{\nabla} \log p(\theta_t | \mathcal{D}) + \mathcal{N}(0, 2\alpha\epsilon_t I)$$

여기서 $\alpha$은 쓸림 값이고 $\hat{\nabla}$은 확률 기울기를 뜻한다.

```python
class SGHMCOptimizer(torch.optim.Optimizer):
    """확률 기울기 해밀턴 몬테카를로."""
    
    def __init__(self, params, lr=1e-4, momentum_decay=0.01,
                 noise_scale=1.0):
        defaults = dict(lr=lr, momentum_decay=momentum_decay,
                       noise_scale=noise_scale)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['velocity'] = torch.zeros_like(p.data)
                
                v = state['velocity']
                lr = group['lr']
                alpha = group['momentum_decay']
                
                # 쓸림 + 기울기 + 잡음
                noise = torch.randn_like(p.data)
                noise_std = (2.0 * alpha * lr) ** 0.5
                
                v.mul_(1 - alpha).add_(
                    p.grad.data, alpha=-lr
                ).add_(noise, alpha=noise_std * group['noise_scale'])
                
                p.data.add_(v, alpha=lr)
```

---

## 5. MCMC 표본으로 미루어 보기

```python
def predict_with_mcmc_samples(
    model: nn.Module,
    samples: List[dict],
    x: torch.Tensor,
    task: str = 'classification'
) -> dict:
    """
    모아 둔 MCMC 뒷분포 표본으로 미루어 본다.
    """
    all_outputs = []
    
    model.eval()
    with torch.no_grad():
        for sample in samples:
            # 뽑아 둔 짐을 얹는다
            for name, param in model.named_parameters():
                param.data.copy_(sample[name])
            
            output = model(x)
            all_outputs.append(output)
    
    outputs = torch.stack(all_outputs)  # (S, batch, dim)
    
    if task == 'classification':
        probs = torch.softmax(outputs, dim=-1)
        mean_probs = probs.mean(dim=0)
        pred_class = mean_probs.argmax(dim=-1)
        epistemic = probs.var(dim=0).mean(dim=-1)
        
        return {
            'probs': mean_probs,
            'pred_class': pred_class,
            'epistemic_uncertainty': epistemic
        }
    else:
        mean = outputs.mean(dim=0)
        epistemic_var = outputs.var(dim=0)
        
        return {
            'mean': mean,
            'epistemic_var': epistemic_var,
            'total_std': torch.sqrt(epistemic_var)
        }
```

---

## 6. 참으로 헤아릴 것

### 베이즈 신경 그물에 MCMC를 쓸 때

- 으뜸 잣대가 되는 뒷분포 어림이 있어야 하는 연구
- 작거나 가운데 크기의 모형(매개변수 1000만 미만)
- 셈 값보다 아리송함의 됨됨이가 더 종요로울 때
- 어림 방법의 밑금을 잡을 때

### 한계

- **크게 늘리기**: 온전한 HMC는 자료 꾸러미 온통의 기울기가 있어야 한다
- **섞임**: 차수가 높으면 잘 섞이지 않아 표본이 서로 얽힌다
- **살펴보기**: 신경 그물에서는 모였는지 따지기가 만만치 않다
- **여러 봉우리**: 여느 MCMC는 뒷분포의 봉우리를 다 둘러보지 못할 수 있다

### 즐겨 쓸 길

| 형편 | 방법 | 붙임말 |
|---------|--------|-------|
| 작은 모형, 으뜸 잣대 | HMC | 가장 맞으나 값이 비쌈 |
| 가운데 모형, 크게 늘리기 | SGLD | 잔 묶음과 어울림 |
| 더 잘 둘러봐야 할 때 | SGHMC | 밀어 나감이 섞임을 돕는다 |
| 큰 서비스 | 모둠이나 SWAG를 쓴다 | MCMC는 너무 비싸다 |

---

## 연습문제

**연습문제 1.**
ReLU 살림과 가우스 짐 앞선 분포를 지닌 두 켜 신경 그물에서, 이 마디에서 밝힌 방법에 따른 어림 뒷분포의 꼴을 이끌어 내어라.

??? success "연습문제 1 풀이"
    짐이 $W_1, W_2$이고 가우스 앞선 분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 뒷분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 어림 방법은 다룰 수 있는 꼴을 낸다. 변이 미루어 봄이면 짐마다 서로 남남인 가우스 뒷분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 어림이면 뒷분포가 MAP 어림을 가운데로 삼고 함께 바뀜이 헤세 행렬의 거꿀인 가우스 하나이며, MC 드롭아웃이면 뒷분포가 드롭아웃 가리개 분포로 넌지시 세워진다. 어림마다 참 뒷분포 모습의 서로 다른 결을 담는다. $\square$

---

**연습문제 2.**
이 방법에서 얻은 아리송함 어림의 눈금 맞음을 MC 드롭아웃, 깊은 모둠과 견주는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

??? success "연습문제 2 풀이"
    자: (1) 통 15개의 바라는 눈금 맞음 어긋남(ECE), (2) 브라이어 점수, (3) 음수 로그 그럴듯함(NLL), (4) 밖 분포 알아내기의 AUROC. 그림: 방법마다 본 잦기를 미루어 본 자신함에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 자료에서 눈금 맞음을, SVHN에서 밖 분포 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 밑금으로 쓴다. 아무렇게나 하는 씨앗 5개에 걸친 평균과 잣대 어긋남을 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

**연습문제 3.**
베이즈 신경 그물의 미루어 봄 흩어짐이 앎의 아리송함과 타고난 아리송함으로 쪼개짐을 증명하여라. 익힘 자료 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

??? success "연습문제 3 풀이"
    미루어 봄 흩어짐은 온 흩어짐 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 타고난 몫은 자료를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 앎의 몫은 매개변수의 아리송함을 드러내며 뒷분포가 참 매개변수 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 타고난 아리송함만 남는다. 이 쪼갬은 자료를 더 모아야 할 때(앎의 아리송함이 클 때)와 타고난 잡음을 받아들여야 할 때(타고난 아리송함이 클 때)를 가르는 데 종요롭다. $\square$

---

**연습문제 4.**
이 마디의 아리송함 재기 방법을 거래 얼개의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

??? success "연습문제 4 풀이"
    판단 규칙: 자리 크기를 앎의 아리송함에 반비례하게 잡는다. $\hat{y}$을 미루어 본 돌아옴, $\sigma_e^2$을 앎의 흩어짐이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 무릅씀 꺼림 값이다. 앎의 아리송함이 크면(낯선 저자 형편) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 앎의 아리송함 위끝을 두어 삼갈 수 있다. 이 틀은 모형의 자신함으로 잣대를 잡은 켈리 잣대 결의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$

## 정리하며

이 마당은 두루 보기、해밀턴 몬테카를로(HMC)、확률 기울기 랑주뱅 움직임(SGLD)、SGHMC: 확률 기울기 해밀턴 몬테카를로을 차례로 짚었다.

**살펴볼 거리**

- Welling, M., & Teh, Y. W. (2011). "Bayesian Learning via Stochastic Gradient Langevin Dynamics." ICML.
- Chen, T., et al. (2014). "Stochastic Gradient Hamiltonian Monte Carlo." ICML.
- Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." Handbook of MCMC.
