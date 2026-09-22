# 변분 자기 부호기

변분 자기 부호기(VAE) 다시 매개변수화 재주로 확률 숨은 공간을 짠다

자기 부호기와 변분 자기 부호기는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

```python
"""
변분 자기 부호기(VAE)
다시 매개변수화 재주로 확률 숨은 공간을 짠다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class VAE(nn.Module):
    """
    정규 숨은 공간을 가진 여느 변분 자기 부호기.
    
    인수:
        input_dim (int): 들임 차원(예컨대 MNIST는 784)
        hidden_dim (int): 숨은 층 차원
        latent_dim (int): 숨은 공간 차원
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 부호기: 숨은 분포의 매개변수를 내놓는다
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 숨은 분포의 평균과 로그 흩어짐
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 복호기
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        들임을 숨은 분포 매개변수로 부호화한다.
        
        인수:
            x: 입력 텐서
            
        반환값:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        다시 매개변수화 재주: z = mu + sigma * epsilon
        여기서 epsilon ~ N(0, 1)
        
        그러면 뒤먹임 퍼뜨리기를 위해 뽑기를 미분할 수 있다.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        숨은 나타냄을 내놓기로 푼다.
        
        인수:
            z: 숨은 벡터
            
        반환값:
            reconstruction: 다시 세운 내놓기
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        온전한 앞먹임: 부호화 -> 다시 매개변수화 -> 풀기
        
        인수:
            x: 입력 텐서
            
        반환값:
            reconstruction: 다시 세운 내놓기
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 자기 부호기 손실 = 다시 세우기 손실 + β * KL 벌어짐
        
        인수:
            reconstruction: 다시 세운 내놓기
            x: 본디 들임
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            beta: KL 벌어짐 항의 무게
            
        반환값:
            loss: 전체 변분 자기 부호기 손실
            bce: 다시 세우기 손실
            kld: KL 벌어짐
        """
        # 다시 세우기 손실(두 값 엇갈린 엔트로피)
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL 벌어짐: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        숨은 공간에서 표본을 만든다.
        
        인수:
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


if __name__ == '__main__':
    # 모델을 시험한다
    model = VAE(input_dim=784, latent_dim=32)
    # BCE 손실은 목표가 [0,1]이어야 한다. randn은 음수를 내므로 rand를 쓴다
    x = torch.rand(32, 784)
    
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 뽑기를 시험한다
    samples = model.sample(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
```

**출력:**

```
Input shape: torch.Size([32, 784])
Reconstruction shape: torch.Size([32, 784])
Latent mu shape: torch.Size([32, 32])
Latent logvar shape: torch.Size([32, 32])
Total Loss: 17424.4863
Reconstruction Loss: 17419.7031
KL Divergence: 4.7822
Generated samples shape: torch.Size([10, 784])
```

## 2. 논의

`VAE` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다. `forward` 방법이 셈 그래프를 정하며, 그래서 PyTorch의 저절로 미분하기가 익히는 동안 기울기 셈하기를 알아서 다룬다. 이 모듈 설계 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 넣기가 쉽다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 나타냄 배우기 일에 대한 실전 직관이 선다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
붙박이 첫자리매김에서 `VAE`의 배울 수 있는 매개변수의 총수를 셈하라. 무게와 치우침을 모두 넣어 층마다 나누어 세어라.

</div>

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

</div>

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

</div>

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
층이나 덩이의 수를 자리매김할 수 있도록 `VAE`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 층 2, 4, 8개로 시험하라.

</div>

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이한다. (수수한 파이썬 목록이 아니라) `nn.ModuleList`을 써야 PyTorch가 모든 매개변수를 가장 좋게 하기에 올린다. 다음으로 시험하라: `for n in [2, 4, 8]: model = VAE(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
이 모듈의 `forward`가 셋을 돌려주는 까닭은 무엇인가?

</div>

??? success "연습문제 5 풀이"
    손실이 출력만으로는 셈해지지 않기 때문이다.

    ```python
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar     # 셋
    ```

    KL 항이 $\mu$와 $\log\sigma^2$을 필요로 하므로 밖으로 내보내야 한다. 자기 부호기라면
    출력 하나로 충분했다([24장 모듈 연습문제 3](../../ch24/architecture/autoencoder.md)).

    그래서 익히기 반복문의 모양도 달라진다.

    ```python
    out, mu, logvar = model(x)                # 셋을 받는다
    loss = model.loss_function(out, x, mu, logvar)
    ```

    $z$도 함께 돌려주면 편리한 경우가 있다. 코드를 살펴보거나 기록하려면 필요한데,
    다시 뽑은 값이라 매번 달라진다는 점을 잊지 말아야 한다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
`reparameterize`를 따로 메서드로 두는 것이 왜 좋은가?

</div>

??? success "연습문제 6 풀이"
    익히기와 평가에서 다르게 쓰고 싶기 때문이다.

    ```python
    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu                    # 평가할 때는 뽑지 않는다
        std = (0.5 * logvar).exp()
        return mu + torch.randn_like(std) * std
    ```

    이렇게 두면 `model.eval()`이 알아서 처리해 준다. 다시 세우기 그림을 보일 때 매번
    다른 결과가 나오는 일을 막아 준다.

    다만 이 선택에는 한 가지 위험이 있다. **익힐 때 뽑지 않으면 모델이 자기 부호기가
    된다.** `model.train()`을 빠뜨리면 조용히 그렇게 되고 오류가 나지 않는다. KL이
    비정상적으로 작으면 이것을 의심할 만하다.

    그래서 뽑는지 여부를 `self.training`에 맡기지 않고 인자로 받는 설계도 쓴다. 뜻이
    더 또렷해지는 대신 부르는 곳마다 적어야 한다. 어느 쪽이든 **이 선택이 있다는 것을
    알고 고르는 것**이 중요하다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
이 모듈을 자기 부호기 모듈과 견주면 무엇이 늘었는가?

</div>

??? success "연습문제 7 풀이"
    매개변수는 거의 안 늘고 코드가 조금 는다.

    | | 자기 부호기 | 변분 자기 부호기 |
    |---|---|---|
    | 부호기 마지막 층 | $256 \times 16$ | $256 \times 32$ |
    | 매개변수 합 | 1,075,488 | 1,079,600 |
    | 늘어난 몫 | | +4,112 (0.4%) |

    **0.4% 늘려서 만들어 내는 모델이 된다.** 표본이 숫자로 보이는 비율이 0.2%에서
    57.4%로 가는 값이 이것이다.

    코드로는 다시 뽑기 한 줄, KL 항 두 줄, `forward`가 셋을 돌려주기가 늘었다.

    이 값싼 변화가 두 장을 나누어 다루는 까닭이기도 하다. 구현의 차이는 사소한데 모델의
    성격이 달라진다. 한쪽은 압축기이고 한쪽은 만들어 내는 모델이다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
이 모듈로 $\beta$-VAE를 만들려면 어디를 고치는가?

</div>

??? success "연습문제 8 풀이"
    손실 함수 한 곳이다.

    ```python
    def loss_function(self, out, x, mu, logvar, beta=1.0):
        rec = F.binary_cross_entropy(out, x, reduction='sum') / x.size(0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.size(0)
        return rec + beta * kl                        # 여기
    ```

    얼개는 손대지 않는다. 그래서 익힌 모델을 서로 갈아 끼울 수 있고, $\beta$만 바꾸어
    쓸어 보기가 쉽다([베타 VAE](beta_vae.md)).

    같은 자리에 자유 비트도 얹힌다.

    ```python
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, k)
    kl = torch.clamp(kl_dim.mean(0), min=lam).sum()
    ```

    두 방법이 모두 손실에서 끝난다는 점이 편리하다. 반면 조건부로 만들려면 얼개를
    손봐야 한다. 부호기와 풀개의 입력 차원이 달라지기 때문이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
이 모듈에 증거 하한을 좀 더 정확히 어림하는 메서드를 붙여라.

</div>

??? success "연습문제 9 풀이"
    익히기에서는 표본 하나로 어림하지만, 값매김에서는 여러 번 뽑아 평균 내는 것이 낫다.
    더 나아가 **중요도 무게를 준 하한**이 훨씬 빡빡하다.

    ```python
    @torch.no_grad()
    def iwae_bound(self, x, K=64):
        # K개 표본으로 중요도 무게를 준 하한. K=1이면 보통 ELBO.
        mu, logvar = self.encode(x)
        mu = mu.unsqueeze(0).expand(K, -1, -1)          # (K, B, k)
        logvar = logvar.unsqueeze(0).expand(K, -1, -1)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + std * eps
        out = self.decode(z.flatten(0, 1)).view(K, x.size(0), -1)
        log_pxz = -F.binary_cross_entropy(out, x.expand(K, -1, -1),
                                          reduction='none').sum(-1)
        log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
        log_qz = -0.5 * (eps.pow(2) + math.log(2 * math.pi) + logvar).sum(-1)
        w = log_pxz + log_pz - log_qz                   # (K, B)
        return torch.logsumexp(w, 0) - math.log(K)      # 표본마다
    ```

    핵심이 마지막 줄의 `logsumexp`다. 로그를 **평균 뒤에** 취하므로
    $\log \mathbb{E}[w] \ge \mathbb{E}[\log w]$에 따라 보통 ELBO보다 크고, $K \to \infty$에서
    $\log p(x)$로 간다.

    이 메서드가 값진 까닭이 하나 더 있다. [부호기 연습문제 5](encoder.md)에서 본
    고르게 나누기의 벌어짐(3.537)처럼, 부호기가 얼마나 손실을 보고 있는지 재는 잣대가
    된다. $K$를 키웠을 때 값이 크게 좋아지면 $q$가 참된 사후 분포와 많이 다르다는 뜻이다.

    `math`를 들여와야 하고 `@torch.no_grad()`를 붙여 두는 것을 잊지 말 것이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 모듈을 저장하고 되불러 올 때 무엇을 함께 적어야 하는가?

</div>

??? success "연습문제 10 풀이"
    자기 부호기에서 챙긴 것에 두 가지가 더 붙는다.

    ```python
    torch.save({'state_dict': model.state_dict(),
                'config': {'input_dim': 784, 'hidden_dim': 256, 'latent_dim': 16},
                'preprocess': 'x / 255.0',
                'beta': 1.0,                   # 어떤 무게로 익혔는가
                'free_bits': 0.0}, path)       # 자유 비트를 썼는가
    ```

    $\beta$와 자유 비트를 적어 두어야 하는 까닭은 **손실 값을 견줄 수 있어야** 하기
    때문이다. $\beta \ne 1$이거나 자유 비트를 썼다면 그 손실은 증거 하한이 아니므로
    다른 모델의 ELBO와 나란히 놓으면 안 된다([베타 VAE 연습문제 3](beta_vae.md)).

    전처리를 적는 것은 자기 부호기와 같은 이유다
    ([24장 모듈 연습문제 6](../../ch24/architecture/autoencoder.md)).

    되불러 올 때 `weights_only=True`를 쓰는 편이 안전하다.

## 정리하며

**다룬 것** — 변분 자기 부호기

`VAE` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다.

고갱이 갈래는 `VAE`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
