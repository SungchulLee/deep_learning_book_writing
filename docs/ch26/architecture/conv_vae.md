# 누비기 변분 오토인코더

누비기 변분 오토인코더(ConvVAE) 공간 특징을 더 잘 뽑으려 누비기 층을 쓴다

오토인코더와 변분 오토인코더는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

```python
"""
누비기 변분 오토인코더(ConvVAE)
공간 특징을 더 잘 뽑으려 누비기 층을 쓴다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ConvVAE(nn.Module):
    """
    그림 자료를 위한 누비기 변분 오토인코더.
    
    인수:
        latent_dim (int): 숨은 공간 차원
        img_channels (int): 들임 그림 채널 수(회색조 1, RGB 3)
        img_size (int): 들임 그림 크기(네모 그림이라 여긴다)
    """
    
    def __init__(self, latent_dim=128, img_channels=1, img_size=28):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 인코더
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 7x7 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # 펼친 크기를 셈한다
        self.flatten_size = 128 * 4 * 4
        
        # 숨은 공간 매개변수
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # 디코더 들임
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # 디코더
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            # 4x4 -> 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """
        들임 그림을 숨은 분포 매개변수로 부호화한다.
        
        인수:
            x: 들임 그림 텐서 [배치 크기, 채널, 높이, 너비]
            
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
        뽑기를 위한 다시 매개변수화 재주.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        숨은 나타냄을 그림으로 푼다.
        
        인수:
            z: 숨은 벡터
            
        반환값:
            reconstruction: 다시 세운 그림
        """
        h = self.decoder_input(z)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        """
        온전한 앞먹임.
        
        인수:
            x: 들임 그림 텐서
            
        반환값:
            reconstruction: 다시 세운 그림
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 오토인코더 손실 함수.
        
        인수:
            reconstruction: 다시 세운 내놓기
            x: 본디 들임
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            beta: KL 벌어짐 항의 무게
            
        반환값:
            loss: 전체 변분 오토인코더 손실
            bce: 다시 세우기 손실
            kld: KL 벌어짐
        """
        # 되살림 손실
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        
        # KL 발산
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, num_samples, device='cpu'):
        """
        숨은 공간에서 표본을 만든다.
        
        인수:
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 그림 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


if __name__ == '__main__':
    # 모델을 시험한다
    model = ConvVAE(latent_dim=128, img_channels=1, img_size=28)
    # BCE 손실은 목표가 [0,1]이어야 한다
    x = torch.rand(32, 1, 28, 28)  # 회색조 28x28 그림 32개 배치
    
    reconstruction, mu, logvar = model(x)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 뽑기를 시험한다
    samples = model.sample(num_samples=10)
    print(f"Generated samples shape: {samples.shape}")
```

**출력:**

```
Input shape: torch.Size([32, 1, 28, 28])
Reconstruction shape: torch.Size([32, 1, 28, 28])
Latent mu shape: torch.Size([32, 128])
Total Loss: 21310.0566
Reconstruction Loss: 20787.1797
KL Divergence: 522.8765
Generated samples shape: torch.Size([10, 1, 28, 28])
```

## 2. 논의

`ConvVAE` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다. `forward` 방법이 셈 그래프를 정하며, 그래서 PyTorch의 저절로 미분하기가 익히는 동안 기울기 셈하기를 알아서 다룬다. 이 모듈 설계 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 넣기가 쉽다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 나타냄 배우기 일에 대한 실전 직관이 선다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`ConvVAE`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 배치에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

</div>

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "연습문제 2 풀이"
    첫 누비기 층의 `in_channels`을 지금 값에서 3으로 바꿔라. 식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 누비기와 모으기 층마다 뒤의 공간 차원을 다시 셈하라. 마지막 누비기/모으기 층의 펼친 내놓기에 맞도록 첫 선형 층의 `in_features`을 고쳐라. 다음으로 확인하라: `model = ConvVAE(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

</div>

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
층이나 덩이의 수를 자리매김할 수 있도록 `ConvVAE`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 층 2, 4, 8개로 시험하라.

</div>

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이한다. (수수한 파이썬 목록이 아니라) `nn.ModuleList`을 써야 PyTorch가 모든 매개변수를 가장 좋게 하기에 올린다. 다음으로 시험하라: `for n in [2, 4, 8]: model = ConvVAE(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
누비기 층을 쓰면 매개변수가 얼마나 줄어드는가?

</div>

??? success "연습문제 5 풀이"
    같은 숨은 차원 16에서 재면 이렇다.

    | | 완전 연결 | 누비기 |
    |---|---|---|
    | 매개변수 | 1,079,600 | **205,825** |
    | 비 | 5.2배 | 1 |

    5분의 1로 줄어든다. 까닭은 [3.4절](../../ch03/mnist/04_cnn.md)에서 본 것과 같다.
    누비기는 필터를 온 자리에서 나누어 쓰므로 매개변수가 그림 크기에 안 딸린다.

    오토인코더에서 잰 값과 견줄 만하다. 거기서는 61,329개였다
    ([25장](../../ch25/architecture/02_ae_cnn.md)). 여기가 더 큰 것은 $2k$차원을
    내놓는 인코더 머리와 $7 \times 7 \times 64$로 가는 디코더 머리가 큰 선형층이기 때문이다.
    누비기 몸통은 작고 그 두 머리가 대부분을 차지한다.

    머리를 줄이려면 마지막 특징 지도를 더 작게 만들거나(누비기를 한 층 더) 전역 평균
    모으기를 쓸 수 있다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
누비기 변분 오토인코더가 다시 세우기와 표본 가운데 어느 쪽에서 이기는가?

</div>

??? success "연습문제 6 풀이"
    **다시 세우기는 이기고 표본은 진다.**

    | | 완전 연결 | 누비기 |
    |---|---|---|
    | 매개변수 | 1,079,600 | **205,825** |
    | 다시 세우기 | 81.22 | **78.12** |
    | KL | 20.32 | 24.49 |
    | 표본이 확신도 0.9를 넘는 비율 | **57.4%** | 42.5% |

    다섯 배 적은 매개변수로 되돌리기를 더 잘한다. 공간 구조를 쓰는 얼개의 값어치다.

    그런데 표본은 오히려 나쁘다. 57.4% 대 42.5%다. 그리고 KL이 더 크다(24.49 대 20.32).
    두 사실이 이어져 있다. **코드가 정보를 더 많이 담고 있고, 그만큼 사전 분포와 덜
    맞는다.**

    되돌리기를 잘하는 쪽이 뽑기를 잘하지 못하는 이 모습이 이 사다리의 되풀이되는 주제다.
    [25장에서](../../ch25/limits/latent_sampling.md) 오토인코더가 변분 오토인코더보다
    되돌리기를 잘하면서 뽑기에서 완패한 것과 같은 구조다.

    **되돌리기 오차는 만들어 내기 품질의 잣대가 아니다.** 얼개를 고를 때도 마찬가지다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
디코더에서 `ConvTranspose2d`를 쓰는데, 대신 무엇을 쓸 수 있는가?

</div>

??? success "연습문제 7 풀이"
    키우기와 누비기를 나누는 방법이 있다.

    ```python
    # 방법 1: 옮겨 누비기
    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)

    # 방법 2: 키운 뒤 누비기
    nn.Upsample(scale_factor=2, mode='nearest')
    nn.Conv2d(64, 32, 3, padding=1)
    ```

    두 번째가 격자 무늬(checkerboard artifact)를 피하는 데 낫다고 알려져 있다. 옮겨
    누비기는 보폭과 낟알 크기가 맞지 않으면 출력 자리마다 겹치는 횟수가 달라져 규칙적인
    얼룩이 생긴다.

    그 겹침을 고르게 하려면 **낟알 크기가 보폭의 배수**여야 한다. 이 장의 코드가
    `kernel_size=4, stride=2`를 쓴 까닭이 그것이다. 3과 2로 두면 얼룩이 생기기 쉽다.

    모양 셈은 [3.4절](../../ch03/mnist/04_cnn.md)의 식을 거꾸로 한 것이다.

    $$H' = (H - 1)s - 2p + k$$

    $H=7$, $s=2$, $p=1$, $k=4$이면 $H' = 12 - 2 + 4 = 14$다. 한 번 더 하면 28이 된다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
인코더에서 모으기(pooling) 대신 보폭 2를 쓴 까닭은 무엇인가?

</div>

??? success "연습문제 8 풀이"
    둘 다 크기를 절반으로 줄이지만 성격이 다르다.

    | | 최대 모으기 | 보폭 2 누비기 |
    |---|---|---|
    | 배울 것 | 없다 | 있다 |
    | 되돌리기 | 어디가 최대였는지 잃는다 | 정보가 가중치에 남는다 |

    오토인코더 계열에서 두 번째 줄이 중요하다. 디코더가 크기를 되돌려야 하는데, 최대
    모으기는 **어느 자리가 최대였는지**를 버리므로 그 정보를 복구할 수 없다.

    보폭 누비기는 줄이는 방식을 배우므로 되돌리기에 필요한 것을 남길 수 있다. 그래서
    오토인코더와 변분 오토인코더에서는 보폭이 관례다.

    최대 모으기를 쓰면서 자리를 기억해 두는 방법도 있다(`MaxUnpool2d`에 인덱스를
    넘긴다). 다만 그 인덱스가 곧 정보이므로 코드 밖으로 정보가 새는 셈이라, 병목의
    뜻이 흐려진다. 그 점이 오토인코더에서 꺼리는 까닭이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
누비기 쪽 KL이 더 큰 것(24.49 대 20.32)을 어떻게 읽어야 하는가?

</div>

??? success "연습문제 9 풀이"
    두 가지로 읽을 수 있고, 이 경우 어느 쪽인지 가릴 수 있다.

    **읽기 1: 코드가 정보를 더 많이 담는다.** KL은 코드가 나르는 정보량의 상한이므로
    ([KL 항 연습문제 6](../theory/kl_term.md)), 크다는 것은 더 많이 담는다는 뜻일 수 있다.
    다시 세우기가 더 좋은 것(78.12 대 81.22)이 이를 뒷받침한다.

    **읽기 2: 사전 분포와 덜 맞는다.** KL은 $q(z\mid x)$와 $p(z)$의 벌어짐이므로, 크다는
    것은 덜 맞는다는 뜻일 수도 있다. 표본이 나쁜 것(42.5% 대 57.4%)이 이를 뒷받침한다.

    사실 둘 다 맞고, [사전 분포 연습문제 3](../architecture/prior.md)의 분해가 왜
    그런지 설명한다.

    $$\mathbb{E}_{p(x)}[\text{KL}] = I(x;z) + D_{\mathrm{KL}}(q(z) \| p(z))$$

    KL이 커진 것이 두 항 가운데 어디서 왔는지가 물음이다. 다시 세우기가 좋아진 것은
    앞 항이 컸다는 뜻이고, 표본이 나빠진 것은 뒤 항도 컸다는 뜻이다. **둘 다 늘었다.**

    그래서 총 KL만 보고 좋다 나쁘다 말할 수 없다. 이 장이 되풀이하는 말이 여기서도
    같다. 총 KL은 여러 가지를 한 수에 뭉쳐 놓은 값이므로, **차원별로 뜯고 다른 잣대와
    함께 보아야** 뜻이 생긴다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
MNIST가 아닌 큰 그림에서는 이 견줌이 어떻게 달라지겠는가?

</div>

??? success "연습문제 10 풀이"
    누비기 쪽이 훨씬 유리해진다.

    까닭은 매개변수 셈에 있다. 완전 연결 인코더의 첫 층은 화소 수에 비례해 커지는데
    누비기는 그렇지 않다.

    | | 28×28 | 128×128 |
    |---|---|---|
    | 완전 연결 첫 층 | $784 \times 512$ | $16{,}384 \times 512$ (21배) |
    | 누비기 첫 층 | $3\times3\times1\times32$ | 그대로 |

    그리고 큰 그림에서는 공간 구조가 더 중요해진다. MNIST는 28×28이라 완전 연결로도
    화소 사이 관계를 외울 만하지만, 큰 그림에서는 불가능하다.

    그래서 실무의 그림 모델은 거의 모두 누비기 계열이고, 이 장에서 완전 연결이 표본에서
    이긴 것을 일반적인 결론으로 읽으면 안 된다. **MNIST 크기에서 이 설정으로 그렇다는
    것**이다.

    표본 품질의 차이가 얼개 탓인지 익히기 설정 탓인지도 확인해 볼 가치가 있다. 같은
    20 에포크를 주었지만 두 얼개의 수렴 속도가 다를 수 있고, $\beta$의 최적값도 다를
    수 있다. 이 장의 수치는 그 확인까지 하지 않았다.

## 정리하며

**다룬 것** — 누비기 변분 오토인코더

`ConvVAE` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다.

고갱이 갈래는 `ConvVAE`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
