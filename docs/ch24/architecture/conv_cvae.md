# 누비기 조건부 변분 자기 부호기

누비기 조건부 변분 자기 부호기(ConvCVAE) 누비기 얼개와 조건부 만들어 내기를 아우른다

자기 부호기와 변분 자기 부호기는 눌러 담은 나타냄을 배우고 새 자료를 만들어 내는 힘 있는 연장이다. 이 짜기는 고갱이 얼개와 익히기 절차를 보이며 수학 얼거리를 도는 PyTorch 부호에 잇는다.

## 1. 코드

```python
"""
누비기 조건부 변분 자기 부호기(ConvCVAE)
누비기 얼개와 조건부 만들어 내기를 아우른다
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ConvConditionalVAE(nn.Module):
    """
    조건부 그림 만들어 내기를 위한 누비기 조건부 변분 자기 부호기.
    
    인수:
        latent_dim (int): 숨은 공간 차원
        num_classes (int): 조건 지을 갈래의 수
        img_channels (int): 들임 그림 채널 수
        img_size (int): 들임 그림 크기(네모 그림이라 여긴다)
    """
    
    def __init__(self, latent_dim=128, num_classes=10, img_channels=1, img_size=28):
        super(ConvConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 공간 조건 짓기를 위한 이름표 묻힘
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        
        # 부호기 - 그림 + 묻은 이름표를 채널 하나로 더 받는다
        self.encoder = nn.Sequential(
            # 들임: img_channels + 1(묻은 이름표용)
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten()
        )
        
        # 펼친 크기를 셈한다
        self.flatten_size = 128 * 4 * 4
        
        # 숨은 분포 매개변수
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # 풀개 들임: 숨은 것 + 하나만 뜨거운 갈래
        self.decoder_input = nn.Linear(latent_dim + num_classes, self.flatten_size)
        
        # 복호기
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x, labels):
        """
        갈래 이름표에 조건 지어 들임 그림을 부호화한다.
        
        인수:
            x: 들임 그림 텐서 [묶음 크기, 채널, 높이, 너비]
            labels: 갈래 이름표 [묶음 크기]
            
        반환값:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        batch_size = x.size(0)
        
        # 이름표를 묻고 공간 꼴로 바꾼다
        c_embedded = self.label_embedding(labels)
        c_embedded = c_embedded.view(batch_size, 1, self.img_size, self.img_size)
        
        # 그림과 묻은 이름표를 잇는다
        x_combined = torch.cat([x, c_embedded], dim=1)
        
        # 부호화
        h = self.encoder(x_combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        다시 매개변수화 재주.
        
        인수:
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
            
        반환값:
            z: 뽑은 숨은 벡터
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        """
        갈래 이름표에 조건 지어 숨은 나타냄을 푼다.
        
        인수:
            z: 숨은 벡터 [묶음 크기, 숨은 차원]
            labels: 갈래 이름표 [묶음 크기]
            
        반환값:
            reconstruction: 다시 세운 그림
        """
        # 이름표를 하나만 뜨겁게 부호화한다
        c_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # 숨은 부호와 조건을 잇는다
        z_combined = torch.cat([z, c_onehot], dim=1)
        
        # 디코딩
        h = self.decoder_input(z_combined)
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x, labels):
        """
        조건을 곁들인 온전한 앞먹임.
        
        인수:
            x: 들임 그림 텐서
            labels: 갈래 이름표
            
        반환값:
            reconstruction: 다시 세운 그림
            mu: 숨은 분포의 평균
            logvar: 숨은 분포의 로그 흩어짐
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, labels)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstruction, x, mu, logvar, beta=1.0):
        """
        변분 자기 부호기 손실 함수.
        
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
        BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD, BCE, KLD
    
    def sample(self, class_label, num_samples, device='cpu'):
        """
        특정 갈래에 조건 지어 표본을 만든다.
        
        인수:
            class_label: 만들 갈래(int)
            num_samples: 만들 표본의 개수
            device: 표본을 만들 기기
            
        반환값:
            samples: 만든 그림 표본
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        samples = self.decode(z, labels)
        return samples
    
    def interpolate_classes(self, z, class1, class2, num_steps=10):
        """
        숨은 부호를 고정한 채 두 갈래 사이를 사이 끼움한다.
        
        인수:
            z: 고정한 숨은 부호 [1, 숨은 차원]
            class1: 시작 갈래
            class2: 끝 갈래
            num_steps: 사이 끼움 걸음 수
            
        반환값:
            interpolations: 사이 끼움한 표본
        """
        device = z.device
        interpolations = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # 사이 끼움을 위한 부드러운 이름표를 만든다
            c1_onehot = F.one_hot(torch.tensor([class1]), num_classes=self.num_classes).float().to(device)
            c2_onehot = F.one_hot(torch.tensor([class2]), num_classes=self.num_classes).float().to(device)
            c_interpolated = (1 - alpha) * c1_onehot + alpha * c2_onehot
            
            # 디코딩
            z_combined = torch.cat([z, c_interpolated], dim=1)
            h = self.decoder_input(z_combined)
            sample = self.decoder(h)
            interpolations.append(sample)
        
        return torch.cat(interpolations, dim=0)


if __name__ == '__main__':
    # 모델을 시험한다
    model = ConvConditionalVAE(latent_dim=128, num_classes=10, img_channels=1, img_size=28)
    # BCE 손실은 목표가 [0,1]이어야 한다
    x = torch.rand(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    
    reconstruction, mu, logvar = model(x, labels)
    loss, bce, kld = model.loss_function(reconstruction, x, mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {bce.item():.4f}")
    print(f"KL Divergence: {kld.item():.4f}")
    
    # 조건부 뽑기를 시험한다
    samples = model.sample(class_label=7, num_samples=10)
    print(f"Generated samples (class 7) shape: {samples.shape}")
```

**출력:**

```
Input shape: torch.Size([32, 1, 28, 28])
Labels shape: torch.Size([32])
Reconstruction shape: torch.Size([32, 1, 28, 28])
Total Loss: 21192.6875
Reconstruction Loss: 20655.6250
KL Divergence: 537.0617
Generated samples (class 7) shape: torch.Size([10, 1, 28, 28])
```

## 2. 논의

`ConvConditionalVAE` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다. `forward` 방법이 셈 그래프를 정하며, 그래서 PyTorch의 저절로 미분하기가 익히는 동안 기울기 셈하기를 알아서 다룬다. 이 모듈 설계 덕분에 낱낱의 조각을 고치거나 모델을 더 큰 물길에 넣기가 쉽다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 결은 더 복잡한 경우로 자연스레 넓어진다. 웃매개변수, 얼개 변형, 여러 자료 묶음을 시험해 보면 이해가 깊어지고 나타냄 배우기 일에 대한 실전 직관이 선다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`ConvConditionalVAE`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

</div>

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

</div>

??? success "연습문제 2 풀이"
    첫 누비기 층의 `in_channels`을 지금 값에서 3으로 바꿔라. 식 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 누비기와 모으기 층마다 뒤의 공간 차원을 다시 셈하라. 마지막 누비기/모으기 층의 펼친 내놓기에 맞도록 첫 선형 층의 `in_features`을 고쳐라. 다음으로 확인하라: `model = ConvConditionalVAE(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

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
층이나 덩이의 수를 자리매김할 수 있도록 `ConvConditionalVAE`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 만들어라. 층 2, 4, 8개로 시험하라.

</div>

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되풀이한다. (수수한 파이썬 목록이 아니라) `nn.ModuleList`을 써야 PyTorch가 모든 매개변수를 가장 좋게 하기에 올린다. 다음으로 시험하라: `for n in [2, 4, 8]: model = ConvConditionalVAE(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
누비기 얼개에서 부류 조건을 어떻게 넣는가? 완전 연결과 무엇이 다른가?

</div>

??? success "연습문제 5 풀이"
    완전 연결에서는 벡터를 이어 붙이면 되었지만, 누비기에서는 특징 지도에 붙여야 하므로
    모양을 맞추어야 한다.

    가장 흔한 방법은 **채널로 펴서 붙이는 것**이다.

    ```python
    # y: (B, 10) 원-핫 -> (B, 10, 28, 28) 로 펴서 채널에 붙인다
    c = y.view(-1, 10, 1, 1).expand(-1, 10, 28, 28)
    x = torch.cat([x, c], dim=1)          # (B, 11, 28, 28)
    ```

    첫 누비기 층의 입력 채널이 1에서 11로 는다. 자리마다 같은 값이 들어가므로 낭비처럼
    보이지만, 누비기가 자리마다 조건을 볼 수 있게 해 준다.

    풀개는 더 쉽다. 코드가 벡터이므로 완전 연결과 같이 이어 붙인 뒤 특징 지도로 펴면 된다.

    ```python
    h = self.fc(torch.cat([z, y], dim=1)).view(-1, 64, 7, 7)
    ```

    부호기의 머리에서 붙이는 방법도 있다. 누비기 몸통을 지난 뒤 펴진 벡터에 이어 붙이면
    채널을 늘리지 않아 싸다. 대신 누비기 층이 조건을 못 본다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
조건을 넣는 자리를 어디로 할지가 왜 중요한가?

</div>

??? success "연습문제 6 풀이"
    조건이 **어느 층부터** 영향을 줄 수 있는지가 달라진다.

    | 넣는 자리 | 누비기 층이 조건을 보는가 | 비용 |
    |---|---|---|
    | 입력 채널에 | 본다 | 첫 층이 커진다 |
    | 몸통 뒤 머리에 | 못 본다 | 싸다 |
    | 층마다 | 모두 본다 | 가장 비싸다 |

    첫 번째가 관례다. 그런데 깊은 그물에서는 입력에서만 준 조건이 위층까지 잘 전달되지
    않는 일이 있다. 그래서 층마다 다시 넣어 주는 방식이 쓰인다.

    층마다 넣는 방법 가운데 널리 쓰이는 것이 조건에 따라 정규화의 눈금과 옮김을 정하는
    것이다(FiLM, 조건부 배치 정규화). 채널을 늘리지 않고 층마다 조건을 먹인다.

    ```python
    gamma, beta = self.film(y).chunk(2, dim=1)      # (B, C) 둘
    h = gamma.view(-1, C, 1, 1) * h + beta.view(-1, C, 1, 1)
    ```

    MNIST의 얕은 그물에서는 입력에 한 번 넣는 것으로 충분하다. 조건이 잘 먹고 있는지는
    지정한 부류대로 나오는 비율로 확인하면 된다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
누비기 조건부 모델에서 코드와 부류가 분리되는지 어떻게 확인하는가?

</div>

??? success "연습문제 7 풀이"
    완전 연결에서와 같다([조건부 VAE 연습문제 5](conditional_vae.md)). 코드 $z$만으로
    부류를 맞히는 분류기를 익혀 정확도가 10% 근처인지 본다.

    눈으로 보는 검사가 더 알려 주는 것이 많다. **$z$를 고정하고 $y$만 0에서 9까지 바꾸어
    한 줄로 그린다.** 열 개의 숫자가 같은 글씨체(굵기, 기울기)로 나오면 분리된 것이다.

    누비기에서 한 가지 더 볼 것이 있다. 조건을 입력 채널에 넣었다면 그 채널이 실제로
    쓰이는지 확인할 수 있다. 첫 층의 가중치에서 조건 채널에 딸린 몫의 크기를 그림 채널의
    것과 견주면 된다. 거의 0이면 조건이 무시되고 있다.

    ```python
    w = model.enc[0].weight            # (32, 11, 3, 3)
    print("그림 채널:", w[:, :1].abs().mean().item())
    print("조건 채널:", w[:, 1:].abs().mean().item())
    ```

    이 검사가 조건이 안 먹는 버그를 조용히 넘기지 않게 해 준다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
누비기와 조건부를 함께 쓰면 두 이득이 더해지는가?

</div>

??? success "연습문제 8 풀이"
    더해질 이유가 있고, 이 장의 수치로는 **확인하지 않았다.**

    잰 것은 이 둘뿐이다.

    | | 다시 세우기 | 표본 |
    |---|---|---|
    | 완전 연결 + 조건부 | 80.74 | 지정한 부류로 90.1% |
    | 누비기 (조건 없음) | 78.12 | 숫자로 보이는 것 42.5% |

    누비기 조건부를 같은 방식으로 재지 않았으므로 표의 빈 칸을 채울 수 없고, 두 수는
    **다른 것을 재고 있어** 나란히 놓을 수도 없다. 하나는 "지정한 그 숫자인가"이고
    다른 하나는 "무엇이든 숫자인가"다.

    기대할 수 있는 것은 이렇다. 누비기가 되돌리기를 돕고 조건부가 뽑기를 돕는데, 둘이
    손대는 곳이 다르므로(얼개와 입력) 부딪힐 이유가 없다.

    걸리는 점도 있다. 누비기 쪽은 표본이 나빴고 그 원인이 사전 분포와 덜 맞는 것으로
    보이는데([conv_vae 연습문제 5](conv_vae.md)), 조건부가 그 문제를 고쳐 주지는 않는다.
    조건은 코드가 나를 정보를 줄여 주므로 도움이 될 수는 있다.

    **이 물음은 재어 보아야 답할 수 있다.** 네 조합(완전 연결/누비기 × 조건 유무)을
    같은 잣대로 재는 것이 이 장에 빠진 실험이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
네 조합을 견주는 실험을 설계하라. 무엇을 잣대로 삼겠는가?

</div>

??? success "연습문제 9 풀이"
    완전 연결/누비기 × 조건 유무의 네 칸을 채우는 실험이다.

    **같게 맞출 것**: 숨은 차원 16, 20 에포크, 묶음 256, 학습률 1e-3, 씨앗 42를 모델
    만들기 직전에 고정, 같은 판정 분류기, 표본 뽑기 씨앗 고정.

    **잣대**는 네 가지를 함께 본다.

    | 잣대 | 왜 |
    |---|---|
    | 다시 세우기 | 얼개의 표현력 |
    | 차원별 KL과 살아 있는 차원 | 무너짐을 가리려면 총합으로는 안 된다 |
    | 표본 품질 | 다시 세우기로는 만들어 내기를 못 잰다 |
    | 매개변수 수 | 공정한 견줌인지 보려면 |

    조건 있는 것과 없는 것을 견줄 때 잣대를 맞추는 데 주의가 필요하다. 조건부에서
    "지정한 부류로 판정된 비율"을 쓰면 조건 없는 쪽에는 대응하는 값이 없다. 그래서
    **두 잣대를 함께** 적는 것이 맞다.

    - 무엇이든 숫자로 보이는 비율 → 네 칸 모두 잴 수 있다
    - 지정한 부류로 나온 비율 → 조건부 두 칸만

    앞쪽으로 네 칸을 견주고 뒤쪽으로 조건부 둘을 견주면 된다.

    그리고 $\beta$를 1로 고정할지 칸마다 쓸어 최적값을 쓸지 정해야 한다. 고정하면 공정
    하지만 어떤 얼개에 불리할 수 있고, 쓸면 공정하되 비용이 네 배가 된다. 쓸어서
    칸마다 최적으로 견주는 편이 결론이 튼튼하다. 여섯 값을 쓸면 24번 익히는 셈인데
    MNIST에서는 감당할 만하다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
이 모델로 '같은 글씨체의 0부터 9까지'를 만들려면 어떻게 하는가?

</div>

??? success "연습문제 10 풀이"
    $z$를 하나 고정하고 $y$만 바꾸어 열 번 푼다.

    ```python
    z = torch.randn(1, 16).repeat(10, 1)              # 같은 코드 열 벌
    y = F.one_hot(torch.arange(10), 10).float()       # 0..9
    imgs = model.decode(z, y)                         # (10, 1, 28, 28)
    ```

    이것이 조건부로 만든 값어치를 가장 잘 보이는 그림이다. 부호기와 풀개 양쪽에 $y$를
    넣었으므로 코드에 부류가 담기지 않고, 그래서 코드는 "부류를 뺀 나머지"를 나른다
    ([조건부 VAE 연습문제 1](conditional_vae.md)).

    잘되면 열 숫자의 굵기와 기울기가 비슷하게 나온다. 안 되면 두 가지 모습을 보게 된다.

    - 글씨체가 제멋대로다 → 코드가 부류와 얽혀 있다. 분리가 덜 되었다
    - 부류가 지정한 대로 안 나온다 → 조건이 풀개에 제대로 안 들어간다

    $z$를 여러 개 뽑아 여러 줄로 그리면 더 낫다. 줄마다 글씨체가 다르고 열마다 숫자가
    다른 격자가 되므로, 두 축이 정말 나뉘었는지 한눈에 보인다.

## 정리하며

**다룬 것** — 누비기 조건부 변분 자기 부호기

`ConvConditionalVAE` 갈래는 PyTorch의 `nn.Module` 겉면으로 모델 얼개를 감싼다.

고갱이 갈래는 `ConvConditionalVAE`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
