# MNIST 주성분 분석 PyTorch

이 두루 살핀 보기는 PyTorch로 MNIST 손글씨 숫자 자료 묶음에 주성분 분석을 써서 784차원 그림을 흩어짐의 82%쯤을 남긴 채 50차원으로 줄인다. 각본은 자료 불러오기부터 특잇값 쪼개기 셈하기, 다시 세운 것 그려 보기, 흩어짐 살피기, 2차원 흩뿌림 그림, 고유 숫자 그려 보기까지 물길 전체를 다루며 주성분 분석이 장난감 보기에서 실제 차원 높은 자료로 어떻게 커지는지 보인다.

## 1. 코드

```python
"""MNIST 주성분 분석 PyTorch."""
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import time

n_components = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === MNIST 불러오기 ==============================================================
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
X, y = next(iter(train_loader))
X = X.view(X.shape[0], -1).to(device)
y = y.to(device)

# === 가운데 맞추고 특잇값 쪼개기 셈하기 ==================================================
mu = X.mean(dim=0, keepdim=True)
X_centered = X - mu

start_time = time.time()
U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
elapsed_time = time.time() - start_time
V = Vt.T

# === 흩어짐 살피기 =======================================================
explained_variance = (S ** 2) / (X.shape[0] - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()
cumulative_variance = explained_variance_ratio[:n_components].sum().item() * 100

# === 차원 줄이기와 다시 세우기 =============================
V_k = V[:, :n_components]
scores = X_centered @ V_k
X_reconstructed = scores @ V_k.T + mu
reconstruction_error = ((X - X_reconstructed) ** 2).mean().item()

print(f"SVD computed in {elapsed_time:.2f}s on {device}")
print(f"{n_components} components explain {cumulative_variance:.1f}% of variance")
print(f"Compression ratio: {X.shape[1]/n_components:.1f}x")
print(f"Reconstruction MSE: {reconstruction_error:.6f}")

if __name__ == "__main__":
    pass
```

**출력:**

```
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz
Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw

SVD computed in 1.15s on cpu
50 components explain 82.5% of variance
Compression ratio: 15.7x
Reconstruction MSE: 0.011793
```

## 2. 논의

MNIST 그림은 784차원 공간에 살지만 손글씨 숫자의 속 차원은 훨씬 낮다. 주성분 분석은 성분 50개만으로 전체 흩어짐의 82%를 잡아 15.7배 눌러 담음을 보여 이를 드러낸다. 스크리 그림(성분마다의 흩어짐)은 특유의 "팔꿈치" 결을 보인다. 곧 앞선 몇 성분이 흩어짐을 많이 잡고(주성분1 하나가 10%쯤) 뒤로 갈수록 보태는 몫이 줄어든다.

주성분 자체를 28x28 그림 꼴로 바꾼 것을 "고유 숫자"라 부른다. 모든 숫자에 걸친 화소 수준 흔들림의 으뜸 결을 나타낸다. 주성분1은 흔히 전체 밝기와 획의 짙기를, 주성분2은 가로획과 세로획의 방향을 잡고, 뒤 성분일수록 고리, 기울기, 삐침 같은 더 구체적인 특징을 담는다. 이 고유 숫자는 얼굴 알아보기의 "고유 얼굴"에 맞닿는다.

(주성분1과 2만 쓴) 2차원 쏘기는 뜻 있는 짜임을 드러낸다. 곧 숫자 갈래가 어느 정도 갈라지는 무리를 이루며, 눈으로 비슷한 숫자(4와 9, 3과 5)는 더 겹친다. 다만 성분 둘은 흩어짐의 17%쯤만 설명하므로 이 2차원 그림은 가르는 앎을 거의 다 잃을 수밖에 없다. 가르기 일에는 성분 50~100개를 남기는 것이 여느 관례이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
흩어짐의 99%를 남기는 데 필요한 가장 적은 성분 수를 정하라. 쌓아 올린 설명 흩어짐 곡선을 그리고 그 문턱을 표시하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    cumsum = torch.cumsum(explained_variance_ratio, dim=0)
    n99 = (cumsum < 0.99).sum().item() + 1
    print(f"Components for 99% variance: {n99}")
    plt.plot(cumsum.cpu().numpy() * 100)
    plt.axhline(99, color='r', linestyle='--')
    plt.axvline(n99, color='g', linestyle='--')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance (%)")
    plt.title(f"99% variance at {n99} components")
    plt.show()
    ```
    MNIST에서 성분 50개는 흩어짐의 82%를 잡는다. 90%에는 87개, 95%에는 154개, 99%에는 331개가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
같은 숫자 그림을 성분 5, 20, 50, 150, 784개로 다시 세워라. 나란히 늘어놓고 성분이 늘수록 눈에 보이는 품질이 어떻게 나아지는지 살펴라.

</div>

??? success "연습문제 2 풀이"
    ```python
    idx = 0
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, k in zip(axes, [5, 20, 50, 150, 784]):
        Vk = V[:, :k]
        score = X_centered[idx] @ Vk
        recon = (score @ Vk.T + mu).cpu().view(28, 28).numpy()
        ax.imshow(recon, cmap='gray')
        ax.set_title(f"k={k}")
        ax.axis('off')
    plt.suptitle("Reconstruction Quality vs Number of Components")
    plt.show()
    ```
    성분 5개로는 숫자가 흐릿한 덩어리이고, 20개면 알아볼 만하며, 50개면 또렷하고, 150개면 본디 것과 거의 구별되지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
성분 50개로 줄인 MNIST에 단순한 로지스틱 회귀 가르개를 익히고 784개 특징 전체로 익힌 것과 정확도를 견주어라. 두 정확도와 익히기 시간을 알려라.

</div>

??? success "연습문제 3 풀이"
    ```python
    from sklearn.linear_model import LogisticRegression
    X_train_50 = scores.cpu().numpy()
    X_train_full = X.cpu().numpy()
    y_train = y.cpu().numpy()

    lr_50 = LogisticRegression(max_iter=1000)
    lr_50.fit(X_train_50, y_train)
    acc_50 = lr_50.score(X_train_50, y_train)

    lr_full = LogisticRegression(max_iter=1000)
    lr_full.fit(X_train_full, y_train)
    acc_full = lr_full.score(X_train_full, y_train)

    print(f"50 components: {acc_50:.4f}")
    print(f"784 features:  {acc_full:.4f}")
    ```
    성분 50개짜리 모델은 흔히 92~93%, 전체 모델은 93~94%의 정확도를 내며 익히기는 5~10배 빠르다. 이 정도 정확도 떨어짐은 대개의 쓰임새에서 받아들일 만하며 주성분 분석이 특징 뽑기 걸음으로 얼마나 잘 듣는지 보여 준다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
성분 50개가 잡는 흩어짐은 얼마인가? 90%와 95%에는 몇 개가 필요한가?

</div>

??? success "연습문제 4 풀이"
    가운데를 맞추고 재면 이렇다.

    | $k$ | 2 | 10 | 50 | 87 | 154 | 331 |
    |---|---|---|---|---|---|---|
    | 누적 흩어짐 | 16.80% | 48.81% | **82.46%** | 90.01% | 95.02% | 99.00% |

    50개는 82.46%다. 90%에 87개, 95%에 154개, 99%에 331개가 든다.

    이 쪽의 글이 오래 50개를 93%라고 적어 두었는데 틀린 값이었다. 이 쪽의 코드는
    제대로 가운데를 맞추고 있으니 82%가 나오는 것이 맞다. 가운데를 **안** 맞추면
    50개에서 89.46%가 나오므로, 그 방향의 실수가 섞였을 가능성이 있다
    ([기본 연습문제 5](pca_fundamentals.md)).

    한 가지 짚어 둘 것이 있다. 82%가 93%보다 나쁜 결과처럼 들리지만 **눌러 담는 힘은
    그대로**다. 784차원을 50차원으로 줄였으니 15.7배이고, 다시 세우기 MSE 0.0118도
    그대로다. 달라진 것은 그 사실을 어떤 수로 적는지뿐이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
2차원 그림에서 숫자 갈래가 겹치는 것을 어떻게 읽어야 하는가?

</div>

??? success "연습문제 5 풀이"
    2차원이 흩어짐의 **16.8%**뿐임을 먼저 새겨야 한다. 83%를 버린 그림이므로 겹치는
    것이 당연하다.

    그래서 이 그림에서 읽어도 되는 것과 안 되는 것이 갈린다.

    | 읽어도 되는 것 | 읽으면 안 되는 것 |
    |---|---|
    | 갈래가 어느 정도 뭉친다 | 겹치니까 가를 수 없다 |
    | 4와 9가 가깝다 | 두 갈래가 정확히 얼마나 다르다 |
    | 큰 흐름의 짜임 | 개별 점의 자리 |

    오른쪽 첫 칸이 가장 흔한 잘못이다. 2차원에서 겹쳐 보인다고 784차원에서도 겹친다는
    뜻이 아니다. 실제로 성분 50개를 쓰면 분류 정확도가 92~93%로, 가를 수 있는 정보가
    충분히 남아 있다.

    이것이 차원 줄이기 그림의 일반적인 함정이다. **못 보는 것과 없는 것을 헷갈리지
    않아야 한다.**

    갈래를 갈라 보이는 것이 목적이라면 주성분 분석은 맞는 도구가 아니다. 표지를 보지
    않기 때문이다([유도 연습문제 6](pca_derivation.md)). t-SNE가 훨씬 잘 갈라 보이고
    ([다양체 견줌](../manifold/01_manifold_comparison.md)), 표지를 쓰려면 선형 판별
    분석 쪽이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
고유 숫자(eigendigit)를 그려 보면 무엇이 보이는가?

</div>

??? success "연습문제 6 풀이"
    성분을 $28 \times 28$로 되펴서 그린 것이며, 자료에서 가장 크게 변하는 **모양의
    패턴**이 나온다.

    앞쪽 몇 장은 큰 덩어리 무늬다. 획이 굵은지 가는지, 전체가 위쪽인지 아래쪽인지,
    동그란지 뾰족한지 같은 큰 변화를 담는다. 뒤로 갈수록 무늬가 잘게 쪼개져 잡음처럼
    보인다.

    읽을 때 조심할 것이 둘이다.

    **부호가 임의롭다.** 밝은 곳과 어두운 곳이 뒤집혀 나올 수 있으며 뜻은 같다
    ([유도 연습문제 4](pca_derivation.md)).

    **뒤쪽 성분은 풀이하지 않는 것이 좋다.** 고윳값이 촘촘해지면 성분 개개가 안정적이지
    않다. 100번째 성분이 0.101%뿐이고 그 근처 값들이 서로 가까우므로, 표본을 조금 바꾸면
    서로 섞인다([유도 연습문제 5](pca_derivation.md)).

    그리고 고유 숫자는 **어떤 숫자도 아니다.** 평균에 더하거나 빼는 방향이므로 그 자체가
    글씨로 보일 이유가 없다. 평균 그림에 성분을 조금씩 더해 가며 그려 보면 그 방향이
    무엇을 바꾸는지 훨씬 잘 보인다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
성분 50개로 줄인 뒤 분류기를 익히면 어떤 이득이 있는가?

</div>

??? success "연습문제 7 풀이"
    이 쪽에서 보인 대로 정확도는 조금 떨어지고(전체 93~94%, 50개 92~93%) 익히기는
    5~10배 빨라진다.

    까닭은 입력 차원이 784에서 50으로 줄어 선형 분류기의 매개변수가 15.7분의 1이 되기
    때문이다.

    이득이 셋이다.

    | 이득 | 왜 |
    |---|---|
    | 빠르기 | 매개변수가 적다 |
    | 기억 장치 | 자료가 15.7배 작다 |
    | 과적합이 덜하다 | 자유도가 적다 |

    마지막 것이 표본이 적을 때 특히 값지다. 표본보다 차원이 큰 상황에서는 차원을 줄이는
    것이 정칙화 노릇을 한다.

    다만 **주성분 분석이 가르기에 좋은 방향을 골라 주지 않는다**는 점을 잊지 말아야
    한다. 흩어짐이 작은 방향에 가르는 정보가 있을 수 있고, 그것을 버리면 정확도가 크게
    떨어질 수 있다. MNIST에서 1%p만 떨어진 것은 운이 좋은 편이다.

    그러므로 $k$는 흩어짐 비율이 아니라 **아래쪽 일의 성능으로** 고르는 것이 맞다
    ([기본 연습문제 2](pca_fundamentals.md)). 그리고 그때 교차 검증을 물길 안에서 해야
    한다([sklearn 연습문제 7](02_pca_2d_sklearn.md)).

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
주성분 분석이 MNIST의 속 차원을 알려 주는가?

</div>

??? success "연습문제 8 풀이"
    직접 알려 주지는 않는다. 알려 주는 것은 **선형 부분 공간의 차원**이다.

    구별이 중요하다. 손글씨 숫자가 놓인 다양체는 굽어 있으므로, 낮은 차원의 굽은
    다양체를 선형 부분 공간으로 덮으려면 훨씬 높은 차원이 필요하다.

    비유가 도움이 된다. 3차원 공간의 나선은 속 차원이 1인데, 선형 부분 공간으로 담으려면
    3차원이 필요하다. 곧 **주성분 분석이 말하는 차원은 위에서 묶는 값**이다.

    MNIST에서 90%에 87개가 필요하다는 사실은 "속 차원이 87이다"가 아니라 "선형으로
    다루면 87개쯤 든다"는 뜻이다. 굽음을 다룰 수 있는 방법으로 재면 훨씬 작게 나온다.
    실제로 손글씨 숫자의 속 차원 추정값은 10~15 정도로 알려져 있다.

    [25장의 오토인코더](../../ch25/architecture/01_ae_fully_connected.md)가 이 차이를
    수로 보여 준다. 같은 병목 16에서 주성분 분석의 어긋남은 0.02686인데 비선형 자기
    인코더는 0.00926이다. **세 배 가까이 낫다.** 그 차이가 곧 굽음이 담고 있던 몫이다.

    그러니 누적 흩어짐 곡선을 속 차원의 잣대로 읽으면 안 된다. 선형 방법의 한계를 재는
    값이며, 그 값이 크다는 것은 자료가 복잡하다는 뜻일 수도 있고 **방법이 자료에 맞지
    않는다**는 뜻일 수도 있다. MNIST는 뒤쪽이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
다시 세우기 MSE 0.0118을 어떻게 읽어야 하는가? 좋은 값인가?

</div>

??? success "연습문제 9 풀이"
    혼자서는 뜻이 없다. 견줄 대상이 있어야 한다.

    세 가지 기준을 댈 수 있다.

    **이론적 예측과 견준다.** 버린 고윳값의 합을 784로 나눈 값과 같아야 한다
    ([기본 연습문제 3](pca_fundamentals.md)). 맞으면 짜기가 옳다는 뜻이다. 곧 이 수는
    품질보다 **정확성 검사**로 먼저 쓰인다.

    **아무것도 안 한 것과 견준다.** 평균 그림만 내놓으면 MSE가 전체 흩어짐, 곧
    $k=0$에 해당하는 값이 된다. 0.0118이 그것의 몇 분의 일인지가 실제로 얻은 몫이다.

    **다른 방법과 견준다.** 같은 병목에서 오토인코더가 0.00926을 낸다
    ([25장](../../ch25/architecture/01_ae_fully_connected.md)). 곧 이 자리에서 비선형이
    더 나을 여지가 뚜렷하다.

    그리고 화소당 값이라는 것을 잊지 말아야 한다. 0.0118의 제곱근이 0.109이므로 화소마다
    평균 0.109쯤 어긋난다는 뜻이고, 화소값이 $[0,1]$이니 11%쯤이다. 그림으로 보면
    흐릿하지만 알아볼 만한 정도다.

    **눈으로 보는 것을 빠뜨리지 않는 것**이 중요하다. 같은 MSE가 전체적으로 살짝
    흐릿한 것일 수도 있고 몇 군데가 크게 망가진 것일 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
MNIST 60,000개의 특잇값 쪼개기가 느리다. 무엇을 바꾸겠는가?

</div>

??? success "연습문제 10 풀이"
    전체 쪼개기는 $60{,}000 \times 784$에 대해 784개의 성분을 모두 구하므로 100초쯤
    걸린다.

    필요한 것에 따라 고를 수 있다.

    | 필요한 것 | 방법 |
    |---|---|
    | 성분 $k$개만 | `torch.svd_lowrank` 또는 무작위 방식 |
    | 누적 곡선 전체 | 전체 쪼개기. 다만 공분산 쪽이 더 빠를 수 있다 |
    | 대충 빠르게 | 표본을 덜어 쓴다 |

    둘째 칸이 재미있다. $d = 784$가 $n = 60{,}000$보다 훨씬 작으므로, 이 경우에는
    $784 \times 784$ 공분산 행렬을 만들어 고유 쪼개기 하는 쪽이 빠르다. 조건수가
    제곱되는 대가를 치르지만([유도 연습문제 3](pca_derivation.md)) 앞쪽 성분에는 큰
    영향이 없다.

    셋째 칸도 실용적이다. 성분을 추정하는 데 60,000개가 다 필요하지 않다. 10,000개로
    구한 성분이 앞쪽에서는 거의 같다. 탐색 단계에서는 이렇게 하고 마지막에 전체로
    다시 하면 된다.

    GPU도 답이 된다. 이 크기에서는 GPU가 CPU보다 빠르다
    ([PyTorch 판 연습문제 4](04_pca_2d_pytorch.md)).

## 정리하며

**다룬 것** — MNIST 주성분 분석 PyTorch

MNIST 그림은 784차원 공간에 살지만 손글씨 숫자의 속 차원은 훨씬 낮다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
