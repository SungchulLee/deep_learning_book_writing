# MNIST 주성분 분석 PyTorch

이 두루 살핀 보기는 PyTorch로 MNIST 손글씨 숫자 자료 묶음에 주성분 분석을 써서 784차원 그림을 흩어짐의 93%쯤을 남긴 채 50차원으로 줄인다. 각본은 자료 불러오기부터 특잇값 쪼개기 셈하기, 다시 세운 것 그려 보기, 흩어짐 살피기, 2차원 흩뿌림 그림, 고유 숫자 그려 보기까지 물길 전체를 다루며 주성분 분석이 장난감 보기에서 실제 차원 높은 자료로 어떻게 커지는지 보인다.

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

## 2. 논의

MNIST 그림은 784차원 공간에 살지만 손글씨 숫자의 속 차원은 훨씬 낮다. 주성분 분석은 성분 50개만으로 전체 흩어짐의 93%를 넘게 잡아 15.7배 눌러 담음을 보여 이를 드러낸다. 스크리 그림(성분마다의 흩어짐)은 특유의 "팔꿈치" 결을 보인다. 곧 앞선 몇 성분이 흩어짐을 많이 잡고(주성분1 하나가 10%쯤) 뒤로 갈수록 보태는 몫이 줄어든다.

주성분 자체를 28x28 그림 꼴로 바꾼 것을 "고유 숫자"라 부른다. 모든 숫자에 걸친 화소 수준 흔들림의 으뜸 결을 나타낸다. 주성분1은 흔히 전체 밝기와 획의 짙기를, 주성분2은 가로획과 세로획의 방향을 잡고, 뒤 성분일수록 고리, 기울기, 삐침 같은 더 구체적인 특징을 담는다. 이 고유 숫자는 얼굴 알아보기의 "고유 얼굴"에 맞닿는다.

(주성분1과 2만 쓴) 2차원 쏘기는 뜻 있는 짜임을 드러낸다. 곧 숫자 갈래가 어느 정도 갈라지는 무리를 이루며, 눈으로 비슷한 숫자(4와 9, 3과 5)는 더 겹친다. 다만 성분 둘은 흩어짐의 17%쯤만 설명하므로 이 2차원 그림은 가르는 앎을 거의 다 잃을 수밖에 없다. 가르기 일에는 성분 50~100개를 남기는 것이 여느 관례이다.

## 연습문제

**연습문제 1.**
흩어짐의 99%를 남기는 데 필요한 가장 적은 성분 수를 정하라. 쌓아 올린 설명 흩어짐 곡선을 그리고 그 문턱을 표시하라.

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
    MNIST에서 93%에는 50개면 되지만 99%에는 흔히 300~350개쯤이 필요하다.

---

**연습문제 2.**
같은 숫자 그림을 성분 5, 20, 50, 150, 784개로 다시 세워라. 나란히 늘어놓고 성분이 늘수록 눈에 보이는 품질이 어떻게 나아지는지 살펴라.

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

**연습문제 3.**
성분 50개로 줄인 MNIST에 단순한 로지스틱 회귀 가르개를 익히고 784개 특징 전체로 익힌 것과 정확도를 견주어라. 두 정확도와 익히기 시간을 알려라.

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

## 정리하며

**다룬 것** — MNIST 주성분 분석 PyTorch

MNIST 그림은 784차원 공간에 살지만 손글씨 숫자의 속 차원은 훨씬 낮다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
