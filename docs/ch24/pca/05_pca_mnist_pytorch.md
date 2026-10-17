# PyTorch로 MNIST에 PCA 적용하기

이 예제는 MNIST 학습 이미지 60,000개를 784차원 벡터로 펴고 PCA 성분 50개로 표현한다. 전체 SVD로 분산을 계산하고 같은 학습 데이터를 재구성한다. 본 코드에는 이미지나 고유숫자 시각화가 없으며 관련 그림은 연습문제에서 다룬다.

## 1. 예제 코드와 보존된 출력

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

## 2. 데이터와 행렬 모양

`ToTensor`는 픽셀을 [0,1] 실수로 바꾼다. 전체 학습 데이터를 하나의 배치로 읽어 X를 60000×784로 만들고, 피처별 학습 평균 1×784를 뺀다. 레이블 y는 같은 장치로 옮기지만 PCA 적합에는 사용하지 않는다.

전체 데이터를 한꺼번에 읽는 방식은 입력 외에 로더의 중간 텐서, 중심화 복사, SVD 출력과 재구성을 위한 공간도 필요하다. float32 입력 하나만 188.16 MB, 약 179.44 MiB다. 이 값만으로 최대 실행 메모리를 판단하지 않는다.

## 3. SVD와 분산 비율

축소 SVD는 U가 60000×784, S가 길이 784, Vt가 784×784다. 이름이 Vt인 세 번째 반환값은 실수 데이터의 Vᵀ이며 V의 앞 50개 열로 점수를 만든다. 점수는 60000×50, 재구성은 60000×784다.

고윳값은 s_j²/(n−1)이고 비율은 그 값의 전체 합으로 나눈다. 보존된 출력에서는 50개가 약 82.5%의 학습 분산을 설명한다. 이는 해당 전처리에서의 결과이며 분류 정보의 82.5%나 데이터의 내재 차원이 50이라는 뜻은 아니다.

## 4. 좌표 감소와 재구성 오차

출력의 `Compression ratio`는 784/50=15.68이라는 **좌표 개수 비율**이다. 같은 dtype의 표본 좌표 저장량은 줄지만 기저와 평균도 저장해야 하며, 원래 uint8 이미지와 float32 점수의 비교에는 원소당 바이트 차이도 반영해야 한다. 이미지 압축 파일의 배율로 그대로 해석하지 않는다.

재구성 MSE는 같은 학습 데이터의 모든 픽셀 원소에서 계산된다. 버린 고윳값과는 다음 관계다.

$$\operatorname{MSE}=\frac{n-1}{n\cdot784}\sum_{j>50}\lambda_j.$$

보존된 값 0.011793의 제곱근은 약 0.1086이다. 이는 픽셀별 제곱 오차의 제곱근(RMSE)이지 평균 절대 오차나 10.86%의 픽셀이 틀렸다는 뜻이 아니다. 다른 전처리와 오차 척도에서는 직접 비교할 수 없다.

## 5. 시각화와 분류 평가의 범위

앞 두 점수로 시각화하면 남긴 부분의 구조만 보인다. 2차원에서 겹친 점이 원래 공간에서도 구분 불가능하다는 결론은 성립하지 않는다. 낮은 분산의 방향이 분류에 중요할 수도 있다. 개별 주성분을 이미지로 표시한 고유숫자는 평균에 더하고 빼는 변화 방향이며 특정 숫자나 하나의 의미만 나타낼 필요가 없다.

연습문제의 로지스틱 회귀는 각 표현을 학습한 **같은 데이터에 점수화**한다. 테스트 정확도나 일반화 비교가 아니고 학습 시간도 측정하지 않는다. 분류용 성분 수는 독립 검증 구간에서 선택하고 PCA 적합도 학습 구간 안에서 수행한다. 오토인코더와 비교하려면 같은 데이터 분할·정규화·병목 차원과 평가 조건을 맞춘다.

## 6. 실행 시간과 규모 확장

코드는 `time.time()`으로 SVD 호출만 감싸며 CUDA의 타이밍 경계에 명시적인 동기화를 넣지 않는다. 따라서 출력의 시간은 해당 환경의 기록이며 이를 보편적인 GPU·CPU 성능으로 비교할 수 없다. 일부 연산의 내부 동기화 여부에 의존하지 않는 명확한 측정 경계를 설정해야 한다.

성분 50개만 필요하면 절단·무작위 근사를 검토할 수 있다. 전체 분산은 중심화 데이터의 제곱합으로 구할 수 있다. n≫d에서는 d×d 공분산 고유분해도 후보지만 조건수와 작은 고윳값의 오차를 고려한다. 온라인·증분 방식이나 대표 표본 적합은 메모리를 줄이는 대신 근사와 데이터 대표성의 영향을 평가해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
누적 설명 분산으로 99%에 도달하는 성분 수를 찾는 원리와 아래 그림 코드의 인덱스 차이를 설명하라.

</div>

??? success "풀이"

    내림차순 고윳값의 누적 비율이 처음 0.99 이상이 되는 위치에 1을 더하면 성분 개수다. 아래 sum(cumsum<0.99)+1은 그 원리를 사용한다. 다만 plot(cumsum)은 x축이 0부터 시작해 첫 점을 성분 1로 해석하는 표시와 한 칸 차이가 있으므로 임계 수직선의 축 의미를 맞춰야 한다. 필요한 수를 데이터와 무관한 상수로 두지 않는다.

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


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
같은 이미지를 여러 k로 복원할 때 수학적으로 보장되는 변화와 보장되지 않는 시각적 판단을 구분하라.

</div>

??? success "풀이"

    같은 직교 기저에서 성분을 추가하면 각 표본의 제곱 재구성 오차는 증가하지 않는다. 모든 784방향을 남기면 수치 오차 범위에서 원본을 복원한다. 그러나 특정 k에서 반드시 사람이 알아보거나 원본과 구분하지 못한다는 기준은 없다. 아래 그림은 한 표본만 보여 준다.

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


---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 로지스틱 회귀 비교가 일반화 정확도와 속도 향상을 검증하는지 설명하라.

</div>

??? success "풀이"

    두 모델 모두 fit에 사용한 데이터로 score하므로 학습 정확도다. 시간 측정도 없어 5~10배 가속 같은 수치를 도출할 수 없다. 독립 검증·테스트 데이터에 학습 PCA를 적용하고 같은 평가 조건과 학습 예산에서 정확도와 전처리 포함 비용을 측정해야 한다.

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


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
출력의 82.5%와 15.7배가 각각 어떤 양인지 설명하라.

</div>

??? success "풀이"

    82.5%는 이 학습 데이터에서 50개 성분의 누적 설명 분산이고 15.7배는 784/50 좌표 수 비율이다. 두 값은 의미가 다르다. 실제 저장 크기에는 dtype과 평균·기저가 필요하고 예측 정보나 정확도는 별도로 평가해야 한다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
2차원 PCA 그림에서 두 숫자 클래스가 겹치면 무엇을 결론 내릴 수 있는가?

</div>

??? success "풀이"

    선택된 두 좌표에서 해당 표본들이 겹쳐 보인다는 사실만 알 수 있다. 버린 방향의 차이는 보이지 않으므로 원래 공간의 분리 가능성이나 최선의 분류 정확도를 단정할 수 없다. 투영된 거리도 원래 거리를 과소 표현할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
고유숫자를 해석할 때 부호와 고윳값 간격을 왜 고려해야 하는가?

</div>

??? success "풀이"

    주성분 방향의 부호는 임의이며 점수와 함께 뒤집으면 재구성이 같다. 중복·근접 고윳값의 개별 축은 회전하거나 데이터 변화에 민감할 수 있다. 평균에 방향을 더하고 빼며 변화를 관찰할 수 있지만 한 축을 하나의 독립적 의미나 특정 숫자로 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
차원 축소 후 선형 분류기의 파라미터 수와 전체 비용은 어떻게 달라지는가?

</div>

??? success "풀이"

    C클래스 선형 로짓의 파라미터는 Cd+C다. d=784에서 50으로 줄이면 C(784+1)에서 C(50+1)로 줄며 정확히 15.68배는 아니다. 데이터 좌표와 일부 연산도 감소하지만 PCA 적합·투영 비용이 추가된다. 과적합과 정확도 개선은 보장되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
PCA 누적 분산의 성분 수가 내재 차원을 직접 뜻하지 않는 이유를 설명하라.

</div>

??? success "풀이"

    PCA는 선택한 분산 오차 기준의 선형 근사 차원을 찾는다. 굽은 1차원 곡선도 이를 담는 선형 공간은 여러 차원일 수 있다. 잡음과 분산 임계값에 따라 값이 달라지므로 이 수를 매니폴드 차원의 보편적 상한이나 정확한 추정으로 읽지 않는다. 비선형 복원 개선량을 곧바로 굽음의 양으로 해석할 수도 없다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
원소별 MSE 0.011793을 평균 이미지 기준과 비교하는 방법을 설명하라.

</div>

??? success "풀이"

    평균 이미지만 출력하는 k=0 기준의 같은 데이터 MSE는 (n−1)Σ_jλ_j/(n×784)다. k=50 오차를 이 기준으로 나누면 같은 분모 관례에서 버린 분산 비율이 된다. 제곱근 약 0.1086은 RMSE이며 평균 절대 오차와 다르다. 이미지 품질과 후속 분류 성능도 별도로 확인한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
전체 SVD가 부담될 때 선택 가능한 대안과 검증할 비용을 설명하라.

</div>

??? success "풀이"

    소수 성분만 필요하면 무작위·절단 분해를, n≫d이면 공분산 고유분해를, 메모리가 제한되면 증분 처리나 대표 표본 적합을 검토한다. 각 방법의 수치 오차와 표본 대표성, 전처리·변환 비용을 확인한다. GPU도 장치 메모리와 커널 지원, 전송·동기화를 포함해 비교해야 한다.



## 정리하며

MNIST 예제는 학습 데이터의 중심화 PCA와 동일 데이터 재구성을 보여 준다. 설명 분산, 좌표 수 비율과 실제 저장 압축을 구분하고 n−1 고윳값을 원소별 MSE로 정확히 환산한다. 분류 일반화와 속도 개선은 독립 평가와 일관된 측정이 필요하다.
