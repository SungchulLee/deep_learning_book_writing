# Matplotlib

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- `Figure`와 `Axes`의 관계를 알고 여러 그림을 나란히 그리기
- 학습 곡선을 그려 과적합을 눈으로 알아내기
- 이미지 자료와 가중치를 그림으로 확인하기
- 그림을 파일로 저장하기

---

## 2. 왜 그려 보아야 하는가

학습이 잘못되어도 프로그램은 오류를 내지 않는다. 손실이 줄지 않아도, 과적합이 일어나도, 자료가 뒤집혀 들어가도 코드는 얌전히 끝까지 돈다. **숫자만 보아서는 알기 어려운 것을 그림은 한눈에 보여 준다.**

Matplotlib의 구조는 두 층이다.

- **Figure**: 도화지 한 장
- **Axes**: 그 위에 놓인 그래프 하나. 한 도화지에 여럿 놓을 수 있다

```python
import matplotlib
matplotlib.use("Agg")            # 화면 없이 파일로만 그릴 때 쓰는 설정
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 200)

# 도화지 하나에 그래프 두 개를 가로로 놓는다
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

axes[0].plot(x, np.sin(x), label="sin")
axes[0].plot(x, np.cos(x), label="cos", linestyle="--")
axes[0].set_title("두 곡선")
axes[0].set_xlabel("x")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(np.random.default_rng(0).normal(size=1000), bins=30)
axes[1].set_title("정규분포 표본 1000개")

plt.tight_layout()

print(f"Figure 하나에 Axes {len(axes)}개")
print(f"첫 번째 Axes에 그려진 선: {len(axes[0].lines)}개")
print(f"figsize: {fig.get_size_inches()} 인치")
```

**출력:**

```
Figure 하나에 Axes 2개
첫 번째 Axes에 그려진 선: 2개
figsize: [10.   3.5] 인치
```

`plt.plot(...)`처럼 곧바로 부르는 방식도 있지만, `fig, axes = plt.subplots(...)`로 그릴 곳을 분명히 정하는 편이 그림이 여럿일 때 헷갈리지 않는다.

---

## 3. 학습 곡선

딥러닝에서 가장 자주 그리는 그림은 **에포크에 따른 손실**이다. 학습 손실과 검증 손실을 함께 그리면 과적합이 눈에 보인다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 학습을 흉내 낸 손실 값. 실제로는 학습 루프에서 모아 둔 값을 쓴다.
epochs = np.arange(1, 31)
train_loss = 1.6 * np.exp(-epochs / 6) + 0.04
val_loss = 1.6 * np.exp(-epochs / 6) + 0.04 + np.maximum(0, (epochs - 12)) * 0.012

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, train_loss, label="학습 손실", marker="o", markersize=3)
ax.plot(epochs, val_loss, label="검증 손실", marker="s", markersize=3)

# 검증 손실이 가장 낮은 지점 — 여기서 멈추는 것이 조기 종료이다
best = val_loss.argmin()
ax.axvline(epochs[best], color="gray", linestyle=":", linewidth=1)
ax.annotate(f"가장 좋은 곳 (에포크 {epochs[best]})",
            xy=(epochs[best], val_loss[best]),
            xytext=(epochs[best] + 3, val_loss[best] + 0.25),
            arrowprops=dict(arrowstyle="->", color="gray"))

ax.set_xlabel("에포크")
ax.set_ylabel("손실")
ax.set_title("학습 곡선")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

print(f"검증 손실이 가장 낮은 에포크: {epochs[best]}")
print(f"그때 학습 손실 {train_loss[best]:.4f}, 검증 손실 {val_loss[best]:.4f}")
print(f"마지막 에포크    학습 손실 {train_loss[-1]:.4f}, 검증 손실 {val_loss[-1]:.4f}")
print("\n학습 손실은 계속 줄지만 검증 손실은 어느 지점부터 늘어난다. 이것이 과적합이다.")
```

**출력:**

```
검증 손실이 가장 낮은 에포크: 19
그때 학습 손실 0.1074, 검증 손실 0.1914
마지막 에포크    학습 손실 0.0508, 검증 손실 0.2668

학습 손실은 계속 줄지만 검증 손실은 어느 지점부터 늘어난다. 이것이 과적합이다.
```

두 곡선이 벌어지기 시작하는 지점이 과적합이 시작되는 곳이다. 이 그림 하나가 "몇 에포크나 돌려야 하는가"라는 물음에 답을 준다.

---

## 4. 이미지와 가중치 보기

이미지 자료를 다룰 때는 자료 자체를 눈으로 확인하는 일이 특히 중요하다. 축이 뒤바뀌거나 값의 범위가 어긋난 실수를 즉시 잡을 수 있다.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)

# 손글씨 숫자를 흉내 낸 8x8 그림 6장
images = rng.random((6, 8, 8))
labels = rng.integers(0, 10, 6)

fig, axes = plt.subplots(1, 6, figsize=(11, 2))
for ax, img, lab in zip(axes, images, labels):
    ax.imshow(img, cmap="gray")      # 회색조 그림은 cmap='gray'
    ax.set_title(f"{lab}")
    ax.axis("off")                    # 눈금은 그림 볼 때 방해가 된다
plt.tight_layout()

print(f"그림 묶음 모양: {images.shape}   <- (장수, 높이, 너비)")
print(f"값의 범위: [{images.min():.3f}, {images.max():.3f}]")
print(f"이름표: {labels}")
```

**출력:**

```
그림 묶음 모양: (6, 8, 8)   <- (장수, 높이, 너비)
값의 범위: [0.000, 0.997]
이름표: [4 8 0 8 7 3]
```

!!! tip "모양을 늘 확인하라"
    PyTorch는 이미지를 `(묶음, 채널, 높이, 너비)`로 담지만 Matplotlib의 `imshow`는 `(높이, 너비)` 또는 `(높이, 너비, 채널)`을 받는다.
    그래서 텐서를 그리려면 `img.permute(1, 2, 0)`처럼 축 순서를 바꾸어야 할 때가 많다. 그림이 이상하게 나온다면 먼저 모양부터 찍어 본다.

---

## 5. 파일로 저장하기

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(np.arange(10), np.arange(10) ** 2)
ax.set_title("저장 예시")

# dpi를 올리면 더 또렷해진다. 문서에 넣을 그림은 150 이상이 무난하다.
# bbox_inches='tight'는 둘레의 빈 자리를 잘라 낸다.
path = "example_plot.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)                        # 다 쓴 그림은 닫아 메모리를 돌려준다

print(f"저장됨: {path}")
print(f"파일 크기: {os.path.getsize(path)} 바이트")
os.remove(path)                       # 예시이므로 지운다
```

**출력:**

```
저장됨: example_plot.png
파일 크기: 17057 바이트
```

`plt.close(fig)`를 잊으면 그림을 많이 그리는 반복문에서 메모리가 계속 쌓인다. 학습 루프 안에서 에포크마다 그림을 그린다면 반드시 닫아야 한다.

---

## 연습문제

**연습문제 1.**
학습 손실과 검증 손실이 둘 다 높은 채로 평평하다면 무엇을 의심해야 하는가?

??? success "연습문제 1 풀이"
    과적합이 아니라 **과소적합**이다. 모델이 자료를 담아내지 못하고 있다.

    모델을 키우거나, 학습률이 너무 작지는 않은지, 특징을 제대로 만들었는지 살펴야 한다. 손실이 아예 줄지 않는다면 학습률이 너무 커서 발산하거나 이름표가 잘못 붙었을 가능성도 있다.

---

**연습문제 2.**
`imshow`로 그린 그림이 온통 검거나 하얗게 나온다. 무엇을 확인하겠는가?

??? success "연습문제 2 풀이"
    값의 범위를 확인한다. `imshow`는 기본적으로 자료의 최솟값과 최댓값을 검정과 하양에 맞추므로, 값이 거의 일정하면 밋밋하게 나온다.

    또 흔한 실수는 정규화를 마친 자료를 그대로 그리는 것이다. 평균을 빼고 표준편차로 나눈 값은 음수를 포함하므로 원래 범위로 되돌린 뒤 그려야 한다.

---

**연습문제 3.**
그림을 100개 그리는 반복문에서 메모리가 계속 늘어난다. 까닭은?

??? success "연습문제 3 풀이"
    `plt.close()`를 부르지 않아 `Figure` 객체가 쌓이고 있다. Matplotlib은 만든 그림을 안에 붙들고 있으므로 명시적으로 닫아 주어야 한다.

    반복문에서는 `fig.savefig(...)` 뒤에 곧바로 `plt.close(fig)`를 붙이는 것이 안전한 버릇이다.

## 정리하며

**다룬 것** — 결과를 눈으로 확인하는 법

학습이 잘못되어도 프로그램은 오류를 내지 않는다. 그래서 그려 보아야 한다.

`Figure`는 도화지이고 `Axes`는 그 위의 그래프 하나이다. `fig, axes = plt.subplots(...)`로 그릴 곳을 분명히 정하는 버릇이 그림이 여럿일 때 도움이 된다.

가장 값어치 있는 그림은 **학습 곡선**이다. 학습 손실과 검증 손실을 함께 그리면 과적합이 시작되는 지점이 눈에 보이고, 몇 에포크에서 멈출지가 정해진다. 이미지 자료는 모델에 넣기 전에 눈으로 확인하는 것이 좋으며, 이때 축 순서와 값의 범위를 함께 살핀다.

앞의 연습문제 3개로 직접 확인할 수 있다.
