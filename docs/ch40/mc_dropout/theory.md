# 몬테카를로 드롭아웃의 이론 밑바탕

---

## 1. 두루 보기

몬테카를로(MC) 드롭아웃은 드롭아웃을 어림 변이 미루어 봄으로 이치에 닿게 풀이한다. 이 글은 그 이론 밑바탕을 엄밀히 펼쳐, 드롭아웃 익힘과 신경 그물 짐에 대한 뒷분포 미루어 봄 사이의 이어짐을 세운다.

---

## 2. 변이 미루어 봄 틀

### 베이즈 신경 그물 문제

짐이 $\omega$이고 자료 꾸러미가 $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$인 신경 그물을 생각하자. 베이즈 길은 뒷분포를 찾는다.

$$
p(\omega | \mathcal{D}) = \frac{p(\mathcal{D} | \omega) p(\omega)}{p(\mathcal{D})}
$$

여기서

- $p(\omega)$은 짐의 앞선 분포
- $p(\mathcal{D} | \omega) = \prod_{i=1}^N p(\mathbf{y}_i | \mathbf{x}_i, \omega)$은 그럴듯함
- $p(\mathcal{D}) = \int p(\mathcal{D} | \omega) p(\omega) \, d\omega$은 가장자리 그럴듯함(밑거리)

밑거리 적분은 신경 그물에서 차수가 높고 곧지 않아 다룰 수 없다. 변이 미루어 봄은 참 뒷분포를 다룰 수 있는 분포로 어림해 이를 풀어낸다.

### 변이 어림

$\theta$으로 매개변수를 잡은 다룰 수 있는 갈래에서 어림 분포 $q_\theta(\omega)$을 찾는다. 목표는 쿨백-라이블러(KL) 갈림을 가장 작게 하는 것이다.

$$
\text{KL}(q_\theta(\omega) \| p(\omega | \mathcal{D})) = \int q_\theta(\omega) \log \frac{q_\theta(\omega)}{p(\omega | \mathcal{D})} \, d\omega
$$

**KL 갈림 펼치기**:

$$
\begin{aligned}
\text{KL}(q_\theta \| p(\cdot | \mathcal{D})) &= \int q_\theta(\omega) \log q_\theta(\omega) \, d\omega - \int q_\theta(\omega) \log p(\omega | \mathcal{D}) \, d\omega \\
&= \int q_\theta(\omega) \log q_\theta(\omega) \, d\omega - \int q_\theta(\omega) \log \frac{p(\mathcal{D} | \omega) p(\omega)}{p(\mathcal{D})} \, d\omega \\
&= \int q_\theta(\omega) \log q_\theta(\omega) \, d\omega - \int q_\theta(\omega) \log p(\mathcal{D} | \omega) \, d\omega \\
&\quad - \int q_\theta(\omega) \log p(\omega) \, d\omega + \log p(\mathcal{D})
\end{aligned}
$$

로그 밑거리에 대해 다시 쓰면

$$
\log p(\mathcal{D}) = \text{KL}(q_\theta \| p(\cdot | \mathcal{D})) + \mathcal{L}(\theta)
$$

여기서 **밑거리 아래끝(ELBO)**은

$$
\mathcal{L}(\theta) = \mathbb{E}_{q_\theta(\omega)}[\log p(\mathcal{D} | \omega)] - \text{KL}(q_\theta(\omega) \| p(\omega))
$$

$\text{KL} \geq 0$이므로 $\log p(\mathcal{D}) \geq \mathcal{L}(\theta)$이다. ELBO를 가장 크게 하는 일은 참 뒷분포에 대한 KL 갈림을 가장 작게 하는 일과 같다.

### ELBO 쪼개기

ELBO에는 서로 겨루는 두 항이 있다.

1. **자료 맞춤 항**: $\mathbb{E}_{q_\theta(\omega)}[\log p(\mathcal{D} | \omega)]$ — 그럴듯함이 큰 자리에 $q_\theta$이 무게를 두게 이끈다
2. **번거로움 벌**: $\text{KL}(q_\theta(\omega) \| p(\omega))$ — $q_\theta$이 앞선 분포에 가까이 머물게 이끈다

이는 베이즈 오컴의 면도날을 절로 이룬다.

---

## 3. 변이 미루어 봄으로서의 드롭아웃

### 드롭아웃 변이 갈래

갈과 가라마니(2016)는 드롭아웃이 넌지시 변이 분포를 매김을 밝혔다. 짐 행렬이 $\mathbf{W} \in \mathbb{R}^{K \times Q}$인 켜 하나를 생각하자. 여기서 $K$은 들임 차수, $Q$은 날임 차수다.

**변이 분포 매기기**:

$\mathbf{W}$의 줄 $\mathbf{w}_k$마다($k$번째 들임 낱자리에 이어짐)

$$
q(\mathbf{w}_k) = p \cdot \delta_{\mathbf{0}}(\mathbf{w}_k) + (1-p) \cdot \delta_{\mathbf{m}_k}(\mathbf{w}_k)
$$

여기서

- $p$은 드롭아웃 낌새
- $\delta_{\mathbf{a}}$은 $\mathbf{a}$에 놓인 점 무게(디랙 델타)
- $\mathbf{m}_k \in \mathbb{R}^Q$은 변이 매개변수("평균" 짐 줄)

이는 두 값 가리개 $z_k \sim \text{Bernoulli}(1-p)$으로 이렇게 적을 수 있다.

$$
\mathbf{w}_k = z_k \cdot \mathbf{m}_k
$$

**온 짐 행렬에 대해서는**:

$$
\mathbf{W} = \text{diag}(\mathbf{z}) \cdot \mathbf{M}
$$

여기서 $\mathbf{z} \in \{0, 1\}^K$은 가리개 벡터이고 $\mathbf{M} \in \mathbb{R}^{K \times Q}$은 변이 매개변수를 담는다.

### 깊은 그물로 넓히기

켜가 $L$개인 깊은 그물에서 짐은 $\omega = \{\mathbf{W}_1, \ldots, \mathbf{W}_L\}$이다. 변이 분포는 곱으로 갈라진다.

$$
q_\theta(\omega) = \prod_{\ell=1}^{L} q(\mathbf{W}_\ell)
$$

여기서 $\theta = \{\mathbf{M}_1, \ldots, \mathbf{M}_L\}$은 변이 매개변수(우리가 참으로 배우는 짐 행렬)다.

$q_\theta(\omega)$에서 뽑은 표본 하나는 다음에 맞물린다.

$$
\mathbf{W}_\ell = \text{diag}(\mathbf{z}_\ell) \cdot \mathbf{M}_\ell, \quad \mathbf{z}_\ell \sim \text{Bernoulli}(1-p)^{K_\ell}
$$

이것이 바로 익히는 동안 드롭아웃이 하는 일이다.

### 목표 함수

**정리(갈 & 가라마니, 2016):** 드롭아웃 익힘 목표

$$
\mathcal{L}_{\text{dropout}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{E}_{\mathbf{z}} \left[ \mathcal{L}(f_{\mathbf{z}}(\mathbf{x}_i; \theta), \mathbf{y}_i) \right] + \lambda \sum_{\ell=1}^{L} \|\mathbf{M}_\ell\|_F^2
$$

을 가장 작게 하는 일은 어떤 앞선 분포를 지닌 ELBO를 가장 크게 하는 일과 같다.

**증명 얼거리**:

1. **그럴듯함 항**: 가우스 잡음을 지닌 되돌이 $p(\mathbf{y} | f(\mathbf{x}), \sigma^2) = \mathcal{N}(\mathbf{y}; f(\mathbf{x}), \sigma^2 \mathbf{I})$에서

$$
\log p(\mathcal{D} | \omega) = -\frac{1}{2\sigma^2} \sum_{i=1}^{N} \|\mathbf{y}_i - f(\mathbf{x}_i; \omega)\|^2 + \text{const}
$$

2. **앞선 분포**: 가우스 앞선 분포 $p(\omega) = \prod_\ell \mathcal{N}(\text{vec}(\mathbf{W}_\ell); \mathbf{0}, \sigma_p^2 \mathbf{I})$에서

$$
\log p(\omega) = -\frac{1}{2\sigma_p^2} \sum_{\ell} \|\mathbf{W}_\ell\|_F^2 + \text{const}
$$

3. **KL 항**: 드롭아웃 변이 갈래에서는

$$
\text{KL}(q_\theta(\omega) \| p(\omega)) \propto \sum_\ell \|\mathbf{M}_\ell\|_F^2
$$

(정확한 꼴에는 베르누이의 엔트로피가 들지만 $\theta$에 대해서는 붙박이다.)

4. **모으기**: ELBO는 이렇게 된다.

$$
\mathcal{L}(\theta) = -\frac{1}{2\sigma^2} \mathbb{E}_{q_\theta} \left[ \sum_i \|\mathbf{y}_i - f(\mathbf{x}_i; \omega)\|^2 \right] - \frac{1}{2\sigma_p^2} \sum_\ell \|\mathbf{M}_\ell\|_F^2
$$

$\lambda = \frac{\sigma^2}{N \sigma_p^2}$으로 두면 드롭아웃 목표가 되살아난다. $\square$

### 앞선 분포 정하기

짐 줄이기와 앞선 분포의 맞물림은 이렇다.

$$
\lambda = \frac{p \ell^2}{2N \tau}
$$

여기서

- $p$은 드롭아웃 낌새
- $\ell^2$은 앞선 분포의 길이 잣대($\sigma_p^2$에 이어짐)
- $N$은 자료 꾸러미 크기
- $\tau$은 모형의 촘촘함(살핌 잡음의 거꿀 $1/\sigma^2$)

이는 드롭아웃 비율과 앞선 믿음이 주어졌을 때 짐 줄이기를 이치에 닿게 정하는 길을 준다.

---

## 4. 미루어 보는 분포

### 뒷분포 미루어 봄

새 들임 $\mathbf{x}^*$에 대한 베이즈 미루어 보는 분포는

$$
p(\mathbf{y}^* | \mathbf{x}^*, \mathcal{D}) = \int p(\mathbf{y}^* | \mathbf{x}^*, \omega) p(\omega | \mathcal{D}) \, d\omega
$$

변이 어림 $q_\theta(\omega) \approx p(\omega | \mathcal{D})$을 쓰면

$$
p(\mathbf{y}^* | \mathbf{x}^*, \mathcal{D}) \approx \int p(\mathbf{y}^* | \mathbf{x}^*, \omega) q_\theta(\omega) \, d\omega = \mathbb{E}_{q_\theta(\omega)} [p(\mathbf{y}^* | \mathbf{x}^*, \omega)]
$$

### 몬테카를로 어림

$q_\theta(\omega)$에 대한 바람은 몬테카를로 표본 뽑기로 어림한다.

$$
\mathbb{E}_{q_\theta(\omega)} [f(\mathbf{x}^*; \omega)] \approx \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}^*; \hat{\omega}_t)
$$

여기서 $\hat{\omega}_t \sim q_\theta(\omega)$은 드롭아웃 가리개를 뽑는 일에 맞물린다.

**미루어 본 평균(되돌이)**:

$$
\mathbb{E}[\mathbf{y}^* | \mathbf{x}^*, \mathcal{D}] \approx \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}^*; \hat{\omega}_t)
$$

**미루어 본 흩어짐**:

온 흩어짐 법칙을 쓰면

$$
\text{Var}[\mathbf{y}^* | \mathbf{x}^*, \mathcal{D}] = \underbrace{\mathbb{E}_{q_\theta}[\text{Var}[\mathbf{y}^* | \mathbf{x}^*, \omega]]}_{\text{타고난}} + \underbrace{\text{Var}_{q_\theta}[\mathbb{E}[\mathbf{y}^* | \mathbf{x}^*, \omega]]}_{\text{앎의}}
$$

MC 어림은 다음을 준다.

$$
\text{Var}[\mathbf{y}^* | \mathbf{x}^*, \mathcal{D}] \approx \sigma^2 \mathbf{I} + \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}^*; \hat{\omega}_t) f(\mathbf{x}^*; \hat{\omega}_t)^\top - \bar{f}(\mathbf{x}^*) \bar{f}(\mathbf{x}^*)^\top
$$

여기서 $\bar{f}(\mathbf{x}^*) = \frac{1}{T} \sum_t f(\mathbf{x}^*; \hat{\omega}_t)$이다.

---

## 5. 가름으로 넓히기

### 소프트맥스 그럴듯함

소프트맥스 날임을 쓰는 $C$갈래 가름에서

$$
p(\mathbf{y} = c | \mathbf{x}, \omega) = \text{softmax}(f(\mathbf{x}; \omega))_c = \frac{\exp(f_c(\mathbf{x}; \omega))}{\sum_{c'} \exp(f_{c'}(\mathbf{x}; \omega))}
$$

미루어 보는 분포는

$$
p(\mathbf{y}^* = c | \mathbf{x}^*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^{T} \text{softmax}(f(\mathbf{x}^*; \hat{\omega}_t))_c
$$

**종요로움:** 로짓이 아니라 소프트맥스 날임을 평균한다.

$$
\bar{p}_c = \frac{1}{T} \sum_{t=1}^{T} p_c^{(t)} \quad \text{where } p_c^{(t)} = \text{softmax}(f(\mathbf{x}^*; \hat{\omega}_t))_c
$$

### 가름에서의 아리송함 재기

**미루어 본 엔트로피**은 온 아리송함을 잰다.

$$
\mathbb{H}[\mathbf{y}^* | \mathbf{x}^*, \mathcal{D}] = -\sum_{c=1}^{C} \bar{p}_c \log \bar{p}_c
$$

**서로 나눈 소식**(앎의 아리송함):

$$
\mathbb{I}[\mathbf{y}^*, \omega | \mathbf{x}^*, \mathcal{D}] = \mathbb{H}[\mathbf{y}^* | \mathbf{x}^*, \mathcal{D}] - \mathbb{E}_{q_\theta(\omega)}[\mathbb{H}[\mathbf{y}^* | \mathbf{x}^*, \omega]]
$$

MC 어림:

$$
\mathbb{I}[\mathbf{y}^*, \omega | \mathbf{x}^*, \mathcal{D}] \approx -\sum_c \bar{p}_c \log \bar{p}_c + \frac{1}{T} \sum_{t=1}^{T} \sum_c p_c^{(t)} \log p_c^{(t)}
$$

---

## 6. 이론의 한계

### 어림의 됨됨이

1. **평균 마당 가정:** 변이 분포는 켜와 낱자리에 걸친 남남임을 가정하므로 뒷분포의 함께 바뀜을 놓친다.

2. **점 무게 섞음:** 드롭아웃 변이 갈래는 이어지는 분포가 아니라 점 무게를 쓰므로 얽힌 뒷분포에는 어림이 나쁠 수 있다.

3. **붙박인 드롭아웃 비율:** 여느 MC 드롭아웃은 붙박인 $p$을 쓰지만, 가장 좋은 비율은 켜마다 다르거나 자료에 매일 수 있다.

### 참으로 걸리는 것

1. **낮게 본 아리송함:** MC 드롭아웃은 온전한 베이즈 방법(MCMC, HMC)보다 아리송함을 낮게 보는 일이 잦다.

2. **눈금 맞음 문제:** 미루어 본 낌새는 재주를 더하지 않으면(온도 잣대 잡기 따위) 눈금이 잘 맞지 않을 수 있다.

3. **앞선 분포 예민함:** 넌지시 세워진 앞선 분포는 그물 얼개와 하이퍼파라미터에 매이므로 밭의 앎과 맞지 않을 수 있다.

---

## 7. 다른 방법과의 이어짐

### 가우스 드롭아웃

베르누이 대신 곱하는 가우스 잡음을 쓰면

$$
\mathbf{w}_k = \mathbf{m}_k \odot \boldsymbol{\epsilon}_k, \quad \boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{1}, \alpha \mathbf{I})
$$

이는 이어지는 밑자리를 지닌 다른 변이 갈래에 맞물린다. 흩어짐 $\alpha$은 $\alpha = \frac{p}{1-p}$으로 드롭아웃 비율에 이어진다.

### 변이 드롭아웃

킹마 등(2015)은 짐마다 드롭아웃 비율 $p$(또는 그와 같은 $\alpha$)을 배우자고 내놓았다. 이는 배운 $\alpha \to \infty$일 때 짐이 아주 "떨어져" 나가므로 걸림새를 절로 가려낸다.

### 깊은 모둠

엄밀히 베이즈는 아니지만, 깊은 모둠(첫값을 달리해 그물 여럿을 익히기)은 참으로 MC 드롭아웃보다 나은 아리송함 어림을 주는 일이 잦다. MC 드롭아웃은 모둠을 셈이 더 싸게 어림한 것으로 볼 수 있다.

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

이 마당은 두루 보기、변이 미루어 봄 틀、변이 미루어 봄으로서의 드롭아웃、미루어 보는 분포을 차례로 짚었다.

**살펴볼 거리**

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML*.

2. Kingma, D. P., Salimans, T., & Welling, M. (2015). Variational Dropout and the Local Reparameterization Trick. *NeurIPS*.

3. Blundell, C., et al. (2015). Weight Uncertainty in Neural Networks. *ICML*.

4. Gal, Y. (2016). Uncertainty in Deep Learning. *PhD Thesis, University of Cambridge*.
