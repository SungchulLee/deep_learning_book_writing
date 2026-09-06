# 퍼짐 모델의 프레셰 인셉션 거리(FID)
FID is the standard metric for evaluating diffusion model sample quality. For the full mathematical derivation, implementation, and best practices, see [FID in §24.6](../../ch25/gan_evaluation/fid.md). This page focuses on diffusion-specific considerations.

## 뜻매김 다시 보기

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

FID이 낮을수록 만든 표본이 Inception-v3 특징 공간에서 실제 자료 분포에 가깝다.

## 퍼짐에 딸린 살핌

### 뽑기 걸음과 FID

퍼짐 모델은 잡음 없애기 걸음 수가 다스리는 저만의 품질과 빠르기의 맞바꿈을 마주한다.

| 뽑개 | 걸음 | 흔한 CIFAR-10 FID |
|---------|-------|---------------------|
| DDPM | 1000 | 약 3.17 |
| DDIM | 50 | 약 4.67 |
| DDIM | 10 | 약 13.36 |
| DPM-Solver++ | 20 | 약 2.80 |

걸음이 많을수록 흔히 FID이 나아지지만 추론 비용이 선형으로 는다. 요즘 풀개(DPM-Solver, DEIS)는 본디 DDPM 뽑개보다 훨씬 적은 걸음으로 좋은 FID을 이룬다.

### 이끌기 잣수와 FID

가름개 없는 이끌기는 다양함을 충실함과 맞바꾸어 특징적인 FID 곡선을 만든다.

| 이끌기 잣수 $w$ | FID | 정밀도 | 재현율 |
|-------------------|-----|-----------|--------|
| 1.0(이끌기 없음) | 더 높음 | 더 낮음 | 더 높음 |
| 2.0–4.0 | **가장 좋음** | 좋음 | 좋음 |
| 7.5(흔한 기본값) | 보통 | 높음 | 더 낮음 |
| 15 이상 | 나빠짐 | 가장 높음 | 낮음 |

가장 좋은 이끌기 잣수는 FID을 가장 작게 하며 표본 품질과 다양함의 가장 좋은 균형을 나타낸다. 그 너머에서는 재현율(다양함)이 정밀도가 나아지는 것보다 빨리 떨어져 FID이 나빠진다.

### 잡음 차례표의 영향

잡음 차례표 $\beta_t$은 배운 점수 함수에, 따라서 표본 품질에 영향을 준다.

- **선형 차례표**: 여느 고르기이며 FID 기준값이 흔히 이를 쓴다
- **코사인 차례표**: 작은 그림에서 흔히 더 나은 FID을 낸다(Nichol와 Dhariwal, 2021)
- **배운 차례표**: 처음부터 끝까지 가장 좋게 하여 FID을 더 줄일 수 있다

### FID-50K 약속

퍼짐 모델을 따지는 여느 방식:

1. 온전한 뽑기 물길로 표본 **50,000**개를 만든다
2. 실제 묶음과 만든 묶음 모두에 Inception-v3 pool3 특징(2048차원)을 셈한다
3. **한결같은 미리 다듬기**를 쓴다. 곧 쌍선형으로 299×299으로 크기를 바꾸고 ImageNet 고르게 맞추기를 한다
4. 뽑개의 정확한 짜임새(걸음, 이끌기 잣수, 잡음 차례표)와 함께 FID을 알린다

!!! warning "미리 다듬기가 중요하다"
    크기 바꾸기의 사이 메우기나 고르게 맞추기가 조금만 달라도 FID이 몇 점 움직일 수 있다. 견주는 기준과 늘 같은 미리 다듬기 물길을 쓰라. `torch-fidelity`과 `clean-fid` 꾸러미가 한결같음을 지키는 데 도움이 된다.

## State-of-the-Art Benchmarks

| Model | CIFAR-10 FID ↓ | ImageNet 256×256 FID ↓ |
|-------|----------------|------------------------|
| DDPM (Ho et al., 2020) | 3.17 | — |
| ADM (Dhariwal & Nichol, 2021) | — | 10.94 |
| ADM + classifier guidance | — | 4.59 |
| ADM + classifier-free guidance | — | 3.94 |
| LDM / Stable Diffusion | — | ~3.60 |
| DiT-XL/2 (Peebles & Xie, 2023) | — | 2.27 |
| Consistency Models (Song et al., 2023) | 2.93 | 3.55 |

## When FID Falls Short for Diffusion

FID captures overall distributional similarity but may not reflect:

- **Text-image alignment** in conditional generation → use CLIP Score instead
- **Fine-grained perceptual quality** → complement with human evaluation
- **Likelihood fit** → use [BPD/NLL](likelihood.md) for models with tractable ELBO

A complete diffusion model evaluation should report FID alongside complementary metrics. See the comprehensive FID treatment in [§24.6](../../ch25/gan_evaluation/fid.md) for implementation details, sample size analysis, and bootstrap confidence intervals.

## 참고 문헌

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
2. Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS*.
3. Peebles, W., & Xie, S. (2023). "Scalable Diffusion Models with Transformers." *ICCV*.
4. Parmar, G., et al. (2022). "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." *CVPR*.

## 연습문제

**연습문제 1.**
Explain why log-likelihood is a useful metric for evaluating diffusion models. What are its limitations?

??? success "연습문제 1 풀이"
    Log-likelihood measures how well the model assigns probability to held-out test data: $\mathcal{L} = \frac{1}{N}\sum_i \log p_\theta(x_i)$. It is useful because: (1) it is a proper scoring rule (maximized by the true distribution), (2) it provides a single number for model comparison, (3) it penalizes both poor sample quality and mode collapse. **Limitations**: (1) models with high likelihood can produce poor samples (e.g., mixtures that assign mass to unrealistic regions), (2) exact computation is often intractable for diffusion models (requires the ELBO or expensive ODE-based evaluation), (3) it does not directly measure perceptual quality.

---

**연습문제 2.**
Compare FID, Inception Score, and log-likelihood as evaluation metrics for generative models.

??? success "연습문제 2 풀이"
    | Metric | Measures | Requires Real Data | Detects Mode Collapse | Perceptual Quality |
    |--------|---------|-------------------|----------------------|-------------------|
    | **FID** | 분포의 닮음 | 그렇다 | 그렇다 | 좋음 |
    | **인셉션 점수** | 품질 + 다양함 | 아니다 | 일부 | 보통 |
    | **로그 가능도** | 밀도의 정확함 | 그렇다(시험 묶음) | 그렇다 | 여림 |

    FID은 사람의 판단과 잘 이어지고 품질과 다양함을 모두 담아 가장 널리 쓰인다. 인셉션 점수는 만든 표본만 따진다. 로그 가능도는 이론으로 원칙이 있지만 느낌의 품질과 어긋날 수 있다. 가장 좋은 방식은 셋 다 알리는 것이다.

---

**연습문제 3.**
차원마다 비트(BPD) 잣대란 무엇인가? 퍼짐 모델에서 어떻게 셈하는가?

??? success "연습문제 3 풀이"
    차원마다 비트는 음의 로그 가능도를 자료 차원으로 고르게 맞추고 비트로 바꾼다. 곧 $\text{BPD} = -\frac{\log_2 p(x)}{d}$이며 $d$은 차원의 수이다(예컨대 CIFAR-10은 $3 \times 32 \times 32 = 3072$). 퍼짐 모델에서 로그 가능도는 증거 하한으로 가둬진다. 곧 $\log p(x) \geq \text{ELBO} = -\sum_t L_t$이며 $L_t$은 쿨백-라이블러 벌어짐 항이다. 정확한 셈은 확률 흐름 상미분 방정식과 순간 변수 바꿈 공식을 쓴다. 차원마다 비트가 낮을수록 좋은 모델이다. 최고 수준의 퍼짐 모델은 CIFAR-10에서 약 2.5을 이룬다.

---

**연습문제 4.**
FID이 뛰어난 만들어 내는 모델이 왜 실제 쓰임새에서 실패할 수 있는가? 어떤 따지기를 더 해야 하는가?

??? success "연습문제 4 풀이"
    FID은 평균 분포 품질을 재지만 다음을 놓친다. (1) **꼬리 움직임**: 드물지만 중요한 잘못됨(흠, 불쾌한 내용)이 평균에 묻힌다. (2) **조건 충실함**: FID을 흔히 조건 없이 셈하는데 갈래나 글 조건 FID은 다를 수 있다. (3) **외우기**: 익히기 자료를 외운 모델은 FID이 낮지만 만들어 내기에는 쓸모없다. (4) **조건 안의 다양함**: 비슷한 채근에 같은 그림을 내도 FID이 낮을 수 있다. 더 따질 것: 정밀도와 재현율 곡선, 갈래마다 FID, 외우기 알아내기(익히기 묶음까지의 가장 가까운 이웃 거리), 품질과 다양함에 대한 사람 따지기, 쓰임새에 맞춘 잣대(예컨대 글에서 그림으로 모델의 글과 그림 맞음).
