# 입력 변환 기반 방어

**입력 바꾸어 막기**은 가르기에 앞서 입력을 미리 다뤄 적대적 섭동을 없애거나 줄인다. 디노이징, 눌러 담기, 복원 같은 기법으로 입력을 고쳐, 클린 분류의 정확도는 지키면서 적대적 샘플을 "맑히려" 한다.

---

## 1. 흔한 바꿈

### JPEG 눌러 담기

JPEG 눌러 담기는 DCT 분야에서 고주파 기여도를 없애는데, 거기에 적대적 섭동이 실려 있는 일이 잦다.

$$
\mathbf{x}_{\text{clean}} \approx \text{JPEG}(\mathbf{x}_{\text{adv}}, q)
$$

여기서 $q$은 성능 값이다. 성능이 낮을수록 섭동을 더 없애지만 신호도 더 잃는다.

### 무작위로 크기 바꾸고 덧대기

셰 등(2018)은 가르기에 앞서 입력의 크기를 무작위로 바꾸고 덧대자고 내놓았다. 그 무작위로임이 그래디언트를 흔들어 그래디언트 기반의 공격을 가로막는다.

```python
import torch
import torch.nn.functional as F

def random_resize_pad(x, target_size=32, resize_range=(28, 36)):
    """막으려고 들임을 아무렇게나 크기 바꾸고 덧댄다."""
    rnd_size = torch.randint(resize_range[0], resize_range[1], (1,)).item()
    
    # 크기를 바꾼다
    x_resized = F.interpolate(
        x, size=(rnd_size, rnd_size), mode='bilinear', align_corners=False
    )
    
    # 타깃 크기가 되도록 아무렇게나 덧댄다
    pad_top = torch.randint(0, target_size - rnd_size + 1, (1,)).item()
    pad_left = torch.randint(0, target_size - rnd_size + 1, (1,)).item()
    pad_bottom = target_size - rnd_size - pad_top
    pad_right = target_size - rnd_size - pad_left
    
    return F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom))
```

### 잡음 지우는 오토인코더

적대적 입력에서 클린 입력을 복원하도록 잡음 지우는 오토인코더를 학습한다.

$$
\hat{\mathbf{x}} = D_\phi(\mathbf{x}_{\text{adv}}) \approx \mathbf{x}
$$

잡음 지우개는 (적대적 샘플, 원본) 짝으로 학습한다. 그다음 분류기는 잡음을 지운 입력으로 움직인다.

### 확산에 기댄 맑히기

요즘 일은 **확산 모델**로 잡음을 더했다가 지워 적대적 샘플을 맑힌다.

1. 가우스 잡음을 더한다: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_{\text{adv}} + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$
2. 확산 모델로 잡음을 지운다: $\hat{\mathbf{x}}_0 = \text{denoise}(\mathbf{x}_t, t)$
3. 가른다: $\hat{y} = f(\hat{\mathbf{x}}_0)$

잡음을 섞으면 적대적 구조가 무너지고, 확산 모델이 클린 입력에 가깝게 복원한다.

---

## 2. 한계

### 그래디언트 가리기 문제

입력 바꾸어 막기는 참 강건성이 아니라 **그래디언트 가리기**에 그치는 일이 잦다.

- 변환이 미분할 수 없어 그래디언트 기반의 공격을 막을 수 있다
- 그러나 다음에는 여전히 무를 수 있다.
    - 되돌아 스텝을 미분할 수 있게 근사하기(BPDA)
    - 방어가 없는 모델에서 전이되는 공격
    - 변환에 대한 기댓값(EOT) 공격

### 정확도가 떨어짐

적대적 섭동을 없애는 변환은 클린 입력도 망가뜨려 클린 정확도를 떨어뜨린다. 섭동을 없애는 일과 소식을 지키는 일 사이의 트레이드오프는 타고난 것이다.

---

## 3. 좋은 버릇

1. **늘 맞추어 오는 공격으로 따져라**: BPDA이나 EOT으로 변환을 뚫고 쳐 본다
2. **다른 방어와 아울러라**: 변환을 적대적 학습의 대체가 아니라 보완으로 쓴다
3. **클린 정확도를 지켜보라**: 변환이 클린 입력의 성능을 받아들일 수 없을 만큼 떨어뜨리지 않게 한다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 네트워크의 강건성과 어떻게 이어지는지 설명하여라.

</div>

??? success "풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 네트워크에서는 국소 선형 근사 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 통하는지를 설명해 준다. 차원이 높은 모델이 취약한 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 절에서 다룬 공격이나 방어를 CIFAR-10의 ResNet-18 모델에 구현하여라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 보고하여라.

</div>

??? success "풀이"
    일반 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 절의 방법을 적용한 결과는 기법에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 바운드를 준다. 정확도와 강건성의 트레이드오프는 근본적인 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 떨어진다. 무작위 시드 3개의 평균과 표준편차로 보고하여라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 서포트가 서로 겹친다고 볼 때, 모델이 용량을 키우지 않고서는 어떤 방어도 클린 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "풀이"
    두 클래스의 서포트가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 서포트의 비중이 피할 수 없는 정확도 손실을 정한다. 자연 이미지 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 관찰되는 10~15%의 정확도 하락을 설명해 준다. 모델 용량을 키우면(더 넓은 네트워크) 복잡한 강건 결정 경계를 더 잘 그려 트레이드오프를 어느 정도 완화할 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 시스템(사기 탐지나 거래 신호 생성 등)에서 적대적 강건성이 어떻게 드러나는지 논하여라. 위협 모델이 비전과 어떻게 다른가?

</div>

??? success "풀이"
    금융에서 적대자는 탐지 시스템에 맞추어 전략을 바꾸는 교묘한 집단(사기꾼, 시장 조작자)이다. 비전과 다른 핵심은 이렇다. (1) 섭동할 수 있는 피처가 경제적으로 그럴듯한 범위로 제약된다(사기꾼이 자기 거래 이력을 통째로 바꿀 수는 없다). (2) 공격이 순차적이고 적응적이다(적대자가 시스템의 응답을 보고 고친다). (3) 거짓 양성과 거짓 음성의 비용이 비대칭이다(정상 거래를 막는 것과 사기를 놓치는 것). (4) $\ell_p$ 노름은 의미가 없고 도메인에 맞는 섭동 모델이 있어야 한다. 방어는 적응형 적대자에게도 강건해야 하므로, 탐지 기준이 알려지면 회피할 수 있는 탐지 기반 방법은 쓸모가 크게 줄어든다. $\square$

## 정리하며

입력 바꾸어 막기는 알기 쉽고 짜기도 쉽지만 그래디언트 가리기 문제를 안고 있다. 여러 레이어 방어 전략의 한 기여도일 때 가장 잘 들며, 맞추어 오는 적대자에 대해 늘 따져 봐야 한다.

**살펴볼 거리**

1. Xie, C., et al. (2018). "Mitigating Adversarial Effects Through Randomization." ICLR.
2. Nie, W., et al. (2022). "Diffusion Models for Adversarial Purification." ICML.
3. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples." ICML.
