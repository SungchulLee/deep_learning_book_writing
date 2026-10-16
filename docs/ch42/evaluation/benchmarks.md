# 평가와 벤치마크

맞섬에 강건성을 제대로 따지는 일은 중요하다. "든든하다"던 방어가 잘못 따진 탓에 무너진 일이 많다. 이 마디는 좋은 버릇, 자주 빠지는 함정, 그리고 **오토어택**과 **RobustBench**을 아우른 잣대 잡힌 따짐 절차를 다룬다.

---

## 1. 문제: 그래디언트 가리기

### 그래디언트 가리기란

**그래디언트 가리기**은 방어가 참 강건성은 주지 못한 채 그래디언트를 감추거나 흐릴 때 생긴다. 모델이 그래디언트 기반의 공격에는 든든해 보이지만 다른 공격에는 여전히 무르다.

### 그래디언트 가리기의 클래스

1. **부서진 그래디언트**: 미분할 수 없는 계산(JPEG 눌러 담기, 수 줄이기)
2. **흔들리는 그래디언트**: 잡음 섞인 그래디언트를 낳는 아무렇게나 하는 방어
3. **사라지거나 터지는 그래디언트**: 다듬기를 가로막는 계산의 탈
4. **흐려 놓은 그래디언트**: 그래디언트 기반의 공격을 헷갈리게 하려고 일부러 꾸민 방어

### 그래디언트 가리기를 짚어내는 길

**붉은 깃발:**

1. **한 스텝 공격이 회귀 공격보다 잘 먹힌다**: FGSM > PGD이면 수상하다
2. **블랙박스 공격이 화이트박스보다 잘 먹힌다**: 전이되는 공격 > 곧바로 공격
3. **마디 없는 공격이 안 먹힌다**: $\varepsilon$을 키워도 더 먹히지 않는다
4. **아무 잡음이 공격과 맞먹는다**: 그래디언트가 쓸모 있는 신호를 못 준다

### 보기: 그래디언트 가리기 짚어내기

```python
def check_gradient_masking(model, x, y, epsilon=8/255):
    """
    그래디언트 가리기의 확률을 살핀다.
    
    진단 결과를 담은 사전을 돌려준다.
    """
    results = {}
    
    # 시험 1: FGSM과 PGD
    fgsm = FGSM(model, epsilon=epsilon)
    pgd = PGD(model, epsilon=epsilon, num_iter=40)
    
    x_fgsm = fgsm.generate(x, y)
    x_pgd = pgd.generate(x, y)
    
    fgsm_success = evaluate_attack_success(model, x_fgsm, y)
    pgd_success = evaluate_attack_success(model, x_pgd, y)
    
    results['fgsm_success'] = fgsm_success
    results['pgd_success'] = pgd_success
    results['fgsm_stronger'] = fgsm_success > pgd_success + 0.05  # 수상하다
    
    # 시험 2: 엡실론 키우기
    success_by_eps = []
    for eps in [0.01, 0.02, 0.04, 0.08, 0.16]:
        attack = PGD(model, epsilon=eps, num_iter=20)
        x_adv = attack.generate(x, y)
        success = evaluate_attack_success(model, x_adv, y)
        success_by_eps.append(success)
    
    results['success_increases'] = all(s1 <= s2 + 0.02 
                                        for s1, s2 in zip(success_by_eps[:-1], success_by_eps[1:]))
    
    # 시험 3: 아무 잡음과 견주기
    noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_random = torch.clamp(x + noise, 0, 1)
    random_success = evaluate_attack_success(model, x_random, y)
    
    results['random_success'] = random_success
    results['random_competitive'] = random_success > 0.5 * pgd_success  # 수상하다
    
    # 간추림
    results['gradient_masking_suspected'] = (
        results['fgsm_stronger'] or 
        not results['success_increases'] or 
        results['random_competitive']
    )
    
    return results
```

---

## 2. AutoAttack

### 두루 보기

**오토어택**(크로체 & 하인, 2020)은 강건성을 미덥게 따지려고 꾸민 파라메터 없는 앙상블 공격이다. 서로 채워 주는 공격 여럿을 아울러 든든하다는 거짓 말을 가장 적게 만든다.

### 이루는 몫

오토어택은 공격 넷으로 이루어진다.

1. **APGD-CE**: 엇갈린 엔트로피 로스를 쓰는 오토-PGD
2. **APGD-DLR**: 로짓 차 비교 로스를 쓰는 오토-PGD
3. **FAB**: 빠르게 맞추어 가는 금 공격
4. **네모**: 블랙박스 스코어 기반 공격

### 오토-PGD(APGD)

APGD은 여느 PGD을 다음으로 낫게 한다.

**맞추어 가는 스텝 크기:**

$$
\alpha^{(t+1)} = \begin{cases}
\alpha^{(t)} / 2 & \text{if loss hasn't improved in } w \text{ steps} \\
\alpha^{(t)} & \text{otherwise}
\end{cases}
$$

**밀어 나감:**

$$
\mathbf{z}^{(t+1)} = \rho \cdot \mathbf{z}^{(t)} + \nabla_\mathbf{x} \mathcal{L}
$$

**찰칵 자리로 되돌리기:** 스텝 크기가 줄면 가장 좋은 점으로 되돌린다.

### DLR 로스

공격에는 **로짓 차 비교** 로스가 엇갈린 엔트로피보다 잘 듣는다.

$$
\mathcal{L}_{\text{DLR}} = -\frac{z_y - \max_{i \neq y} z_i}{z_{\pi_1} - z_{\pi_3}}
$$

여기서 $\pi$은 로짓을 줄 세운 차례다.

### 오토어택 쓰기

```python
# 깔기: pip install autoattack
from autoattack import AutoAttack

# 여느 따짐
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
x_adv = adversary.run_standard_evaluation(x, y, bs=100)

# 더 빠른 따짐(치기 4개 대신 2개)
adversary_fast = AutoAttack(model, norm='Linf', eps=8/255, version='plus')

# 골라 쓰는 치기 배치
adversary_custom = AutoAttack(model, norm='Linf', eps=8/255)
adversary_custom.attacks_to_run = ['apgd-ce', 'apgd-t']  # APGD 치기만
```

### 오토어택 피처와

흔한 따짐 출력은 이렇다.

```
처음 맞음: 87.20%
apgd-ce:         든든한 맞음: 53.40% (- 33.80%)
apgd-t:          든든한 맞음: 51.20% (- 2.20%)
fab-t:           든든한 맞음: 50.80% (- 0.40%)
square:          든든한 맞음: 50.60% (- 0.20%)
```

마지막 강건 정확도(50.60%)은 공격을 모두 이겨 낸 몫이다.

---

## 3. RobustBench

### 두루 보기

**RobustBench**은 적대적 강건성의 잣대 잡힌 잣대 데이터로 다음을 갖췄다.

- CIFAR-10, CIFAR-100, 이미지넷의 가려 뽑은 순위표
- 오토어택으로 미리 따져 둔 모델
- 모델을 쉽게 얹는 API

### RobustBench 쓰기

```python
# 깔기: pip install robustbench
from robustbench import load_model
from robustbench.eval import benchmark

# 미리 익힌 든든한 모형을 얹는다
model = load_model(
    model_name='Carmon2019Unlabeled',  # 순위표의 모형 이름
    dataset='cifar10',
    threat_model='Linf'
)

# 내 모형의 잣대를 잰다
clean_acc, robust_acc = benchmark(
    model,
    dataset='cifar10',
    threat_model='Linf',
    eps=8/255
)

print(f"맑음: {clean_acc:.2%}, 든든함: {robust_acc:.2%}")
```

### 순위표(CIFAR-10, L∞, ε=8/255)

2024년 기준이다.

| 순위 | 모델 | 클린 정확도 | 강건 정확도 |
|------|-------|-----------|------------|
| 1 | Wang2023Better | 93.25% | 70.69% |
| 2 | Cui2023... | 92.16% | 67.73% |
| 3 | Peng2023... | 93.27% | 67.31% |
| ... | ... | ... | ... |
| 밑금 | 매드리(2018) | 87.14% | 44.04% |

---

## 4. 두루 갖춘 따짐 절차

### 스텝1: 기초 살피기

```python
def basic_evaluation(model, test_loader, epsilon=8/255):
    """치기 여럿으로 하는 밑바탕 따짐."""
    results = {}
    
    # 맑은 맞음
    results['clean'] = compute_accuracy(model, test_loader)
    
    # FGSM(빠르고 여림)
    fgsm = FGSM(model, epsilon=epsilon)
    results['fgsm'] = evaluate_robust_accuracy(model, test_loader, fgsm)
    
    # PGD-20(여느 것)
    pgd20 = PGD(model, epsilon=epsilon, num_iter=20)
    results['pgd20'] = evaluate_robust_accuracy(model, test_loader, pgd20)
    
    # PGD-100(셈)
    pgd100 = PGD(model, epsilon=epsilon, num_iter=100)
    results['pgd100'] = evaluate_robust_accuracy(model, test_loader, pgd100)
    
    return results
```

### 스텝2: 그래디언트 가리기 살피기

```python
def gradient_masking_check(model, x, y, epsilon=8/255):
    """그래디언트 가리기 표시를 살핀다."""
    # 위에서 이미 짜 놓았다
    return check_gradient_masking(model, x, y, epsilon)
```

### 스텝3: 오토어택 따짐

```python
def autoattack_evaluation(model, x, y, epsilon=8/255):
    """여느 오토어택 따짐."""
    from autoattack import AutoAttack
    
    adversary = AutoAttack(model, norm='Linf', eps=epsilon)
    x_adv = adversary.run_standard_evaluation(x, y, bs=100)
    
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1)
        robust_acc = (pred == y).float().mean().item()
    
    return robust_acc
```

### 스텝4: 여러 위협 모델

```python
def multi_threat_evaluation(model, test_loader):
    """으름 구조 여럿에 걸쳐 따진다."""
    results = {}
    
    # L∞ 으름 구조
    for eps in [4/255, 8/255, 16/255]:
        pgd = PGD(model, epsilon=eps, norm='linf')
        results[f'linf_eps={eps:.4f}'] = evaluate_robust_accuracy(model, test_loader, pgd)
    
    # L2 으름 구조
    for eps in [0.25, 0.5, 1.0]:
        pgd = PGD(model, epsilon=eps, norm='l2')
        results[f'l2_eps={eps}'] = evaluate_robust_accuracy(model, test_loader, pgd)
    
    return results
```

---

## 5. 따질 때 흔한 잘못

### 1. 여린 공격만 쓰기

❌ **틀림:** FGSM으로만 시험한다
✅ **옳음:** PGD-100 이상이나 오토어택을 쓴다

### 2. 붙박인 하이퍼파라미터

❌ **틀림:** 모든 방어에 PGD의 기본값을 쓴다
✅ **옳음:** 방어에 맞춰 공격의 하이퍼파라미터를 손본다

### 3. 블랙박스를 못 본 척하기

❌ **틀림:** 화이트박스로만 따진다
✅ **옳음:** 전이되는 공격과 물음 기초 공격을 넣는다

### 4. 작은 시험 꾸러미

❌ **틀림:** 보기 100개로 따진다
✅ **옳음:** 온 시험 꾸러미(CIFAR-10이면 1만 개)

### 5. 맞추어 오는 공격 빠뜨리기

❌ **틀림:** 방어 구조에 공격을 맞추지 않는다
✅ **옳음:** 방어의 피처를 헤아린 공격을 꾸민다

---

## 6. 좋은 버릇 간추림

### 막는 쪽에

1. **오토어택을 쓴다**: 잣대 잡히고 미더운 따짐
2. **그래디언트 가리기를 살핀다**: 진단 시험을 돌린다
3. **여러 위협 모델**: L∞, L2, 여러 ε으로 시험한다
4. **밑금과 견준다**: RobustBench 모델과 견준다
5. **믿음 구간을 알린다**: 시험 꾸러미가 작을수록 더욱

### 치는 쪽에

1. **맞추어 오는 공격**: 그 방어에 맞춘 공격을 꾸민다
2. **여러 번 다시 비롯하기**: 아무 첫자리를 쓴다
3. **스텝 크기 맞추기**: 로스 터에 맞춘다
4. **로스 함수 고르기**: 엇갈린 엔트로피, DLR, 틈 로스
5. **그래디언트 어림**: 미분할 수 없는 방어에

---

## 7. 온전한 따짐 흐름

```python
def full_robustness_evaluation(
    model,
    test_loader,
    epsilon=8/255,
    device='cuda'
):
    """
    온전한 든든함 따짐 흐름.
    """
    model.eval()
    model.to(device)
    
    results = {
        'epsilon': epsilon,
        'model': str(model.__class__.__name__)
    }
    
    # 시험 자료를 얻는다
    x_test, y_test = [], []
    for x, y in test_loader:
        x_test.append(x)
        y_test.append(y)
    x_test = torch.cat(x_test).to(device)
    y_test = torch.cat(y_test).to(device)
    
    print("=" * 60)
    print("든든함 따지기")
    print("=" * 60)
    
    # 1. 맑은 맞음
    with torch.no_grad():
        clean_pred = model(x_test).argmax(dim=1)
        clean_acc = (clean_pred == y_test).float().mean().item()
    results['clean_accuracy'] = clean_acc
    print(f"맑은 맞음: {clean_acc:.2%}")
    
    # 2. 그래디언트 가리기 살피기
    print("\n그래디언트 가리기 살피는 중...")
    gm_results = check_gradient_masking(model, x_test[:1000], y_test[:1000], epsilon)
    results['gradient_masking'] = gm_results
    if gm_results['gradient_masking_suspected']:
        print("⚠️  알림: 그래디언트 가리기가 의심된다!")
    else:
        print("✓ 그래디언트 가리기는 보이지 않는다")
    
    # 3. PGD 따짐
    print("\nPGD 따지는 중...")
    for num_iter in [20, 100]:
        pgd = PGD(model, epsilon=epsilon, num_iter=num_iter, device=device)
        x_adv = pgd.generate(x_test, y_test)
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(dim=1)
            robust_acc = (adv_pred == y_test).float().mean().item()
        results[f'pgd{num_iter}'] = robust_acc
        print(f"  PGD-{num_iter}: {robust_acc:.2%}")
    
    # 4. 오토어택(빠르게 하려고 일부만)
    print("\n오토어택 따지는 중(표본 1000개)...")
    try:
        from autoattack import AutoAttack
        adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
        x_adv_aa = adversary.run_standard_evaluation(
            x_test[:1000], y_test[:1000], bs=100
        )
        with torch.no_grad():
            aa_pred = model(x_adv_aa).argmax(dim=1)
            aa_acc = (aa_pred == y_test[:1000]).float().mean().item()
        results['autoattack'] = aa_acc
        print(f"  오토어택: {aa_acc:.2%}")
    except ImportError:
        print("  오토어택이 깔려 있지 않아 건너뛴다...")
    
    # 5. 간추림
    print("\n" + "=" * 60)
    print("간추림")
    print("=" * 60)
    print(f"맑은 맞음:     {results['clean_accuracy']:.2%}")
    print(f"PGD-20 맞음:    {results.get('pgd20', 'N/A'):.2%}")
    print(f"PGD-100 맞음:   {results.get('pgd100', 'N/A'):.2%}")
    print(f"오토어택:         {results.get('autoattack', 'N/A'):.2%}")
    print("=" * 60)
    
    return results
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
선형 분류기 $f(x) = w^T x + b$에서 예측한 클래스를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 섭동을 계산하여라. 이것이 뉴럴 넷트웍의 강건성과 어떻게 이어지는지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    선형 분류기에서 $\ell_\infty$ 노름으로 잰 결정 경계까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 섭동은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 뉴럴 넷트웍에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(그래디언트의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차원이 높은 모델이 무른 까닭은 $\|w\|_1$은 차원과 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 강건성의 여유가 줄어들기 때문이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 마디에서 다룬 공격나 방어를 CIFAR-10의 ResNet-18 모델에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 공격 아래에서 클린 정확도와 강건 정확도를 알려라.

</div>

??? success "연습문제 2 풀이"
    여느 ResNet-18은 클린 정확도가 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 스텝 크기 $2/255$) 아래의 강건 정확도는 $\sim$0%이다. 이 마디의 방법을 걸면 피처와는 재주에 따라 다르다. 적대적 학습은 클린 정확도 $\sim$83%에 강건 정확도 $\sim$50%이고, 인증 가능한 방어는 더 낮지만 증명할 수 있는 테두리를 준다. 정확도와 강건성의 트레이드오프는 기초부터 있는 것이라, 강건성을 높이면 클린 정확도가 흔히 5~15% 든다. 아무렇게나 하는 시드 3개의 평균과 표준편차로 알려라. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
섭동 공 안에서 클래스별 데이터의 밑자리가 서로 겹친다고 볼 때, 모델이 담는 힘을 키우지 않고서는 어떤 방어도 맑은 데이터의 높은 정확도와 $\ell_\infty$ 섭동에 대한 높은 강건성을 함께 이룰 수 없음을 증명하여라.

</div>

??? success "연습문제 3 풀이"
    두 클래스의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{클래스 1}, x_2 \in \text{클래스 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 강건한 분류기는 적어도 하나를 틀리게 가를 수밖에 없다(섭동 공이 겹치기 때문이다). 이것이 정확도와 강건성의 기초 트레이드오프다. 겹치는 밑자리의 몫이 피할 수 없는 정확도 로스를 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 정확도 떨어짐을 밝혀 준다. 모델이 담는 힘을 키우면(더 너른 네트워크) 얽힌 강건한 결정 경계를 더 잘 그려 트레이드오프를 얼마쯤 눅일 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 머신러닝 구조(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 강건성이 어떻게 드러나는지 다루어라. 위협 모델이 보기 다룸과 어떻게 다른가?

</div>

??? success "연습문제 4 풀이"
    금융에서 겨루는 이는 알아내는 구조에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 핵심은 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 공격이 잇따르며 맞추어 간다(겨루는 이가 구조의 되받음을 보고 손본다). (3) 헛 정확도와 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 섭동 모델이 있어야 한다. 방어는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 기초의 길은 많이 걸러진다. $\square$

## 정리하며

이 절에서는 문제: 그래디언트 가리기, 오토어택, RobustBench, 두루 갖춘 따짐 절차를 살펴보았다.

**살펴볼 거리**

1. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
2. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security." ICML.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv.
4. RobustBench: https://robustbench.github.io/
