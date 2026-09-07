# 따지기와 잣대 재기
## 들머리

맞섬에 든든함을 제대로 따지는 일은 종요롭다. "든든하다"던 막이가 잘못 따진 탓에 무너진 일이 많다. 이 마디는 좋은 버릇, 자주 빠지는 함정, 그리고 **오토어택**과 **RobustBench**을 아우른 잣대 잡힌 따짐 절차를 다룬다.

## 문제: 기울기 가리기

### 기울기 가리기란

**기울기 가리기**은 막이가 참 든든함은 주지 못한 채 기울기를 감추거나 흐릴 때 생긴다. 모형이 기울기 바탕의 치기에는 든든해 보이지만 다른 치기에는 여전히 무르다.

### 기울기 가리기의 갈래

1. **부서진 기울기**: 미분할 수 없는 셈(JPEG 눌러 담기, 수 줄이기)
2. **흔들리는 기울기**: 잡음 섞인 기울기를 낳는 아무렇게나 하는 막이
3. **사라지거나 터지는 기울기**: 다듬기를 가로막는 셈의 탈
4. **흐려 놓은 기울기**: 기울기 바탕의 치기를 헷갈리게 하려고 일부러 꾸민 막이

### 기울기 가리기를 짚어내는 길

**붉은 깃발:**

1. **한 걸음 치기가 되돌이 치기보다 잘 먹힌다**: FGSM > PGD이면 수상하다
2. **검은 상자 치기가 흰 상자보다 잘 먹힌다**: 옮아가는 치기 > 곧바로 치기
3. **마디 없는 치기가 안 먹힌다**: $\varepsilon$을 키워도 더 먹히지 않는다
4. **아무 잡음이 치기와 맞먹는다**: 기울기가 쓸모 있는 신호를 못 준다

### 보기: 기울기 가리기 짚어내기

```python
def check_gradient_masking(model, x, y, epsilon=8/255):
    """
    기울기 가리기의 낌새를 살핀다.
    
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

## 오토어택

### 두루 보기

**오토어택**(크로체 & 하인, 2020)은 든든함을 미덥게 따지려고 꾸민 매개변수 없는 모둠 치기다. 서로 채워 주는 치기 여럿을 아울러 든든하다는 거짓 말을 가장 적게 만든다.

### 이루는 몫

오토어택은 치기 넷으로 이루어진다.

1. **APGD-CE**: 엇갈린 엔트로피 잃음을 쓰는 오토-PGD
2. **APGD-DLR**: 로짓 차 견줌 잃음을 쓰는 오토-PGD
3. **FAB**: 빠르게 맞추어 가는 금 치기
4. **네모**: 검은 상자 점수 바탕 치기

### 오토-PGD(APGD)

APGD은 여느 PGD을 다음으로 낫게 한다.

**맞추어 가는 걸음 크기:**

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

**찰칵 자리로 되돌리기:** 걸음 크기가 줄면 가장 좋은 점으로 되돌린다.

### DLR 잃음

치기에는 **로짓 차 견줌** 잃음이 엇갈린 엔트로피보다 잘 듣는다.

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

# 골라 쓰는 치기 묶음
adversary_custom = AutoAttack(model, norm='Linf', eps=8/255)
adversary_custom.attacks_to_run = ['apgd-ce', 'apgd-t']  # APGD 치기만
```

### 오토어택 결과

흔한 따짐 날임은 이렇다.

```
처음 맞음: 87.20%
apgd-ce:         든든한 맞음: 53.40% (- 33.80%)
apgd-t:          든든한 맞음: 51.20% (- 2.20%)
fab-t:           든든한 맞음: 50.80% (- 0.40%)
square:          든든한 맞음: 50.60% (- 0.20%)
```

마지막 든든한 맞음(50.60%)은 치기를 모두 이겨 낸 몫이다.

## RobustBench

### 두루 보기

**RobustBench**은 맞섬에 든든하기의 잣대 잡힌 잣대 자료로 다음을 갖췄다.

- CIFAR-10, CIFAR-100, 이미지넷의 가려 뽑은 순위표
- 오토어택으로 미리 따져 둔 모형
- 모형을 쉽게 얹는 API

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

| 순위 | 모형 | 맑은 맞음 | 든든한 맞음 |
|------|-------|-----------|------------|
| 1 | Wang2023Better | 93.25% | 70.69% |
| 2 | Cui2023... | 92.16% | 67.73% |
| 3 | Peng2023... | 93.27% | 67.31% |
| ... | ... | ... | ... |
| 밑금 | 매드리(2018) | 87.14% | 44.04% |

## 두루 갖춘 따짐 절차

### 1걸음: 밑바탕 살피기

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

### 2걸음: 기울기 가리기 살피기

```python
def gradient_masking_check(model, x, y, epsilon=8/255):
    """기울기 가리기 표시를 살핀다."""
    # 위에서 이미 짜 놓았다
    return check_gradient_masking(model, x, y, epsilon)
```

### 3걸음: 오토어택 따짐

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

### 4걸음: 여러 으름 얼개

```python
def multi_threat_evaluation(model, test_loader):
    """으름 얼개 여럿에 걸쳐 따진다."""
    results = {}
    
    # L∞ 으름 얼개
    for eps in [4/255, 8/255, 16/255]:
        pgd = PGD(model, epsilon=eps, norm='linf')
        results[f'linf_eps={eps:.4f}'] = evaluate_robust_accuracy(model, test_loader, pgd)
    
    # L2 으름 얼개
    for eps in [0.25, 0.5, 1.0]:
        pgd = PGD(model, epsilon=eps, norm='l2')
        results[f'l2_eps={eps}'] = evaluate_robust_accuracy(model, test_loader, pgd)
    
    return results
```

## 따질 때 흔한 잘못

### 1. 여린 치기만 쓰기

❌ **틀림:** FGSM으로만 시험한다
✅ **옳음:** PGD-100 이상이나 오토어택을 쓴다

### 2. 붙박인 하이퍼파라미터

❌ **틀림:** 모든 막이에 PGD의 기본값을 쓴다
✅ **옳음:** 막이에 맞춰 치기의 하이퍼파라미터를 손본다

### 3. 검은 상자를 못 본 척하기

❌ **틀림:** 흰 상자로만 따진다
✅ **옳음:** 옮아가는 치기와 물음 바탕 치기를 넣는다

### 4. 작은 시험 꾸러미

❌ **틀림:** 보기 100개로 따진다
✅ **옳음:** 온 시험 꾸러미(CIFAR-10이면 1만 개)

### 5. 맞추어 오는 치기 빠뜨리기

❌ **틀림:** 막이 얼개에 치기를 맞추지 않는다
✅ **옳음:** 막이의 결을 헤아린 치기를 꾸민다

## 좋은 버릇 간추림

### 막는 쪽에

1. **오토어택을 쓴다**: 잣대 잡히고 미더운 따짐
2. **기울기 가리기를 살핀다**: 진단 시험을 돌린다
3. **여러 으름 얼개**: L∞, L2, 여러 ε으로 시험한다
4. **밑금과 견준다**: RobustBench 모형과 견준다
5. **믿음 구간을 알린다**: 시험 꾸러미가 작을수록 더욱

### 치는 쪽에

1. **맞추어 오는 치기**: 그 막이에 맞춘 치기를 꾸민다
2. **여러 번 다시 비롯하기**: 아무 첫자리를 쓴다
3. **걸음 크기 맞추기**: 잃음 터에 맞춘다
4. **잃음 함수 고르기**: 엇갈린 엔트로피, DLR, 틈 잃음
5. **기울기 어림**: 미분할 수 없는 막이에

## 온전한 따짐 흐름

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
    
    # 2. 기울기 가리기 살피기
    print("\n기울기 가리기 살피는 중...")
    gm_results = check_gradient_masking(model, x_test[:1000], y_test[:1000], epsilon)
    results['gradient_masking'] = gm_results
    if gm_results['gradient_masking_suspected']:
        print("⚠️  알림: 기울기 가리기가 의심된다!")
    else:
        print("✓ 기울기 가리기는 보이지 않는다")
    
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

## 살펴볼 거리

1. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
2. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security." ICML.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv.
4. RobustBench: https://robustbench.github.io/

## 익힘 문제

**익힘 1.**
선형 가름개 $f(x) = w^T x + b$에서 미루어 본 갈래를 바꾸는 데 드는 가장 작은 $\ell_\infty$ 흔듦을 셈하여라. 이것이 신경 그물의 든든함과 어떻게 이어지는지 밝혀라.

??? success "익힘 1 풀이"
    선형 가름개에서 $\ell_\infty$ 노름으로 잰 판단의 금까지의 거리는 $\frac{|w^T x + b|}{\|w\|_1}$이다. 가장 작은 흔듦은 $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$이다. 신경 그물에서는 그 자리의 선형 어림 $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$이 FGSM(기울기의 부호를 쓴다)이 왜 잘 듣는지를 밝혀 준다. 차수가 높은 모형이 무른 까닭은 $\|w\|_1$은 차수와 함께 커지는데 $|w^T x + b|$은 꼭 그렇지 않아 든든함의 여유가 줄어들기 때문이다. $\square$

---

**익힘 2.**
이 마디에서 다룬 치기나 막이를 CIFAR-10의 ResNet-18 모형에 짜 넣어라. $\epsilon = 8/255$의 PGD-20 치기 아래에서 맑은 맞음과 든든한 맞음을 알려라.

??? success "익힘 2 풀이"
    여느 ResNet-18은 맑은 맞음이 $\sim$93%이지만 PGD-20($\epsilon = 8/255$, 걸음 크기 $2/255$) 아래의 든든한 맞음은 $\sim$0%이다. 이 마디의 방법을 걸면 결과는 재주에 따라 다르다. 맞서며 익히기는 맑은 맞음 $\sim$83%에 든든한 맞음 $\sim$50%이고, 밝혀 낸 막이는 더 낮지만 증명할 수 있는 테두리를 준다. 맞음과 든든함의 맞바꿈은 밑바탕부터 있는 것이라, 든든함을 높이면 맑은 맞음이 흔히 5~15% 든다. 아무렇게나 하는 씨앗 3개의 평균과 잣대 어긋남으로 알려라. $\square$

---

**익힘 3.**
흔듦 공 안에서 갈래별 자료의 밑자리가 서로 겹친다고 볼 때, 모형이 담는 힘을 키우지 않고서는 어떤 막이도 맑은 자료의 높은 맞음과 $\ell_\infty$ 흔듦에 대한 높은 든든함을 함께 이룰 수 없음을 증명하여라.

??? success "익힘 3 풀이"
    두 갈래의 밑자리가 거리 $\epsilon$ 안에서 겹치면(곧 $\|x_1 - x_2\|_\infty \leq 2\epsilon$인 $x_1 \in \text{갈래 1}, x_2 \in \text{갈래 2}$이 있으면), $x_1$과 $x_2$ 둘 다에서 든든한 가름개는 적어도 하나를 틀리게 가를 수밖에 없다(흔듦 공이 겹치기 때문이다). 이것이 맞음과 든든함의 밑바탕 맞바꿈이다. 겹치는 밑자리의 몫이 피할 수 없는 맞음 잃음을 정한다. 여느 그림 분포에서는 $\epsilon = 8/255$에서 겹침이 꽤 있어, 살펴본 10~15%의 맞음 떨어짐을 밝혀 준다. 모형이 담는 힘을 키우면(더 너른 그물) 얽힌 든든한 판단의 금을 더 잘 그려 맞바꿈을 얼마쯤 눅일 수 있다. $\square$

---

**익힘 4.**
금융 기계 배움 얼개(속임수 알아내기나 거래 신호 만들기 따위)에서 맞섬의 든든함이 어떻게 드러나는지 다루어라. 으름 얼개가 보기 다룸과 어떻게 다른가?

??? success "익힘 4 풀이"
    금융에서 겨루는 이는 알아내는 얼개에 맞추어 스스로 움직이는 꾀 많은 무리(속임수꾼, 저자 흔드는 이)다. 보기 다룸과 다른 고갱이는 이렇다. (1) 흔들 수 있는 밭이 돈으로 될 만한 것에 옭매인다(속임수꾼이 제 거래 자취를 통째로 바꿀 수는 없다). (2) 치기가 잇따르며 맞추어 간다(겨루는 이가 얼개의 되받음을 보고 손본다). (3) 헛 맞음과 놓침의 값이 서로 어긋난다(옳은 거래를 막는 것과 속임수를 놓치는 것). (4) $\ell_p$ 노름은 뜻이 없고 밭에 맞는 흔듦 모형이 있어야 한다. 막이는 맞추어 오는 겨루는 이에게도 든든해야 하므로, 알아내는 잣대가 알려지면 비껴갈 수 있는 알아내기 바탕의 길은 많이 걸러진다. $\square$
