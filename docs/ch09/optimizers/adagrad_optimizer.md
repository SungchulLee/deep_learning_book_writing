# Adagrad 실습

NumPy로 Adagrad의 상태 누적과 파라메터 갱신을 구현한다. 이차 함수와 드문 좌표 갱신 예제를 통해 누적된 그래디언트 제곱이 다음 갱신에 미치는 영향을 살펴본다.

## 1. 코드

```python
"""
AdaGrad(적응형 그래디언트) 최적화기
======================================

AdaGrad는 지난 그래디언트 제곱의 합에 따라 파라메터마다 러닝레이트를 맞춘다.
그래디언트가 큰 파라메터는 러닝레이트가 작아지고, 그래디언트가 작은 파라메터는
러닝레이트가 커진다.

주요 기능:
- 모든 시각에 걸쳐 그래디언트 제곱을 누적한다
- 자주 갱신되는 파라메터의 러닝레이트를 저절로 줄인다
- 희소한 그래디언트에 좋다 (예: 자연어 처리, 추천 시스템)
- 러닝레이트가 너무 작아질 수 있다 (단조 감소한다)

논문: Duchi 등(2011), "Adaptive Subgradient Methods for Online Learning"
"""

import numpy as np

# ========================================================================
# 메인
# ========================================================================


class AdaGrad:
    """
    AdaGrad 최적화기의 구현.
    
    파라메터:
    -----------
    learning_rate : float, 기본값=0.01
        처음 러닝레이트 (전역 스텝 크기)
    epsilon : float, 기본값=1e-8
        수치 안정성을 위한 작은 상수
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        # 상태 변수
        self.cache = {}  # 누적된 그래디언트 제곱의 합
    
    def update(self, params, grads):
        """
        AdaGrad 알고리즘으로 파라메터를 갱신한다.
        
        파라메터:
        -----------
        params : dict
            갱신할 파라메터의 사전
        grads : dict
            파라메터별 그래디언트의 사전
        
        반환값:
        --------
        dict : 갱신된 파라메터
        """
        updated_params = {}
        
        for key in params.keys():
            # 캐시가 없으면 초기화
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # 그래디언트 제곱 누적
            self.cache[key] += grads[key] ** 2
            
            # 파라메터 갱신
            # 러닝레이트를 누적된 그래디언트 제곱의 제곱근으로 나눈다
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return updated_params


def demo_adagrad():
    """
    간단한 이차 함수에서 AdaGrad 최적화기 시연.
    f(x, y) = x^2 + y^2 최소화
    """
    print("=" * 60)
    print("AdaGrad Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x^2 + y^2")
    print()
    
    # 파라메터를 초기화한다
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    optimizer = AdaGrad(learning_rate=1.0)  # AdaGrad는 처음 러닝레이트를 더 높게 쓸 수 있다
    
    # 최적화 루프
    print(f"{'Iteration':<12} {'x':<12} {'y':<12} {'f(x,y)':<12}")
    print("-" * 60)
    
    for i in range(50):
        # 그래디언트 계산: df/dx = 2x, df/dy = 2y
        grads = {
            'x': 2 * params['x'],
            'y': 2 * params['y']
        }
        
        # 파라메터 갱신
        params = optimizer.update(params, grads)
        
        # 함수값 계산
        f_val = params['x']**2 + params['y']**2
        
        if i % 10 == 0:
            print(f"{i:<12} {params['x'][0]:<12.6f} {params['y'][0]:<12.6f} {f_val[0]:<12.6f}")
    
    print()
    print(f"Final values: x = {params['x'][0]:.8f}, y = {params['y'][0]:.8f}")
    print(f"Function value: f(x,y) = {f_val[0]:.8f}")
    print()


def demo_sparse_gradients():
    """
    희소한 그래디언트에서 AdaGrad의 이점을 보인다.
    어떤 파라메터가 드물게 갱신되는 상황을 흉내 낸다.
    """
    print("=" * 60)
    print("AdaGrad with Sparse Gradients")
    print("=" * 60)
    print("Parameters x, y, z where z is rarely updated (sparse)")
    print()
    
    # 파라메터를 초기화한다
    params = {
        'x': np.array([5.0]),
        'y': np.array([5.0]),
        'z': np.array([5.0])  # 이것은 드물게 갱신된다
    }
    
    # 최적화기 초기화
    optimizer = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'x':<12} {'y':<12} {'z':<12}")
    print("-" * 60)
    
    for i in range(50):
        # 대부분의 반복에서는 x와 y에만 그래디언트가 있다
        # 10번째 반복마다 z에도 그래디언트가 생긴다
        grads = {
            'x': 2 * params['x'],
            'y': 2 * params['y'],
            'z': 2 * params['z'] if i % 10 == 0 else np.array([0.0])
        }
        
        # 파라메터 갱신
        params = optimizer.update(params, grads)
        
        if i % 10 == 0:
            print(f"{i:<12} {params['x'][0]:<12.6f} {params['y'][0]:<12.6f} {params['z'][0]:<12.6f}")
    
    print()
    print("Notice: z converges slower because it's updated less frequently,")
    print("but AdaGrad gives it a relatively larger effective learning rate!")
    print()


def show_learning_rate_decay():
    """
    AdaGrad의 실효 러닝레이트가 시간이 갈수록 줄어드는 모습을 보인다.
    """
    print("=" * 60)
    print("AdaGrad Learning Rate Decay")
    print("=" * 60)
    print("Effective learning rate = lr / sqrt(sum of squared gradients)")
    print()
    
    # 파라메터 하나에 대한 최적화
    param = np.array([10.0])
    optimizer = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Param':<15} {'Effective LR':<15}")
    print("-" * 60)
    
    effective_lrs = []
    
    for i in range(50):
        # 상수 그래디언트
        grad = 2 * param
        
        # 갱신 전에 실효 러닝레이트 계산
        if i == 0:
            # 아직 cache가 차지 않았다. 아래 클래스와 모양을 맞추어 배열로 둔다.
            effective_lr = np.full_like(param, optimizer.learning_rate)
        else:
            effective_lr = optimizer.learning_rate / np.sqrt(optimizer.cache['param'] + optimizer.epsilon)
        
        effective_lrs.append(effective_lr)
        
        # 파라메터 갱신
        params = {'param': param}
        grads = {'param': grad}
        updated = optimizer.update(params, grads)
        param = updated['param']
        
        if i % 10 == 0:
            print(f"{i:<12} {param[0]:<15.6f} {effective_lr[0]:<15.6f}")
    
    print()
    print("Notice: The effective learning rate monotonically decreases.")
    print("This can cause AdaGrad to stop learning prematurely in some cases.")
    print()


if __name__ == "__main__":
    demo_adagrad()
    print("\n")
    demo_sparse_gradients()
    print("\n")
    show_learning_rate_decay()
```

**출력:**

```
============================================================
AdaGrad Optimizer Demo
============================================================
Minimizing f(x, y) = x^2 + y^2

Iteration    x            y            f(x,y)      
------------------------------------------------------------
0            9.000000     9.000000     162.000000  
10           5.311021     5.311021     56.413889   
20           3.632809     3.632809     26.394607   
30           2.568478     2.568478     13.194154   
40           1.840387     1.840387     6.774052    

Final values: x = 1.37083081, y = 1.37083081
Function value: f(x,y) = 3.75835423



============================================================
AdaGrad with Sparse Gradients
============================================================
Parameters x, y, z where z is rarely updated (sparse)

Iteration    x            y            z           
------------------------------------------------------------
0            4.000000     4.000000     4.000000    
10           1.090040     1.090040     3.375305    
20           0.354721     0.354721     2.908992    
30           0.116876     0.116876     2.536090    
40           0.038559     0.038559     2.226918    

Notice: z converges slower because it's updated less frequently,
but AdaGrad gives it a relatively larger effective learning rate!



============================================================
AdaGrad Learning Rate Decay
============================================================
Effective learning rate = lr / sqrt(sum of squared gradients)

Iteration    Param           Effective LR   
------------------------------------------------------------
0            9.000000        1.000000       
10           5.311021        0.020977       
20           3.632809        0.017811       
30           2.568478        0.016728       
40           1.840387        0.016251       

Notice: The effective learning rate monotonically decreases.
This can cause AdaGrad to stop learning prematurely in some cases.

```

## 2. 논의

각 좌표의 상태는 $G_t=G_{t-1}+g_t^2$이고, 구현의 갱신식은 $\theta_t=\theta_{t-1}-\eta g_t/(\sqrt{G_t}+\epsilon)$이다. 작은 상수는 제곱근 밖에 있다. 상태가 단조 증가하므로 고정된 전역 러닝레이트에서 좌표별 계수는 증가하지 않는다. 다만 그래디언트 제곱의 합이 유한하면 그 계수가 반드시 0으로 수렴하는 것은 아니다.

희소한 갱신에서는 사용되지 않은 좌표의 상태가 그대로 남는다. 드물게 갱신된다는 이유만으로 항상 계수가 더 큰 것은 아니며, 그래디언트의 크기까지 포함한 누적합을 비교해야 한다. 코드의 z 좌표는 10회마다 한 번만 갱신되므로, 매번 갱신되는 좌표보다 목표에 느리게 접근할 수 있다.

`show_learning_rate_decay`는 갱신 전의 캐시를 이용해 표시값을 만든다. 첫 표시값은 전역 러닝레이트고 이후에는 epsilon을 제곱근 안에 넣는다. 따라서 표시 열은 현재 그래디언트를 반영한 실제 update의 분모와 정확히 같지 않다. 주석의 표현과 달리 이 예제의 기울기 2×param도 상수가 아니다. 클래스는 NumPy 배열의 사전을 반환하며 PyTorch 옵티마이저나 자동 미분을 사용하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
첫 기울기가 2, 전역 러닝레이트가 1일 때 첫 갱신량을 구하라.

</div>

??? success "연습문제 1 풀이"
    초기 캐시가 0이면 새 캐시는 4이고 갱신량은 2/(2+epsilon)이다. epsilon을 무시하면 약 1이다. 표시용 전역 러닝레이트 자체와 그래디언트에 곱하는 계수를 구분한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
한 좌표의 기울기가 0인 단계에서 캐시와 파라메터는 어떻게 되는가?

</div>

??? success "연습문제 2 풀이"
    그래디언트 제곱의 추가분이 0이므로 캐시는 그대로이고 갱신량도 0이다. 사용하지 않은 동안 캐시가 줄어들거나 초기화되지는 않는다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
같은 최적화기 객체를 새 문제에 재사용하면 어떤 상태를 주의해야 하는가?

</div>

??? success "연습문제 3 풀이"
    cache에 이전 문제의 누적합이 남는다. 새 문제에서는 새 객체를 만들거나 상태를 명시적으로 초기화해야 한다. 배열의 키·모양·부동소수점 자료형도 일치시킨다.

## 정리하며

Adagrad의 적응은 좌표별 누적 그래디언트 제곱에 의해 정해진다. 갱신에 쓰는 실제 분모와 진단용 표시값의 시점을 구분하고, 오래 학습했을 때의 감소를 무조건적인 0 수렴으로 해석하지 않는다.
