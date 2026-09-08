# Adagrad 최적화기

AdaGrad는 누적된 기울기 제곱의 합에 따라 매개변수마다 학습률을 맞춘다. 자주 갱신되는 매개변수는 작은 학습률을, 드물게 갱신되는 매개변수는 큰 학습률을 유지한다. 덕분에 희소한 데이터에 특히 효과적이지만, 학습률이 단조 감소하므로 너무 일찍 수렴해 버릴 수 있다.

## 1. 코드

```python
"""
AdaGrad(적응형 기울기) 최적화기
======================================

AdaGrad는 지난 기울기 제곱의 합에 따라 매개변수마다 학습률을 맞춘다.
기울기가 큰 매개변수는 학습률이 작아지고, 기울기가 작은 매개변수는
학습률이 커진다.

주요 기능:
- 모든 시각에 걸쳐 기울기 제곱을 누적한다
- 자주 갱신되는 매개변수의 학습률을 저절로 줄인다
- 희소한 기울기에 좋다 (예: 자연어 처리, 추천 시스템)
- 학습률이 너무 작아질 수 있다 (단조 감소한다)

논문: Duchi 등(2011), "Adaptive Subgradient Methods for Online Learning"
"""

import numpy as np

# ========================================================================
# 메인
# ========================================================================


class AdaGrad:
    """
    AdaGrad 최적화기의 구현.
    
    매개변수:
    -----------
    learning_rate : float, 기본값=0.01
        처음 학습률 (전역 걸음 크기)
    epsilon : float, 기본값=1e-8
        수치 안정성을 위한 작은 상수
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        # 상태 변수
        self.cache = {}  # 누적된 기울기 제곱의 합
    
    def update(self, params, grads):
        """
        AdaGrad 알고리즘으로 매개변수를 갱신한다.
        
        매개변수:
        -----------
        params : dict
            갱신할 매개변수의 사전
        grads : dict
            매개변수별 기울기의 사전
        
        반환값:
        --------
        dict : 갱신된 매개변수
        """
        updated_params = {}
        
        for key in params.keys():
            # 캐시가 없으면 초기화
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            
            # 기울기 제곱 누적
            self.cache[key] += grads[key] ** 2
            
            # 매개변수 갱신
            # 학습률을 누적된 기울기 제곱의 제곱근으로 나눈다
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
    
    # 매개변수를 초기화한다
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    optimizer = AdaGrad(learning_rate=1.0)  # AdaGrad는 처음 학습률을 더 높게 쓸 수 있다
    
    # 최적화 루프
    print(f"{'Iteration':<12} {'x':<12} {'y':<12} {'f(x,y)':<12}")
    print("-" * 60)
    
    for i in range(50):
        # 기울기 계산: df/dx = 2x, df/dy = 2y
        grads = {
            'x': 2 * params['x'],
            'y': 2 * params['y']
        }
        
        # 매개변수 갱신
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
    희소한 기울기에서 AdaGrad의 이점을 보인다.
    어떤 매개변수가 드물게 갱신되는 상황을 흉내 낸다.
    """
    print("=" * 60)
    print("AdaGrad with Sparse Gradients")
    print("=" * 60)
    print("Parameters x, y, z where z is rarely updated (sparse)")
    print()
    
    # 매개변수를 초기화한다
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
        # 대부분의 반복에서는 x와 y에만 기울기가 있다
        # 10번째 반복마다 z에도 기울기가 생긴다
        grads = {
            'x': 2 * params['x'],
            'y': 2 * params['y'],
            'z': 2 * params['z'] if i % 10 == 0 else np.array([0.0])
        }
        
        # 매개변수 갱신
        params = optimizer.update(params, grads)
        
        if i % 10 == 0:
            print(f"{i:<12} {params['x'][0]:<12.6f} {params['y'][0]:<12.6f} {params['z'][0]:<12.6f}")
    
    print()
    print("Notice: z converges slower because it's updated less frequently,")
    print("but AdaGrad gives it a relatively larger effective learning rate!")
    print()


def show_learning_rate_decay():
    """
    AdaGrad의 실효 학습률이 시간이 갈수록 줄어드는 모습을 보인다.
    """
    print("=" * 60)
    print("AdaGrad Learning Rate Decay")
    print("=" * 60)
    print("Effective learning rate = lr / sqrt(sum of squared gradients)")
    print()
    
    # 매개변수 하나에 대한 최적화
    param = np.array([10.0])
    optimizer = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Param':<15} {'Effective LR':<15}")
    print("-" * 60)
    
    effective_lrs = []
    
    for i in range(50):
        # 상수 기울기
        grad = 2 * param
        
        # 갱신 전에 실효 학습률 계산
        if i == 0:
            effective_lr = optimizer.learning_rate
        else:
            effective_lr = optimizer.learning_rate / np.sqrt(optimizer.cache['param'] + optimizer.epsilon)
        
        effective_lrs.append(effective_lr)
        
        # 매개변수 갱신
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

## 2. 논의

AdaGrad는 매개변수마다 기울기 제곱의 누적합 $G_t = G_{t-1} + g_t^2$을 관리한다. 매개변수별 실효 학습률은 $\eta / \sqrt{G_t + \epsilon}$이며 자주 갱신되는 매개변수에서 저절로 줄어든다. 어떤 매개변수가 다른 것보다 훨씬 자주 갱신되는 희소한 특징에서 이러한 매개변수별 적응이 특히 값어치 있다.

AdaGrad의 주된 한계는 누적된 기울기 제곱이 단조 증가하여 실효 학습률이 0으로 줄어든다는 것이다. 오래 학습시키면 현재의 해가 최적과 멀어도 모델이 더 나아가지 못할 수 있다.

RMSprop과 Adam은 누적합 대신 지수 이동 평균을 써서 분모가 오래된 기울기를 "잊게" 하고 학습 내내 뜻있는 학습률을 유지하여 이 한계를 다룬다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

## 정리하며

**다룬 것** — Adagrad 최적화기

AdaGrad는 매개변수마다 기울기 제곱의 누적합 $G_t = G_{t-1} + g_t^2$을 관리한다.

핵심 클래스는 `AdaGrad`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
