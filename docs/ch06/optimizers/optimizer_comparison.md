# 최적화기 비교

기준 함수에서 적응형 최적화기를 견주면 저마다의 강점이 드러난다. 로젠브록 함수 같은 까다로운 지형에서는 대체로 Adam이 가장 빨리 수렴하고, RMSprop은 조건이 나쁜 문제를 잘 다루며, AdaGrad는 희소한 기울기에 뛰어나지만 분모가 계속 쌓여 오랜 학습에서는 멈춰 설 수 있다.

## 1. 코드

```python
"""
최적화기 비교: Adam, RMSprop, AdaGrad
================================================

이 스크립트는 여러 최적화 문제에서 적응형 학습률 최적화기 세 가지를 견주어
저마다의 강점과 차이를 보인다.
"""

import numpy as np
import sys

# ========================================================================
# 메인
# ========================================================================


# 우리가 만든 최적화기 구현 가져오기
from adam_optimizer import Adam
from rmsprop_optimizer import RMSprop
from adagrad_optimizer import AdaGrad


def test_simple_quadratic():
    """
    간단한 이차 함수에서 세 최적화기를 모두 시험한다.
    f(x, y) = x^2 + y^2 최소화
    """
    print("=" * 80)
    print("TEST 1: Simple Quadratic Function")
    print("=" * 80)
    print("Minimizing f(x, y) = x^2 + y^2")
    print("Starting point: x=10, y=10")
    print()
    
    # 최적화기마다 매개변수 초기화
    params_adam = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_adagrad = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    adam = Adam(learning_rate=0.1)
    rmsprop = RMSprop(learning_rate=0.1)
    adagrad = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Adam f(x,y)':<15} {'RMSprop f(x,y)':<15} {'AdaGrad f(x,y)':<15}")
    print("-" * 80)
    
    for i in range(50):
        # 경사를 계산한다
        grads_adam = {'x': 2 * params_adam['x'], 'y': 2 * params_adam['y']}
        grads_rmsprop = {'x': 2 * params_rmsprop['x'], 'y': 2 * params_rmsprop['y']}
        grads_adagrad = {'x': 2 * params_adagrad['x'], 'y': 2 * params_adagrad['y']}
        
        # 매개변수 갱신
        params_adam = adam.update(params_adam, grads_adam)
        params_rmsprop = rmsprop.update(params_rmsprop, grads_rmsprop)
        params_adagrad = adagrad.update(params_adagrad, grads_adagrad)
        
        # 함수값 계산
        f_adam = params_adam['x']**2 + params_adam['y']**2
        f_rmsprop = params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_adagrad = params_adagrad['x']**2 + params_adagrad['y']**2
        
        if i % 10 == 0:
            print(f"{i:<12} {f_adam[0]:<15.8f} {f_rmsprop[0]:<15.8f} {f_adagrad[0]:<15.8f}")
    
    print("\nConclusion: All three converge well on this simple problem.")
    print()


def test_ill_conditioned():
    """
    기울기의 척도가 서로 다른, 조건이 나쁜 문제에서 시험한다.
    f(x, y) = 100*x^2 + y^2 최소화
    """
    print("=" * 80)
    print("TEST 2: Ill-Conditioned Problem")
    print("=" * 80)
    print("Minimizing f(x, y) = 100*x^2 + y^2")
    print("Starting point: x=10, y=10")
    print("(x direction has much larger gradients than y direction)")
    print()
    
    # 매개변수를 초기화한다
    params_adam = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_adagrad = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    adam = Adam(learning_rate=0.1)
    rmsprop = RMSprop(learning_rate=0.1)
    adagrad = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Adam f(x,y)':<15} {'RMSprop f(x,y)':<15} {'AdaGrad f(x,y)':<15}")
    print("-" * 80)
    
    for i in range(100):
        # 기울기 계산: df/dx = 200x, df/dy = 2y
        grads_adam = {'x': 200 * params_adam['x'], 'y': 2 * params_adam['y']}
        grads_rmsprop = {'x': 200 * params_rmsprop['x'], 'y': 2 * params_rmsprop['y']}
        grads_adagrad = {'x': 200 * params_adagrad['x'], 'y': 2 * params_adagrad['y']}
        
        # 매개변수 갱신
        params_adam = adam.update(params_adam, grads_adam)
        params_rmsprop = rmsprop.update(params_rmsprop, grads_rmsprop)
        params_adagrad = adagrad.update(params_adagrad, grads_adagrad)
        
        # 함수값 계산
        f_adam = 100 * params_adam['x']**2 + params_adam['y']**2
        f_rmsprop = 100 * params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_adagrad = 100 * params_adagrad['x']**2 + params_adagrad['y']**2
        
        if i % 20 == 0:
            print(f"{i:<12} {f_adam[0]:<15.6f} {f_rmsprop[0]:<15.6f} {f_adagrad[0]:<15.6f}")
    
    print("\nConclusion: Adaptive methods handle different gradient scales automatically!")
    print()


def test_noisy_gradients():
    """
    잡음이 섞인 기울기로 시험하여 최적화기마다 얼마나 견고한지 본다.
    기울기에 잡음을 더한 채로 f(x, y) = x^2 + y^2 최소화
    """
    print("=" * 80)
    print("TEST 3: Noisy Gradients")
    print("=" * 80)
    print("Minimizing f(x, y) = x^2 + y^2 with noisy gradient estimates")
    print("Starting point: x=10, y=10")
    print()
    
    np.random.seed(42)
    
    # 매개변수를 초기화한다
    params_adam = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_adagrad = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    adam = Adam(learning_rate=0.1)
    rmsprop = RMSprop(learning_rate=0.1)
    adagrad = AdaGrad(learning_rate=1.0)
    
    print(f"{'Iteration':<12} {'Adam f(x,y)':<15} {'RMSprop f(x,y)':<15} {'AdaGrad f(x,y)':<15}")
    print("-" * 80)
    
    for i in range(100):
        # 잡음을 섞어 기울기 계산
        noise_x = np.random.randn() * 0.5
        noise_y = np.random.randn() * 0.5
        
        grads_adam = {
            'x': 2 * params_adam['x'] + noise_x,
            'y': 2 * params_adam['y'] + noise_y
        }
        grads_rmsprop = {
            'x': 2 * params_rmsprop['x'] + noise_x,
            'y': 2 * params_rmsprop['y'] + noise_y
        }
        grads_adagrad = {
            'x': 2 * params_adagrad['x'] + noise_x,
            'y': 2 * params_adagrad['y'] + noise_y
        }
        
        # 매개변수 갱신
        params_adam = adam.update(params_adam, grads_adam)
        params_rmsprop = rmsprop.update(params_rmsprop, grads_rmsprop)
        params_adagrad = adagrad.update(params_adagrad, grads_adagrad)
        
        # 함수값 계산
        f_adam = params_adam['x']**2 + params_adam['y']**2
        f_rmsprop = params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_adagrad = params_adagrad['x']**2 + params_adagrad['y']**2
        
        if i % 20 == 0:
            print(f"{i:<12} {f_adam[0]:<15.6f} {f_rmsprop[0]:<15.6f} {f_adagrad[0]:<15.6f}")
    
    print("\nConclusion: Adam's momentum helps smooth out noisy gradients better.")
    print()


def test_rosenbrock():
    """
    고전적인 최적화 기준인 로젠브록 함수에서 시험한다.
    f(x, y) = (1-x)^2 + 100*(y - x^2)^2 최소화
    """
    print("=" * 80)
    print("TEST 4: Rosenbrock Function (Challenging Benchmark)")
    print("=" * 80)
    print("Minimizing f(x, y) = (1-x)^2 + 100*(y - x^2)^2")
    print("Global minimum at (1, 1)")
    print("Starting point: x=-1, y=-1")
    print()
    
    # 매개변수를 초기화한다
    params_adam = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    params_rmsprop = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    params_adagrad = {'x': np.array([-1.0]), 'y': np.array([-1.0])}
    
    # 최적화기 초기화
    adam = Adam(learning_rate=0.01)
    rmsprop = RMSprop(learning_rate=0.01)
    adagrad = AdaGrad(learning_rate=0.1)
    
    print(f"{'Iteration':<12} {'Adam f(x,y)':<18} {'RMSprop f(x,y)':<18} {'AdaGrad f(x,y)':<18}")
    print("-" * 80)
    
    for i in range(1000):
        # 로젠브록 함수의 기울기 계산
        # df/dx = -2(1-x) - 400x(y - x^2)
        # df/dy = 200(y - x^2)
        
        grad_x_adam = -2*(1 - params_adam['x']) - 400*params_adam['x']*(params_adam['y'] - params_adam['x']**2)
        grad_y_adam = 200*(params_adam['y'] - params_adam['x']**2)
        
        grad_x_rmsprop = -2*(1 - params_rmsprop['x']) - 400*params_rmsprop['x']*(params_rmsprop['y'] - params_rmsprop['x']**2)
        grad_y_rmsprop = 200*(params_rmsprop['y'] - params_rmsprop['x']**2)
        
        grad_x_adagrad = -2*(1 - params_adagrad['x']) - 400*params_adagrad['x']*(params_adagrad['y'] - params_adagrad['x']**2)
        grad_y_adagrad = 200*(params_adagrad['y'] - params_adagrad['x']**2)
        
        grads_adam = {'x': grad_x_adam, 'y': grad_y_adam}
        grads_rmsprop = {'x': grad_x_rmsprop, 'y': grad_y_rmsprop}
        grads_adagrad = {'x': grad_x_adagrad, 'y': grad_y_adagrad}
        
        # 매개변수 갱신
        params_adam = adam.update(params_adam, grads_adam)
        params_rmsprop = rmsprop.update(params_rmsprop, grads_rmsprop)
        params_adagrad = adagrad.update(params_adagrad, grads_adagrad)
        
        # 함수값 계산
        f_adam = (1 - params_adam['x'])**2 + 100*(params_adam['y'] - params_adam['x']**2)**2
        f_rmsprop = (1 - params_rmsprop['x'])**2 + 100*(params_rmsprop['y'] - params_rmsprop['x']**2)**2
        f_adagrad = (1 - params_adagrad['x'])**2 + 100*(params_adagrad['y'] - params_adagrad['x']**2)**2
        
        if i % 200 == 0:
            print(f"{i:<12} {f_adam[0]:<18.6f} {f_rmsprop[0]:<18.6f} {f_adagrad[0]:<18.6f}")
    
    print()
    print(f"Final positions:")
    print(f"  Adam:    x={params_adam['x'][0]:.6f}, y={params_adam['y'][0]:.6f}")
    print(f"  RMSprop: x={params_rmsprop['x'][0]:.6f}, y={params_rmsprop['y'][0]:.6f}")
    print(f"  AdaGrad: x={params_adagrad['x'][0]:.6f}, y={params_adagrad['y'][0]:.6f}")
    print(f"  (Target: x=1.0, y=1.0)")
    print()
    print("Conclusion: Adam often performs best on challenging optimization landscapes.")
    print()


def print_summary():
    """
    최적화기의 특성 요약을 출력한다.
    """
    print("=" * 80)
    print("OPTIMIZER SUMMARY")
    print("=" * 80)
    print()
    
    print("AdaGrad (2011):")
    print("  ✓ Adapts learning rate per parameter")
    print("  ✓ Good for sparse gradients")
    print("  ✗ Learning rate monotonically decreases (can stop learning)")
    print("  • Best for: Sparse data, NLP, recommender systems")
    print()
    
    print("RMSprop (2012):")
    print("  ✓ Uses moving average of squared gradients")
    print("  ✓ Fixes AdaGrad's diminishing learning rates")
    print("  ✓ Works well on non-stationary problems")
    print("  • Best for: RNNs, non-stationary objectives")
    print()
    
    print("Adam (2014):")
    print("  ✓ Combines RMSprop + Momentum")
    print("  ✓ Includes bias correction")
    print("  ✓ Usually works well with default hyperparameters")
    print("  ✓ Most popular optimizer in deep learning")
    print("  • Best for: General purpose, default choice")
    print()
    
    print("Hyperparameter Recommendations:")
    print("  Adam:    lr=0.001, beta1=0.9, beta2=0.999")
    print("  RMSprop: lr=0.001, rho=0.9")
    print("  AdaGrad: lr=0.01")
    print()


if __name__ == "__main__":
    print("\n")
    test_simple_quadratic()
    print("\n")
    test_ill_conditioned()
    print("\n")
    test_noisy_gradients()
    print("\n")
    test_rosenbrock()
    print("\n")
    print_summary()
```

## 2. 논의

이차 함수 시험($f(x,y) = x^2 + y^2$)은 가장 단순한 기준이며 세 최적화기 모두 쉽게 수렴한다. 조건이 나쁜 시험($f(x,y) = 100x^2 + y^2$)은 적응형 방법의 이점을 드러낸다. 학습률이 하나뿐인 SGD는 가파른 $x$ 방향과 완만한 $y$ 방향에서 동시에 잘 나아갈 수 없지만, 적응형 방법은 저절로 서로 다른 학습률을 쓴다.

잡음 섞인 기울기 시험은 참 기울기에 정규 잡음을 더해 미니배치 학습의 확률성을 흉내 낸다. Adam의 모멘텀이 이 잡음을 가장 잘 다듬는 반면, AdaGrad의 누적 분모는 결국 갱신을 지나치게 억누른다.

로젠브록 함수 $f(x,y) = (1-x)^2 + 100(y-x^2)^2$은 좁고 굽은 골짜기를 가진 고전적인 최적화 난제이다. 모멘텀과 적응을 결합한 Adam이 이 골짜기를 가장 효율적으로 지나가고, 평범한 기울기 방법은 골짜기 벽 사이를 오가며 진동하는 경향이 있다.

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

**다룬 것** — 최적화기 비교

이차 함수 시험($f(x,y) = x^2 + y^2$)은 가장 단순한 기준이며 세 최적화기 모두 쉽게 수렴한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
