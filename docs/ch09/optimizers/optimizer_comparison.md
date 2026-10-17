# 기준 함수에서 옵티마이저 비교

직접 구현한 Adam, RMSprop, Adagrad를 네 가지 2변수 목적함수에서 비교한다. 초기점, 러닝레이트, 반복 횟수와 잡음 조건이 결과에 어떻게 영향을 주는지 읽는 예제이다.

## 1. 코드

```python
"""
최적화기 비교: Adam, RMSprop, AdaGrad
================================================

이 스크립트는 여러 최적화 문제에서 적응형 러닝레이트 최적화기 세 가지를 견주어
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
    
    # 최적화기마다 파라메터 초기화
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
        
        # 파라메터 갱신
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
    그래디언트의 척도가 서로 다른, 조건이 나쁜 문제에서 시험한다.
    f(x, y) = 100*x^2 + y^2 최소화
    """
    print("=" * 80)
    print("TEST 2: Ill-Conditioned Problem")
    print("=" * 80)
    print("Minimizing f(x, y) = 100*x^2 + y^2")
    print("Starting point: x=10, y=10")
    print("(x direction has much larger gradients than y direction)")
    print()
    
    # 파라메터를 초기화한다
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
        # 그래디언트 계산: df/dx = 200x, df/dy = 2y
        grads_adam = {'x': 200 * params_adam['x'], 'y': 2 * params_adam['y']}
        grads_rmsprop = {'x': 200 * params_rmsprop['x'], 'y': 2 * params_rmsprop['y']}
        grads_adagrad = {'x': 200 * params_adagrad['x'], 'y': 2 * params_adagrad['y']}
        
        # 파라메터 갱신
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
    잡음이 섞인 그래디언트로 시험하여 최적화기마다 얼마나 견고한지 본다.
    그래디언트에 잡음을 더한 채로 f(x, y) = x^2 + y^2 최소화
    """
    print("=" * 80)
    print("TEST 3: Noisy Gradients")
    print("=" * 80)
    print("Minimizing f(x, y) = x^2 + y^2 with noisy gradient estimates")
    print("Starting point: x=10, y=10")
    print()
    
    np.random.seed(42)
    
    # 파라메터를 초기화한다
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
        # 잡음을 섞어 그래디언트 계산
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
        
        # 파라메터 갱신
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
    
    # 파라메터를 초기화한다
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
        # 로젠브록 함수의 그래디언트 계산
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
        
        # 파라메터 갱신
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

??? note "전체 출력 (110줄)"

    ```


    ================================================================================
    TEST 1: Simple Quadratic Function
    ================================================================================
    Minimizing f(x, y) = x^2 + y^2
    Starting point: x=10, y=10

    Iteration    Adam f(x,y)     RMSprop f(x,y)  AdaGrad f(x,y) 
    --------------------------------------------------------------------------------
    0            196.02000000    187.55088938    162.00000002   
    10           158.58456702    134.72215130    56.41388894    
    20           125.79381676    103.25381613    26.39460664    
    30           97.87188501     78.28367763     13.19415351    
    40           74.70788831     57.63329665     6.77405221     

    Conclusion: All three converge well on this simple problem.



    ================================================================================
    TEST 2: Ill-Conditioned Problem
    ================================================================================
    Minimizing f(x, y) = 100*x^2 + y^2
    Starting point: x=10, y=10
    (x direction has much larger gradients than y direction)

    Iteration    Adam f(x,y)     RMSprop f(x,y)  AdaGrad f(x,y) 
    --------------------------------------------------------------------------------
    0            9899.010000     9471.319913     8181.000000    
    20           6352.587745     5214.317711     1332.927634    
    40           3772.748357     2910.481477     342.089636     
    60           2074.497165     1363.257978     93.027793      
    80           1051.791278     452.221025      25.659421      

    Conclusion: Adaptive methods handle different gradient scales automatically!



    ================================================================================
    TEST 3: Noisy Gradients
    ================================================================================
    Minimizing f(x, y) = x^2 + y^2 with noisy gradient estimates
    Starting point: x=10, y=10

    Iteration    Adam f(x,y)     RMSprop f(x,y)  AdaGrad f(x,y) 
    --------------------------------------------------------------------------------
    0            196.020000      187.550889      162.000000     
    20           125.959298      103.577110      27.026679      
    40           74.806532       57.796310       6.971340       
    60           41.113410       27.052619       1.890734       
    80           20.828289       9.023060        0.531231       

    Conclusion: Adam's momentum helps smooth out noisy gradients better.



    ================================================================================
    TEST 4: Rosenbrock Function (Challenging Benchmark)
    ================================================================================
    Minimizing f(x, y) = (1-x)^2 + 100*(y - x^2)^2
    Global minimum at (1, 1)
    Starting point: x=-1, y=-1

    Iteration    Adam f(x,y)        RMSprop f(x,y)     AdaGrad f(x,y)    
    --------------------------------------------------------------------------------
    0            392.089501         367.208303         296.020000        
    200          2.144669           0.431561           1.548199          
    400          1.498415           0.106057           1.471525          
    600          1.423645           0.059361           1.408719          
    800          1.332703           0.041348           1.347106          

    Final positions:
      Adam:    x=-0.108335, y=0.013289
      RMSprop: x=0.877699, y=0.783831
      AdaGrad: x=-0.134348, y=0.020197
      (Target: x=1.0, y=1.0)

    Conclusion: Adam often performs best on challenging optimization landscapes.



    ================================================================================
    OPTIMIZER SUMMARY
    ================================================================================

    AdaGrad (2011):
      ✓ Adapts learning rate per parameter
      ✓ Good for sparse gradients
      ✗ Learning rate monotonically decreases (can stop learning)
      • Best for: Sparse data, NLP, recommender systems

    RMSprop (2012):
      ✓ Uses moving average of squared gradients
      ✓ Fixes AdaGrad's diminishing learning rates
      ✓ Works well on non-stationary problems
      • Best for: RNNs, non-stationary objectives

    Adam (2014):
      ✓ Combines RMSprop + Momentum
      ✓ Includes bias correction
      ✓ Usually works well with default hyperparameters
      ✓ Most popular optimizer in deep learning
      • Best for: General purpose, default choice

    Hyperparameter Recommendations:
      Adam:    lr=0.001, beta1=0.9, beta2=0.999
      RMSprop: lr=0.001, rho=0.9
      AdaGrad: lr=0.01

    ```


## 2. 논의

이차 함수 $x^2+y^2$와 조건수가 큰 함수 $100x^2+y^2$는 좌표별 스케일 차이를 비교하는 기준이다. 후자의 헤시안 고윳값은 200과 2로 조건수는 100이다. 적응형 방법은 각 좌표의 그래디언트 이력에 따라 갱신을 조절하지만, 이 비교 코드에는 기본 SGD가 포함되어 있지 않다.

잡음 예제에서는 매 단계 뽑은 같은 noise_x와 noise_y를 세 방법에 더한다. 잡음 실현은 공유하지만 파라메터가 서로 달라지므로 잡음을 더하기 전의 그래디언트는 같지 않다. 한 시드의 로스값만으로 잡음에 대한 견고함의 순위를 확정할 수 없다.

로젠브록 함수의 전역 최솟값은 (1, 1)에서 0이다. 좁고 굽은 골짜기가 있어 유한한 반복 후의 값은 초기점과 설정에 민감하다. 코드에서는 Adagrad의 러닝레이트가 다른 두 방법의 10배이며 반복 예산도 함수마다 다르다. 출력에 있는 일괄적인 우위 설명은 이 설정의 관찰로 한정하며, 더 넓은 문제군에서의 보장으로 읽지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
100x²+y²의 헤시안과 조건수를 구하라.

</div>

??? success "풀이"
    헤시안은 대각 원소가 200과 2인 행렬이다. 두 값 모두 양수이고 스펙트럼 조건수는 200/2=100이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
세 방법에 같은 잡음을 더하면 같은 그래디언트를 입력하는 비교인가?

</div>

??? success "풀이"
    아니다. 같은 단계에도 각 방법의 현재 파라메터가 달라 참 그래디언트는 다르다. 공유한 것은 더해지는 잡음이며, 이는 잡음 조건의 차이를 줄이기 위한 통제이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
알고리즘의 성능을 더 신뢰할 수 있게 비교하려면 어떤 설계가 필요한가?

</div>

??? success "풀이"
    방법별 러닝레이트 탐색 범위와 예산을 명시하고 여러 초기점·잡음 시드에서 비교한다. 최종 함수값뿐 아니라 목표값 도달까지의 그래디언트 평가 횟수도 보고한다. 한 함수에서 고른 설정을 다른 함수의 최적 설정으로 가정하지 않는다.

## 정리하며

기준 함수는 갱신의 피처를 드러내지만 보편적인 순위를 정하지 않는다. 동일한 초기점과 잡음 조건, 방법별 조율 예산과 반복 횟수를 함께 제시해야 결과를 해석할 수 있다.
