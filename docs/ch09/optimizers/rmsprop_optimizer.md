# RMSprop 실습

NumPy로 그래디언트 제곱의 지수 이동평균을 유지하는 기본 RMSprop을 구현한다. 이차 함수와 좌표별 곡률 차이가 큰 함수에서 갱신을 살펴본다.

## 1. 코드

```python
"""
RMSprop(제곱평균제곱근 전파) 최적화기
================================================

RMSprop은 지난 그래디언트 제곱을 모두 누적하는 대신 이동 평균을 써서 AdaGrad의
한계를 다루는 적응형 러닝레이트 방법이다.


주요 기능:
- 그래디언트 제곱의 지수 이동 평균을 쓴다
- 러닝레이트를 이 평균의 제곱근으로 나눈다
- (AdaGrad와 달리) 비정상 문제에서 잘 통한다
- (Adam과 달리) 바이어스 보정이 없다

만든 사람: 제프리 힌턴 (코세라 강의)
"""

import numpy as np

# ========================================================================
# 메인
# ========================================================================


class RMSprop:
    """
    RMSprop 최적화기의 구현.
    
    파라메터:
    -----------
    learning_rate : float, 기본값=0.001
        파라메터 갱신의 스텝 크기
    rho : float, 기본값=0.9
        그래디언트 제곱의 이동 평균에 대한 감쇠율
    epsilon : float, 기본값=1e-8
        수치 안정성을 위한 작은 상수
    """
    
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        
        # 상태 변수
        self.cache = {}  # 그래디언트 제곱의 이동 평균
    
    def update(self, params, grads):
        """
        RMSprop 알고리즘으로 파라메터를 갱신한다.
        
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
            
            # 그래디언트 제곱의 이동 평균 갱신
            self.cache[key] = self.rho * self.cache[key] + (1 - self.rho) * (grads[key] ** 2)
            
            # 파라메터 갱신
            # 러닝레이트를 이동 평균의 제곱근으로 나눈다
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        return updated_params


def demo_rmsprop():
    """
    간단한 이차 함수에서 RMSprop 최적화기 시연.
    f(x, y) = x^2 + y^2 최소화
    """
    print("=" * 60)
    print("RMSprop Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x^2 + y^2")
    print()
    
    # 파라메터를 초기화한다
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    optimizer = RMSprop(learning_rate=0.1)
    
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


def compare_with_without_rmsprop():
    """
    RMSprop 적응이 있을 때와 없을 때의 경사 하강법을 견준다.
    RMSprop이 서로 다른 그래디언트 척도를 다루는 모습을 보인다.
    """
    print("=" * 60)
    print("RMSprop vs Standard Gradient Descent")
    print("=" * 60)
    print("Minimizing f(x, y) = 100*x^2 + y^2 (ill-conditioned)")
    print()
    
    # 파라메터를 초기화한다
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_sgd = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    optimizer_rmsprop = RMSprop(learning_rate=0.1)
    lr_sgd = 0.001  # 조건이 나쁜 문제에서 SGD는 훨씬 작은 러닝레이트가 필요하다
    
    print(f"{'Iteration':<12} {'RMSprop f(x,y)':<20} {'SGD f(x,y)':<20}")
    print("-" * 60)
    
    for i in range(100):
        # 그래디언트 계산: df/dx = 200x, df/dy = 2y
        grads_rmsprop = {
            'x': 200 * params_rmsprop['x'],
            'y': 2 * params_rmsprop['y']
        }
        grads_sgd = {
            'x': 200 * params_sgd['x'],
            'y': 2 * params_sgd['y']
        }
        
        # 파라메터 갱신
        params_rmsprop = optimizer_rmsprop.update(params_rmsprop, grads_rmsprop)
        params_sgd['x'] = params_sgd['x'] - lr_sgd * grads_sgd['x']
        params_sgd['y'] = params_sgd['y'] - lr_sgd * grads_sgd['y']
        
        # 함수값 계산
        f_rmsprop = 100 * params_rmsprop['x']**2 + params_rmsprop['y']**2
        f_sgd = 100 * params_sgd['x']**2 + params_sgd['y']**2
        
        if i % 20 == 0:
            print(f"{i:<12} {f_rmsprop[0]:<20.6f} {f_sgd[0]:<20.6f}")
    
    print()
    print("Notice: RMSprop converges faster on this ill-conditioned problem!")
    print()


if __name__ == "__main__":
    demo_rmsprop()
    print("\n")
    compare_with_without_rmsprop()
```

**출력:**

```
============================================================
RMSprop Optimizer Demo
============================================================
Minimizing f(x, y) = x^2 + y^2

Iteration    x            y            f(x,y)      
------------------------------------------------------------
0            9.683772     9.683772     187.550889  
10           8.207379     8.207379     134.722151  
20           7.185187     7.185187     103.253816  
30           6.256344     6.256344     78.283678   
40           5.368114     5.368114     57.633297   

Final values: x = 4.59238901, y = 4.59238901
Function value: f(x,y) = 42.18007367



============================================================
RMSprop vs Standard Gradient Descent
============================================================
Minimizing f(x, y) = 100*x^2 + y^2 (ill-conditioned)

Iteration    RMSprop f(x,y)       SGD f(x,y)          
------------------------------------------------------------
0            9471.319913          6499.600400         
20           5214.317711          92.786098           
40           2910.481477          84.860378           
60           1363.257978          78.329623           
80           452.221025           72.301563           

Notice: RMSprop converges faster on this ill-conditioned problem!
```

## 2. 논의

캐시는 $v_t=\rho v_{t-1}+(1-\rho)g_t^2$로 갱신되고 파라메터에서는 $\eta g_t/(\sqrt{v_t}+\epsilon)$을 뺀다. 이 클래스에는 모멘텀, 중심화와 바이어스 보정이 없다. 그래디언트 제곱이 일정 범위 안에 있으면 캐시도 유계이지만, 지수 이동평균이라는 이유만으로 무한히 커지는 입력까지 제한되지는 않는다.

비교 함수 $100x^2+y^2$에서 기본 경사 하강법은 x 방향의 안정성을 위해 러닝레이트가 0.01보다 작아야 한다. 코드에서 선택한 0.001만이 유일한 안정적인 값은 아니다. RMSprop에는 0.1을 사용하므로 결과는 이 두 설정의 비교이며 러닝레이트를 충분히 조율한 모든 방법에 대한 우위를 보이지 않는다.

이 구현의 기본 learning_rate는 0.001이다. 라이브러리 구현의 기본값과 기능이 같다고 가정하지 말고 생성자를 확인한다. 또한 update는 갱신된 사전을 반환하므로 반환값을 다음 단계의 파라메터로 사용한다. 상수 러닝레이트에서 최솟값 주변을 오갈 수 있으므로 제한된 반복 후 감소한 로스와 수렴 보장을 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
rho=0.9, 초기 캐시 0, 첫 기울기 2일 때 새 캐시와 갱신량을 구하라.

</div>

??? success "연습문제 1 풀이"
    캐시는 $0.1\times4=0.4$이고 갱신량은 $2\eta/(\sqrt{0.4}+\epsilon)$이다. 바이어스 보정이 없어 첫 분모는 그래디언트 크기 2보다 작다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
기울기가 0인 단계에서 RMSprop과 Adagrad의 캐시는 어떻게 다른가?

</div>

??? success "연습문제 2 풀이"
    RMSprop은 캐시에 rho를 곱해 줄이고 Adagrad는 그대로 유지한다. 두 경우 모두 현재 갱신량은 0이며, 줄어든 RMSprop 캐시는 이후의 0이 아닌 그래디언트에 영향을 준다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
100x²+y²에 대한 기본 경사 하강법의 안정적인 러닝레이트 범위를 도출하라.

</div>

??? success "연습문제 3 풀이"
    x는 $(1-200\eta)x$, y는 $(1-2\eta)y$로 갱신된다. 두 절댓값이 모두 1보다 작으려면 $0<\eta<0.01$이어야 한다. 이는 이 이차 함수와 고정 러닝레이트에 대한 결과이다.

## 정리하며

RMSprop은 그래디언트 제곱의 오래된 기여를 감쇠시킨다. 안정성과 속도는 전역 러닝레이트와 목적함수에도 달려 있으므로 적응형 분모만으로 수렴이나 우위를 보장하지 않는다.
