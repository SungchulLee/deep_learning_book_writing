# Adam 최적화기

Adam은 모멘텀(일차 모멘트)과 RMSprop(이차 모멘트)에 편향 보정을 더하여 아주 다양한 문제에서 잘 통하는 적응형 최적화기를 만든다. 매개변수마다 학습률을 관리하고 초매개변수 조율이 거의 필요 없어 많은 딥러닝 실무자의 기본 선택이 되었다.

## 1. 코드

```python
"""
Adam(적응형 모멘트 추정) 최적화기
============================================

Adam은 RMSprop과 모멘텀 기반 경사 하강법의 장점을 결합한다.
기울기의 일차 모멘트(평균)와 이차 모멘트(분산)를 모두 써서 매개변수마다
적응형 학습률을 계산한다.

주요 기능:
- 기울기의 지수 이동 평균을 관리한다 (일차 모멘트)
- 기울기 제곱의 지수 이동 평균을 관리한다 (이차 모멘트)
- 두 모멘트 모두에 편향 보정을 한다
- 기본 초매개변수로도 대체로 잘 통한다

논문: Kingma & Ba(2014), "Adam: A Method for Stochastic Optimization"
"""

import numpy as np

# ========================================================================
# 메인
# ========================================================================


class Adam:
    """
    Adam 최적화기의 구현.
    
    매개변수:
    -----------
    learning_rate : float, 기본값=0.001
        매개변수 갱신의 걸음 크기
    beta1 : float, 기본값=0.9
        일차 모멘트 추정값의 지수 감쇠율
    beta2 : float, 기본값=0.999
        이차 모멘트 추정값의 지수 감쇠율
    epsilon : float, 기본값=1e-8
        수치 안정성을 위한 작은 상수
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # 상태 변수
        self.m = {}  # 일차 모멘트 벡터 (기울기의 평균)
        self.v = {}  # 이차 모멘트 벡터 (기울기의 분산)
        self.t = 0   # 시각
    
    def update(self, params, grads):
        """
        Adam 알고리즘으로 매개변수를 갱신한다.
        
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
        self.t += 1
        
        updated_params = {}
        
        for key in params.keys():
            # 모멘트 벡터가 없으면 초기화
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # 치우친 일차 모멘트 추정값 갱신
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # 치우친 이차 원시 모멘트 추정값 갱신
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # 편향을 보정한 일차 모멘트 추정값 계산
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # 편향을 보정한 이차 원시 모멘트 추정값 계산
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 매개변수 갱신
            updated_params[key] = params[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return updated_params


def demo_adam():
    """
    간단한 이차 함수에서 Adam 최적화기 시연.
    f(x, y) = x^2 + y^2 최소화
    """
    print("=" * 60)
    print("Adam Optimizer Demo")
    print("=" * 60)
    print("Minimizing f(x, y) = x^2 + y^2")
    print()
    
    # 매개변수를 초기화한다
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    optimizer = Adam(learning_rate=0.1)
    
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


if __name__ == "__main__":
    demo_adam()
```

**출력:**

```
============================================================
Adam Optimizer Demo
============================================================
Minimizing f(x, y) = x^2 + y^2

Iteration    x            y            f(x,y)      
------------------------------------------------------------
0            9.900000     9.900000     196.020000  
10           8.904621     8.904621     158.584567  
20           7.930757     7.930757     125.793817  
30           6.995423     6.995423     97.871885   
40           6.111787     6.111787     74.707888   

Final values: x = 5.36821171, y = 5.36821171
Function value: f(x,y) = 57.63539398
```

## 2. 논의

Adam은 지수 이동 평균 두 개를 관리한다. 일차 모멘트 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$(기울기의 평균으로 모멘텀을 준다)과 이차 모멘트 $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$(기울기의 분산으로 적응을 준다)이다. 갱신에는 두 모멘트를 0으로 초기화한 것을 메우려고 편향을 보정한 추정값 $\hat{m}_t$과 $\hat{v}_t$을 쓴다.

편향 보정은 각 모멘트를 $(1 - \beta^t)$으로 나누며, 이 값은 $t$이 커질수록 1에 가까워진다. 보정이 없으면 지수 이동 평균이 0에서 시작하여 참 통계에 이르기까지 여러 단계가 걸리므로 초반 추정값이 0 쪽으로 치우친다.

기본 초매개변수($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$)는 거의 모든 딥러닝 과제에서 잘 통한다. 문제마다 조율해야 하는 것은 보통 학습률(기본값 0.001)뿐이다.

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

**다룬 것** — Adam 최적화기

Adam은 지수 이동 평균 두 개를 관리한다.

핵심 클래스는 `Adam`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
