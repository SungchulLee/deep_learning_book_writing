# Adam 실습

NumPy 배열의 사전을 사용해 Adam의 두 모멘트와 편향 보정을 구현한다. 이차 함수의 기울기를 직접 계산하므로 자동 미분이나 신경망 학습 루프 없이 상태 변화를 읽을 수 있다.

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

일차 모멘트 $m_t$는 기울기의 지수 이동평균이고, 이차 모멘트 $v_t$는 기울기 제곱의 지수 이동평균이다. $v_t$는 평균을 빼지 않은 이차 모멘트이므로 코드의 일부 주석처럼 분산과 동일시하면 안 된다. 갱신은 $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$를 사용한다.

0 초기화에 대한 보정은 일정한 기대 모멘트를 가정했을 때 초반 추정치의 축소를 보완한다. 기울기 분포가 시간에 따라 바뀌는 실제 최적화에서 모든 통계적 편향을 없앤다는 뜻은 아니다. learning_rate와 beta, epsilon은 문제의 스케일과 잡음에 영향을 받으므로 기본값의 보편적인 최적성을 가정하지 않는다.

`update`는 호출마다 하나의 t를 증가시키고 모든 키를 갱신한다. 키별로 m과 v를 유지하지만 지연되어 새로 추가된 키도 공통 t로 보정한다. 이 구현은 처음부터 같은 키 집합을 유지하는 예제에 맞춰져 있다. 원래 params 사전의 값을 직접 덮어쓰는 대신 새 배열을 담은 사전을 반환하므로 호출자는 반환값을 받아야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
상수 기울기 g에서 첫 단계의 보정된 모멘트를 구하라.

</div>

??? success "연습문제 1 풀이"
    $m_1=(1-\beta_1)g$, $v_1=(1-\beta_2)g^2$이므로 $\hat m_1=g$, $\hat v_1=g^2$이다. 첫 갱신량은 $\eta g/(|g|+\epsilon)$이다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
v가 기울기의 분산이 아닌 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    분산은 $E[g^2]-E[g]^2$인데 이 구현의 v는 제곱의 이동평균만 유지한다. 기울기가 일정하고 0이 아니어도 v는 양수인 값에 접근하지만 분산은 0이다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
Adam 학습을 정확히 이어가려면 매개변수 외에 무엇을 저장해야 하는가?

</div>

??? success "연습문제 3 풀이"
    m, v, t와 learning_rate, beta1, beta2, epsilon을 함께 기록한다. 이 예제에서는 배열의 키와 모양도 유지해야 한다. 매개변수만 복원하고 모멘트를 초기화하면 이후 갱신 경로가 달라진다.

## 정리하며

Adam의 핵심은 두 지수 이동평균과 0 초기화 보정이다. 구현을 읽을 때 이차 모멘트와 분산을 구별하고, 상태를 포함한 복원과 반환값의 사용 방식을 확인한다.
