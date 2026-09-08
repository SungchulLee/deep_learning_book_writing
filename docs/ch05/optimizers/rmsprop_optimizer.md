# RMSprop 최적화기

RMSprop은 AdaGrad의 누적합 대신 기울기 제곱의 지수 이동 평균을 써서 학습률이 단조롭게 0으로 줄어드는 것을 막는다. 덕분에 비정상 최적화 문제에 효과적이며, 역사적으로 순환 신경망 학습에 즐겨 쓰인 최적화기였다.

## 1. 코드

```python
"""
RMSprop(제곱평균제곱근 전파) 최적화기
================================================

RMSprop은 지난 기울기 제곱을 모두 누적하는 대신 이동 평균을 써서 AdaGrad의
한계를 다루는 적응형 학습률 방법이다.


주요 기능:
- 기울기 제곱의 지수 이동 평균을 쓴다
- 학습률을 이 평균의 제곱근으로 나눈다
- (AdaGrad와 달리) 비정상 문제에서 잘 통한다
- (Adam과 달리) 편향 보정이 없다

만든 사람: 제프리 힌턴 (코세라 강의)
"""

import numpy as np

# ========================================================================
# 메인
# ========================================================================


class RMSprop:
    """
    RMSprop 최적화기의 구현.
    
    매개변수:
    -----------
    learning_rate : float, 기본값=0.001
        매개변수 갱신의 걸음 크기
    rho : float, 기본값=0.9
        기울기 제곱의 이동 평균에 대한 감쇠율
    epsilon : float, 기본값=1e-8
        수치 안정성을 위한 작은 상수
    """
    
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        
        # 상태 변수
        self.cache = {}  # 기울기 제곱의 이동 평균
    
    def update(self, params, grads):
        """
        RMSprop 알고리즘으로 매개변수를 갱신한다.
        
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
            
            # 기울기 제곱의 이동 평균 갱신
            self.cache[key] = self.rho * self.cache[key] + (1 - self.rho) * (grads[key] ** 2)
            
            # 매개변수 갱신
            # 학습률을 이동 평균의 제곱근으로 나눈다
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
    
    # 매개변수를 초기화한다
    params = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    optimizer = RMSprop(learning_rate=0.1)
    
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


def compare_with_without_rmsprop():
    """
    RMSprop 적응이 있을 때와 없을 때의 경사 하강법을 견준다.
    RMSprop이 서로 다른 기울기 척도를 다루는 모습을 보인다.
    """
    print("=" * 60)
    print("RMSprop vs Standard Gradient Descent")
    print("=" * 60)
    print("Minimizing f(x, y) = 100*x^2 + y^2 (ill-conditioned)")
    print()
    
    # 매개변수를 초기화한다
    params_rmsprop = {'x': np.array([10.0]), 'y': np.array([10.0])}
    params_sgd = {'x': np.array([10.0]), 'y': np.array([10.0])}
    
    # 최적화기 초기화
    optimizer_rmsprop = RMSprop(learning_rate=0.1)
    lr_sgd = 0.001  # 조건이 나쁜 문제에서 SGD는 훨씬 작은 학습률이 필요하다
    
    print(f"{'Iteration':<12} {'RMSprop f(x,y)':<20} {'SGD f(x,y)':<20}")
    print("-" * 60)
    
    for i in range(100):
        # 기울기 계산: df/dx = 200x, df/dy = 2y
        grads_rmsprop = {
            'x': 200 * params_rmsprop['x'],
            'y': 2 * params_rmsprop['y']
        }
        grads_sgd = {
            'x': 200 * params_sgd['x'],
            'y': 2 * params_sgd['y']
        }
        
        # 매개변수 갱신
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

## 2. 논의

AdaGrad에 견준 RMSprop의 핵심 혁신은 기울기 제곱의 누적에 지수 이동 평균 $v_t = \rho v_{t-1} + (1-\rho) g_t^2$을 쓴다는 점이다. $\rho=0.9$이면 최근 기울기가 오래된 것보다 많이 기여하므로 분모가 한없이 커지지 않는다.

조건이 나쁜 문제 $f(x,y) = 100x^2 + y^2$에서 기본 SGD와 견주어 보면 RMSprop의 이점이 드러난다. SGD는 가파른 $x$ 방향에서 발산하지 않으려면 아주 작은 학습률(0.001)을 써야 하는데, 그러면 완만한 $y$ 방향에서 견딜 수 없이 느려진다. RMSprop은 같은 기본 학습률을 쓰되 매개변수마다 기울기의 크기에 따라 갱신의 배율을 저절로 맞춘다.

제프리 힌턴이 코세라 강의에서 발표하지 않은 채로 RMSprop을 제안했으므로, 정식 논문 없이 널리 쓰이게 된 드문 알고리즘 가운데 하나이다. Adam은 RMSprop에 모멘텀과 편향 보정을 더한 것으로 볼 수 있다.

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

**다룬 것** — RMSprop 최적화기

AdaGrad에 견준 RMSprop의 핵심 혁신은 기울기 제곱의 누적에 지수 이동 평균 $v_t = \rho v_{t-1} + (1-\rho) g_t^2$을 쓴다는 점이다.

핵심 클래스는 `RMSprop`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
