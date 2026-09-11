"""rmsprop_optimizer — rmsprop_optimizer 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch07/optimizers/rmsprop_optimizer.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

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
