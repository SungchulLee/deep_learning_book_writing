"""adam_optimizer — adam_optimizer 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch07/optimizers/adam_optimizer.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

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
