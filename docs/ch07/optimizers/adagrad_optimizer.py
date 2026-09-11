"""adagrad_optimizer — adagrad_optimizer 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch07/optimizers/adagrad_optimizer.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

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
            # 아직 cache가 차지 않았다. 아래 갈래와 모양을 맞추어 배열로 둔다.
            effective_lr = np.full_like(param, optimizer.learning_rate)
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
