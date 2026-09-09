# 날씨 모형

weather_model.py (모듈 07) 마르코프 사슬로 날씨 본뜨기

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 1. 코드

```python
"""
weather_model.py (단원 07)

마르코프 사슬로 날씨 본뜨기
====================================

Location: 06_markov_chain/03_applications/
난이도: ⭐⭐ 기초
예상 시간: 2~3시간

학습 목표:
- 마르코프 사슬로 실제 현상 본뜨기
- 자료로 옮김 행렬 어림하기
- 날씨 미리보기
- 오래 뒤의 날씨 무늬 살피기

수학적 바탕:
다음을 놓으면 날씨를 마르코프 사슬로 본뜰 수 있다:
- 내일 날씨는 오늘 날씨에만 달렸다
- 옮김 확률이 시간에 고르다(일정하다)

이는 간추린 것이지만 무늬를 이해하는 데 쓸모 있다.

쓰임새:
지난 날씨 자료가 있으면 다음을 할 수 있다:
1. 옮김 확률 어림하기
2. 앞으로의 날씨 미리보기
3. 날씨 갈래별로 오래 뒤의 잦기 셈하기
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ========================================================================
# 메인
# ========================================================================


class WeatherMarkovChain:
    """
    날씨 미리보기를 위한 마르코프 사슬 모형.
    
    상태에는 보통 맑음, 흐림, 비 따위가 들어간다.
    """
    
    def __init__(self, states):
        """
        날씨 모형 첫값 잡기.
        
        매개변수:
            states (list): 날씨 상태의 이름(이를테면 ['Sunny', 'Rainy'])
        """
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        self.transition_matrix = None
    
    def estimate_from_data(self, weather_sequence):
        """
        관측한 날씨 자료로 옮김 행렬 어림하기.
        
        매개변수:
            weather_sequence (list): 관측한 날씨 상태의 늘어놓음
        
        반환값:
            np.ndarray: 어림한 옮김 행렬
        
        수학 방법:
        최대 가능도 어림(MLE):
        P̂[i][j] = (i에서 j로 간 옮김의 수) / (상태 i에 머문 횟수)
        
        이것이 잦기 어림꼴이다:
        P̂[i][j] = N_{ij} / Σ_k N_{ik}
        여기서 N_{ij} = 관측한 옮김 i → j의 횟수
        """
        # 옮김 세기
        # transition_counts[i][j] = 상태 i에서 상태 j으로 간 옮김의 수
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for t in range(len(weather_sequence) - 1):
            current_state = weather_sequence[t]
            next_state = weather_sequence[t + 1]
            
            # 색인으로 바꾸기
            i = self.state_to_idx[current_state]
            j = self.state_to_idx[next_state]
            
            transition_counts[i, j] += 1
        
        # 확률을 얻으려고 고르게 하기
        # 행마다 합이 1
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        
        # 한 번도 나오지 않은 상태 다루기(0으로 나누기 피하기)
        row_sums[row_sums == 0] = 1
        
        self.transition_matrix = transition_counts / row_sums
        
        return self.transition_matrix
    
    def predict_next_day(self, current_weather):
        """
        내일 날씨 미리보기(확률로).
        
        매개변수:
            current_weather (str): 오늘의 날씨
        
        반환값:
            dict: 내일 날씨의 확률 분포
        
        수학의 바탕:
        P(X_{t+1} = j | X_t = i) = P[i][j]
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition matrix first")
        
        i = self.state_to_idx[current_weather]
        probabilities = self.transition_matrix[i, :]
        
        # 사전으로 돌려주기
        return {state: prob for state, prob in zip(self.states, probabilities)}
    
    def predict_n_days(self, current_weather, n_days):
        """
        n일 뒤의 날씨 분포 미리보기.
        
        매개변수:
            current_weather (str): 지금의 날씨 상태
            n_days (int): 며칠 뒤인가
        
        반환값:
            dict: n일 뒤의 확률 분포
        
        수학의 바탕:
        P(X_{t+n} = j | X_t = i) = [P^n]_{i,j}
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition matrix first")
        
        # 첫 분포 만들기(100% current_weather)
        initial_dist = np.zeros(self.n_states)
        initial_dist[self.state_to_idx[current_weather]] = 1.0
        
        # P^n 곱하기
        P_n = np.linalg.matrix_power(self.transition_matrix, n_days)
        future_dist = initial_dist @ P_n
        
        return {state: prob for state, prob in zip(self.states, future_dist)}
    
    def simulate_weather(self, n_days, initial_weather):
        """
        n일 동안의 날씨 늘어놓음 흉내내기.
        
        매개변수:
            n_days (int): 흉내낼 날의 수
            initial_weather (str): 시작 날씨
        
        반환값:
            list: 흉내낸 날씨 늘어놓음
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition matrix first")
        
        sequence = [initial_weather]
        current_idx = self.state_to_idx[initial_weather]
        
        for _ in range(n_days):
            # 옮김 확률에 따라 다음 상태 표집
            probs = self.transition_matrix[current_idx, :]
            next_idx = np.random.choice(self.n_states, p=probs)
            
            sequence.append(self.states[next_idx])
            current_idx = next_idx
        
        return sequence
    
    def stationary_distribution(self, method='eigenvector'):
        """
        멈춘 분포 셈하기.
        
        매개변수:
            method (str): 'eigenvector' 또는 'power'
        
        반환값:
            dict: 상태마다의 멈춘 확률
        
        수학적 바탕:
        멈춘 분포 π은 π = π × P을 만족한다
        곧 π은 고윳값이 1인 P의 왼쪽 고유벡터라는 뜻이다.
        
        물리로 풀이하기:
        날씨 상태마다 오래 보았을 때 머문 시간의 비율.
        """
        if self.transition_matrix is None:
            raise ValueError("Must estimate transition matrix first")
        
        if method == 'eigenvector':
            # 고윳값이 1인 왼쪽 고유벡터 찾기
            # P^T × v = 1 × v이므로 P^T의 고유벡터를 찾는다
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            
            # 고윳값 1에 딸린 고유벡터 찾기
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])
            
            # 합이 1이 되도록 고르게 하기
            stationary = stationary / stationary.sum()
        
        elif method == 'power':
            # 큰 n에 대해 P^n 셈하기
            P_n = np.linalg.matrix_power(self.transition_matrix, 1000)
            stationary = P_n[0, :]  # 아무 행이나 멈춘 분포를 준다
        
        return {state: prob for state, prob in zip(self.states, stationary)}


def example_simple_weather_model():
    """
    보기 1: 단순한 세 상태 날씨 모형.
    
    자료로 어림하고 미리보는 것을 보인다.
    """
    print("=" * 70)
    print("Example 1: Three-State Weather Model")
    print("=" * 70)
    
    # 관측한 날씨 자료(30일)
    observed_weather = [
        'Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy', 'Cloudy',
        'Sunny', 'Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy',
        'Cloudy', 'Cloudy', 'Sunny', 'Sunny', 'Sunny', 'Cloudy',
        'Cloudy', 'Rainy', 'Rainy', 'Rainy', 'Cloudy', 'Sunny',
        'Sunny', 'Cloudy', 'Rainy', 'Cloudy', 'Sunny', 'Sunny'
    ]
    
    print(f"\nObserved weather sequence ({len(observed_weather)} days):")
    print(observed_weather)
    
    # 잦기 세기
    counter = Counter(observed_weather)
    print(f"\nObserved frequencies:")
    for state, count in sorted(counter.items()):
        print(f"  {state}: {count}/{len(observed_weather)} = {count/len(observed_weather):.3f}")
    
    # 모형을 만들고 옮김 어림하기
    states = ['Sunny', 'Cloudy', 'Rainy']
    model = WeatherMarkovChain(states)
    P = model.estimate_from_data(observed_weather)
    
    print(f"\nEstimated Transition Matrix:")
    print(f"{'':10s} {'Sunny':>10s} {'Cloudy':>10s} {'Rainy':>10s}")
    for i, state in enumerate(states):
        row = " ".join(f"{P[i,j]:10.4f}" for j in range(len(states)))
        print(f"{state:10s} {row}")
    
    # 예측한다
    print(f"\n" + "-" * 70)
    print("Predictions if today is Sunny:")
    tomorrow = model.predict_next_day('Sunny')
    for state, prob in sorted(tomorrow.items()):
        print(f"  P(Tomorrow = {state} | Today = Sunny) = {prob:.4f}")
    
    print(f"\nPredictions 7 days ahead if today is Sunny:")
    week_ahead = model.predict_n_days('Sunny', 7)
    for state, prob in sorted(week_ahead.items()):
        print(f"  P(Day 7 = {state} | Today = Sunny) = {prob:.4f}")


def example_stationary_distribution():
    """
    보기 2: 멈춘 분포 셈하고 풀이하기.
    
    오래 뒤의 날씨 무늬를 보인다.
    """
    print("\n" + "=" * 70)
    print("Example 2: Stationary Distribution Analysis")
    print("=" * 70)
    
    # 미리 정한 옮김 행렬 쓰기
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = np.array([
        [0.7, 0.25, 0.05],   # 맑음에서
        [0.3, 0.4, 0.3],      # 흐림에서
        [0.2, 0.3, 0.5]       # 비에서
    ])
    
    print("\nTransition Matrix:")
    print(f"{'':10s} {'Sunny':>10s} {'Cloudy':>10s} {'Rainy':>10s}")
    for i, state in enumerate(states):
        row = " ".join(f"{P[i,j]:10.4f}" for j in range(len(states)))
        print(f"{state:10s} {row}")
    
    # 모델 생성
    model = WeatherMarkovChain(states)
    model.transition_matrix = P
    
    # 멈춘 분포 셈하기
    print("\n" + "-" * 70)
    print("Stationary Distribution (long-run frequencies):")
    
    # 방법 1: 고유벡터
    stationary_eig = model.stationary_distribution(method='eigenvector')
    print("\nUsing eigenvector method:")
    for state, prob in sorted(stationary_eig.items()):
        print(f"  π({state}) = {prob:.6f}")
    
    # 방법 2: 거듭제곱 되풀이
    stationary_pow = model.stationary_distribution(method='power')
    print("\nUsing matrix power method:")
    for state, prob in sorted(stationary_pow.items()):
        print(f"  π({state}) = {prob:.6f}")
    
    # 흉내내기로 확인하기
    print("\n" + "-" * 70)
    print("Verification via simulation (10,000 days):")
    
    long_sim = model.simulate_weather(10000, initial_weather='Sunny')
    simulated_freq = Counter(long_sim)
    
    print("\nSimulated frequencies:")
    for state in sorted(states):
        freq = simulated_freq[state] / len(long_sim)
        theoretical = stationary_eig[state]
        print(f"  {state}: {freq:.6f} (theoretical: {theoretical:.6f})")


def example_seasonal_weather():
    """
    보기 3: 인공 계절 날씨 자료 만들기.
    
    옮김 확률이 다른 계절을 본뜬다.
    """
    print("\n" + "=" * 70)
    print("Example 3: Seasonal Weather Patterns")
    print("=" * 70)
    
    states = ['Sunny', 'Cloudy', 'Rainy']
    
    # 여름 옮김 행렬(맑은 날이 더 많음)
    P_summer = np.array([
        [0.8, 0.15, 0.05],
        [0.5, 0.3, 0.2],
        [0.4, 0.4, 0.2]
    ])
    
    # 겨울 옮김 행렬(비 오는 날이 더 많음)
    P_winter = np.array([
        [0.5, 0.3, 0.2],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    
    print("\nSummer Transition Matrix:")
    print(P_summer)
    
    print("\nWinter Transition Matrix:")
    print(P_winter)
    
    # 여름 흉내내기
    model_summer = WeatherMarkovChain(states)
    model_summer.transition_matrix = P_summer
    summer_weather = model_summer.simulate_weather(90, 'Sunny')
    
    # 겨울 흉내내기
    model_winter = WeatherMarkovChain(states)
    model_winter.transition_matrix = P_winter
    winter_weather = model_winter.simulate_weather(90, 'Cloudy')
    
    # 잦기 견주기
    summer_freq = Counter(summer_weather)
    winter_freq = Counter(winter_weather)
    
    print("\n" + "-" * 70)
    print("Simulated 90-day frequencies:")
    print(f"{'State':<10s} {'Summer':<15s} {'Winter':<15s}")
    
    for state in states:
        s_freq = summer_freq[state] / len(summer_weather)
        w_freq = winter_freq[state] / len(winter_weather)
        print(f"{state:<10s} {s_freq:<15.4f} {w_freq:<15.4f}")


def visualize_weather_model():
    """
    날씨 모형의 그림 만들기.
    """
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    states = ['Sunny', 'Cloudy', 'Rainy']
    P = np.array([
        [0.7, 0.25, 0.05],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    
    model = WeatherMarkovChain(states)
    model.transition_matrix = P
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 작은 그림 1: 옮김 행렬 열지도
    ax = axes[0, 0]
    im = ax.imshow(P, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_xlabel('To State', fontsize=11)
    ax.set_ylabel('From State', fontsize=11)
    ax.set_title('Transition Probability Matrix', fontsize=12)
    
    # 글자 주석을 추가한다
    for i in range(len(states)):
        for j in range(len(states)):
            text = ax.text(j, i, f'{P[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax)
    
    # 작은 그림 2: 날씨 늘어놓음 표본
    ax = axes[0, 1]
    weather_seq = model.simulate_weather(60, 'Sunny')
    
    # 그리기 위해 수로 바꾸기
    state_to_num = {state: i for i, state in enumerate(states)}
    num_seq = [state_to_num[w] for w in weather_seq]
    
    ax.step(range(len(num_seq)), num_seq, where='post', linewidth=2)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states)
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Weather', fontsize=11)
    ax.set_title('Simulated 60-Day Weather Sequence', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 작은 그림 3: 오래 뒤 미리봄의 모임
    ax = axes[1, 0]
    
    days_ahead = range(1, 31)
    sunny_probs = []
    cloudy_probs = []
    rainy_probs = []
    
    for n in days_ahead:
        dist = model.predict_n_days('Sunny', n)
        sunny_probs.append(dist['Sunny'])
        cloudy_probs.append(dist['Cloudy'])
        rainy_probs.append(dist['Rainy'])
    
    ax.plot(days_ahead, sunny_probs, 'o-', label='Sunny', linewidth=2)
    ax.plot(days_ahead, cloudy_probs, 's-', label='Cloudy', linewidth=2)
    ax.plot(days_ahead, rainy_probs, '^-', label='Rainy', linewidth=2)
    
    # 멈춘 분포 선들 더하기
    stationary = model.stationary_distribution()
    ax.axhline(y=stationary['Sunny'], color='C0', linestyle='--', alpha=0.5)
    ax.axhline(y=stationary['Cloudy'], color='C1', linestyle='--', alpha=0.5)
    ax.axhline(y=stationary['Rainy'], color='C2', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Days Ahead', fontsize=11)
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('Prediction Convergence to Stationary Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 작은 그림 4: 멈춘 분포 막대그래프
    ax = axes[1, 1]
    stationary = model.stationary_distribution()
    
    colors = ['#FFD700', '#87CEEB', '#4682B4']
    bars = ax.bar(states, [stationary[s] for s in states], color=colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel('Long-run Frequency', fontsize=11)
    ax.set_title('Stationary Distribution', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 막대에 값 이름표를 추가한다
    for bar, state in zip(bars, states):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{stationary[state]:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/weather_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Weather model visualizations saved to weather_model.png")


def main():
    """
    날씨 본뜨기 보기 모두 돌리기.
    """
    print("WEATHER MODELING WITH MARKOV CHAINS")
    print("====================================\n")
    
    # 예제 실행
    example_simple_weather_model()
    example_stationary_distribution()
    example_seasonal_weather()
    
    # 시각화 만들기
    visualize_weather_model()
    
    print("\n" + "=" * 70)
    print("Practical Applications:")
    print("=" * 70)
    print("1. Short-term weather prediction (1-7 days)")
    print("2. Long-term climate pattern analysis")
    print("3. Agricultural planning")
    print("4. Event planning based on weather probabilities")
    print("5. Understanding stationary behavior of weather systems")


if __name__ == "__main__":
    main()
```

## 2. 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 확률 과정 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 죽 읽고 핵심 설계 결정을 가려내어라. 구체적인 구현 고름 셋을 적고 저마다 왜 마르코프 사슬에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
날씨 모형 구현을 확인하는 두루 갖춘 시험 함수를 적어라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단적인 값(0, 아주 큰 수)이 든 입력 같은 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_weathermarkovchain():
        model = WeatherMarkovChain(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 날씨 모형

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

고갱이 갈래는 `WeatherMarkovChain`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
