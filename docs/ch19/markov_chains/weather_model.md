# 날씨 모델

관측한 날씨 상태의 전이 도수로 1차 마르코프 모형을 추정한다. 상태별 다음 날 확률과 여러 날 뒤의 상태분포를 계산하고, 정상분포·계절별 전이 가정을 비교한다. 단순 상태 모형의 예측과 실제 기상 예보의 정확성은 구분해야 한다.

## 1. 코드

??? note "코드 (525줄)"

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

    # 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
    # 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
    np.random.seed(0)

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
        plt.savefig('weather_model.png', dpi=150, bbox_inches='tight')
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


??? note "전체 출력 (97줄)"

    ```
    WEATHER MODELING WITH MARKOV CHAINS
    ====================================

    ======================================================================
    Example 1: Three-State Weather Model
    ======================================================================

    Observed weather sequence (30 days):
    ['Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy', 'Cloudy', 'Sunny', 'Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Rainy', 'Cloudy', 'Cloudy', 'Sunny', 'Sunny', 'Sunny', 'Cloudy', 'Cloudy', 'Rainy', 'Rainy', 'Rainy', 'Cloudy', 'Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Cloudy', 'Sunny', 'Sunny']

    Observed frequencies:
      Cloudy: 10/30 = 0.333
      Rainy: 8/30 = 0.267
      Sunny: 12/30 = 0.400

    Estimated Transition Matrix:
                    Sunny     Cloudy      Rainy
    Sunny          0.6364     0.3636     0.0000
    Cloudy         0.4000     0.2000     0.4000
    Rainy          0.0000     0.5000     0.5000

    ----------------------------------------------------------------------
    Predictions if today is Sunny:
      P(Tomorrow = Cloudy | Today = Sunny) = 0.3636
      P(Tomorrow = Rainy | Today = Sunny) = 0.0000
      P(Tomorrow = Sunny | Today = Sunny) = 0.6364

    Predictions 7 days ahead if today is Sunny:
      P(Day 7 = Cloudy | Today = Sunny) = 0.3432
      P(Day 7 = Rainy | Today = Sunny) = 0.2672
      P(Day 7 = Sunny | Today = Sunny) = 0.3895

    ======================================================================
    Example 2: Stationary Distribution Analysis
    ======================================================================

    Transition Matrix:
                    Sunny     Cloudy      Rainy
    Sunny          0.7000     0.2500     0.0500
    Cloudy         0.3000     0.4000     0.3000
    Rainy          0.2000     0.3000     0.5000

    ----------------------------------------------------------------------
    Stationary Distribution (long-run frequencies):

    Using eigenvector method:
      π(Cloudy) = 0.307692
      π(Rainy) = 0.230769
      π(Sunny) = 0.461538

    Using matrix power method:
      π(Cloudy) = 0.307692
      π(Rainy) = 0.230769
      π(Sunny) = 0.461538

    ----------------------------------------------------------------------
    Verification via simulation (10,000 days):

    Simulated frequencies:
      Cloudy: 0.302470 (theoretical: 0.307692)
      Rainy: 0.221578 (theoretical: 0.230769)
      Sunny: 0.475952 (theoretical: 0.461538)

    ======================================================================
    Example 3: Seasonal Weather Patterns
    ======================================================================

    Summer Transition Matrix:
    [[0.8  0.15 0.05]
     [0.5  0.3  0.2 ]
     [0.4  0.4  0.2 ]]

    Winter Transition Matrix:
    [[0.5 0.3 0.2]
     [0.3 0.4 0.3]
     [0.2 0.3 0.5]]

    ----------------------------------------------------------------------
    Simulated 90-day frequencies:
    State      Summer          Winter         
    Sunny      0.6813          0.4066         
    Cloudy     0.2088          0.2747         
    Rainy      0.1099          0.3187         

    ======================================================================
    Creating Visualizations
    ======================================================================
    Weather model visualizations saved to weather_model.png

    ======================================================================
    Practical Applications:
    ======================================================================
    1. Short-term weather prediction (1-7 days)
    2. Long-term climate pattern analysis
    3. Agricultural planning
    4. Event planning based on weather probabilities
    5. Understanding stationary behavior of weather systems
    ```


## 2. 논의

### 전이 도수 추정

상태 i에서 j로 전이한 횟수를 N_ij라 하면 출발 도수 N_i>0인 행의 최대 우도 추정은 다음과 같다.

$$\hat P_{ij}=N_{ij}/\sum_jN_{ij}$$

관측 길이가 T이면 전이는 T−1개다. 상태별 전체 출현 횟수에는 마지막 날도 포함되지만 출발 전이 도수에는 포함되지 않는다. 코드가 0합 행의 분모를 1로 바꾸면 행 전체는 0으로 남으므로 유효한 확률행렬이 되지 않는다. 미관측 출발 상태에는 사전·평활화·추정 불가 표시 등 별도 규칙이 필요하다.

### 여러 단계 예측과 정상분포

현재 날씨를 한 상태로 고정하면 n일 뒤 상태분포는 해당 행벡터에 Pⁿ을 곱한 값이다. 이것은 향후 n일 전체 경로나 비가 한 번이라도 올 확률과 다르다. 양의 정수·0의 예측 기간, 고유한 상태명과 관측값의 유효성을 확인한다.

stationary_distribution의 거듭제곱 방식은 P¹⁰⁰⁰의 첫 행을 반환할 뿐 수렴을 확인하지 않는다. 기약·비주기 조건과 정상성 잔차를 점검해야 하며, 고유벡터 방식도 여러 정상분포나 수치 오차가 있으면 별도 검증이 필요하다. 지원하지 않는 method 이름은 처리하지 않는다.

### 계절성과 추정 불확실성

계절 예제는 여름·겨울에 별도로 가정한 전이행렬을 시뮬레이션한다. 연중 변하는 모형을 자료에서 공동 추정하거나 계절 경계를 연속적으로 연결하지 않는다. simulate_weather(90)는 초기 날을 포함해 91개 상태를 반환하므로 빈도의 분모와 기간을 맞춘다.

표본 전이를 대입한 예측은 전이행렬 추정 오차를 포함하지 않는다. 실제 적용에는 시간 순서에 맞춘 검증, 계절별 안정성, 상태 정의와 미관측 전이의 처리가 필요하다. 짧은 가상 목록과 미리 정한 행렬의 결과를 실제 날씨 성능으로 읽지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
날씨 관측 T개의 전이 도수와 상태 출현 도수는 어떻게 다른가?

</div>

??? success "연습문제 1 풀이"

    전이는 인접한 T−1쌍을 센다. 마지막 날은 관측 빈도에는 들어가지만 다음 날을 관측하지 않았으므로 출발 전이 도수에는 들어가지 않는다. 조건부 행확률의 분모는 전체 출현 횟수가 아니라 해당 상태에서 출발한 전이 수다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
출발 전이가 없는 상태의 행을 코드처럼 모두 0으로 남기면 어떤 문제가 생기는가?

</div>

??? success "연습문제 2 풀이"

    행합이 0이어서 확률행렬이 아니다. 샘플링이 실패하거나 행렬 예측의 총 확률이 1보다 작아질 수 있다. 상태 공간을 다시 정의하거나 명시한 사전·평활화를 적용해야 하며 분모만 1로 바꾸는 것으로 해결되지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
7일 뒤 비일 확률과 7일 동안 한 번 이상 비가 올 확률은 같은가?

</div>

??? success "연습문제 3 풀이"

    다르다. 전자는 e_iP⁷의 비 상태 성분이다. 후자는 비 상태를 피한 모든 경로의 확률을 1에서 빼거나 보조 상태를 사용해 계산한다. 또한 정상분포는 이 특정 유한 기간 경로 사건의 답이 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 신경망 시험 조각 대신 어떤 날씨 모형 조건을 점검해야 하는가?

</div>

??? success "연습문제 4 풀이"

    이 클래스에는 신경망 호출·경사 인터페이스가 없다. 짧은 알려진 관측열의 전이 도수, 모든 행의 비음성·합 1, 미관측 상태, n=0의 현재 상태 분포, 시뮬레이션의 n+1 길이와 지원하지 않는 method 처리를 점검해야 한다. 일반 시험 조각은 이 규약을 구현하지 않는다.

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


## 정리하며

전이 도수 기반 날씨 모형은 유효한 행확률과 시간 동질성 가정이 필요하다. 미관측 행·초기 상태·예측 사건을 명시하고 유한 단계 예측과 정상분포, 계절별 가상 실험과 실제 자료 검증을 구분한다.
