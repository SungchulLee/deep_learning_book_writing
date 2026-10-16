# 수렴 정리

유한 마르코프 사슬의 구조와 수렴 속도를 행렬로 분석한다. 기약성·비주기성을 확인하고 정상분포까지의 총변동 거리와 혼합 시간을 계산한다. 코드의 간단한 구조 진단과 스펙트럼 근사가 각각 어떤 조건에서 유효한지 구분한다.

## 1. 코드

??? note "코드 (718줄)"

    ```python
    """
    ================================================================================
    모임 정리와 섞임 시간
    ================================================================================

    난이도: 중급-상급
    걸리는 시간: 90-120분
    먼저 볼 것: 파일 01-03, 극한과 모임에 대한 이해

    학습 목표:
    1. 줄일 수 없음과 주기 없음의 조건 이해하기
    2. 모임 정리를 서술하고 증명하기
    3. 섞임 시간 정하고 셈하기
    4. 고윳값으로 모이는 빠르기 살피기
    5. MCMC로 잇는 다리: 섞임 시간이 왜 중요한가

    수학의 바탕:
    ==========================

    정의:

    1. 줄일 수 없음:
       어느 상태에서든 다른 모든 상태에 닿을 수 있으면 그 사슬은 줄일 수 없다고 한다.
       엄밀히: 모든 i,j에 대해 P^n[i,j] > 0인 n > 0이 있다.

       직관: 상태 공간 전체가 "이어져" 있고 동떨어진 부분이 없다.

    2. 주기 없음:
       상태 i의 주기는 d = gcd{n: P^n[i,i] > 0}이다.
       d = 1이면 상태 i은 주기가 없다.

       직관: 사슬이 정해진 순환에 "갇히지" 않는다.

    3. 에르고드성:
       사슬이 줄일 수 없고 주기가 없으면 에르고드적이라 한다.

       이것이 모임에 필요한 바로 그 조건이다!

    근본 모임 정리:
    --------------------------------
    P을 에르고드 마르코프 사슬의 옮김 행렬이라 하자. 그러면:

    1. 멈춘 분포 π이 오직 하나 있다
    2. 모든 첫 상태 i에 대해:
           모든 j에 대해 lim_{n→∞} P^n[i,j] = π[j]

    3. 그 모임은 지수꼴이다:
           |P^n[i,j] - π[j]| ≤ C · ρ^n
       여기서 ρ < 1은 둘째로 큰 고윳값이다

    섞임 시간:
    ------------
    섞임 시간은 사슬이 멈춘 분포에 "가까워질" 때까지의 시간이다
    (멈춘 분포까지의 시간이다).

    정의(ε 섞임 시간):
    τ_mix(ε) = min{n : ||P^n(i,·) - π||_{TV} ≤ ε for all i}

    여기서 ||·||_{TV}은 총변동 거리이다:
    ||μ - ν||_{TV} = (1/2) Σ_j |μ[j] - ν[j]|

    흔한 고름: ε = 1/4(그러면 τ_mix ≡ τ_mix(1/4))

    MCMC에서 왜 중요한가:
    - π의 어림 표본을 얻으려면 사슬을 τ_mix스텝쯤 돌려야 한다
    - 섞임 시간이 길수록 → MCMC 표집이 느리다
    - 섞임을 이해하는 것이 실전 MCMC에 결정적이다!

    스펙트럼 틈:
    -------------
    스펙트럼 틈은 다음과 같다:
       γ = 1 - |λ_2|
    여기서 λ_2은 P의 둘째로 큰 고윳값이다.

    정리: τ_mix ≈ O(1/γ)(스펙트럼 틈에 반비례한다)

    틈이 클수록 → 빨리 섞인다!

    ================================================================================
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import linalg
    import pandas as pd
    from typing import List, Tuple, Dict
    import warnings
    import os

    # ========================================================================
    # 메인
    # ========================================================================
    warnings.filterwarnings('ignore')


    # 그 자리 실행을 위한 내임 디렉터리 차리기
    OUTPUT_DIR = './outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    np.random.seed(42)
    plt.style.use('seaborn-v0_8-darkgrid')


    ################################################################################
    # 1절: 에르고드 조건 살피기
    ################################################################################

    def check_irreducibility(P: np.ndarray, verbose: bool = True) -> bool:
        """
        마르코프 사슬이 줄일 수 없는지 살피기.

        알고리즘:
        1. n을 키워 가며 P^n 셈하기
        2. 성분이 끝내 모두 양수가 되는지 살피기
        3. 그럴듯한 시간 안에 그렇다면 → 줄일 수 없다

        실전 살피기:
        n = n_states에 대해 P + P^2 + ... + P^n 셈하기
        성분이 모두 > 0이면 사슬은 줄일 수 없다.

        까닭: i에서 j으로 가는 길이 있으면 n스텝 안에 나타난다.

        파라메터:
        -----------
        P : np.ndarray
            옮김 행렬
        verbose : bool
            진단 정보 찍기

        반환값:
        --------
        is_irreducible : bool
        """

        n_states = P.shape[0]

        # 거듭제곱의 합 셈하기: Σ_{k=1}^n P^k
        P_sum = np.zeros_like(P)
        P_k = P.copy()

        for k in range(1, n_states + 1):
            P_sum += P_k
            P_k = P_k @ P

        # 성분이 모두 양수인지 살피기
        is_irreducible = np.all(P_sum > 0)

        if verbose:
            print("\nChecking Irreducibility:")
            print("-" * 60)
            print(f"States can reach each other: {is_irreducible}")

            if not is_irreducible:
                # 닿을 수 없는 짝 찾기
                unreachable = np.argwhere(P_sum == 0)
                print(f"\nUnreachable state pairs:")
                for i, j in unreachable[:5]:  # 처음 5개 보이기
                    print(f"  State {i} → State {j}")
                if len(unreachable) > 5:
                    print(f"  ... and {len(unreachable) - 5} more")

        return is_irreducible


    def check_aperiodicity(P: np.ndarray, verbose: bool = True) -> bool:
        """
        마르코프 사슬이 주기가 없는지 살피기.

        실전 살피기:
        주기가 없기에 넉넉한 조건: 적어도 하나의 i에 대해 P[i,i] > 0.
        (어느 상태든 자기 고리가 있으면 사슬은 주기가 없다)

        엄밀한 살피기:
        상태마다 {n: P^n[i,i] > 0}의 최대공약수 셈하기.
        최대공약수가 모두 1이면 사슬은 주기가 없다.

        간단히 하려고 넉넉한 조건을 쓴다.

        파라메터:
        -----------
        P : np.ndarray
            옮김 행렬
        verbose : bool
            진단 정보 찍기

        반환값:
        --------
        is_aperiodic : bool
        """

        # 대각 성분 살피기
        has_self_loop = np.any(np.diag(P) > 0)

        if verbose:
            print("\nChecking Aperiodicity:")
            print("-" * 60)

            if has_self_loop:
                self_loop_states = np.where(np.diag(P) > 0)[0]
                print(f"Has self-loops at states: {self_loop_states}")
                print(f"✓ Chain is APERIODIC (sufficient condition)")
            else:
                print("No self-loops detected")
                print("Need more sophisticated check for periodicity...")
                # 온전함을 위해 여기서 최대공약수 살피기를 넣을 수도 있다
                print("(Assuming aperiodic for now - more rigorous check omitted)")

        return has_self_loop  # 간추린 살피기


    def check_ergodicity(P: np.ndarray, verbose: bool = True) -> Dict[str, bool]:
        """
        마르코프 사슬이 에르고드적인지 살피기(줄일 수 없음 + 주기 없음).

        에르고드성이 모임 정리의 핵심 조건이다!

        파라메터:
        -----------
        P : np.ndarray
            옮김 행렬
        verbose : bool
            진단 정보 찍기

        반환값:
        --------
        results : dict
            'irreducible', 'aperiodic', 'ergodic' 열쇠를 갖는 사전
        """

        if verbose:
            print("\n" + "=" * 80)
            print("ERGODICITY CHECK")
            print("=" * 80)

        is_irr = check_irreducibility(P, verbose=verbose)
        is_aper = check_aperiodicity(P, verbose=verbose)

        is_erg = is_irr and is_aper

        if verbose:
            print("\n" + "-" * 60)
            print("SUMMARY:")
            print(f"  Irreducible: {is_irr}")
            print(f"  Aperiodic:   {is_aper}")
            print(f"  ✓ ERGODIC:   {is_erg}")
            print("-" * 60)

        return {
            'irreducible': is_irr,
            'aperiodic': is_aper,
            'ergodic': is_erg
        }


    ################################################################################
    # 2절: 섞임 시간 살피기
    ################################################################################

    def compute_total_variation_distance(pi1: np.ndarray, pi2: np.ndarray) -> float:
        """
        두 분포 사이의 총변동 거리 셈하기.

        정의:
        ||μ - ν||_{TV} = (1/2) Σ_i |μ[i] - ν[i]|

        풀이:
        - 아무 사건에 대한 확률 차이의 최댓값
        - 0(같음)에서 1(받침이 겹치지 않음)까지이다
        - 확률 분포를 견주는 표준 잣대

        파라메터:
        -----------
        pi1, pi2 : np.ndarray
            확률 분포

        반환값:
        --------
        tv_distance : float
            총변동 거리
        """
        return 0.5 * np.sum(np.abs(pi1 - pi2))


    def compute_mixing_time(P: np.ndarray,
                           pi_stationary: np.ndarray,
                           epsilon: float = 0.25,
                           max_steps: int = 10000) -> Tuple[int, np.ndarray]:
        """
        마르코프 사슬의 ε 섞임 시간 셈하기.

        정의:
        τ_mix(ε) = min{n : 모든 시작 상태 i에 대해 ||P^n(i,·) - π||_{TV} ≤ ε}

        알고리즘:
        1. n = 1, 2, 3, ...에 대해
        2. P^n 셈하기
        3. 시작 상태 i마다 π까지의 TV 거리 셈하기
        4. 모든 i에 대해 최대 TV 거리가 ε 이하이면 n 돌려주기

        파라메터:
        -----------
        P : np.ndarray
            옮김 행렬
        pi_stationary : np.ndarray
            멈춘 분포
        epsilon : float
            너그러움 임계값(표준: 0.25 또는 0.01)
        max_steps : int
            살필 최대 스텝 수

        반환값:
        --------
        mixing_time : int
            ε 섞임에 이르는 스텝 수
        tv_distances : np.ndarray
            스텝마다의 최대 TV 거리
        """

        n_states = P.shape[0]
        tv_distances = []

        P_n = P.copy()

        for n in range(1, max_steps + 1):
            # 모든 시작 상태에 걸친 최대 TV 거리 셈하기
            max_tv = 0.0
            for i in range(n_states):
                # 상태 i에서 시작해 n스텝 뒤의 분포
                pi_n = P_n[i, :]
                tv = compute_total_variation_distance(pi_n, pi_stationary)
                max_tv = max(max_tv, tv)

            tv_distances.append(max_tv)

            # 섞였는지 살피기
            if max_tv <= epsilon:
                return n, np.array(tv_distances)

            # 다음 거듭제곱 셈하기
            P_n = P_n @ P

        warnings.warn(f"Did not reach ε={epsilon} mixing in {max_steps} steps")
        return max_steps, np.array(tv_distances)


    def compute_spectral_gap(P: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        옮김 행렬의 스펙트럼 틈 셈하기.

        정의:
        스펙트럼 틈 = 1 - |λ_2|
        여기서 λ_2은 (크기로) 둘째로 큰 고윳값이다

        풀이:
        - 틈이 클수록 → 빨리 모인다
        - 틈 = 0 → 모이지 않는다(이를테면 주기 사슬)
        - 이론의 섞임 시간: τ_mix ≈ O(1/gap)

        파라메터:
        -----------
        P : np.ndarray
            옮김 행렬

        반환값:
        --------
        gap : float
            스펙트럼 틈
        eigenvalues : np.ndarray
            모든 고윳값(크기로 정렬)
        """

        # 고윳값 셈하기
        eigenvalues = linalg.eigvals(P)

        # 크기(절댓값)로 정렬
        eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]

        # 가장 큰 값은 1이어야 한다(수치 오차 범위 안에서)
        lambda_1 = eigenvalues[0].real
        assert np.abs(lambda_1 - 1.0) < 1e-6, "Largest eigenvalue should be 1"

        # 크기로 둘째
        lambda_2 = np.abs(eigenvalues[1])

        # 스펙트럼 틈
        gap = 1.0 - lambda_2

        return gap, eigenvalues


    def analyze_mixing(P: np.ndarray,
                      pi_stationary: np.ndarray,
                      states: List[str]):
        """
        섞임 두루 살피기.

        셈하는 것:
        1. 스펙트럼 틈
        2. ε에 따른 섞임 시간
        3. 시작 상태에 따른 모임 빠르기
        """

        print("\n" + "=" * 80)
        print("MIXING TIME ANALYSIS")
        print("=" * 80)

        # 스펙트럼 틈
        gap, eigenvalues = compute_spectral_gap(P)

        print(f"\nSpectral Gap: {gap:.6f}")
        print(f"\nTop 5 eigenvalues (by magnitude):")
        for i, lam in enumerate(eigenvalues[:5]):
            print(f"  λ_{i+1} = {lam:.6f}")

        # 너그러움에 따른 섞임 시간
        print("\n" + "-" * 60)
        print("MIXING TIMES:")
        print("-" * 60)

        epsilons = [0.5, 0.25, 0.1, 0.01]
        mixing_times = {}

        for eps in epsilons:
            t_mix, _ = compute_mixing_time(P, pi_stationary, epsilon=eps, max_steps=1000)
            mixing_times[eps] = t_mix
            print(f"τ_mix({eps:.2f}) = {t_mix:4d} steps")

        # 스펙트럼 틈에서 나온 이론의 미리봄
        if gap > 0:
            t_mix_theory = np.log(1/0.25) / gap  # 거친 어림값
            print(f"\nTheoretical estimate (ε=0.25): ≈ {t_mix_theory:.1f} steps")
            print(f"Actual: {mixing_times[0.25]} steps")

        return mixing_times, gap, eigenvalues


    ################################################################################
    # 3절: 모임 그려 보기
    ################################################################################

    def visualize_convergence_rate(P: np.ndarray,
                                   pi_stationary: np.ndarray,
                                   states: List[str],
                                   max_steps: int = 100):
        """
        멈춘 분포로 모이는 것 그려 보기.

        보이는 것:
        1. 시간에 따른 TV 거리(로그 눈금)
        2. 성분별 모임
        3. 지수 사그라듦 빠르기와 견주기
        """

        n_states = len(states)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # TV 거리와 성분별 오차 셈하기
        tv_distances_by_start = np.zeros((n_states, max_steps))

        for start_idx in range(n_states):
            pi_t = np.zeros(n_states)
            pi_t[start_idx] = 1.0

            for t in range(max_steps):
                pi_t = pi_t @ P
                tv = compute_total_variation_distance(pi_t, pi_stationary)
                tv_distances_by_start[start_idx, t] = tv

        # 그림 1: TV 거리(로그 눈금)
        ax1 = axes[0, 0]
        for start_idx, state_name in enumerate(states):
            ax1.semilogy(range(max_steps), tv_distances_by_start[start_idx, :],
                        marker='o', markersize=3, label=f'Start: {state_name}',
                        linewidth=2, alpha=0.7)

        # 이론의 지수 사그라듦 더하기
        gap, _ = compute_spectral_gap(P)
        if gap > 0:
            theoretical_decay = np.exp(-gap * np.arange(max_steps))
            ax1.semilogy(range(max_steps), theoretical_decay,
                        'k--', linewidth=2, label='Theoretical exp(-γt)', alpha=0.5)

        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Total Variation Distance', fontsize=12)
        ax1.set_title('Convergence Rate (Log Scale)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 그림 2: 최대 TV 거리
        ax2 = axes[0, 1]
        max_tv = np.max(tv_distances_by_start, axis=0)
        ax2.semilogy(range(max_steps), max_tv, 'b-', linewidth=3, label='Max over starts')

        # 섞임 시간 표시하기
        for eps in [0.5, 0.25, 0.1]:
            idx = np.where(max_tv <= eps)[0]
            if len(idx) > 0:
                t_mix = idx[0]
                ax2.axvline(x=t_mix, color='r', linestyle='--', alpha=0.5)
                ax2.text(t_mix, 0.5, f'ε={eps}', rotation=90, va='bottom')

        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Max TV Distance', fontsize=12)
        ax2.set_title('Worst-Case Convergence', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 그림 3: 성분별 모임(한 시작 상태에서)
        ax3 = axes[1, 0]
        start_idx = 0
        pi_t = np.zeros(n_states)
        pi_t[start_idx] = 1.0

        component_evolution = np.zeros((max_steps, n_states))
        for t in range(max_steps):
            pi_t = pi_t @ P
            component_evolution[t, :] = pi_t

        for j in range(n_states):
            ax3.plot(range(max_steps), component_evolution[:, j],
                    marker='o', markersize=3, label=f'π[{states[j]}]',
                    linewidth=2, alpha=0.7)
            ax3.axhline(y=pi_stationary[j], color='gray', linestyle='--', alpha=0.3)

        ax3.set_xlabel('Time Step', fontsize=12)
        ax3.set_ylabel('Probability', fontsize=12)
        ax3.set_title(f'Component Convergence (start: {states[start_idx]})',
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 그림 4: 고윳값 스펙트럼
        ax4 = axes[1, 1]
        gap, eigenvalues = compute_spectral_gap(P)

        # 복소평면에 그리기
        for i, lam in enumerate(eigenvalues):
            if i == 0:
                ax4.plot(lam.real, lam.imag, 'ro', markersize=15, 
                        label='λ₁ = 1', zorder=3)
            else:
                ax4.plot(lam.real, lam.imag, 'bo', markersize=10, alpha=0.7)

        # 단위원 그리기
        theta = np.linspace(0, 2*np.pi, 100)
        ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')

        # 둘째 고윳값 도드라지게 하기
        lam_2 = eigenvalues[1]
        ax4.plot(lam_2.real, lam_2.imag, 'go', markersize=15,
                label=f'λ₂ = {lam_2:.3f}', zorder=3)

        ax4.set_xlabel('Real', fontsize=12)
        ax4.set_ylabel('Imaginary', fontsize=12)
        ax4.set_title(f'Eigenvalue Spectrum (gap = {gap:.4f})',
                     fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')

        plt.tight_layout()
        return fig


    ################################################################################
    # 4절: 보기
    ################################################################################

    def example_fast_mixing_chain():
        """
        보기: 빨리 섞이는 사슬(스펙트럼 틈이 큼).
        """

        print("\n" + "=" * 80)
        print("EXAMPLE 1: FAST MIXING CHAIN")
        print("=" * 80)
        print("\nChain with strong connectivity and self-loops")

        # 자기 고리 확률이 크고 강하게 이어짐
        P = np.array([[0.8, 0.1, 0.1],
                      [0.1, 0.8, 0.1],
                      [0.1, 0.1, 0.8]])

        states = ['A', 'B', 'C']

        # 에르고드성 살피기
        check_ergodicity(P)

        # 멈춘 분포 찾기
        from scipy.linalg import eig
        eigenvalues, eigenvectors = eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = eigenvectors[:, idx].real
        pi = pi / pi.sum()

        print(f"\nStationary distribution: {dict(zip(states, np.round(pi, 4)))}")

        # 섞임 살피기
        mixing_times, gap, _ = analyze_mixing(P, pi, states)

        return P, pi, states


    def example_slow_mixing_chain():
        """
        보기: 느리게 섞이는 사슬(스펙트럼 틈이 작음).
        """

        print("\n\n" + "=" * 80)
        print("EXAMPLE 2: SLOW MIXING CHAIN")
        print("=" * 80)
        print("\nChain with weak connectivity (bottleneck structure)")

        # 약하게 이어진 두 무리
        P = np.array([[0.45, 0.45, 0.1],
                      [0.45, 0.45, 0.1],
                      [0.1,  0.1,  0.8]])

        states = ['A₁', 'A₂', 'B']

        # 에르고드성 살피기
        check_ergodicity(P)

        # 멈춘 분포 찾기
        from scipy.linalg import eig
        eigenvalues, eigenvectors = eig(P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = eigenvectors[:, idx].real
        pi = pi / pi.sum()

        print(f"\nStationary distribution: {dict(zip(states, np.round(pi, 4)))}")

        # 섞임 살피기
        mixing_times, gap, _ = analyze_mixing(P, pi, states)

        print("\n" + "-" * 60)
        print("INTERPRETATION:")
        print("Bottleneck between {A₁,A₂} and {B} slows mixing!")
        print("-" * 60)

        return P, pi, states


    ################################################################################
    # 주된 보임
    ################################################################################

    if __name__ == "__main__":
        print("""
        ╔════════════════════════════════════════════════════════════════════════╗
        ║             모임 정리와 섞임 시간                                      ║
        ║                    교육 단원 04                                        ║
        ║                난이도: 중급-상급                                       ║
        ╚════════════════════════════════════════════════════════════════════════╝
        """)

        # 보기 1: 빠른 섞임
        P_fast, pi_fast, states_fast = example_fast_mixing_chain()

        fig1 = visualize_convergence_rate(P_fast, pi_fast, states_fast, max_steps=50)
        plt.savefig(f'{OUTPUT_DIR}/04_fast_mixing_convergence.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

        # 보기 2: 느린 섞임
        P_slow, pi_slow, states_slow = example_slow_mixing_chain()

        fig2 = visualize_convergence_rate(P_slow, pi_slow, states_slow, max_steps=200)
        plt.savefig(f'{OUTPUT_DIR}/04_slow_mixing_convergence.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

        # 요약
        print("\n\n" + "=" * 80)
        print("SUMMARY: KEY TAKEAWAYS")
        print("=" * 80)
        print("""
        1. 에르고드성 = 줄일 수 없음 + 주기 없음
           → 오직 하나인 π으로 모이는 데 필요하다
           → 그래프의 짜임을 살펴 확인한다

        2. 섞임 시간 τ_mix(ε):
           → 멈춘 분포의 ε 안으로 들어가는 데 걸리는 시간
           → 스펙트럼 틈에 달렸다: τ_mix ≈ O(1/gap)
           → MCMC의 효율에 결정적이다!

        3. 스펙트럼 틈 γ = 1 - |λ_2|:
           → 틈이 클수록 → 빨리 섞인다
           → 상태 공간의 "이어짐"과 관계있다
           → 병목이 섞임을 느리게 한다

        4. 모이는 빠르기:
           → 지수꼴: ||P^n(i,·) - π||_{TV} ≤ C·ρⁿ
           → 빠르기 ρ = |λ_2| < 1
           → 그려 보고 경험으로 잴 수 있다

        5. 이것이 MCMC에 왜 중요한가:
           → 표집하기 전에 사슬을 τ_mix스텝쯤 돌려야 한다
           → 느린 섞임 → 쓸 수 없는 MCMC
           → 다음 단원: 마르코프 사슬로 아무 분포에서나 표집하기!

        ✓ 파일을 {os.path.abspath(OUTPUT_DIR)}/에 저장했다

        다음 스텝(파일 05): 표집 쓰임새 - MCMC로 잇는 다리!

        다음 단원의 결정적인 물음:
        "우리가 표집하려는 타깃 분포 π(x)이 주어졌을 때,
         멈춘 분포가 π인 마르코프 사슬을 우리가 지어낼 수 있는가?
         그렇다면 얼마나 빨리 섞이는가?"

        이것이 MCMC 방법의 핵심 물음이다!
        """)

        print("\n" + "=" * 80)
        print("END OF MODULE 04")
        print("=" * 80)
    ```


??? note "전체 출력 (159줄)"

    ```
    Output directory: /Users/sungchul/Documents/book/deep_learning_in_korean/outputs

        ╔════════════════════════════════════════════════════════════════════════╗
        ║             모임 정리와 섞임 시간                                      ║
        ║                    교육 단원 04                                        ║
        ║                난이도: 중급-상급                                       ║
        ╚════════════════════════════════════════════════════════════════════════╝
        

    ================================================================================
    EXAMPLE 1: FAST MIXING CHAIN
    ================================================================================

    Chain with strong connectivity and self-loops

    ================================================================================
    ERGODICITY CHECK
    ================================================================================

    Checking Irreducibility:
    ------------------------------------------------------------
    States can reach each other: True

    Checking Aperiodicity:
    ------------------------------------------------------------
    Has self-loops at states: [0 1 2]
    ✓ Chain is APERIODIC (sufficient condition)

    ------------------------------------------------------------
    SUMMARY:
      Irreducible: True
      Aperiodic:   True
      ✓ ERGODIC:   True
    ------------------------------------------------------------

    Stationary distribution: {'A': 0.3333, 'B': 0.3333, 'C': 0.3333}

    ================================================================================
    MIXING TIME ANALYSIS
    ================================================================================

    Spectral Gap: 0.300000

    Top 5 eigenvalues (by magnitude):
      λ_1 = 1.000000+0.000000j
      λ_2 = 0.700000+0.000000j
      λ_3 = 0.700000+0.000000j

    ------------------------------------------------------------
    MIXING TIMES:
    ------------------------------------------------------------
    τ_mix(0.50) =    1 steps
    τ_mix(0.25) =    3 steps
    τ_mix(0.10) =    6 steps
    τ_mix(0.01) =   12 steps

    Theoretical estimate (ε=0.25): ≈ 4.6 steps
    Actual: 3 steps


    ================================================================================
    EXAMPLE 2: SLOW MIXING CHAIN
    ================================================================================

    Chain with weak connectivity (bottleneck structure)

    ================================================================================
    ERGODICITY CHECK
    ================================================================================

    Checking Irreducibility:
    ------------------------------------------------------------
    States can reach each other: True

    Checking Aperiodicity:
    ------------------------------------------------------------
    Has self-loops at states: [0 1 2]
    ✓ Chain is APERIODIC (sufficient condition)

    ------------------------------------------------------------
    SUMMARY:
      Irreducible: True
      Aperiodic:   True
      ✓ ERGODIC:   True
    ------------------------------------------------------------

    Stationary distribution: {'A₁': 0.3333, 'A₂': 0.3333, 'B': 0.3333}

    ================================================================================
    MIXING TIME ANALYSIS
    ================================================================================

    Spectral Gap: 0.300000

    Top 5 eigenvalues (by magnitude):
      λ_1 = 1.000000+0.000000j
      λ_2 = 0.700000+0.000000j
      λ_3 = 0.000000+0.000000j

    ------------------------------------------------------------
    MIXING TIMES:
    ------------------------------------------------------------
    τ_mix(0.50) =    1 steps
    τ_mix(0.25) =    3 steps
    τ_mix(0.10) =    6 steps
    τ_mix(0.01) =   12 steps

    Theoretical estimate (ε=0.25): ≈ 4.6 steps
    Actual: 3 steps

    ------------------------------------------------------------
    INTERPRETATION:
    Bottleneck between {A₁,A₂} and {B} slows mixing!
    ------------------------------------------------------------


    ================================================================================
    SUMMARY: KEY TAKEAWAYS
    ================================================================================

        1. 에르고드성 = 줄일 수 없음 + 주기 없음
           → 오직 하나인 π으로 모이는 데 필요하다
           → 그래프의 짜임을 살펴 확인한다
        
        2. 섞임 시간 τ_mix(ε):
           → 멈춘 분포의 ε 안으로 들어가는 데 걸리는 시간
           → 스펙트럼 틈에 달렸다: τ_mix ≈ O(1/gap)
           → MCMC의 효율에 결정적이다!
        
        3. 스펙트럼 틈 γ = 1 - |λ_2|:
           → 틈이 클수록 → 빨리 섞인다
           → 상태 공간의 "이어짐"과 관계있다
           → 병목이 섞임을 느리게 한다
        
        4. 모이는 빠르기:
           → 지수꼴: ||P^n(i,·) - π||_{TV} ≤ C·ρⁿ
           → 빠르기 ρ = |λ_2| < 1
           → 그려 보고 경험으로 잴 수 있다
        
        5. 이것이 MCMC에 왜 중요한가:
           → 표집하기 전에 사슬을 τ_mix스텝쯤 돌려야 한다
           → 느린 섞임 → 쓸 수 없는 MCMC
           → 다음 단원: 마르코프 사슬로 아무 분포에서나 표집하기!
        
        ✓ 파일을 {os.path.abspath(OUTPUT_DIR)}/에 저장했다
        
        다음 스텝(파일 05): 표집 쓰임새 - MCMC로 잇는 다리!
        
        다음 단원의 결정적인 물음:
        "우리가 표집하려는 타깃 분포 π(x)이 주어졌을 때,
         멈춘 분포가 π인 마르코프 사슬을 우리가 지어낼 수 있는가?
         그렇다면 얼마나 빨리 섞이는가?"
        
        이것이 MCMC 방법의 핵심 물음이다!
        

    ================================================================================
    END OF MODULE 04
    ================================================================================
    ```


## 2. 논의

### 유한 사슬의 수렴

유한 기약 사슬에는 유일한 정상분포 π가 있다. 비주기성까지 있으면 모든 초기 상태의 분포가 π로 수렴한다. 정상분포 존재만으로 기약성이나 비주기성이 따라오지는 않는다. 구조 진단은 충분조건과 필요충분조건을 구분해야 한다.

check_irreducibility는 P¹부터 P^N까지 합해 양수 경로를 검사한다. 유효한 유한 확률행렬에서는 간선 그래프의 강연결성으로도 판단할 수 있으며 수치 거듭제곱의 언더플로·허용오차를 주의한다. check_aperiodicity는 자기 루프의 존재만 검사한다. 기약 사슬에서 자기 루프는 충분조건이지만 필요조건이 아니므로 반환 False를 주기성의 확정 진단으로 읽지 않는다. 가약 사슬의 한 자기 루프도 전체 비주기성을 보장하지 않는다. 출력의 assuming 문구와 실제 반환값도 구분한다.

### 총변동 거리와 혼합 시간

확률벡터 p·q 사이의 거리는 TV(p,q)=∑|p_i−q_i|/2다. compute_mixing_time은 각 시점에서 모든 초기 상태 행과 주어진 π의 TV 거리를 비교한다. π가 실제 정상분포인지 먼저 확인해야 한다. 함수는 n=1부터 시작하므로 n=0에 이미 조건을 만족하는 경우를 세지 않고, 제한 내 미도달 시에도 max_steps를 반환한다. 이 값이 실제 혼합 시간인지 미도달 상한인지 출력만으로 구분하지 않으면 안 된다. 코드가 경고를 전역 억제하는 점도 해석 시 고려한다.

### 스펙트럼 해석

유한 기약·비주기 가역 사슬의 절대 스펙트럼 갭은 γ*=1−max_(j≥2)|λ_j|다. π_min과 오차 ε를 포함하는 TV 상계를 사용할 수 있다. analyze_mixing의 log(1/ε)/gap은 이 상수·초기 질량을 생략한 휴리스틱이며 일반적인 엄밀 상계가 아니다. 비가역 행렬에 가역 사슬의 정확한 C|λ₂|ⁿ 식을 자동 적용하지 않는다.

compute_spectral_gap은 고유값 절댓값으로 정렬한 첫 값이 1이라고 검사한다. 주기 행렬에서는 절댓값 1인 다른 고유값과 동점이어서 이 정렬 가정이 불안정할 수 있고, 한 상태 행렬에는 두 번째 값이 없다. 그림은 첫 계산값이 1단계 분포지만 축을 0부터 표시하므로 이론 곡선과 시점도 맞춰 읽는다. 빠른·느린 예제의 비교는 주어진 가상 전이행렬의 비교이며 임의 MCMC의 수렴 보장은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
유한 기약 사슬에서 자기 루프 검사의 True·False는 각각 무엇을 뜻하는가?

</div>

??? success "연습문제 1 풀이"

    양의 자기 루프를 찾으면 비주기성의 충분조건을 확인한 것이다. 찾지 못해도 길이 2·3 등 귀환 경로의 최대공약수가 1이면 비주기일 수 있다. 따라서 False는 비주기성이 거짓이라는 확정 판정이 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
혼합 시간을 계산하기 전에 π와 입력 행렬에서 확인할 조건은?

</div>

??? success "연습문제 2 풀이"

    P가 비어 있지 않은 유한 확률행렬인지 확인하고 π의 길이·비음성·합 1·πP=π 잔차를 확인한다. ε와 최대 단계도 유효해야 한다. 시작 시점 0 포함 여부와 최대 반복 미도달의 반환 규약을 보고해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
log(1/ε)/gap을 모든 행렬의 혼합 시간 상계로 사용해도 되는가?

</div>

??? success "연습문제 3 풀이"

    아니다. 가역성 등의 조건과 정상 질량·초기조건에 따른 상수가 필요하다. 비가역의 곡률·비대각화 문제와 주기성도 고려해야 한다. 코드의 이론 추정 곡선은 제한된 휴리스틱이며 엄밀한 인증은 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
아래 일반 시험 조각이 이 모듈에 맞는지 평가하라.

</div>

??? success "연습문제 4 풀이"

    아래 조각에는 실제로 정의하지 않은 클래스와 공백이 포함된 함수명, 뉴럴 넷트웍 호출·경사 검사가 있어 실행 가능한 검증이 아니다. 정상성 잔차, 알려진 주기·비주기·가약 행렬, 작은 행렬의 직접 TV 값과 혼합 시간의 미도달 표시를 점검하는 것이 적합하다. 코드는 요청대로 보존했다.

    ```python
    def test_convergence theorems():
        model = Convergence Theorems(...)
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

수렴 분석은 유효한 전이행렬·정상분포·구조 조건에서 시작한다. 자기 루프 검사는 비주기성의 일부 충분조건이며, 수치 종료·미도달 반환값과 엄밀한 혼합 시간은 다르다. 스펙트럼 근사의 조건과 그림의 시점 규약을 확인한다.
