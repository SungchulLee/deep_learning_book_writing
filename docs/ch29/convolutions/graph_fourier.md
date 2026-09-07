# 그래프 푸리에 변환

그래프 푸리에 변환은 그래프 겹말기 연산의 중요한 개념이다. 이 짜기는 얽힌 핵심 알고리즘과 자료 얼개를 손으로 만져 보게 하며 이론 바탕과 실제로 펼칠 때 살필 것을 함께 보여 준다.

## 코드

```python
"""
29.3.2: 그래프 푸리에 변환
그래프 푸리에 바꿈 짜기, 스펙트럼 엮음, 파세발 정리.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


def compute_gft_basis(A):
    """그래프 푸리에 변환의 바탕(라플라스 고유 벡터)을 셈한다."""
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues, U = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], U[:, idx], L


def gft(signal, U):
    """앞으로 가는 그래프 푸리에 변환."""
    return U.T @ signal


def igft(spectral_coeffs, U):
    """거꾸로 가는 그래프 푸리에 변환."""
    return U @ spectral_coeffs


def spectral_convolution(f, g, U):
    """스펙트럼 자리를 거친 그래프 위 겹말기."""
    f_hat = gft(f, U)
    g_hat = gft(g, U)
    return igft(f_hat * g_hat, U)


def demo_gft():
    """그래프 푸리에 변환을 보여 준다."""
    print("=" * 60)
    print("Graph Fourier Transform")
    print("=" * 60)

    G = nx.path_graph(50)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, L = compute_gft_basis(A)
    n = 50

    # 서로 다른 신호
    signals = {
        'Low-freq (sin 1 cycle)': np.sin(np.linspace(0, 2*np.pi, n)),
        'High-freq (sin 10 cycles)': np.sin(np.linspace(0, 20*np.pi, n)),
        'Step function': np.concatenate([np.ones(25), -np.ones(25)]),
        'Random': np.random.randn(n),
    }

    for name, f in signals.items():
        f_hat = gft(f, U)
        # 에너지 분포
        total_energy = np.sum(f_hat**2)
        low_energy = np.sum(f_hat[:n//4]**2) / total_energy
        high_energy = np.sum(f_hat[3*n//4:]**2) / total_energy
        print(f"  {name:30s}: low={low_energy:.3f}, high={high_energy:.3f}")


def demo_parseval():
    """그래프에서 파세발 정리를 따져 본다."""
    print("\n" + "=" * 60)
    print("Parseval's Theorem")
    print("=" * 60)

    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    _, U, _ = compute_gft_basis(A)

    np.random.seed(42)
    for trial in range(5):
        f = np.random.randn(A.shape[0])
        f_hat = gft(f, U)
        spatial_energy = np.sum(f**2)
        spectral_energy = np.sum(f_hat**2)
        print(f"  Trial {trial+1}: spatial={spatial_energy:.4f}, "
              f"spectral={spectral_energy:.4f}, "
              f"match={np.isclose(spatial_energy, spectral_energy)}")


def demo_spectral_convolution():
    """스펙트럼 자리에서의 그래프 겹말기를 보여 준다."""
    print("\n" + "=" * 60)
    print("Spectral Convolution")
    print("=" * 60)

    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, L = compute_gft_basis(A)
    n = A.shape[0]

    np.random.seed(42)
    signal = np.random.randn(n) + np.array(
        [0 if G.nodes[i].get('club', '') == 'Mr. Hi' else 2 for i in range(n)],
        dtype=float)

    # 배울 수 있는 스펙트럼 거르개(흉내)
    # 낮은 진동수 통과 거르개
    theta_lowpass = np.exp(-eigenvalues / eigenvalues[-1])
    filtered = U @ (theta_lowpass * (U.T @ signal))

    # 같은 뜻: g_theta(L) @ signal
    g_L = U @ np.diag(theta_lowpass) @ U.T
    filtered_matrix = g_L @ signal

    print(f"Spectral filtering match: {np.allclose(filtered, filtered_matrix)}")
    print(f"Original signal range: [{signal.min():.2f}, {signal.max():.2f}]")
    print(f"Filtered signal range: [{filtered.min():.2f}, {filtered.max():.2f}]")


def demo_learnable_spectral_filter():
    """스펙트럼 거르개 잡을 배우는 것을 흉내 낸다."""
    print("\n" + "=" * 60)
    print("Learnable Spectral Filter")
    print("=" * 60)

    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, L = compute_gft_basis(A)
    n = A.shape[0]

    # 과녁: 무리 이름표
    labels = np.array([0 if G.nodes[i].get('club', '') == 'Mr. Hi' else 1
                        for i in range(n)], dtype=float)

    # 들임: 잡소리 섞인 신호
    np.random.seed(42)
    signal = labels + np.random.randn(n) * 0.5

    # 기울기 내려가기로 theta 배우기
    theta = np.random.randn(n) * 0.1
    lr = 0.01

    for epoch in range(200):
        filtered = U @ (theta * (U.T @ signal))
        error = filtered - labels
        loss = np.mean(error**2)

        # 기울기: d_loss/d_theta
        f_hat = U.T @ signal
        grad = 2 * (U.T @ error) * f_hat / n
        theta -= lr * grad

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}")

    # 마지막 거르개 꼴
    print(f"\nLearned filter (first 5): {np.round(theta[:5], 3)}")
    print(f"Ideal low-pass (first 5): {np.round(np.exp(-eigenvalues[:5]), 3)}")


def visualize_gft():
    """그래프 푸리에 변환 성분을 그린다."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    G = nx.path_graph(50)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, _ = compute_gft_basis(A)

    # 고유 벡터(바탕 함수)
    for k in range(6):
        ax = axes[k // 3, k % 3]
        ax.plot(U[:, k], 'b-', linewidth=1.5)
        ax.set_title(f"Eigenvector u_{k} (λ={eigenvalues[k]:.3f})")
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle("Graph Fourier Basis (Path Graph)", fontsize=14)
    plt.tight_layout()
    plt.savefig("graph_fourier.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to graph_fourier.png")


if __name__ == "__main__":
    demo_gft()
    demo_parseval()
    demo_spectral_convolution()
    demo_learnable_spectral_filter()
    visualize_gft()```

## 논의

이 짜기는 그래프 푸리에 변환의 핵심 연산을 짜는 여러 도구 함수를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수들은 잘 알려진 그래프 자료 묶음에서 이 조각들의 실제 쓰임을 보인다. 내놓기를 살펴보면 윗매개변수를 어떻게 고르고 문제를 어떻게 차리느냐에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실제 관점에서 이 짜기는 순수한 성능보다 또렷함을 앞세운다. 실제로 쓰는 얼개는 보통 묶음 셈, GPU 빠르게 하기, 더 정교한 윗매개변수 맞추기 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 생각은 큰 규모의 쓰임새로 곧바로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌려 핵심 내놓기 잣대를 적어라. 윗매개변수 하나(배움 빠르기, 숨은 차원, 층 개수 같은 것)를 고치고 결과가 어떻게 바뀌는지 적어라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 나머지를 붙박아 두고 고른 윗매개변수를 차근히 바꾼다. 보기로 숨은 차원을 두 배로 하면 보통 나타냄 담이가 늘지만 셈 시간이 커진다. 배움 빠르기는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 모이고 너무 크면 흔들린다. 고른 윗매개변수의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 적어 두라.

---

**연습문제 2.**
이 짜기에서 핵심 얼개 고르기의 몫을 밝혀라. 왜 그 깨움 함수, 고르게 맞추기 셈속, 손실 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

??? success "연습문제 2 풀이"
    이 얼개 고르기는 그래프 겹말기 연산에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.
