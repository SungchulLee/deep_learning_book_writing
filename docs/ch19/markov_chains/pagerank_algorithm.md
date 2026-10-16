# 페이지랭크 알고리즘

페이지랭크는 링크를 따라 이동하고 일정 확률로 임의 페이지로 이동하는 랜덤 서퍼의 정상분포로 페이지 점수를 정의한다. 이 예제는 열벡터 점수와 열 확률행렬을 사용한다. 뉴럴 넷트웍 학습·로스·옵티마이저 대신 고정점 반복으로 점수를 계산한다.

## 1. 코드

??? note "코드 (417줄)"

    ```python
    """
    pagerank_algorithm.py (단원 08)

    페이지랭크 알고리즘(구글의 본래 알고리즘)
    =================================================

    Location: 06_markov_chain/03_applications/
    난이도: ⭐⭐⭐ 중급
    걸리는 시간: 3-4시간

    학습 목표:
    - 마르코프 사슬의 쓰임새로서 페이지랭크 이해하기
    - 페이지랭크 알고리즘 구현하기
    - 순간이동과 감쇠 인자 다루기
    - 웹 쪽을 중요도로 차례 매기기

    수학적 바탕:
    페이지랭크는 웹 서핑을 웹 그래프 위의 무작위 스텝으로 본뜬다:
    - 상태 = 웹 쪽
    - 옮김 = 이음을 고르게 무작위로 누르기
    - 확률 α으로: 무작위 이음을 따라간다
    - 확률 (1-α)으로: 무작위 쪽으로 뛴다(순간이동)

    페이지랭크 방정식:
    PR(p) = (1-α)/N + α × Σ_{q→p} PR(q)/L(q)

    여기서 각 기호는 다음과 같다.
    - PR(p) = 쪽 p의 페이지랭크
    - α = 감쇠 인자(보통 0.85)
    - N = 쪽의 전체 수
    - q→p은 쪽 q이 쪽 p으로 이음을 뜻한다
    - L(q) = q에서 나가는 이음의 수

    행렬로 쓰면: r = (1-α)/N × e + α × P^T × r
    여기서 r은 페이지랭크 벡터이다(멈춘 분포)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx

    # ========================================================================
    # 메인
    # ========================================================================


    class PageRank:
        """
        페이지랭크 알고리즘의 구현.
        """

        def __init__(self, adjacency_matrix, damping_factor=0.85):
            """
            페이지랭크 셈하개 첫값 잡기.

            파라메터:
                adjacency_matrix (np.ndarray): 쪽 i이 쪽 j으로 이으면 A[i][j] = 1
                damping_factor (float): 이음을 따라갈 확률(보통 0.85)

            수학의 차림:
            구글 행렬을 만든다:
            G = α × P^T + (1-α) × E
            여기서 P^T은 열 확률 이음 행렬이다
            그리고 E은 고른 순간이동 행렬이다
            """
            self.A = np.array(adjacency_matrix, dtype=float)
            self.n_pages = self.A.shape[0]
            self.alpha = damping_factor

            # 옮김 행렬 만들기
            self._build_transition_matrix()

        def _build_transition_matrix(self):
            """
            페이지랭크 옮김 행렬 세우기.

            수학 과정:
            1. 이웃 행렬 A으로 이음 행렬 P 만들기
               P[i][j] = A[i][j] / L(i) if L(i) > 0
               여기서 L(i) = 쪽 i에서 나가는 이음의 수

            2. 매달린 마디 다루기(나가는 이음이 없는 쪽)
               그 행들을 고른 분포로 바꾸기

            3. 순간이동 더하기:
               G = α × P^T + (1-α)/N × E
               여기서 E은 1로 채운 행렬이다
            """
            # 쪽마다 나가는 이음 세기
            out_degrees = self.A.sum(axis=1)

            # 옮김 행렬 P 만들기
            # P[i][j] = 이음을 따라 i에서 j으로 갈 확률
            P = np.zeros_like(self.A)

            for i in range(self.n_pages):
                if out_degrees[i] > 0:
                    # 나가는 이음의 수로 고르게 하기
                    P[i, :] = self.A[i, :] / out_degrees[i]
                else:
                    # 매달린 마디: 고른 분포
                    P[i, :] = 1.0 / self.n_pages

            # 열 확률 행렬을 얻으려고 전치하기
            P_T = P.T

            # 순간이동 더하기(구글 행렬)
            # G = α × P^T + (1-α)/N × E
            E = np.ones((self.n_pages, self.n_pages)) / self.n_pages
            self.G = self.alpha * P_T + (1 - self.alpha) * E

        def compute_pagerank_power_iteration(self, max_iter=100, tol=1e-8):
            """
            거듭제곱 되풀이 방법으로 페이지랭크 셈하기.

            파라메터:
                max_iter (int): 최대 되풀이 횟수
                tol (float): 모임 너그러움

            반환값:
                tuple: (페이지랭크 벡터, 되풀이 횟수)

            수학 방법:
            거듭제곱 되풀이: r^{(k+1)} = G × r^{(k)}
            r^{(0)} = 1/N × e으로 시작한다(고른 분포)
            모일 때까지 되풀이: ||r^{(k+1)} - r^{(k)}|| < tol
            """
            # 고른 분포로 첫스텝 잡기
            r = np.ones(self.n_pages) / self.n_pages

            for iteration in range(max_iter):
                r_new = self.G @ r

                # 모임 살피기
                if np.linalg.norm(r_new - r, ord=1) < tol:
                    return r_new, iteration + 1

                r = r_new

            return r, max_iter

        def compute_pagerank_eigenvector(self):
            """
            고유벡터 방법으로 페이지랭크 셈하기.

            반환값:
                np.ndarray: 페이지랭크 벡터

            수학 방법:
            페이지랭크는 G의 으뜸 고유벡터이다:
            λ = 1일 때 G × r = λ × r

            고윳값 1에 딸린 고유벡터를 찾는다
            """
            eigenvalues, eigenvectors = np.linalg.eig(self.G.T)

            # 고윳값이 1인 고유벡터 찾기
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            r = np.real(eigenvectors[:, idx])

            # 합이 1이 되도록 고르게 하기
            r = r / r.sum()

            return r

        def rank_pages(self, pagerank_vector, page_names=None):
            """
            페이지랭크 점수로 쪽의 차례 매기기.

            파라메터:
                pagerank_vector (np.ndarray): 페이지랭크 점수
                page_names (list): 쪽 이름(없어도 된다)

            반환값:
                list: 정렬한 (쪽, 점수) 짝의 목록
            """
            if page_names is None:
                page_names = [f"Page {i}" for i in range(self.n_pages)]

            # (쪽, 점수) 짝의 목록 만들기
            page_scores = list(zip(page_names, pagerank_vector))

            # 점수로 정렬 (내림차순)
            page_scores.sort(key=lambda x: x[1], reverse=True)

            return page_scores


    def example_simple_web_graph():
        """
        보기 1: 쪽 4개짜리 단순한 웹 그래프.

        이음 짜임:
        A → B, C
        B → C
        C → A
        D → A, B, C  (D은 권위 쪽이다)
        """
        print("=" * 70)
        print("Example 1: Simple Web Graph (4 Pages)")
        print("=" * 70)

        # 이웃 행렬
        # 쪽 i이 쪽 j으로 이으면 A[i][j] = 1
        pages = ['A', 'B', 'C', 'D']
        A = np.array([
            [0, 1, 1, 0],  # A이 B, C으로 이음
            [0, 0, 1, 0],  # B이 C으로 이음
            [1, 0, 0, 0],  # C이 A으로 이음
            [1, 1, 1, 0]   # D이 A, B, C으로 이음
        ])

        print("\nAdjacency Matrix:")
        print(f"{'':5s} " + " ".join(f"{p:3s}" for p in pages))
        for i, page in enumerate(pages):
            row = " ".join(f"{int(A[i,j]):3d}" for j in range(len(pages)))
            print(f"{page:5s} {row}")

        print("\nLink structure:")
        for i, page_i in enumerate(pages):
            links_to = [pages[j] for j in range(len(pages)) if A[i,j] == 1]
            if links_to:
                print(f"  {page_i} → {', '.join(links_to)}")

        # 페이지랭크 셈하기
        pr = PageRank(A, damping_factor=0.85)

        print("\n" + "-" * 70)
        print("Computing PageRank...")

        # 방법 1: 거듭제곱 되풀이
        r_power, iterations = pr.compute_pagerank_power_iteration()
        print(f"\nPower iteration (converged in {iterations} iterations):")
        for page, score in zip(pages, r_power):
            print(f"  {page}: {score:.6f}")

        # 방법 2: 고유벡터
        r_eig = pr.compute_pagerank_eigenvector()
        print(f"\nEigenvector method:")
        for page, score in zip(pages, r_eig):
            print(f"  {page}: {score:.6f}")

        # 쪽 차례 매기기
        print("\n" + "-" * 70)
        print("Page Rankings:")
        ranked = pr.rank_pages(r_power, pages)
        for rank, (page, score) in enumerate(ranked, 1):
            print(f"  {rank}. {page}: {score:.6f}")

        print("\nInterpretation:")
        print("  Page A has highest PageRank because:")
        print("  - It's linked by C (which is linked by B and A)")
        print("  - It's linked by D (authority page)")


    def example_damping_factor_effect():
        """
        보기 2: 감쇠 인자가 페이지랭크에 주는 영향.

        α이 차례 매기기에 어떤 영향을 주는지 보인다.
        """
        print("\n" + "=" * 70)
        print("Example 2: Effect of Damping Factor")
        print("=" * 70)

        pages = ['A', 'B', 'C', 'D']
        A = np.array([
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0]
        ])

        print("\nComparing different damping factors:")
        print(f"{'α':<8s} " + " ".join(f"{p:>12s}" for p in pages))

        for alpha in [0.5, 0.75, 0.85, 0.95]:
            pr = PageRank(A, damping_factor=alpha)
            r, _ = pr.compute_pagerank_power_iteration()

            row = " ".join(f"{score:12.6f}" for score in r)
            print(f"{alpha:<8.2f} {row}")

        print("\nObservation:")
        print("  Higher α → more influence from link structure")
        print("  Lower α → closer to uniform distribution")


    def example_larger_network():
        """
        보기 3: 쪽 8개짜리 더 큰 넷트웍.

        더 복잡한 짜임에서 차례 매기기를 보인다.
        """
        print("\n" + "=" * 70)
        print("Example 3: Larger Network (8 Pages)")
        print("=" * 70)

        pages = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        # 더 복잡한 이음 짜임 만들기
        A = np.array([
            [0, 1, 1, 0, 0, 0, 0, 0],  # A → B, C
            [1, 0, 1, 1, 0, 0, 0, 0],  # B → A, C, D
            [0, 0, 0, 1, 1, 0, 0, 0],  # C → D, E
            [0, 0, 0, 0, 1, 1, 0, 0],  # D → E, F
            [0, 0, 0, 0, 0, 1, 1, 0],  # E → F, G
            [0, 0, 0, 0, 0, 0, 1, 1],  # F → G, H
            [0, 0, 0, 0, 0, 0, 0, 1],  # G → H
            [1, 0, 0, 0, 0, 0, 0, 0]   # H → A(순환을 만듦)
        ])

        pr = PageRank(A, damping_factor=0.85)
        r, iterations = pr.compute_pagerank_power_iteration()

        print(f"\nPageRank scores (converged in {iterations} iterations):")
        ranked = pr.rank_pages(r, pages)

        for rank, (page, score) in enumerate(ranked, 1):
            bar = '█' * int(score * 500)
            print(f"  {rank}. {page}: {score:.6f} {bar}")


    def visualize_pagerank():
        """
        넷트웍 그래프로 페이지랭크 그려 보기.
        """
        print("\n" + "=" * 70)
        print("Creating PageRank Visualization")
        print("=" * 70)

        pages = ['A', 'B', 'C', 'D', 'E']
        A = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0]
        ])

        pr = PageRank(A, damping_factor=0.85)
        r, _ = pr.compute_pagerank_power_iteration()

        # 넷트웍 그래프 만들기
        G = nx.DiGraph()
        for i, page in enumerate(pages):
            G.add_node(page)

        for i in range(len(pages)):
            for j in range(len(pages)):
                if A[i,j] == 1:
                    G.add_edge(pages[i], pages[j])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 그림 1: 페이지랭크 크기로 그린 넷트웍
        ax = axes[0]
        pos = nx.spring_layout(G, seed=42)

        # 마디 크기를 페이지랭크에 비례하게
        node_sizes = [r[i] * 5000 for i in range(len(pages))]

        nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes,
               node_color='lightblue', font_size=12, font_weight='bold',
               arrows=True, arrowsize=20, edge_color='gray', width=2)

        ax.set_title('Web Graph (Node size = PageRank)', fontsize=13)

        # 그림 2: 페이지랭크 막대그래프
        ax = axes[1]

        ranked = pr.rank_pages(r, pages)
        pages_sorted = [p for p, _ in ranked]
        scores_sorted = [s for _, s in ranked]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(pages)))
        bars = ax.barh(pages_sorted, scores_sorted, color=colors, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('PageRank Score', fontsize=12)
        ax.set_title('PageRank Rankings', fontsize=13)
        ax.grid(True, alpha=0.3, axis='x')

        for bar, score in zip(bars, scores_sorted):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{score:.4f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('pagerank.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("PageRank visualization saved")


    def main():
        """
        페이지랭크 보기 모두 돌리기.
        """
        print("PAGERANK ALGORITHM")
        print("==================\n")

        example_simple_web_graph()
        example_damping_factor_effect()
        example_larger_network()
        visualize_pagerank()

        print("\n" + "=" * 70)
        print("Key Insights:")
        print("=" * 70)
        print("1. PageRank = stationary distribution of random surfer")
        print("2. Damping factor (typically 0.85) balances link-following and teleportation")
        print("3. Pages with many incoming links from important pages rank higher")
        print("4. Power iteration typically converges in ~50-100 iterations")


    if __name__ == "__main__":
        main()
    ```


**출력:**

```
PAGERANK ALGORITHM
==================

======================================================================
Example 1: Simple Web Graph (4 Pages)
======================================================================

Adjacency Matrix:
      A   B   C   D  
A       0   1   1   0
B       0   0   1   0
C       1   0   0   0
D       1   1   1   0

Link structure:
  A → B, C
  B → C
  C → A
  D → A, B, C

----------------------------------------------------------------------
Computing PageRank...

Power iteration (converged in 35 iterations):
  A: 0.373248
  B: 0.206755
  C: 0.382497
  D: 0.037500

Eigenvector method:
  A: 0.250000
  B: 0.250000
  C: 0.250000
  D: 0.250000

----------------------------------------------------------------------
Page Rankings:
  1. C: 0.382497
  2. A: 0.373248
  3. B: 0.206755
  4. D: 0.037500

Interpretation:
  Page A has highest PageRank because:
  - It's linked by C (which is linked by B and A)
  - It's linked by D (authority page)

======================================================================
Example 2: Effect of Damping Factor
======================================================================

Comparing different damping factors:
α                   A            B            C            D
0.50         0.314103     0.224359     0.336538     0.125000
0.75         0.355769     0.211538     0.370192     0.062500
0.85         0.373248     0.206755     0.382497     0.037500
0.95         0.391025     0.202195     0.394280     0.012500

Observation:
  Higher α → more influence from link structure
  Lower α → closer to uniform distribution

======================================================================
Example 3: Larger Network (8 Pages)
======================================================================

PageRank scores (converged in 64 iterations):
  1. A: 0.183305 ███████████████████████████████████████████████████████████████████████████████████████████
  2. H: 0.161376 ████████████████████████████████████████████████████████████████████████████████
  3. C: 0.124040 ██████████████████████████████████████████████████████████████
  4. E: 0.113479 ████████████████████████████████████████████████████████
  5. G: 0.113300 ████████████████████████████████████████████████████████
  6. F: 0.108991 ██████████████████████████████████████████████████████
  7. D: 0.098853 █████████████████████████████████████████████████
  8. B: 0.096655 ████████████████████████████████████████████████

======================================================================
Creating PageRank Visualization
======================================================================
PageRank visualization saved

======================================================================
Key Insights:
======================================================================
1. PageRank = stationary distribution of random surfer
2. Damping factor (typically 0.85) balances link-following and teleportation
3. Pages with many incoming links from important pages rank higher
4. Power iteration typically converges in ~50-100 iterations
```

## 2. 논의

### 링크 전이와 순간이동

A_ij가 i에서 j로 가는 링크 웨이트라면 출발 행을 정규화해 행 확률행렬 P를 만든다. 나가는 링크가 없는 dangling 페이지는 여기서 균등 행으로 처리한다. 균등 순간이동 벡터 v와 감쇠계수 0≤α<1을 사용하면 열 확률행렬은 다음과 같다.

$$G=\alpha P^\top+(1-\alpha)v\mathbf1^\top,\qquad r=Gr$$

모든 v_i>0이면 G는 양의 행렬이어서 유일한 정상 점수를 갖는다. 여기서 r은 열벡터이며 행 확률행렬에서의 π=πP와 전치 규약이 다르다. α가 1에 가까울수록 링크 구조의 비중이 커지고 수렴이 느려질 수 있다. α=1에서는 일반적인 유일성·비주기 보장이 사라진다.

### 코드의 두 계산 방법

거듭제곱 반복은 r_new=G@r을 사용하므로 위 열벡터 정의에 맞는다. 하지만 compute_pagerank_eigenvector는 G가 아니라 G.T의 오른쪽 고유벡터를 구한다. G.T는 행 확률행렬이어서 고유값 1의 상수 오른쪽 고유벡터를 선택하면 균등 점수를 주며 일반 페이지랭크와 다르다. 두 결과의 불일치를 난수 오차로 설명하지 않는다. 올바른 점수는 Gr−r 잔차·비음성·합 1을 확인해야 한다.

종료 조건은 인접 반복의 L1 차이다. 0≤α<1이면 같은 확률 단체에서 α 수축을 이용해 잔차를 실제 고정점 오차와 연결할 수 있다. 예를 들어 α<1에서 ||r−r*||₁≤||Gr−r||₁/(1−α)다. 고정된 50~100회가 모든 그래프·허용오차에 충분한 것은 아니다. 최대 반복 도달 여부도 보고한다.

### 입력·계산 규모와 해석

A는 비어 있지 않은 유한 비음수 정방행렬이고 페이지 이름 수가 맞아야 한다. 조밀한 N×N G를 만드는 방식은 큰 웹 그래프에 비싸므로 희소 링크 곱과 별도 순간이동 항으로 계산할 수 있다. 높은 페이지랭크는 정한 링크 모형의 정상 방문 비율이며 콘텐츠의 진실성·품질·현재 검색 순위 전체를 뜻하지 않는다. 아래 가상 예제의 고정 순위 해설도 실제 반환 점수와 대조해 읽는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 페이지랭크 구현에 optimizer.zero_grad·backward가 필요한가?

</div>

??? success "연습문제 1 풀이"

    필요하지 않다. 링크와 감쇠계수로 고정된 확률행렬을 구성하고 거듭제곱 반복으로 정상 점수를 계산한다. 경사 기반 학습 루프나 학습 가능한 PyTorch 모델은 이 구현에 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
거듭제곱 반복과 보존된 고유벡터 함수가 다른 결과를 낼 수 있는 이유는?

</div>

??? success "연습문제 2 풀이"

    G는 열 확률행렬이고 점수는 G의 오른쪽 고유값 1 벡터다. 보존된 함수는 G.T의 오른쪽 고유벡터를 선택하므로 일반적으로 균등 벡터를 반환한다. 전치 방향을 일치시키고 Gr=r를 검사해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
dangling 페이지와 α=1이 만드는 문제를 설명하라.

</div>

??? success "연습문제 3 풀이"

    링크가 없는 행은 그대로 두면 행합이 0이므로 별도 전이 규칙이 필요하고 코드는 균등 행을 쓴다. α<1과 양의 순간이동은 유일한 정상 점수와 수렴을 보장하지만 α=1에서는 링크 그래프의 가약성·주기성이 다시 문제가 될 수 있다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 보존된 조기 종료 조각이 페이지랭크 반복의 종료 조건으로 적합한지 평가하라.

</div>

??? success "연습문제 4 풀이"

    이 조각은 검증 로스와 뉴럴 넷트웍 state_dict를 사용하는 별개 학습 예제이며 PageRank에는 해당 인터페이스가 없다. state_dict().copy()는 텐서 저장을 독립 복제하지 않고 중단 분기 외에는 최적 상태를 복원하지 않는 한계도 있다. 페이지랭크에서는 확률 제약과 고정점 잔차·수축 상계 및 최대 반복 여부를 사용해 종료를 평가해야 한다.

    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```


## 정리하며

페이지랭크는 감쇠된 랜덤 서퍼의 정상분포다. 행·열 규약을 맞추고 링크 없는 페이지·순간이동·감쇠계수의 조건을 명시한다. 코드의 고유벡터 전치 오류와 무관한 뉴럴 넷트웍 조기 종료 조각을 올바른 점수 계산으로 해석하지 않는다.
