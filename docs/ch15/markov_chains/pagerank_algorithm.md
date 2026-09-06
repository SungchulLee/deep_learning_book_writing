# 페이지랭크 알고리즘

pagerank_algorithm.py (모듈 08) 페이지랭크 알고리즘(구글의 원래 알고리즘)

확률 과정과 표집 방법은 확률적 기계 학습의 바탕 도구이다. 이 모듈은 마르코프 사슬의 개념을 보이며 수학 이론과 셈 구현 사이의 틈을 잇는다.

## 코드

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
페이지랭크는 웹 서핑을 웹 그래프 위의 무작위 걸음으로 본뜬다:
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
        
        매개변수:
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
        
        매개변수:
            max_iter (int): 최대 되풀이 횟수
            tol (float): 모임 너그러움
        
        반환값:
            tuple: (페이지랭크 벡터, 되풀이 횟수)
        
        수학 방법:
        거듭제곱 되풀이: r^{(k+1)} = G × r^{(k)}
        r^{(0)} = 1/N × e으로 시작한다(고른 분포)
        모일 때까지 되풀이: ||r^{(k+1)} - r^{(k)}|| < tol
        """
        # 고른 분포로 첫걸음 잡기
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
        
        매개변수:
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
    보기 3: 쪽 8개짜리 더 큰 그물.
    
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
    그물 그래프로 페이지랭크 그려 보기.
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
    
    # 그물 그래프 만들기
    G = nx.DiGraph()
    for i, page in enumerate(pages):
        G.add_node(page)
    
    for i in range(len(pages)):
        for j in range(len(pages)):
            if A[i,j] == 1:
                G.add_edge(pages[i], pages[j])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 그림 1: 페이지랭크 크기로 그린 그물
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
    plt.savefig('/mnt/user-data/outputs/pagerank.png', dpi=150, bbox_inches='tight')
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
    main()```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 무늬는 더 복잡한 상황으로 자연스럽게 넓어진다. 웃매개변수, 구조의 변형, 서로 다른 자료 묶음을 이리저리 시험해 보면 이해가 깊어지고 확률 과정 일감에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
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
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.
