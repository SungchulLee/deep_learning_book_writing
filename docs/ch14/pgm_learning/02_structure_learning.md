# 짜임 배우기

짜임 배우기는 관측 자료에서 베이즈 망의 그래프 짜임을 찾아내는 문제이다. 있을 수 있는 DAG의 공간이 변수 개수에 따라 지수보다 빠르게 커지므로 기계 학습에서 가장 까다로운 문제 가운데 하나이다. 크게 두 갈래가 있다. 조건부 독립 관계를 검정하는 제약 기반 방법과, BIC 같은 점수 함수를 가장 크게 하는 짜임을 뒤지는 점수 기반 방법이다.

## 1. 코드

```python
"""
베이즈 망의 짜임 배우기
========================================

이 모듈은 자료에서 베이즈 망의 짜임(그래프)을 배우는 일을 다룬다.

학습 목표:
-------------------
1. 짜임 배우기 문제 이해하기
2. 제약 기반 방법(PC 알고리즘) 배우기
3. 점수 기반 방법(BIC를 쓴 언덕 오르기) 배우기
4. 짜임 배우기의 어려움 이해하기
5. 실제 자료에 짜임 배우기 쓰기

수학의 바탕:
------------------------
짜임 배우기 문제: 자료 D이 주어졌을 때 D을 가장 잘 설명하는 짜임 G 찾기

두 큰 갈래:
1. 제약 기반: 조건부 독립을 검정한다
   - PC 알고리즘: d-가름을 검정한다
   - 통계적 독립 검정에 기댄다

2. 점수 기반: 점수 함수를 최적화한다
   - 점수: BIC, AIC, BDe 등
   - 뒤지기: 언덕 오르기, 욕심, 유전 알고리즘
   
BIC 점수: BIC(G, D) = log P(D | G, θ_MLE) - (d/2) log n
여기서 d = 매개변수의 개수, n = 표본의 개수

지은이: 교육용 ML 팀
수준: 나아간 단계
미리 볼 것: 첫걸음과 중간 수준의 모든 모듈
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations
import pandas as pd
from scipy.stats import chi2_contingency
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================


class IndependenceTest:
    """
    (조건부) 독립의 통계 검정.
    
    독립 검정에 카이제곱 검정을 쓴다.
    """
    
    @staticmethod
    def test_independence(data: pd.DataFrame,
                         var1: str,
                         var2: str,
                         alpha: float = 0.05) -> bool:
        """
        var1과 var2이 독립인지 검정한다.
        
        독립의 카이제곱 검정을 쓴다.
        
        인수:
            data: 변수를 담은 DataFrame
            var1, var2: 변수의 이름
            alpha: 유의수준
        
        반환값:
            독립이면(H0을 물리치지 못하면) True, 아니면 False
        """
        # 분할표 만들기
        contingency = pd.crosstab(data[var1], data[var2])
        
        # 카이제곱 검정
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # p값 > alpha이면 귀무가설(독립)을 물리치지 못한다
        return p_value > alpha
    
    @staticmethod
    def test_conditional_independence(data: pd.DataFrame,
                                     var1: str,
                                     var2: str,
                                     given: List[str],
                                     alpha: float = 0.05) -> bool:
        """
        `given`이 주어졌을 때 var1과 var2이 조건부 독립인지 검정한다.
        
        검정: X ⊥ Y | Z
        
        인수:
            data: DataFrame
            var1, var2: 검정할 변수
            given: 조건 변수
            alpha: 유의수준
        
        반환값:
            조건부 독립이면 True, 아니면 False
        """
        if not given:
            return IndependenceTest.test_independence(data, var1, var2, alpha)
        
        # 조건 변수의 값마다 독립을 검정한다
        # 모든 값에서 독립이면 조건부 독립이다
        
        # 조건 변수로 묶기
        grouped = data.groupby(given)
        
        p_values = []
        for name, group in grouped:
            if len(group) < 5:  # 너무 작은 무리는 건너뛴다
                continue
            
            # 이 무리에서 독립 검정하기
            contingency = pd.crosstab(group[var1], group[var2])
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                p_values.append(p_value)
            except:
                continue
        
        if not p_values:
            return True  # 자료가 모자라니 독립이라고 놓는다
        
        # 가장 작은 p값을 쓴다(가장 깐깐하다)
        min_p_value = min(p_values)
        return min_p_value > alpha


class PCAlgorithm:
    """
    짜임 배우기를 위한 PC(피터-클라크) 알고리즘을 구현한다.
    
    PC 알고리즘은 다음과 같은 제약 기반 방법이다:
    1. 완전 무방향 그래프에서 시작한다
    2. 조건부 독립 검정에 따라 변을 지운다
    3. v-짜임에 따라 변의 방향을 정한다
    4. 방향을 퍼뜨린다
    
    CPDAG(온전히 채운 부분 방향 비순환 그래프)을 되돌린다.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        PC 알고리즘의 첫걸음을 잡는다.
        
        인수:
            alpha: 독립 검정의 유의수준
        """
        self.alpha = alpha
        self.tester = IndependenceTest()
    
    def learn_structure(self, data: pd.DataFrame) -> nx.Graph:
        """
        PC 알고리즘으로 베이즈 망의 짜임을 배운다.
        
        인수:
            data: 표본을 담은 DataFrame
        
        반환값:
            망의 무방향 그래프(뼈대)
        """
        variables = list(data.columns)
        n = len(variables)
        
        print(f"\nPC Algorithm: Learning structure from {len(data)} samples")
        print(f"Variables: {variables}")
        print("-" * 70)
        
        # 걸음 1: 완전 무방향 그래프에서 시작
        graph = nx.Graph()
        graph.add_nodes_from(variables)
        for i in range(n):
            for j in range(i+1, n):
                graph.add_edge(variables[i], variables[j])
        
        print(f"\nStep 1: Started with complete graph ({graph.number_of_edges()} edges)")
        
        # 걸음 2: 조건부 독립에 따라 변 지우기
        # 조건 묶음의 크기를 늘려 가며 검정하기
        separating_sets = {}  # 나중을 위해 가름 집합을 담아 둔다
        
        for order in range(n):
            print(f"\nStep 2.{order+1}: Testing conditional independence with |S| = {order}")
            
            edges_to_remove = []
            
            for edge in list(graph.edges()):
                var1, var2 = edge
                
                # 조건 묶음 후보 얻기(var1이나 var2의 이웃)
                neighbors = set(graph.neighbors(var1)) | set(graph.neighbors(var2))
                neighbors -= {var1, var2}
                
                # 크기가 `order`인 조건 묶음 모두 검정하기
                if len(neighbors) >= order:
                    for cond_set in combinations(neighbors, order):
                        cond_list = list(cond_set)
                        
                        # 조건부 독립 검정하기
                        if self.tester.test_conditional_independence(
                            data, var1, var2, cond_list, self.alpha
                        ):
                            edges_to_remove.append(edge)
                            separating_sets[edge] = cond_list
                            print(f"  {var1} ⊥ {var2} | {{{', '.join(cond_list)}}}")
                            break
            
            # 변 지우기
            for edge in edges_to_remove:
                graph.remove_edge(*edge)
            
            if edges_to_remove:
                print(f"  Removed {len(edges_to_remove)} edges")
            
            # 더 검정할 변이 없으면 멈춘다
            if graph.number_of_edges() == 0:
                break
        
        print(f"\nStep 3: Final skeleton has {graph.number_of_edges()} edges")
        
        return graph


class ScoreBasedLearning:
    """
    BIC 점수를 쓴 언덕 오르기로 하는 점수 기반 짜임 배우기.
    
    점수 기반 방법:
    1. 점수 함수를 정한다(BIC, AIC, BDe 등)
    2. 점수를 가장 크게 하는 짜임을 뒤진다
    3. 어림짐작 뒤지기를 쓴다(언덕 오르기, 유전 알고리즘 등)
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        점수 기반 배우기의 첫걸음을 잡는다.
        
        인수:
            data: 학습 자료
        """
        self.data = data
        self.variables = list(data.columns)
        self.n_samples = len(data)
        
        # 점수 매기기용 통계량 셈하기
        self._compute_statistics()
    
    def _compute_statistics(self):
        """효율적인 점수 매기기를 위해 통계량을 미리 셈한다."""
        self.counts = {}
        
        for var in self.variables:
            # 나온 횟수를 센다
            self.counts[var] = self.data[var].value_counts().to_dict()
    
    def compute_bic_score(self, graph: nx.DiGraph) -> float:
        """
        주어진 DAG의 BIC 점수를 셈한다.
        
        BIC = log P(D | G, θ_MLE) - (d/2) log n
        
        여기서 각 기호는 다음과 같다.
        - P(D | G, θ_MLE)은 MLE 매개변수로 셈한 가능도이다
        - d은 자유 매개변수의 개수이다
        - n은 표본의 개수이다
        
        인수:
            graph: 방향 비순환 그래프
        
        반환값:
            BIC 점수(높을수록 좋다)
        """
        score = 0.0
        num_parameters = 0
        
        for var in graph.nodes():
            parents = list(graph.predecessors(var))
            
            # 이 변수의 국소 BIC 점수 셈하기
            if not parents:
                # 어버이 없음: 앞확률 분포일 뿐
                local_counts = self.data[var].value_counts()
                n_local = local_counts.sum()
                
                # 로그가능도
                for count in local_counts:
                    if count > 0:
                        p = count / n_local
                        score += count * np.log(p)
                
                # 벌점 항
                cardinality = len(local_counts)
                num_parameters += cardinality - 1
            
            else:
                # 어버이 있음: 조건부 분포
                # 어버이 값으로 묶기
                parent_groups = self.data.groupby(parents)[var]
                
                for parent_values, group in parent_groups:
                    counts = group.value_counts()
                    n_local = counts.sum()
                    
                    # 이 어버이 꼴의 로그 가능도
                    for count in counts:
                        if count > 0:
                            p = count / n_local
                            score += count * np.log(p)
                
                # 벌점 항
                var_card = self.data[var].nunique()
                parent_configs = len(parent_groups)
                num_parameters += parent_configs * (var_card - 1)
        
        # BIC 벌점
        penalty = (num_parameters / 2) * np.log(self.n_samples)
        bic = score - penalty
        
        return bic
    
    def hill_climbing(self, max_parents: int = 3, max_iterations: int = 100) -> nx.DiGraph:
        """
        언덕 오르기 뒤지기로 짜임을 배운다.
        
        언덕 오르기:
        1. 빈 그래프(또는 무작위 그래프)에서 시작한다
        2. 걸음마다 변 하나짜리 손질을 모두 해 본다
        3. 점수를 가장 많이 올리는 손질을 한다
        4. 더 나아질 수 없으면 멈춘다
        
        인수:
            max_parents: 마디마다 어버이의 최대 개수
            max_iterations: 되풀이의 최대 횟수
        
        반환값:
            배운 DAG
        """
        print(f"\nHill Climbing Structure Learning")
        print(f"Max parents: {max_parents}, Max iterations: {max_iterations}")
        print("-" * 70)
        
        # 빈 그래프에서 시작
        current_graph = nx.DiGraph()
        current_graph.add_nodes_from(self.variables)
        
        current_score = self.compute_bic_score(current_graph)
        print(f"Initial BIC score: {current_score:.2f}")
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}")
            
            best_graph = None
            best_score = current_score
            best_operation = None
            
            # 변 하나짜리 연산을 모두 해 보기
            
            # 1. 변 더하기
            for var1 in self.variables:
                for var2 in self.variables:
                    if var1 == var2:
                        continue
                    
                    # 변이 이미 있는지 살피기
                    if current_graph.has_edge(var1, var2):
                        continue
                    
                    # 더하면 어버이 최대 개수를 넘는지 살피기
                    if len(list(current_graph.predecessors(var2))) >= max_parents:
                        continue
                    
                    # 변 더해 보기
                    new_graph = current_graph.copy()
                    new_graph.add_edge(var1, var2)
                    
                    # 아직 고리가 없는지 살피기
                    if not nx.is_directed_acyclic_graph(new_graph):
                        continue
                    
                    # 점수 셈하기
                    score = self.compute_bic_score(new_graph)
                    
                    if score > best_score:
                        best_score = score
                        best_graph = new_graph
                        best_operation = f"Add {var1} -> {var2}"
            
            # 2. 변 지우기
            for var1, var2 in current_graph.edges():
                new_graph = current_graph.copy()
                new_graph.remove_edge(var1, var2)
                
                score = self.compute_bic_score(new_graph)
                
                if score > best_score:
                    best_score = score
                    best_graph = new_graph
                    best_operation = f"Remove {var1} -> {var2}"
            
            # 3. 변 뒤집기
            for var1, var2 in current_graph.edges():
                # 뒤집으면 어버이 최대 개수를 넘는지 살피기
                if len(list(current_graph.predecessors(var1))) >= max_parents:
                    continue
                
                new_graph = current_graph.copy()
                new_graph.remove_edge(var1, var2)
                new_graph.add_edge(var2, var1)
                
                # 아직 고리가 없는지 살피기
                if not nx.is_directed_acyclic_graph(new_graph):
                    continue
                
                score = self.compute_bic_score(new_graph)
                
                if score > best_score:
                    best_score = score
                    best_graph = new_graph
                    best_operation = f"Reverse {var1} -> {var2}"
            
            # 나아진 것을 찾았는지 살피기
            if best_graph is None:
                print("No improvement found. Converged!")
                break
            
            print(f"Best operation: {best_operation}")
            print(f"BIC score: {current_score:.2f} -> {best_score:.2f} (Δ = {best_score - current_score:.2f})")
            
            current_graph = best_graph
            current_score = best_score
        
        print(f"\nFinal graph has {current_graph.number_of_edges()} edges")
        print(f"Final BIC score: {current_score:.2f}")
        
        return current_graph


def demonstrate_structure_learning():
    """흉내 낸 자료로 짜임 배우기를 보인다."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Structure Learning")
    print("="*70)
    
    # 알려진 짜임에서 흉내 낸 자료 만들기
    print("\nGenerating synthetic data from known structure...")
    
    # 단순한 망 만들기: A -> B -> C, A -> C
    np.random.seed(42)
    n_samples = 1000
    
    # A 표집(앞확률)
    A = np.random.binomial(1, 0.5, n_samples)
    
    # A이 주어졌을 때 B 표집
    B = np.random.binomial(1, 0.3 + 0.4 * A)
    
    # A과 B이 주어졌을 때 C 표집
    prob_C = 0.2 + 0.3 * A + 0.4 * B
    C = np.random.binomial(1, prob_C)
    
    # DataFrame 만들기
    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    
    print(f"Generated {n_samples} samples")
    print(f"True structure: A -> B -> C, A -> C")
    
    # 언덕 오르기로 짜임 배우기
    print("\n" + "="*70)
    print("Score-Based Learning (Hill Climbing)")
    print("="*70)
    
    learner = ScoreBasedLearning(data)
    learned_graph = learner.hill_climbing(max_parents=2, max_iterations=20)
    
    # 배운 짜임 그려 보기
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(learned_graph)
    nx.draw(learned_graph, pos,
            with_labels=True,
            node_color='lightgreen',
            node_size=3000,
            font_size=16,
            font_weight='bold',
            arrows=True,
            arrowsize=30,
            edge_color='gray',
            width=2)
    plt.title("Learned Bayesian Network Structure", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nLearned edges:", list(learned_graph.edges()))


def main():
    """주된 보여 주기."""
    print("\n" + "="*70)
    print("STRUCTURE LEARNING FOR BAYESIAN NETWORKS")
    print("="*70)
    
    print("\nStructure learning is the problem of discovering")
    print("the graph structure from observational data.")
    
    demonstrate_structure_learning()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. Two main approaches:")
    print("   - Constraint-based (PC algorithm): Test independencies")
    print("   - Score-based: Optimize scoring function")
    
    print("\n2. Structure learning is hard!")
    print("   - Super-exponential space of DAGs")
    print("   - NP-hard in general")
    print("   - Need heuristic search methods")
    
    print("\n3. Challenges:")
    print("   - Observational equivalence (multiple DAGs fit data equally)")
    print("   - Limited data (statistical power)")
    print("   - Computational complexity")
    
    print("\n4. Practical considerations:")
    print("   - Incorporate domain knowledge")
    print("   - Limit search space (max parents)")
    print("   - Use appropriate scoring functions")
    print("   - Validate learned structures")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()```

## 2. 논의

PC 알고리즘 같은 제약 기반 방법은 완전 무방향 그래프에서 시작해 조건부 독립을 검정하며 변을 차근차근 지운다. 변수 짝마다 다른 변수의 여러 부분집합을 조건으로 두었을 때 조건부 독립인지 검정한다. 독립이 드러나면 변을 지우고 나중에 변의 방향을 정하려고 가름 집합을 적어 둔다.

BIC를 쓴 언덕 오르기 같은 점수 기반 방법은 점수 함수를 정하고 DAG의 공간을 뒤진다. BIC 점수는 모형의 맞음새(로그 가능도)와 복잡함(매개변수 개수)의 균형을 잡는다. 곧 $\text{BIC}(G, D) = \log P(D|G, \hat{\theta}) - \frac{d}{2}\log n$이다. 언덕 오르기는 빈 그래프에서 시작해 변을 더하거나 지우거나 뒤집기를 되풀이하며, 늘 점수를 가장 많이 올리는 손질을 고른다.

두 길 모두 근본적인 어려움을 만난다. DAG의 공간이 지수보다 빠르게 커져 남김없이 뒤지기는 할 수 없다. 여러 DAG이 관측으로는 같을(마르코프 같음) 수 있는데, 이는 같은 조건부 독립 묶음을 담고 있어 자료만으로는 갈라낼 수 없다는 뜻이다. 실전에서는 분야 지식을 넣어 뒤지는 공간을 좁히고 배운 짜임을 확인하는 일이 꼭 필요하다.

## 연습문제

**연습문제 1.**
알려진 짜임 $A \to B \to C$에서 뽑은 표본 500개의 자료가 있을 때, 카이제곱 검정으로 $A \perp C | B$이지만 $A \not\perp C$임을 확인하여라.

??? success "연습문제 1 풀이"
    ```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

np.random.seed(42)
n = 500
A = np.random.binomial(1, 0.5, n)
B = np.random.binomial(1, 0.3 + 0.4 * A)
C = np.random.binomial(1, 0.2 + 0.5 * B)
df = pd.DataFrame({'A': A, 'B': B, 'C': C})

# A와 C의 독립 검정(주변)
ct = pd.crosstab(df['A'], df['C'])
_, p_marginal, _, _ = chi2_contingency(ct)
print(f'A indep C (marginal): p={p_marginal:.4f} -> {"Yes" if p_marginal > 0.05 else "No"}')

# A와 C의 독립 검정 | B=0
df0 = df[df['B'] == 0]
ct0 = pd.crosstab(df0['A'], df0['C'])
_, p0, _, _ = chi2_contingency(ct0)

# A와 C의 독립 검정 | B=1
df1 = df[df['B'] == 1]
ct1 = pd.crosstab(df1['A'], df1['C'])
_, p1, _, _ = chi2_contingency(ct1)

print(f'A indep C | B=0: p={p0:.4f}')
print(f'A indep C | B=1: p={p1:.4f}')
print(f'Conditionally independent? {min(p0, p1) > 0.05}')
```


---

**연습문제 2.**
마르코프 같음의 개념을 설명하여라. 마르코프로 같은 서로 다른 DAG 둘의 보기와, 같지 않은 것 하나의 보기를 들어라.

??? success "연습문제 2 풀이"
    DAG 둘이 같은 조건부 독립 관계 묶음을 담고 있으면 마르코프로 같다. 마르코프로 같은 DAG은 뼈대(무방향 변)가 같고 v-짜임($X$과 $Y$이 이웃하지 않는 충돌자 $X \to Z \leftarrow Y$)도 같다.

같은 DAG의 보기: $A \to B \to C$, $A \leftarrow B \leftarrow C$, $A \leftarrow B \to C$. 셋 다 뼈대($A - B - C$)가 같고 v-짜임이 없다. 셋 다 조건부 독립을 꼭 하나 담는다. 곧 $A \perp C | B$이다.

같지 않은 것: $A \to B \leftarrow C$(충돌자/v-짜임). 이는 $A \perp C$이지만 $A \not\perp C | B$(설명해 치우기)을 담는다. 같은 뼈대의 다른 어떤 DAG도 이 무늬를 담지 못한다.

곧 관측 자료로 짜임을 배우면 마르코프 같음 갈래까지만 가려낼 수 있고 특정 DAG은 알 수 없다는 뜻이다. 같음 갈래 안에서 갈라내려면 개입 자료나 분야 지식이 필요하다.


---

**연습문제 3.**
변수 3개의 단순한 망에 대해 BIC 점수를 구현하고, $A \to B, A \to C$에서 만든 자료에 가장 좋은 짜임을 찾도록 마디 3개의 모든 DAG의 점수를 견주어라.

??? success "연습문제 3 풀이"
    ```python
import numpy as np
import pandas as pd
from itertools import product

np.random.seed(42)
n = 1000
A = np.random.binomial(1, 0.5, n)
B = np.random.binomial(1, 0.3 + 0.4*A)
C = np.random.binomial(1, 0.2 + 0.5*A)
df = pd.DataFrame({'A': A, 'B': B, 'C': C})

def local_bic(df, var, parents):
    n = len(df)
    if not parents:
        counts = df[var].value_counts()
        ll = sum(c * np.log(c/n) for c in counts)
        return ll - 0.5 * np.log(n)
    groups = df.groupby(parents)[var]
    ll = 0
    n_params = 0
    for _, group in groups:
        counts = group.value_counts()
        total = counts.sum()
        ll += sum(c * np.log(c/total) for c in counts)
        n_params += len(counts) - 1
    return ll - 0.5 * n_params * np.log(n)

# 핵심 짜임 검정하기
structures = {
    'A->B, A->C': {'A': [], 'B': ['A'], 'C': ['A']},
    'B->A, C->A': {'A': ['B','C'], 'B': [], 'C': []},
    'A->B->C': {'A': [], 'B': ['A'], 'C': ['B']},
    'Empty': {'A': [], 'B': [], 'C': []},
}

for name, struct in structures.items():
    score = sum(local_bic(df, v, p) for v, p in struct.items())
    print(f'{name:20s}: BIC = {score:.2f}')
```
참 짜임 $A \to B, A \to C$이 가장 높은(가장 덜 음인) BIC 점수를 받아야 한다.

## 정리하며

**다룬 것** — 짜임 배우기

PC 알고리즘 같은 제약 기반 방법은 완전 무방향 그래프에서 시작해 조건부 독립을 검정하며 변을 차근차근 지운다.

고갱이 갈래는 `IndependenceTest`, `PCAlgorithm`, `ScoreBasedLearning`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
