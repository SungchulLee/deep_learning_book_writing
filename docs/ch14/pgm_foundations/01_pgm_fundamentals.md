# PGM의 바탕

확률 그래프 모형(PGM)은 그래프 짜임을 써서 복잡한 결합 확률 분포를 간결하게 나타내는 얼개를 준다. 마디와 변으로 독립 관계를 담아, PGM은 분포를 못 박는 데 필요한 매개변수의 개수를 지수에서 다룰 만한 수준으로 줄인다. 이 모듈은 핵심 개념인 확률 분포, 독립, 조건부 독립, 방향 그래프, d-가름, 쪼개기를 들여온다.

## 코드

```python
"""
확률 그래프 모형 — 바탕
==============================================

이 모듈은 확률 그래프 모형(PGM)의 바탕 개념을 들여온다.

학습 목표:
-------------------
1. 확률 그래프 모형이 무엇이고 왜 쓸모 있는지 이해하기
2. 확률 분포의 그래프 표현 배우기
3. 독립과 조건부 독립 개념 익히기
4. 방향 그래프에서의 d-가름 이해하기
5. 결합 확률 분포를 간결하게 나타내는 법 배우기

수학의 바탕:
------------------------
- 결합 확률: P(X1, X2, ..., Xn)
- 조건부 확률: P(X|Y) = P(X,Y) / P(Y)
- 사슬 규칙: P(X1,...,Xn) = ∏ P(Xi | X1,...,Xi-1)
- 독립: P(X,Y) = P(X)P(Y)
- 조건부 독립: P(X,Y|Z) = P(X|Z)P(Y|Z)

지은이: 교육용 ML 팀
수준: 첫걸음
미리 알아야 할 것: 기본 확률 이론
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional
from itertools import product

# ========================================================================
# 메인
# ========================================================================


class ProbabilityDistribution:
    """
    확률 변수 묶음에 걸친 이산 확률 분포를 나타낸다.
    
    이 클래스는 주변화, 조건 걸기, 독립 검정을 비롯해 확률 분포를 다루는
    기본 연산을 준다.
    
    속성:
        variables: 확률 변수 이름의 목록
        cardinalities: 변수 이름을 가짓수에 잇는 사전
        values: 확률 값을 담은 NumPy 배열
    """
    
    def __init__(self, 
                 variables: List[str], 
                 cardinalities: Dict[str, int],
                 values: Optional[np.ndarray] = None):
        """
        확률 분포의 첫걸음을 잡는다.
        
        인수:
            variables: 변수 이름의 목록(이를테면 ['X', 'Y', 'Z'])
            cardinalities: 가짓수 사전(이를테면 {'X': 2, 'Y': 3, 'Z': 2})
            values: 있어도 되고 없어도 되는 확률 값 배열. None이면 고른 분포
        
        보기:
            # 이진 변수 둘에 걸친 분포 만들기
            dist = ProbabilityDistribution(
                variables=['X', 'Y'],
                cardinalities={'X': 2, 'Y': 2},
                values=np.array([[0.3, 0.2], [0.4, 0.1]])
            )
        """
        self.variables = variables
        self.cardinalities = cardinalities
        
        # 분포 배열의 꼴 셈하기
        shape = tuple(cardinalities[var] for var in variables)
        
        if values is None:
            # 고른 분포로 첫걸음 잡기
            total_size = np.prod(shape)
            self.values = np.ones(shape) / total_size
        else:
            self.values = np.array(values)
            # 분포가 고르게 되었는지 보장하기
            self.values = self.values / np.sum(self.values)
        
        # 꼴이 맞는지 확인하기
        assert self.values.shape == shape, \
            f"Values shape {self.values.shape} doesn't match expected {shape}"
    
    def marginalize(self, variables_to_keep: List[str]) -> 'ProbabilityDistribution':
        """
        variables_to_keep에 없는 변수를 주변화한다.
        
        주변화는 변수를 합으로 지워 부분 변수에 걸친 분포를
        얻는 일이다.
        
        수학의 정의:
            P(X) = Σ_Y P(X, Y)
        
        인수:
            variables_to_keep: 분포에 남길 변수의 목록
        
        반환값:
            정해진 변수에 걸친 새 ProbabilityDistribution
        
        보기:
            # P(X, Y, Z)이 있으면 Z을 주변화해 P(X, Y)을 얻을 수 있다
            joint = ProbabilityDistribution(['X', 'Y', 'Z'], ...)
            marginal = joint.marginalize(['X', 'Y'])  # 이것이 P(X, Y)을 준다
        """
        # 합으로 지울 변수 찾기
        variables_to_sum = [var for var in self.variables if var not in variables_to_keep]
        
        # 지울 변수가 없으면 베낀 것을 되돌린다
        if not variables_to_sum:
            return ProbabilityDistribution(
                self.variables, 
                self.cardinalities, 
                self.values.copy()
            )
        
        # 지울 변수에 해당하는 축 찾기
        axes_to_sum = tuple(self.variables.index(var) for var in variables_to_sum)
        
        # 알맞은 축에 걸쳐 합해 주변화하기
        marginalized_values = np.sum(self.values, axis=axes_to_sum)
        
        # 새 가짓수 사전 만들기
        new_cardinalities = {var: self.cardinalities[var] for var in variables_to_keep}
        
        return ProbabilityDistribution(
            variables_to_keep,
            new_cardinalities,
            marginalized_values
        )
    
    def condition(self, evidence: Dict[str, int]) -> 'ProbabilityDistribution':
        """
        관측한 증거에 분포를 조건 건다.
        
        조건 걸기는 어떤 변수를 관측 값으로 붙박아 두고 분포를
        다시 고르게 하는 일이다.
        
        수학의 정의:
            P(X|Y=y) = P(X, Y=y) / P(Y=y)
        
        인수:
            evidence: 변수 이름을 관측 값에 잇는 사전
                     (이를테면 {'X': 0, 'Y': 1})
        
        반환값:
            증거에 조건 건 새 ProbabilityDistribution
        
        보기:
            # P(X, Y, Z)에 Y=1을 조건으로 걸어 P(X, Z | Y=1)을 얻기
            joint = ProbabilityDistribution(['X', 'Y', 'Z'], ...)
            conditional = joint.condition({'Y': 1})
        """
        # 알맞은 값을 고르는 조각 만들기
        slice_indices = []
        remaining_variables = []
        remaining_cardinalities = {}
        
        for var in self.variables:
            if var in evidence:
                # 이 변수는 관측되었으니 그 값을 고른다
                slice_indices.append(evidence[var])
            else:
                # 이 변수는 관측되지 않았으니 모든 값을 남긴다
                slice_indices.append(slice(None))
                remaining_variables.append(var)
                remaining_cardinalities[var] = self.cardinalities[var]
        
        # 조건 건 값 뽑아내기
        conditioned_values = self.values[tuple(slice_indices)]
        
        # 제대로 된 확률 분포가 되도록 고르게 하기
        # P(X|Y=y) = P(X,Y=y) / Σ_X P(X,Y=y)
        total = np.sum(conditioned_values)
        if total > 0:
            conditioned_values = conditioned_values / total
        else:
            # 합이 0이면 그 증거는 있을 수 없다
            print("Warning: Evidence has probability 0!")
            conditioned_values = np.ones_like(conditioned_values) / conditioned_values.size
        
        return ProbabilityDistribution(
            remaining_variables,
            remaining_cardinalities,
            conditioned_values
        )
    
    def is_independent(self, var1: str, var2: str, threshold: float = 1e-6) -> bool:
        """
        변수 둘이 독립인지 검정한다.
        
        다음이면 변수 X과 Y이 독립이다:
            X과 Y의 모든 값에 대해 P(X, Y) = P(X) * P(Y)
        
        인수:
            var1: 첫 변수의 이름
            var2: 둘째 변수의 이름
            threshold: 같음을 검정하는 수치 문턱값
        
        반환값:
            변수가 독립이면 True, 아니면 False
        
        보기:
            dist = ProbabilityDistribution(['X', 'Y'], ...)
            if dist.is_independent('X', 'Y'):
                print("X and Y are independent")
        """
        # 주변 분포 얻기
        p_var1 = self.marginalize([var1])
        p_var2 = self.marginalize([var2])
        
        # 이 변수 둘에만 걸친 결합 분포 얻기
        p_joint = self.marginalize([var1, var2])
        
        # 주변 분포의 곱 셈하기: P(X) * P(Y)
        # 바깥곱을 셈해야 한다
        idx1 = p_joint.variables.index(var1)
        idx2 = p_joint.variables.index(var2)
        
        # 퍼뜨리기를 위해 배열 꼴 바꾸기
        shape1 = [1] * len(p_joint.variables)
        shape1[idx1] = self.cardinalities[var1]
        
        shape2 = [1] * len(p_joint.variables)
        shape2[idx2] = self.cardinalities[var2]
        
        marginal_product = (
            p_var1.values.reshape(shape1) * 
            p_var2.values.reshape(shape2)
        )
        
        # P(X,Y) ≈ P(X)P(Y)인지 살피기
        difference = np.abs(p_joint.values - marginal_product)
        return np.all(difference < threshold)
    
    def is_conditionally_independent(self, 
                                    var1: str, 
                                    var2: str, 
                                    given: List[str],
                                    threshold: float = 1e-6) -> bool:
        """
        다른 변수가 주어졌을 때 변수 둘이 조건부 독립인지 검정한다.
        
        다음이면 Z이 주어졌을 때 X과 Y이 조건부 독립이다:
            모든 값에 대해 P(X, Y | Z) = P(X | Z) * P(Y | Z)
        
        이는 Z의 모든 값 z에 대해 다음을 확인해 살필 수 있다:
            P(X, Y, Z=z) = P(X, Z=z) * P(Y, Z=z) / P(Z=z)
        
        인수:
            var1: 첫 변수의 이름
            var2: 둘째 변수의 이름
            given: 조건 변수 이름의 목록
            threshold: 같음을 검정하는 수치 문턱값
        
        반환값:
            조건부 독립이면 True, 아니면 False
        
        보기:
            # X ⊥ Y | Z인지 검정하기(Z을 조건으로 X과 Y이 독립인지)
            if dist.is_conditionally_independent('X', 'Y', ['Z']):
                print("X and Y are conditionally independent given Z")
        """
        # 조건 변수의 모든 값에 대해 이를 살펴야 한다
        # 조건 변수에 있을 수 있는 대입을 모두 얻기
        given_cardinalities = [self.cardinalities[var] for var in given]
        
        for given_values in product(*[range(card) for card in given_cardinalities]):
            # 증거 사전 만들기
            evidence = {var: val for var, val in zip(given, given_values)}
            
            # 이 증거에 조건 걸기
            conditioned = self.condition(evidence)
            
            # 조건 건 분포에서 var1과 var2이 독립인지 살피기
            if not conditioned.is_independent(var1, var2, threshold):
                return False
        
        return True
    
    def __str__(self) -> str:
        """분포의 문자열 표현."""
        return f"P({', '.join(self.variables)})\nShape: {self.values.shape}\nValues:\n{self.values}"


class DirectedGraph:
    """
    베이즈 망을 위한 방향 비순환 그래프(DAG)를 나타낸다.
    
    DAG은 방향 변을 갖고 고리가 없는 그래프이다.
    이것이 베이즈 망을 떠받치는 근본 짜임이다.
    
    속성:
        graph: NetworkX DiGraph 객체
    """
    
    def __init__(self):
        """빈 방향 그래프의 첫걸음을 잡는다."""
        self.graph = nx.DiGraph()
    
    def add_node(self, node: str):
        """
        그래프에 마디(확률 변수)를 더한다.
        
        인수:
            node: 마디/변수의 이름
        """
        self.graph.add_node(node)
    
    def add_edge(self, parent: str, child: str):
        """
        어버이에서 자식으로 가는 방향 변을 더한다.
        
        이 변은 곧바른 확률 기댐을 나타낸다.
        곧 어버이가 자식에 영향을 준다.
        
        인수:
            parent: 어버이 마디의 이름
            child: 자식 마디의 이름
        
        일으키는 예외:
            ValueError: 그 변을 더하면 고리가 생길 때
        """
        # 이 변을 더하면 고리가 생기는지 살피기
        self.graph.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(parent, child)
            raise ValueError(f"Adding edge {parent}->{child} would create a cycle!")
    
    def get_parents(self, node: str) -> List[str]:
        """
        마디의 어버이를 얻는다.
        
        어버이는 이 마디를 가리키는 방향 변을 갖는 마디이다.
        확률로 말하면 이 마디가 곧바로 기대는
        변수들이다.
        
        인수:
            node: 마디의 이름
        
        반환값:
            어버이 마디 이름의 목록
        """
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """
        마디의 자식을 얻는다.
        
        자식은 이 마디가 방향 변으로 가리키는 마디이다.
        
        인수:
            node: 마디의 이름
        
        반환값:
            자식 마디 이름의 목록
        """
        return list(self.graph.successors(node))
    
    def get_ancestors(self, node: str) -> Set[str]:
        """
        마디의 조상을 모두 얻는다.
        
        조상에는 어버이, 어버이의 어버이 등이 든다.
        이 마디로 가는 방향 길이 있는 모든 마디이다.
        
        인수:
            node: 마디의 이름
        
        반환값:
            조상 마디 이름의 묶음
        """
        return nx.ancestors(self.graph, node)
    
    def get_descendants(self, node: str) -> Set[str]:
        """
        마디의 자손을 모두 얻는다.
        
        자손에는 자식, 자식의 자식 등이 든다.
        이 마디에서 가는 방향 길이 있는 모든 마디이다.
        
        인수:
            node: 마디의 이름
        
        반환값:
            자손 마디 이름의 묶음
        """
        return nx.descendants(self.graph, node)
    
    def topological_order(self) -> List[str]:
        """
        마디의 위상 차례를 얻는다.
        
        위상 차례는 변 (u,v)마다 u이 v보다 앞에 오도록
        마디를 늘어놓은 것이다.
        
        다음에 쓸모 있다:
        1. 사슬 규칙으로 결합 확률 셈하기
        2. 앞먹임 표집
        3. 여러 추론 알고리즘
        
        반환값:
            위상 차례로 늘어놓은 마디의 목록
        """
        return list(nx.topological_sort(self.graph))
    
    def is_d_separated(self, 
                      X: Set[str], 
                      Y: Set[str], 
                      Z: Set[str]) -> bool:
        """
        Z이 주어졌을 때 X과 Y이 d-갈리는지 검정한다.
        
        d-가름(방향 가름)은 베이즈 망에서 조건부 독립을 가리는
        그래프 기준이다.
        
        Z이 주어졌을 때 X과 Y이 d-갈리면 X ⊥ Y | Z이다
        (Z이 주어졌을 때 X과 Y이 조건부 독립이다).
        
        d-가름 규칙:
        1. 사슬: X -> Z -> Y: Z이 주어지면 X과 Y이 d-갈린다
        2. 갈래: X <- Z -> Y: Z이 주어지면 X과 Y이 d-갈린다
        3. 충돌자: X -> Z <- Y: Z이 주어지면 X과 Y이 d-갈리지 않는다
                     그러나 Z을 관측하지 않으면 d-갈린다
        
        인수:
            X: 첫 무리의 마디 묶음
            Y: 둘째 무리의 마디 묶음
            Z: 조건 마디의 묶음
        
        반환값:
            Z이 주어졌을 때 X과 Y이 d-갈리면 True
        
        보기:
            graph = DirectedGraph()
            graph.add_edge('X', 'Z')
            graph.add_edge('Z', 'Y')
            # Z을 조건으로 X과 Y은 d-갈린다(사슬 짜임)
            is_sep = graph.is_d_separated({'X'}, {'Y'}, {'Z'})  # 참
        """
        # NetworkX이 d-가름 검정을 준다
        return nx.d_separated(self.graph, X, Y, Z)
    
    def visualize(self, title: str = "Directed Graph", figsize: Tuple[int, int] = (10, 6)):
        """
        방향 그래프를 그려 본다.
        
        인수:
            title: 그림의 제목
            figsize: 그림 크기(너비, 높이)
        """
        plt.figure(figsize=figsize)
        
        # 더 잘 보이도록 층 배치를 쓴다
        try:
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        except:
            pos = nx.spring_layout(self.graph)
        
        # 그래프 그리기
        nx.draw(self.graph, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=12,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                width=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def demonstrate_independence():
    """
    보기를 들어 독립 개념을 보인다.
    
    이 함수는 독립과 조건부 독립을 보이는 분포를 만들어
    그 개념을 그려 보게 돕는다.
    """
    print("=" * 70)
    print("DEMONSTRATION: Independence vs Conditional Independence")
    print("=" * 70)
    
    # 보기 1: 독립인 변수
    # 동전 두 번 던지기 — 서로 독립이다
    print("\nExample 1: Two Independent Coin Flips")
    print("-" * 70)
    
    # 둘 다 공평하고 독립일 때의 P(Coin1, Coin2)
    # P(C1=H, C2=H) = 0.25, P(C1=H, C2=T) = 0.25 등
    independent_dist = ProbabilityDistribution(
        variables=['Coin1', 'Coin2'],
        cardinalities={'Coin1': 2, 'Coin2': 2},
        values=np.array([[0.25, 0.25],  # Coin1=0(뒷면)
                        [0.25, 0.25]])   # Coin1=1(앞면)
    )
    
    print(f"Joint distribution P(Coin1, Coin2):")
    print(independent_dist.values)
    print(f"\nAre Coin1 and Coin2 independent? {independent_dist.is_independent('Coin1', 'Coin2')}")
    
    # 보기 2: 서로 기대는 변수
    # 날씨가 우산을 드는지에 영향을 준다
    print("\n\nExample 2: Dependent Variables (Weather and Umbrella)")
    print("-" * 70)
    
    # P(Rain, Umbrella)
    # 비가 오면 우산을 들 가능성이 높다
    dependent_dist = ProbabilityDistribution(
        variables=['Rain', 'Umbrella'],
        cardinalities={'Rain': 2, 'Umbrella': 2},
        values=np.array([[0.50, 0.05],  # Rain=0(비 안 옴)
                        [0.05, 0.40]])   # Rain=1(비 옴)
    )
    
    print(f"Joint distribution P(Rain, Umbrella):")
    print(dependent_dist.values)
    print(f"\nAre Rain and Umbrella independent? {dependent_dist.is_independent('Rain', 'Umbrella')}")
    
    # 보기 3: 조건부 독립
    # X -> Z -> Y 짜임(사슬)
    print("\n\nExample 3: Conditional Independence (Chain Structure)")
    print("-" * 70)
    print("Structure: X -> Z -> Y")
    print("X and Y are conditionally independent given Z")
    
    # 조건부 독립을 보이는 분포 만들기
    # X이 Z에, Z이 Y에 영향을 줄 때의 P(X, Z, Y)
    chain_values = np.zeros((2, 2, 2))
    # 사슬 규칙으로 쌓기: P(X,Z,Y) = P(X) P(Z|X) P(Y|Z)
    
    # P(X)
    p_x = np.array([0.6, 0.4])
    
    # P(Z|X) — X이 Z에 영향을 준다
    p_z_given_x = np.array([[0.8, 0.2],  # Z|X=0
                            [0.3, 0.7]])  # Z|X=1
    
    # P(Y|Z) — Z이 Y에 영향을 준다
    p_y_given_z = np.array([[0.7, 0.3],  # Y|Z=0
                            [0.2, 0.8]])  # Y|Z=1
    
    for x in range(2):
        for z in range(2):
            for y in range(2):
                chain_values[x, z, y] = p_x[x] * p_z_given_x[x, z] * p_y_given_z[z, y]
    
    chain_dist = ProbabilityDistribution(
        variables=['X', 'Z', 'Y'],
        cardinalities={'X': 2, 'Z': 2, 'Y': 2},
        values=chain_values
    )
    
    print(f"\nAre X and Y independent? {chain_dist.is_independent('X', 'Y')}")
    print(f"Are X and Y conditionally independent given Z? "
          f"{chain_dist.is_conditionally_independent('X', 'Y', ['Z'])}")
    
    print("\nIntuition: Once we know Z, knowing X doesn't give us additional")
    print("information about Y. All the influence from X to Y goes through Z.")


def demonstrate_d_separation():
    """
    서로 다른 그래프 짜임으로 d-가름을 보인다.
    
    d-가름은 베이즈 망에서 조건부 독립을 이해하는
    핵심 개념이다.
    """
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION: D-Separation in Different Structures")
    print("=" * 70)
    
    # 짜임 1: 사슬(X -> Z -> Y)
    print("\nStructure 1: Chain (X -> Z -> Y)")
    print("-" * 70)
    chain = DirectedGraph()
    chain.add_node('X')
    chain.add_node('Z')
    chain.add_node('Y')
    chain.add_edge('X', 'Z')
    chain.add_edge('Z', 'Y')
    
    print("Graph: X -> Z -> Y")
    print(f"X ⊥ Y | Z? {chain.is_d_separated({'X'}, {'Y'}, {'Z'})} (should be True)")
    print(f"X ⊥ Y | ∅? {chain.is_d_separated({'X'}, {'Y'}, set())} (should be False)")
    print("\nIntuition: Information flows from X to Y through Z.")
    print("If we observe Z, the path is blocked.")
    
    # 짜임 2: 갈래(X <- Z -> Y)
    print("\n\nStructure 2: Fork (X <- Z -> Y)")
    print("-" * 70)
    fork = DirectedGraph()
    fork.add_node('X')
    fork.add_node('Z')
    fork.add_node('Y')
    fork.add_edge('Z', 'X')
    fork.add_edge('Z', 'Y')
    
    print("Graph: X <- Z -> Y")
    print(f"X ⊥ Y | Z? {fork.is_d_separated({'X'}, {'Y'}, {'Z'})} (should be True)")
    print(f"X ⊥ Y | ∅? {fork.is_d_separated({'X'}, {'Y'}, set())} (should be False)")
    print("\nIntuition: Z is a common cause of X and Y.")
    print("If we observe Z, X and Y become independent.")
    
    # 짜임 3: 충돌자(X -> Z <- Y)
    print("\n\nStructure 3: Collider (X -> Z <- Y)")
    print("-" * 70)
    collider = DirectedGraph()
    collider.add_node('X')
    collider.add_node('Z')
    collider.add_node('Y')
    collider.add_edge('X', 'Z')
    collider.add_edge('Y', 'Z')
    
    print("Graph: X -> Z <- Y")
    print(f"X ⊥ Y | Z? {collider.is_d_separated({'X'}, {'Y'}, {'Z'})} (should be False)")
    print(f"X ⊥ Y | ∅? {collider.is_d_separated({'X'}, {'Y'}, set())} (should be True)")
    print("\nIntuition: Z is a common effect of X and Y.")
    print("If we DON'T observe Z, X and Y are independent.")
    print("If we DO observe Z, X and Y become dependent (explaining away effect).")
    
    # 세 짜임 모두 그려 보기
    print("\n\nVisualizing all three structures...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, (graph, title) in zip(axes, 
                                   [(chain.graph, "Chain: X → Z → Y"),
                                    (fork.graph, "Fork: X ← Z → Y"),
                                    (collider.graph, "Collider: X → Z ← Y")]):
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, ax=ax,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=14,
                font_weight='bold',
                arrows=True,
                arrowsize=20)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def demonstrate_factorization():
    """
    베이즈 망이 결합 분포를 어떻게 쪼개는지 보인다.
    
    핵심 통찰: 변수 X1, ..., Xn에 걸친 베이즈 망은
    결합 분포를 다음과 같이 나타낸다:
    
    P(X1, ..., Xn) = ∏ P(Xi | Parents(Xi))
    
    이 쪼개기가 셈을 훨씬 효율적으로 만든다.
    """
    print("\n\n" + "=" * 70)
    print("DEMONSTRATION: Factorization in Bayesian Networks")
    print("=" * 70)
    
    print("\nConsider a simple alarm system:")
    print("- Burglary and Earthquake are independent events")
    print("- Alarm goes off if Burglary OR Earthquake occurs")
    print("- John and Mary call if they hear the Alarm")
    print("\nStructure: Burglary -> Alarm <- Earthquake")
    print("           Alarm -> JohnCalls")
    print("           Alarm -> MaryCalls")
    
    # 그래프 만들기
    graph = DirectedGraph()
    for node in ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls']:
        graph.add_node(node)
    
    graph.add_edge('Burglary', 'Alarm')
    graph.add_edge('Earthquake', 'Alarm')
    graph.add_edge('Alarm', 'JohnCalls')
    graph.add_edge('Alarm', 'MaryCalls')
    
    print("\n\nNaive joint distribution representation:")
    print("-" * 70)
    print("Without structure: P(B, E, A, J, M)")
    print("Number of parameters: 2^5 - 1 = 31 independent parameters")
    print("(We need to store probability for each of 32 possible combinations)")
    
    print("\n\nFactorized representation using Bayesian network:")
    print("-" * 70)
    print("P(B, E, A, J, M) = P(B) × P(E) × P(A|B,E) × P(J|A) × P(M|A)")
    print("\nNumber of parameters:")
    print("- P(B): 1 parameter (probability of burglary)")
    print("- P(E): 1 parameter (probability of earthquake)")
    print("- P(A|B,E): 4 parameters (2×2 combinations of B and E)")
    print("- P(J|A): 2 parameters (2 values of A)")
    print("- P(M|A): 2 parameters (2 values of A)")
    print("Total: 1 + 1 + 4 + 2 + 2 = 10 parameters")
    print("\nSpace savings: 31 vs 10 parameters (68% reduction!)")
    
    print("\n\nThis factorization also enables efficient inference:")
    print("- We can compute conditional probabilities efficiently")
    print("- We can perform reasoning with incomplete information")
    print("- We can identify independence relationships")
    
    # 망 그려 보기
    graph.visualize("Alarm Network: Factorized Representation")


def main():
    """
    모든 보여 주기를 돌리는 주된 함수.
    
    이는 구체적인 보기와 그림으로 PGM의 바탕을
    두루 갖춰 들여온다.
    """
    print("\n" + "=" * 70)
    print("PROBABILISTIC GRAPHICAL MODELS - FUNDAMENTALS")
    print("=" * 70)
    print("\nThis module introduces the core concepts of PGMs:")
    print("1. Probability distributions and their operations")
    print("2. Independence and conditional independence")
    print("3. Graphical representations (directed graphs)")
    print("4. D-separation")
    print("5. Factorization of joint distributions")
    
    # 시연 실행
    demonstrate_independence()
    demonstrate_d_separation()
    demonstrate_factorization()
    
    print("\n\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n1. PGMs provide a compact representation of joint distributions")
    print("   using graph structure to encode independence relationships.")
    
    print("\n2. Independence and conditional independence are different:")
    print("   - Independent: P(X,Y) = P(X)P(Y)")
    print("   - Conditionally independent: P(X,Y|Z) = P(X|Z)P(Y|Z)")
    
    print("\n3. D-separation is a graphical test for conditional independence:")
    print("   - Chain & Fork: Z blocks the path when observed")
    print("   - Collider: Z blocks the path when NOT observed")
    
    print("\n4. Factorization enables efficient computation:")
    print("   P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))")
    
    print("\n5. These concepts are fundamental to all graphical models:")
    print("   - Bayesian networks (next module)")
    print("   - Markov random fields")
    print("   - Factor graphs")
    print("   - And many more...")
    
    print("\n" + "=" * 70)
    print("Next: Learn how to build and use Bayesian networks!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()```

## 논의

PGM은 그래프 이론과 확률 이론을 합쳐 높은 차원의 분포를 간결하게 나타낸다. 핵심 통찰은 실제 세상의 변수 대부분이 서로 곧바로 기대고 있지는 않다는 것이다. 이 독립 관계를 알아내어 그래프 짜임에 담으면 결합 분포를 더 작고 다루기 쉬운 성분으로 쪼갤 수 있다.

코드는 주변화, 조건 걸기, 독립 검정을 받쳐 주는 `ProbabilityDistribution` 클래스를 구현한다. 이 연산들이 모든 PGM 알고리즘의 셈 바탕을 이룬다. 주변화는 변수를 합으로 지운다. 곧 $P(X) = \sum_Y P(X, Y)$이다. 조건 걸기는 관측한 변수를 붙박아 두고 다시 고르게 한다. 곧 $P(X|Y=y) = P(X, Y=y) / P(Y=y)$이다.

d-가름은 망의 짜임에서 조건부 독립을 읽어 내는 그래프 기준이다. 근본이 되는 세 짜임인 사슬($X \to Z \to Y$), 갈래($X \leftarrow Z \to Y$), 충돌자($X \to Z \leftarrow Y$)는 서로 다르게 굴러간다. 가운데 마디에 조건을 걸면 사슬과 갈래에서는 정보의 흐름이 막히지만 충돌자에서는 오히려 열린다. 충돌자의 이 놀라운 굴러감('설명해 치우기' 효과)은 인과 추론에서 가장 중요한 개념 가운데 하나이다.

## 연습문제

**연습문제 1.**
질병이 증상1과 증상2을 함께 일으키는 의료 진단 얼개를 나타내는 이진 변수 셋의 분포를 만들어라. 두 증상이 질병을 조건으로 두면 독립이지만 주변으로는 기대고 있음을 확인하여라.

??? success "연습문제 1 풀이"
    ```python
import numpy as np

# P(D=1)=0.1
# P(S1=1|D=0)=0.05, P(S1=1|D=1)=0.8
# P(S2=1|D=0)=0.1, P(S2=1|D=1)=0.7
values = np.zeros((2, 2, 2))  # D, S1, S2
for d in range(2):
    p_d = 0.1 if d == 1 else 0.9
    for s1 in range(2):
        p_s1 = (0.8 if s1 == 1 else 0.2) if d == 1 else (0.05 if s1 == 1 else 0.95)
        for s2 in range(2):
            p_s2 = (0.7 if s2 == 1 else 0.3) if d == 1 else (0.1 if s2 == 1 else 0.9)
            values[d, s1, s2] = p_d * p_s1 * p_s2

# 주변 분포 P(S1, S2)
p_s1_s2 = values.sum(axis=0)
p_s1 = p_s1_s2.sum(axis=1)
p_s2 = p_s1_s2.sum(axis=0)

# 주변 독립 살피기
print('Marginal P(S1,S2):', p_s1_s2)
print('P(S1)*P(S2):', np.outer(p_s1, p_s2))
print('Marginally independent?', np.allclose(p_s1_s2, np.outer(p_s1, p_s2)))

# D=0을 조건으로 한 조건부 독립 살피기
p_given_d0 = values[0] / values[0].sum()
p_s1_d0 = p_given_d0.sum(axis=1)
p_s2_d0 = p_given_d0.sum(axis=0)
print('Cond. independent given D=0?', np.allclose(p_given_d0, np.outer(p_s1_d0, p_s2_d0)))
```
증상은 질병을 조건으로 두면 독립이지만(갈래 짜임) 주변으로는 기대고 있다. 한 증상을 보면 숨은 질병에 대한 정보를 얻고, 그것이 다시 다른 증상의 확률을 바꾸기 때문이다.


---

**연습문제 2.**
충돌자 짜임에서의 '설명해 치우기' 효과를 설명하여라. 실제 세상의 구체적인 보기를 들어라.

??? success "연습문제 2 풀이"
    충돌자 짜임 $X \to Z \leftarrow Y$에서 변수 $X$과 $Y$은 주변으로는 독립이다. 그러나 함께 낳은 결과 $Z$을 보고 나면(조건을 걸면) $X$과 $Y$이 서로 기대게 된다. 이를 '설명해 치우기'라고 한다.

실제 보기: 머리 좋음($X$)과 공부 시간($Y$)이 함께 시험 점수($Z$)를 낳는다고 하자. 어떤 학생은 머리는 좋으나 게으를 수 있고, 딱히 타고나지는 않았으나 부지런할 수 있으며, 둘 다 높은 점수로 이어질 수 있다. 시험 점수를 모르면 머리 좋음과 공부 시간은 서로 상관없는 성질이다. 그러나 어떤 학생이 아주 높은 점수를 받았음을 알고($Z$에 조건을 걸고) 그 학생이 별로 공부하지 않았음($Y$이 낮음)을 알게 되면, 우리는 그 학생이 머리가 좋음($X$이 높음)을 미루어 알게 된다. $Z$을 보는 일이 $X$과 $Y$ 사이에 기댐을 만든다.

수학으로 쓰면 $P(X) = P(X|Y)$인데도 $P(X | Z) \neq P(X | Y, Z)$이다.


---

**연습문제 3.**
짜임이 없는 이진 변수 10개의 결합 분포와, 변수마다 어버이가 많아야 2개인 베이즈 망을 나타내는 데 필요한 매개변수의 개수를 셈하여라. 줄인 비율은 얼마인가?

??? success "연습문제 3 풀이"
    짜임이 없으면 매개변수 $2^{10} - 1 = 1023$개이다(온전한 결합 분포 표).

변수마다 어버이가 많아야 2개인 베이즈 망에서는, 어버이가 $k$개인 변수의 CPT마다 조건부 분포가 $2^k$개이고 저마다 매개변수 1개가 필요하다(이진이므로). 그러므로 변수마다 매개변수 $2^k$개가 필요하다.

- 어버이가 0개인 변수: 저마다 매개변수 1개
- 어버이가 1개인 변수: 저마다 매개변수 2개
- 어버이가 2개인 변수: 저마다 매개변수 4개

최악의 경우(모두 어버이가 2개): 매개변수 $10 \times 4 = 40$개.
줄인 비율: $1023 / 40 \approx 25.6\times$.

이 최악의 경우에도 쪼갠 표현은 온전한 결합 분포에 필요한 매개변수의 대략 4%만 쓴다. 더 크고 성기게 이어진 망에서는 아끼는 정도가 더욱 극적이다.

