# 베이즈 망의 기초

베이즈 망은 가장 흔한 확률 그래프 모형으로, 방향 비순환 그래프(DAG)와 조건부 확률표(CPT)를 합쳐 변수 사이의 인과 관계를 나타낸다. 이 모듈은 베이즈 망을 맨바닥에서 쌓기, CPT 못 박기, 사슬 규칙 쪼개기로 결합 확률 셈하기, 앞먹임 표집으로 표본 만들기를 다룬다.

## 1. 코드

```python
"""
베이즈 망 — 기초
===========================

이 모듈은 인과 관계를 나타내는 데 가장 흔히 쓰이는 확률 그래프
모형인 베이즈 망을 들여온다.

학습 목표:
-------------------
1. 베이즈 망이 무엇이고 무엇으로 이루어지는지 이해하기
2. 조건부 확률표(CPT) 나타내는 법 배우기
3. 단순한 베이즈 망을 맨바닥에서 쌓기
4. 사슬 규칙으로 결합 확률 셈하기
5. 베이즈 망에 기본 물음 던지기

수학의 바탕:
------------------------
베이즈 망은 다음을 만족하는 튜플 (G, P)이다:
- G = (V, E)은 방향 비순환 그래프(DAG)이다
- P = {P(Xi | Parents(Xi))}은 조건부 확률 분포의 묶음이다

이 망은 결합 분포를 다음과 같이 나타낸다:
    P(X1, ..., Xn) = ∏ P(Xi | Parents(Xi))

지은이: 교육용 ML 팀
수준: 첫걸음
미리 볼 것: 01_pgm_fundamentals.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Optional, Union
from itertools import product
import pandas as pd

# ========================================================================
# 메인
# ========================================================================


class ConditionalProbabilityTable:
    """
    변수의 조건부 확률표(CPT)를 나타낸다.
    
    CPT은 어버이 값의 모든 조합에 대해 P(변수 | 어버이)을 못 박는다.
    이것이 베이즈 망의 근본
    밑돌이다.
    
    속성:
        variable: 이 CPT이 밝히는 변수의 이름
        parents: 어버이 변수 이름의 목록
        cardinalities: 모든 변수를 가짓수에 잇는 사전
        table: 조건부 확률을 담은 NumPy 배열
    """
    
    def __init__(self,
                 variable: str,
                 parents: List[str],
                 cardinalities: Dict[str, int],
                 table: Optional[np.ndarray] = None):
        """
        CPT의 첫걸음을 잡는다.
        
        인수:
            variable: 변수의 이름(이를테면 'Rain')
            parents: 어버이 변수 이름의 목록(이를테면 ['Cloudy'])
            cardinalities: 모든 변수의 가짓수 사전
            table: 있어도 되고 없어도 되는 CPT 값. 꼴은
                   (*어버이_가짓수, 변수_가짓수)이어야 한다
                   None이면 고른 분포로 첫걸음을 잡는다
        
        보기:
            # P(Rain | Cloudy)의 CPT
            # Rain도 이진, Cloudy도 이진
            cpt = ConditionalProbabilityTable(
                variable='Rain',
                parents=['Cloudy'],
                cardinalities={'Rain': 2, 'Cloudy': 2},
                table=np.array([[0.8, 0.2],  # P(Rain | Cloudy=0)
                               [0.2, 0.8]])  # P(Rain | Cloudy=1)
            )
        """
        self.variable = variable
        self.parents = parents
        self.cardinalities = cardinalities
        
        # CPT의 꼴 정하기
        # 꼴: (card(어버이1), card(어버이2), ..., card(변수))
        parent_cards = [cardinalities[p] for p in parents]
        var_card = cardinalities[variable]
        shape = tuple(parent_cards + [var_card])
        
        if table is None:
            # 고른 분포로 첫걸음 잡기
            self.table = np.ones(shape) / var_card
        else:
            self.table = np.array(table)
            # 꼴 확인하기
            assert self.table.shape == shape, \
                f"Table shape {self.table.shape} doesn't match expected {shape}"
            # 조건부 분포마다 합이 1인지 확인하기
            self._verify_normalized()
    
    def _verify_normalized(self):
        """
        조건부 확률의 합이 1인지 확인한다.
        
        어버이 값의 꼴마다 변수 값에 걸친 확률의
        합이 1이어야 한다.
        """
        # 마지막 축(변수 자신)에 걸쳐 합하기
        sums = np.sum(self.table, axis=-1)
        if not np.allclose(sums, 1.0):
            print(f"Warning: CPT for {self.variable} is not normalized!")
            print(f"Sums: {sums}")
    
    def get_probability(self, 
                       variable_value: int,
                       parent_values: Dict[str, int]) -> float:
        """
        P(변수=값 | 어버이=어버이 값)을 얻는다.
        
        인수:
            variable_value: 변수의 값(정수 첨자)
            parent_values: 어버이 이름을 값에 잇는 사전
        
        반환값:
            조건부 확률
        
        보기:
            # P(Rain=1 | Cloudy=0) 얻기
            prob = cpt.get_probability(1, {'Cloudy': 0})
        """
        # 표에 닿을 첨자 튜플 쌓기
        index = []
        for parent in self.parents:
            index.append(parent_values[parent])
        index.append(variable_value)
        
        return self.table[tuple(index)]
    
    def set_probability(self,
                       variable_value: int,
                       parent_values: Dict[str, int],
                       probability: float):
        """
        P(변수=값 | 어버이=어버이 값)을 정한다.
        
        인수:
            variable_value: 변수의 값
            parent_values: 어버이 이름을 값에 잇는 사전
            probability: 정할 확률 값
        """
        index = []
        for parent in self.parents:
            index.append(parent_values[parent])
        index.append(variable_value)
        
        self.table[tuple(index)] = probability
    
    def sample(self, parent_values: Dict[str, int]) -> int:
        """
        어버이 값이 주어졌을 때 변수의 값을 표집한다.
        
        이는 베이즈 망의 앞먹임 표집에 쓸모 있다.
        
        인수:
            parent_values: 어버이 이름을 값에 잇는 사전
        
        반환값:
            표집한 값(정수)
        
        보기:
            # Cloudy=1일 때 Rain 값 표집하기
            rain_value = cpt.sample({'Cloudy': 1})
        """
        # 조건부 분포 얻기
        index = tuple(parent_values[p] for p in self.parents)
        probabilities = self.table[index]
        
        # 범주 분포에서 표집하기
        return np.random.choice(len(probabilities), p=probabilities)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        CPT을 읽기 좋은 DataFrame으로 바꾼다.
        
        이는 그려 보기와 벌레잡기에 쓸모 있다.
        
        반환값:
            어버이 값과 조건부 확률을 담은 DataFrame
        """
        rows = []
        
        # 어버이 값의 모든 조합 만들기
        parent_cards = [self.cardinalities[p] for p in self.parents]
        
        if not self.parents:
            # 어버이 없음 — 앞확률 분포일 뿐이다
            for var_val in range(self.cardinalities[self.variable]):
                row = {self.variable: var_val, 'Probability': self.table[var_val]}
                rows.append(row)
        else:
            for parent_combo in product(*[range(c) for c in parent_cards]):
                for var_val in range(self.cardinalities[self.variable]):
                    row = {}
                    # 어버이 값 더하기
                    for parent, pval in zip(self.parents, parent_combo):
                        row[parent] = pval
                    # 변수 값 더하기
                    row[self.variable] = var_val
                    # 확률 더하기
                    index = parent_combo + (var_val,)
                    row['Probability'] = self.table[index]
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def __str__(self) -> str:
        """CPT의 문자열 표현."""
        if not self.parents:
            return f"P({self.variable})\n{self.to_dataframe().to_string(index=False)}"
        else:
            parent_str = ', '.join(self.parents)
            return f"P({self.variable} | {parent_str})\n{self.to_dataframe().to_string(index=False)}"


class BayesianNetwork:
    """
    베이즈 망을 나타낸다. 곧 CPT이 딸린 DAG이다.
    
    베이즈 망은 다음으로 이루어진다:
    1. 방향 비순환 그래프(DAG) 짜임
    2. 마디마다의 조건부 확률표(CPT)
    
    이 망은 P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))을 나타낸다
    
    속성:
        graph: 짜임을 나타내는 NetworkX DiGraph
        cpts: 변수 이름을 CPT에 잇는 사전
        cardinalities: 변수 이름을 가짓수에 잇는 사전
    """
    
    def __init__(self):
        """빈 베이즈 망의 첫걸음을 잡는다."""
        self.graph = nx.DiGraph()
        self.cpts: Dict[str, ConditionalProbabilityTable] = {}
        self.cardinalities: Dict[str, int] = {}
    
    def add_variable(self, name: str, cardinality: int):
        """
        망에 변수(마디)를 더한다.
        
        인수:
            name: 변수 이름(이를테면 'Weather', 'Traffic')
            cardinality: 있을 수 있는 값의 개수(이진이면 2)
        
        보기:
            bn = BayesianNetwork()
            bn.add_variable('Rain', 2)  # 이진 변수
            bn.add_variable('Season', 4)  # 네 계절
        """
        self.graph.add_node(name)
        self.cardinalities[name] = cardinality
    
    def add_edge(self, parent: str, child: str):
        """
        변수 사이에 방향 변(인과 관계)을 더한다.
        
        인수:
            parent: 어버이 변수의 이름
            child: 자식 변수의 이름
        
        일으키는 예외:
            ValueError: 그 변이 고리를 만들 때
        
        보기:
            bn.add_edge('Rain', 'WetGrass')  # 비가 잔디를 적신다
        """
        self.graph.add_edge(parent, child)
        
        # 고리가 없는지 확인하기
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(parent, child)
            raise ValueError(f"Adding edge {parent}->{child} creates a cycle!")
    
    def set_cpt(self, 
                variable: str,
                table: np.ndarray):
        """
        변수의 조건부 확률표를 정한다.
        
        인수:
            variable: 변수의 이름
            table: 조건부 확률을 담은 NumPy 배열
                   꼴은 (card(어버이1), ..., card(어버이N), card(변수))과 맞아야 한다
        
        보기:
            # 이진 Cloudy을 어버이로 갖는 이진 Rain의 경우
            bn.set_cpt('Rain', np.array([[0.8, 0.2],   # P(Rain | Cloudy=0)
                                         [0.3, 0.7]]))  # P(Rain | Cloudy=1)
        """
        parents = list(self.graph.predecessors(variable))
        parents.sort()  # 어긋나지 않는 차례
        
        cpt = ConditionalProbabilityTable(
            variable=variable,
            parents=parents,
            cardinalities=self.cardinalities,
            table=table
        )
        
        self.cpts[variable] = cpt
    
    def get_cpt(self, variable: str) -> ConditionalProbabilityTable:
        """
        변수의 CPT을 얻는다.
        
        인수:
            variable: 변수의 이름
        
        반환값:
            조건부 확률표
        """
        if variable not in self.cpts:
            raise ValueError(f"No CPT defined for variable {variable}")
        return self.cpts[variable]
    
    def compute_joint_probability(self, assignment: Dict[str, int]) -> float:
        """
        온전한 대입에 대해 P(X1=x1, X2=x2, ..., Xn=xn)을 셈한다.
        
        사슬 규칙 쪼개기를 쓴다:
        P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))
        
        인수:
            assignment: 모든 변수에 값을 준 온전한 대입
        
        반환값:
            결합 확률
        
        보기:
            # P(Cloudy=1, Rain=1, WetGrass=1) 셈하기
            prob = bn.compute_joint_probability({
                'Cloudy': 1,
                'Rain': 1,
                'WetGrass': 1
            })
        """
        probability = 1.0
        
        # 변수마다 조건부 확률 곱하기
        for variable in self.graph.nodes():
            cpt = self.get_cpt(variable)
            parents = list(self.graph.predecessors(variable))
            
            # 어버이 값 얻기
            parent_values = {p: assignment[p] for p in parents}
            
            # 변수 값 얻기
            var_value = assignment[variable]
            
            # P(변수 | 어버이)을 곱하기
            prob = cpt.get_probability(var_value, parent_values)
            probability *= prob
        
        return probability
    
    def forward_sample(self) -> Dict[str, int]:
        """
        앞먹임 표집으로 결합 분포에서 표본을 만든다.
        
        앞먹임 표집은 위상 차례를 따른다:
        1. 변수를 위상 차례로 표집한다
        2. 이미 표집한 어버이 값으로 P(Xi | Parents(Xi))에서 표집한다
        
        반환값:
            P(X1,...,Xn)에서 표집한 온전한 대입
        
        보기:
            # 망에서 표본 1000개 만들기
            samples = [bn.forward_sample() for _ in range(1000)]
        """
        assignment = {}
        
        # 위상 차례로 표집하기(자식보다 어버이를 먼저)
        for variable in nx.topological_sort(self.graph):
            cpt = self.get_cpt(variable)
            parents = list(self.graph.predecessors(variable))
            
            # 이미 표집한 변수에서 어버이 값 얻기
            parent_values = {p: assignment[p] for p in parents}
            
            # 이 변수 표집하기
            assignment[variable] = cpt.sample(parent_values)
        
        return assignment
    
    def visualize(self, figsize: Tuple[int, int] = (12, 8), show_cpts: bool = False):
        """
        베이즈 망의 짜임과, 원하면 CPT까지 그려 본다.
        
        인수:
            figsize: 그림 크기
            show_cpts: CPT 값을 보일지 여부
        """
        plt.figure(figsize=figsize)
        
        # 배치
        try:
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        except:
            pos = nx.spring_layout(self.graph)
        
        # 마디와 변 그리기
        nx.draw(self.graph, pos,
                with_labels=True,
                node_color='lightcoral',
                node_size=3000,
                font_size=12,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                width=2)
        
        # 가짓수를 마디 이름표로 붙이기
        labels = {node: f"{node}\n(card={self.cardinalities[node]})" 
                 for node in self.graph.nodes()}
        pos_labels = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
        nx.draw_networkx_labels(self.graph, pos_labels, labels, 
                               font_size=9, font_color='darkred')
        
        plt.title("Bayesian Network Structure", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # 요청하면 CPT 보이기
        if show_cpts:
            print("\n" + "="*70)
            print("CONDITIONAL PROBABILITY TABLES")
            print("="*70)
            for variable in nx.topological_sort(self.graph):
                cpt = self.get_cpt(variable)
                print(f"\n{cpt}")
                print("-"*70)


def build_weather_network() -> BayesianNetwork:
    """
    단순한 날씨 예측 베이즈 망을 쌓는다.
    
    망의 짜임:
        Cloudy -> Rain -> WetGrass
        Cloudy -> Sprinkler -> WetGrass
    
    이는 다음을 보여 주는 고전적인 보기이다:
    - 함께 낳은 원인(흐림이 비와 물뿌리개에 함께 영향을 준다)
    - 여러 원인(젖은 잔디는 비 때문일 수도 물뿌리개 때문일 수도 있다)
    
    반환값:
        CPT을 갖춘 온전한 베이즈 망
    """
    print("\nBuilding Weather Network...")
    print("-" * 70)
    
    bn = BayesianNetwork()
    
    # 변수 더하기(모두 이진: 0=거짓, 1=참)
    bn.add_variable('Cloudy', 2)
    bn.add_variable('Sprinkler', 2)
    bn.add_variable('Rain', 2)
    bn.add_variable('WetGrass', 2)
    
    # 변 더하기(인과 관계)
    bn.add_edge('Cloudy', 'Sprinkler')  # 흐림이 물뿌리개 씀에 영향을 준다
    bn.add_edge('Cloudy', 'Rain')        # 흐림이 비에 영향을 준다
    bn.add_edge('Sprinkler', 'WetGrass') # 물뿌리개가 잔디를 적실 수 있다
    bn.add_edge('Rain', 'WetGrass')      # 비가 잔디를 적실 수 있다
    
    # CPT 정하기
    
    # P(Cloudy) — 앞확률
    # [P(Cloudy=0), P(Cloudy=1)]
    bn.set_cpt('Cloudy', np.array([0.5, 0.5]))
    
    # P(Sprinkler | Cloudy)
    # 흐리면 물뿌리개를 쓸 가능성이 낮다
    bn.set_cpt('Sprinkler', np.array([
        [0.5, 0.5],  # P(Sprinkler | Cloudy=0)
        [0.9, 0.1]   # P(Sprinkler | Cloudy=1) — 흐리면 잘 안 쓴다
    ]))
    
    # P(Rain | Cloudy)
    # 흐리면 비가 올 가능성이 높다
    bn.set_cpt('Rain', np.array([
        [0.8, 0.2],  # P(Rain | Cloudy=0) — 흐리지 않으면 잘 안 온다
        [0.2, 0.8]   # P(Rain | Cloudy=1) — 흐리면 잘 온다
    ]))
    
    # P(WetGrass | Sprinkler, Rain)
    # 물뿌리개가 켜졌거나 비가 오면 잔디가 젖는다
    # 차례: [Sprinkler=0, Rain=0], [Sprinkler=0, Rain=1],
    #        [Sprinkler=1, Rain=0], [Sprinkler=1, Rain=1]
    bn.set_cpt('WetGrass', np.array([
        [[1.0, 0.0],   # Sprinkler=0, Rain=0: 잔디가 마름
         [0.1, 0.9]],  # Sprinkler=0, Rain=1: 잔디가 젖음(비)
        [[0.1, 0.9],   # Sprinkler=1, Rain=0: 잔디가 젖음(물뿌리개)
         [0.01, 0.99]] # Sprinkler=1, Rain=1: 잔디가 매우 젖음(둘 다)
    ]))
    
    print("Network built successfully!")
    print(f"Variables: {list(bn.graph.nodes())}")
    print(f"Edges: {list(bn.graph.edges())}")
    
    return bn


def demonstrate_joint_probability():
    """
    베이즈 망에서 결합 확률 셈하기를 보인다.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Computing Joint Probabilities")
    print("="*70)
    
    bn = build_weather_network()
    
    # 결합 확률 몇 개 셈하기
    test_cases = [
        {'Cloudy': 0, 'Sprinkler': 0, 'Rain': 0, 'WetGrass': 0},
        {'Cloudy': 1, 'Sprinkler': 0, 'Rain': 1, 'WetGrass': 1},
        {'Cloudy': 1, 'Sprinkler': 1, 'Rain': 1, 'WetGrass': 1},
    ]
    
    print("\nComputing joint probabilities for different scenarios:")
    print("-" * 70)
    
    for i, assignment in enumerate(test_cases, 1):
        prob = bn.compute_joint_probability(assignment)
        
        # 읽기 좋은 설명 만들기
        desc = ", ".join([f"{var}={'Yes' if val else 'No'}" 
                         for var, val in assignment.items()])
        
        print(f"\nScenario {i}: {desc}")
        print(f"P(assignment) = {prob:.6f}")
        
        # 쪼개기 보이기
        print("\nFactorization:")
        print(f"  = P(Cloudy={assignment['Cloudy']})")
        print(f"  × P(Sprinkler={assignment['Sprinkler']} | Cloudy={assignment['Cloudy']})")
        print(f"  × P(Rain={assignment['Rain']} | Cloudy={assignment['Cloudy']})")
        print(f"  × P(WetGrass={assignment['WetGrass']} | Sprinkler={assignment['Sprinkler']}, Rain={assignment['Rain']})")


def demonstrate_sampling():
    """
    베이즈 망에서 앞먹임 표집을 보인다.
    
    앞먹임 표집은 결합 분포를 어림하는
    몬테카를로 방법이다.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Forward Sampling")
    print("="*70)
    
    bn = build_weather_network()
    
    # 표본 만들기
    num_samples = 10000
    print(f"\nGenerating {num_samples} samples from the network...")
    
    samples = [bn.forward_sample() for _ in range(num_samples)]
    
    # 뜯어보려고 DataFrame으로 바꾸기
    df = pd.DataFrame(samples)
    
    print("\nFirst 10 samples:")
    print(df.head(10).to_string(index=False))
    
    # 경험 확률 셈하기
    print("\n" + "-"*70)
    print("Empirical vs. True Probabilities")
    print("-"*70)
    
    # P(Cloudy) 살피기
    empirical_cloudy = df['Cloudy'].mean()
    print(f"\nP(Cloudy=1):")
    print(f"  True: 0.500")
    print(f"  Empirical: {empirical_cloudy:.3f}")
    
    # P(WetGrass) 살피기
    empirical_wet = df['WetGrass'].mean()
    # 모든 꼴에 걸쳐 합해 참 확률 셈하기
    true_wet = 0.0
    for assignment in [dict(zip(['Cloudy', 'Sprinkler', 'Rain', 'WetGrass'], combo))
                      for combo in product([0,1], repeat=4)
                      if combo[3] == 1]:  # WetGrass=1
        true_wet += bn.compute_joint_probability(assignment)
    
    print(f"\nP(WetGrass=1):")
    print(f"  True: {true_wet:.3f}")
    print(f"  Empirical: {empirical_wet:.3f}")
    
    # 조건부 확률: P(Rain=1 | Cloudy=1)
    cloudy_samples = df[df['Cloudy'] == 1]
    empirical_rain_given_cloudy = cloudy_samples['Rain'].mean()
    
    print(f"\nP(Rain=1 | Cloudy=1):")
    print(f"  True: 0.800")
    print(f"  Empirical: {empirical_rain_given_cloudy:.3f}")
    
    print("\nNote: With more samples, empirical probabilities converge to true values!")


def build_student_network() -> BayesianNetwork:
    """
    학생 성적 베이즈 망을 쌓는다.
    
    이 망은 학생의 시험 성적에 영향을 주는 요소를 본뜬다:
    - 시험의 어려움
    - 학생의 머리 좋음
    - 학생의 성적(둘 다에 기댄다)
    - 추천서의 질(성적에 기댄다)
    
    망의 짜임:
        Difficulty -> Grade <- Intelligence
        Grade -> Letter
    
    반환값:
        온전한 베이즈 망
    """
    print("\nBuilding Student Network...")
    print("-" * 70)
    
    bn = BayesianNetwork()
    
    # 변수 더하기
    bn.add_variable('Difficulty', 2)   # 0=쉬움, 1=어려움
    bn.add_variable('Intelligence', 2)  # 0=낮음, 1=높음
    bn.add_variable('Grade', 3)         # 0=A, 1=B, 2=C
    bn.add_variable('Letter', 2)        # 0=약함, 1=셈
    
    # 변 더하기
    bn.add_edge('Difficulty', 'Grade')
    bn.add_edge('Intelligence', 'Grade')
    bn.add_edge('Grade', 'Letter')
    
    # CPT 정하기
    
    # P(Difficulty)
    bn.set_cpt('Difficulty', np.array([0.6, 0.4]))
    
    # P(Intelligence)
    bn.set_cpt('Intelligence', np.array([0.7, 0.3]))
    
    # P(Grade | Intelligence, Difficulty)
    # [Difficulty=0, Intelligence=0] -> 대개 B과 C
    # [Difficulty=0, Intelligence=1] -> 대개 A과 B
    # [Difficulty=1, Intelligence=0] -> 대개 C
    # [Difficulty=1, Intelligence=1] -> 대개 B
    bn.set_cpt('Grade', np.array([
        [[0.3, 0.4, 0.3],   # Difficulty=0, Intelligence=0
         [0.9, 0.08, 0.02]], # Difficulty=0, Intelligence=1
        [[0.05, 0.25, 0.7],  # Difficulty=1, Intelligence=0
         [0.5, 0.3, 0.2]]    # Difficulty=1, Intelligence=1
    ]))
    
    # P(Letter | Grade)
    # 성적이 좋을수록 추천서가 세진다
    bn.set_cpt('Letter', np.array([
        [0.1, 0.9],  # Grade=A -> 센 추천서
        [0.4, 0.6],  # Grade=B -> 보통 추천서
        [0.9, 0.1]   # Grade=C -> 약한 추천서
    ]))
    
    print("Student network built successfully!")
    return bn


def main():
    """
    베이즈 망 개념을 보여 주는 주된 함수.
    """
    print("\n" + "="*70)
    print("BAYESIAN NETWORKS - BASICS")
    print("="*70)
    
    print("\nTopics covered:")
    print("1. Building Bayesian Networks")
    print("2. Conditional Probability Tables (CPTs)")
    print("3. Computing joint probabilities")
    print("4. Forward sampling")
    
    # 날씨 망을 쌓고 그려 보기
    print("\n" + "="*70)
    print("Example 1: Weather Network")
    print("="*70)
    bn = build_weather_network()
    bn.visualize(show_cpts=True)
    
    # 결합 확률 셈하기 보이기
    demonstrate_joint_probability()
    
    # 표집 보이기
    demonstrate_sampling()
    
    # 학생 망을 쌓고 그려 보기
    print("\n" + "="*70)
    print("Example 2: Student Network")
    print("="*70)
    student_bn = build_student_network()
    student_bn.visualize(show_cpts=True)
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. Bayesian Networks = DAG + CPTs")
    print("2. CPTs specify P(Variable | Parents)")
    print("3. Joint distribution: P(X1,...,Xn) = ∏ P(Xi | Parents(Xi))")
    print("4. Forward sampling follows topological order")
    print("5. Network structure encodes conditional independence")
    
    print("\n" + "="*70)
    print("Next: Learn about inference in Bayesian Networks!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
```

## 2. 논의

베이즈 망은 튜플 $(G, P)$이며, 여기서 $G = (V, E)$은 DAG이고 $P = \{P(X_i | \text{Parents}(X_i))\}$은 조건부 확률 분포의 묶음이다. 이 망은 결합 분포를 $P(X_1, \ldots, X_n) = \prod_i P(X_i | \text{Parents}(X_i))$으로 나타낸다. 이 쪼개기가 셈 효율의 열쇠이다.

코드는 고전적인 보기 둘을 구현한다. 날씨 망(흐림이 물뿌리개와 비에 함께 영향을 주고, 그 둘이 젖은 잔디에 영향을 준다)과 학생 망(어려움과 머리 좋음이 성적에 영향을 주고, 성적이 추천서의 질에 영향을 준다)이다. 이 망들은 함께 낳은 원인 짜임, 여러 원인, 그리고 쪼개기로 얻는 큰 매개변수 줄임을 보여 준다.

앞먹임 표집은 DAG의 위상 차례를 따른다. 곧 변수를 차례대로 표집하되, 앞서 표집한 어버이 값으로 조건부 분포를 정한다. 이렇게 하면 결합 분포에서 정확한 표본이 나오고, 몬테카를로 평균 내기로 어떤 주변 확률이나 조건부 확률도 어림할 수 있다.

## 연습문제

**연습문제 1.**
단순한 의료 진단의 베이즈 망을 쌓아라. 독감과 알레르기가 함께 재채기와 기침을 일으킨다. 낱낱 세기로 $P(\text{Flu}=1 | \text{Sneezing}=1, \text{Cough}=1)$을 셈하여라.

??? success "연습문제 1 풀이"
    ```python
import numpy as np
from itertools import product

# CPT
p_flu = {0: 0.95, 1: 0.05}
p_allergy = {0: 0.7, 1: 0.3}
p_sneeze = {(0,0): {0:0.95,1:0.05}, (0,1): {0:0.3,1:0.7},
            (1,0): {0:0.2,1:0.8}, (1,1): {0:0.1,1:0.9}}
p_cough = {(0,0): {0:0.95,1:0.05}, (0,1): {0:0.7,1:0.3},
           (1,0): {0:0.2,1:0.8}, (1,1): {0:0.1,1:0.9}}

# Allergy에 걸쳐 합한 P(Flu=1, Sneeze=1, Cough=1)
numerator = 0
for a in [0, 1]:
    numerator += p_flu[1] * p_allergy[a] * p_sneeze[(1,a)][1] * p_cough[(1,a)][1]

denominator = 0
for f in [0, 1]:
    for a in [0, 1]:
        denominator += p_flu[f] * p_allergy[a] * p_sneeze[(f,a)][1] * p_cough[(f,a)][1]

print(f'P(Flu=1 | Sneeze=1, Cough=1) = {numerator/denominator:.4f}')
```


---

**연습문제 2.**
베이즈 망에 변을 더하면 왜 나타내는 힘이 결코 줄지 않으면서 매개변수의 개수는 늘 수 있는지 설명하여라. 무엇을 주고 무엇을 얻는가?

??? success "연습문제 2 풀이"
    변이 더 많은 베이즈 망은 더 성긴 망이 나타낼 수 있는 분포를 모두 나타낼 수 있고, 덧붙은 조건부 기댐이 시시하지 않은 분포까지 더 나타낼 수 있다. 그러므로 나타내는 힘은 커지거나 그대로일 뿐이다.

그러나 변이 많아지면 어떤 마디의 어버이가 많아지고, 어버이가 하나 늘 때마다 그 마디의 CPT 크기가 두 배가 된다(이진 변수일 때). 그러면 매개변수의 개수가 늘고 값이 두 가지로 든다. (1) 매개변수를 미덥게 어림하려면 자료가 더 필요하고, (2) 추론에 드는 셈이 더 비싸진다.

주고받음은 모형의 충실함과 복잡함 사이에 있다. 온통 이어진 DAG은 어떤 분포든 나타낼 수 있지만 매개변수가 지수만큼 필요하다. 성긴 망은 기댐을 얼마쯤 놓칠 수 있지만 저장하고 배우고 추론하기에 효율적이다. 가장 좋은 망은 그 분야의 참된 조건부 독립 짜임을 담는다.


---

**연습문제 3.**
날씨 망에서 앞먹임 표본 10,000개를 만들고 물리치기 표집으로 $P(\text{Rain}=1 | \text{WetGrass}=1)$을 어림하여라. 낱낱 세기로 셈한 정확한 답과 견주어라.

??? success "연습문제 3 풀이"
    ```python
import numpy as np

np.random.seed(42)
n = 100000
samples = []

for _ in range(n):
    cloudy = np.random.binomial(1, 0.5)
    sprinkler = np.random.binomial(1, 0.1 if cloudy else 0.5)
    rain = np.random.binomial(1, 0.8 if cloudy else 0.2)
    p_wet = [[1.0, 0.0], [0.1, 0.9], [0.1, 0.9], [0.01, 0.99]]
    idx = sprinkler * 2 + rain
    wet = np.random.binomial(1, p_wet[idx][1])
    samples.append((cloudy, sprinkler, rain, wet))

samples = np.array(samples)
wet_mask = samples[:, 3] == 1
rain_given_wet = samples[wet_mask, 2].mean()
print(f'P(Rain=1|WetGrass=1) approx = {rain_given_wet:.4f}')

# 낱낱 세기로 정확히
from itertools import product
p_rain_wet = 0
p_wet = 0
for c, s, r, w in product([0,1], repeat=4):
    pc = [0.5,0.5][c]
    ps = [0.5,0.1][c] if s else [0.5,0.9][c]
    pr = [0.2,0.8][c] if r else [0.8,0.2][c]
    pw_table = [[1,0],[0.1,0.9],[0.1,0.9],[0.01,0.99]]
    pw = pw_table[s*2+r][w]
    joint = pc * ps * pr * pw
    if w == 1:
        p_wet += joint
        if r == 1:
            p_rain_wet += joint

print(f'P(Rain=1|WetGrass=1) exact = {p_rain_wet/p_wet:.4f}')
```

## 정리하며

**다룬 것** — 베이즈 망의 기초

베이즈 망은 튜플 $(G, P)$이며, 여기서 $G = (V, E)$은 DAG이고 $P = \{P(X_i | \text{Parents}(X_i))\}$은 조건부 확률 분포의 묶음이다.

고갱이 갈래는 `ConditionalProbabilityTable`, `BayesianNetwork`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
