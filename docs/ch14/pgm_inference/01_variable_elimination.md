# 변수 없애기

변수 없애기는 베이즈 망에서 정확히 추론하는 표준 알고리즘이다. 결합 분포의 쪼개진 짜임을 써먹어 소박한 낱낱 세기의 지수 폭발을 피한다. 인자 곱하기와 주변화로 숨은 변수를 하나씩 차근차근 없애면, 변수 없애기는 마구잡이 세기보다 훨씬 효율적일 수 있으며, 그 효율은 없애는 차례에 크게 달렸다.

## 코드

```python
"""
변수 없애기 알고리즘
===============================

이 모듈은 베이즈 망의 짜임을 써먹는 효율적인 정확 추론 방법인
변수 없애기 알고리즘을 구현한다.

학습 목표:
-------------------
1. 낱낱 세기가 왜 비효율적인지 이해하기
2. 변수 없애기 알고리즘 배우기
3. 인자 연산(곱, 주변화) 이해하기
4. 없애는 차례와 그 영향 배우기
5. 효율적인 정확 추론 구현하기

수학의 바탕:
------------------------
변수 없애기는 다음과 같이 굴러간다:
1. CPT을 인자로 바꾼다
2. 없앨 변수마다:
   - 그 변수를 담은 인자를 모두 모은다
   - 그것들을 곱한다(인자 곱)
   - 그 변수를 합으로 지운다(주변화)
3. 남은 인자를 곱하고 고르게 한다

시간 복잡도: O(n * d^(w+1))이며 여기서:
- n = 변수의 개수
- d = 가장 큰 가짓수
- w = 이끌린 너비(없애는 차례에 달렸다)

지은이: 교육용 ML 팀
수준: 중간
미리 볼 것: 첫걸음 수준의 01-04
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
import time

# ========================================================================
# 메인
# ========================================================================


class Factor:
    """
    변수 묶음에 걸친 인자(퍼텐셜 함수)를 나타낸다.
    
    인자 φ(X1, ..., Xk)은 (X1, ..., Xk)의 대입마다
    음이 아닌 수에 잇는 함수이다.
    
    인자는 변수 없애기의 근본 자료 짜임이다.
    """
    
    def __init__(self,
                 variables: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[np.ndarray] = None):
        """
        인자의 첫걸음을 잡는다.
        
        인수:
            variables: 이 인자에 든 변수 이름의 목록
            cardinalities: 모든 변수의 가짓수 사전
            values: 인자 값의 NumPy 배열(없어도 된다)
        """
        self.variables = sorted(variables)  # 어긋나지 않도록 정렬해 둔다
        self.cardinalities = cardinalities
        
        shape = tuple(cardinalities[var] for var in self.variables)
        
        if values is None:
            self.values = np.ones(shape)
        else:
            self.values = np.array(values)
            assert self.values.shape == shape, \
                f"Shape mismatch: {self.values.shape} vs {shape}"
    
    def multiply(self, other: 'Factor') -> 'Factor':
        """
        인자 곱 φ1 × φ2을 셈한다.
        
        인자 곱하기는 어긋나지 않는 대입마다 값을 곱해
        인자 둘을 합친다.
        
        이를테면 φ1(A,B)과 φ2(B,C)이 있으면 다음과 같다:
        (φ1 × φ2)(A,B,C) = φ1(A,B) × φ2(B,C)
        
        인수:
            other: 곱할 다른 인자
        
        반환값:
            변수의 합집합에 걸친 곱 인자
        """
        # 변수의 합집합 찾기
        new_variables = sorted(set(self.variables + other.variables))
        
        # 결과 인자 만들기
        result = Factor(new_variables, self.cardinalities)
        
        # 새 변수의 대입마다
        for assignment in self._enumerate_assignments(new_variables):
            # 두 인자에서 값 얻기
            val1 = self._get_value(assignment)
            val2 = other._get_value(assignment)
            
            # 곱하기
            result._set_value(assignment, val1 * val2)
        
        return result
    
    def marginalize(self, variables_to_eliminate: List[str]) -> 'Factor':
        """
        인자에서 변수를 합으로 지운다(주변화한다).
        
        주변화는 φ'(X) = Σ_Y φ(X, Y)을 셈한다
        
        인수:
            variables_to_eliminate: 합으로 지울 변수
        
        반환값:
            변수를 없앤 새 인자
        """
        # 없앤 뒤 남은 변수
        remaining_vars = [v for v in self.variables 
                         if v not in variables_to_eliminate]
        
        if not remaining_vars:
            # 모든 변수에 걸쳐 합하기 = 숫자 하나
            return Factor([], self.cardinalities, 
                         np.array([np.sum(self.values)]))
        
        # 결과 인자 만들기
        result = Factor(remaining_vars, self.cardinalities)
        
        # 남은 변수의 대입마다
        for remaining_assignment in result._enumerate_assignments(remaining_vars):
            # 없앤 변수의 모든 대입에 걸쳐 합하기
            total = 0.0
            
            elim_cards = [self.cardinalities[v] for v in variables_to_eliminate]
            for elim_values in product(*[range(c) for c in elim_cards]):
                elim_assignment = dict(zip(variables_to_eliminate, elim_values))
                full_assignment = {**remaining_assignment, **elim_assignment}
                
                total += self._get_value(full_assignment)
            
            result._set_value(remaining_assignment, total)
        
        return result
    
    def reduce(self, evidence: Dict[str, int]) -> 'Factor':
        """
        증거 변수를 붙박아 인자를 줄인다.
        
        증거와 어긋나지 않는 대입만 헤아리는
        새 인자를 만든다.
        
        인수:
            evidence: 관측한 변수 값의 사전
        
        반환값:
            줄인 인자
        """
        # 증거에 없는 변수
        remaining_vars = [v for v in self.variables if v not in evidence]
        
        if not remaining_vars:
            # 모든 변수가 관측됨
            return Factor([], self.cardinalities, 
                         np.array([self._get_value(evidence)]))
        
        # 결과 인자 만들기
        result = Factor(remaining_vars, self.cardinalities)
        
        # 남은 변수의 대입마다
        for assignment in result._enumerate_assignments(remaining_vars):
            full_assignment = {**assignment, **evidence}
            value = self._get_value(full_assignment)
            result._set_value(assignment, value)
        
        return result
    
    def normalize(self) -> 'Factor':
        """값의 합이 1이 되도록 인자를 고르게 한다."""
        total = np.sum(self.values)
        if total > 0:
            return Factor(self.variables, self.cardinalities, 
                         self.values / total)
        return self
    
    def _get_value(self, assignment: Dict[str, int]) -> float:
        """대입에 대한 인자 값을 얻는다."""
        if not self.variables:
            return self.values[0]
        
        index = tuple(assignment[var] for var in self.variables)
        return self.values[index]
    
    def _set_value(self, assignment: Dict[str, int], value: float):
        """대입에 대한 인자 값을 정한다."""
        if not self.variables:
            self.values[0] = value
        else:
            index = tuple(assignment[var] for var in self.variables)
            self.values[index] = value
    
    def _enumerate_assignments(self, variables: List[str]):
        """변수의 모든 대입을 만든다."""
        if not variables:
            yield {}
            return
        
        cards = [self.cardinalities[var] for var in variables]
        for values in product(*[range(c) for c in cards]):
            yield dict(zip(variables, values))
    
    def __str__(self) -> str:
        """문자열 표현."""
        if not self.variables:
            return f"Factor([]) = {self.values[0]:.4f}"
        return f"Factor({', '.join(self.variables)})\nShape: {self.values.shape}"


class VariableElimination:
    """
    정확 추론을 위한 변수 없애기 알고리즘을 구현한다.
    
    이것이 베이즈 망 정확 추론의 일꾼 알고리즘이다.
    분포의 쪼개진 짜임을 써먹으므로 낱낱 세기보다
    훨씬 효율적이다.
    """
    
    def __init__(self, bn):
        """
        변수 없애기의 첫걸음을 잡는다.
        
        인수:
            bn: BayesianNetwork 사례
        """
        self.bn = bn
        
    def _create_factors_from_cpts(self) -> List[Factor]:
        """
        망의 모든 CPT을 인자로 바꾼다.
        
        반환값:
            변수마다 하나씩인 인자의 목록
        """
        factors = []
        
        for variable in self.bn.graph.nodes():
            cpt = self.bn.get_cpt(variable)
            parents = cpt.parents
            
            # 변수와 그 어버이에 걸친 인자 만들기
            factor_vars = parents + [variable]
            factor = Factor(factor_vars, self.bn.cardinalities)
            
            # CPT 값을 인자로 베끼기
            for assignment in factor._enumerate_assignments(factor_vars):
                parent_vals = {p: assignment[p] for p in parents}
                var_val = assignment[variable]
                prob = cpt.get_probability(var_val, parent_vals)
                factor._set_value(assignment, prob)
            
            factors.append(factor)
        
        return factors
    
    def _choose_elimination_order(self,
                                  variables_to_eliminate: Set[str],
                                  strategy: str = 'min_neighbors') -> List[str]:
        """
        없애는 차례를 고른다.
        
        없애는 차례가 효율을 크게 좌우한다!
        여러 전략:
        - min_neighbors: 이웃이 가장 적은 변수를 먼저 없앤다
        - min_fill: 그래프에 더해지는 변을 가장 적게 한다
        - weighted_min_fill: 변수의 가짓수를 헤아린다
        
        인수:
            variables_to_eliminate: 차례를 매길 변수
            strategy: 차례 매기기 전략
        
        반환값:
            없애는 차례로 늘어놓은 변수의 목록
        """
        if strategy == 'min_neighbors':
            # 단순한 욕심 전략: 이웃이 가장 적은 변수를 없앤다
            order = []
            graph = self.bn.graph.to_undirected()
            remaining = set(variables_to_eliminate)
            
            while remaining:
                # 남은 묶음에서 이웃이 가장 적은 변수 찾기
                min_var = min(remaining, 
                             key=lambda v: len(set(graph.neighbors(v)) & remaining))
                order.append(min_var)
                remaining.remove(min_var)
            
            return order
        else:
            # 기본값: 정렬한 차례를 그대로 쓴다
            return sorted(variables_to_eliminate)
    
    def query(self,
             query_vars: List[str],
             evidence: Optional[Dict[str, int]] = None,
             elimination_order: Optional[List[str]] = None,
             verbose: bool = True) -> Dict[tuple, float]:
        """
        변수 없애기로 추론한다.
        
        인수:
            query_vars: 분포를 셈할 변수
            evidence: 관측한 변수 값
            elimination_order: 손수 정한 없애는 차례(없어도 된다)
            verbose: 진행 상황을 출력할지 여부
        
        반환값:
            물음 변수에 걸친 확률 분포
        """
        if evidence is None:
            evidence = {}
        
        if verbose:
            print(f"\nVariable Elimination Query:")
            print(f"Query: P({', '.join(query_vars)}" +
                  (f" | {', '.join(f'{k}={v}' for k, v in evidence.items())})" 
                   if evidence else ")"))
            print("-" * 60)
        
        # 걸음 1: CPT에서 첫 인자 만들기
        factors = self._create_factors_from_cpts()
        
        if verbose:
            print(f"\nStep 1: Created {len(factors)} initial factors from CPTs")
        
        # 걸음 2: 증거로 인자 줄이기
        if evidence:
            factors = [f.reduce(evidence) for f in factors]
            if verbose:
                print(f"Step 2: Reduced factors with evidence {evidence}")
        
        # 걸음 3: 없앨 변수 정하기
        all_vars = set(self.bn.graph.nodes())
        variables_to_eliminate = all_vars - set(query_vars) - set(evidence.keys())
        
        if verbose:
            print(f"Step 3: Variables to eliminate: {sorted(variables_to_eliminate)}")
        
        # 걸음 4: 없애는 차례 고르기
        if elimination_order is None:
            elimination_order = self._choose_elimination_order(variables_to_eliminate)
        
        if verbose:
            print(f"Step 4: Elimination order: {elimination_order}")
        
        # 걸음 5: 변수를 하나씩 없애기
        if verbose:
            print(f"\nStep 5: Eliminating variables...")
        
        for i, var in enumerate(elimination_order, 1):
            if verbose:
                print(f"\n  Eliminating {var} ({i}/{len(elimination_order)}):")
            
            # 이 변수를 담은 인자 찾기
            relevant_factors = [f for f in factors if var in f.variables]
            other_factors = [f for f in factors if var not in f.variables]
            
            if verbose:
                print(f"    - Found {len(relevant_factors)} factors containing {var}")
            
            if relevant_factors:
                # 해당하는 인자 곱하기
                product_factor = relevant_factors[0]
                for f in relevant_factors[1:]:
                    product_factor = product_factor.multiply(f)
                
                if verbose:
                    print(f"    - Multiplied factors: scope = {product_factor.variables}")
                
                # 변수 주변화하기
                marginalized_factor = product_factor.marginalize([var])
                
                if verbose:
                    print(f"    - Marginalized out {var}: scope = {marginalized_factor.variables}")
                
                # 인자 목록 새로 고치기
                factors = other_factors + [marginalized_factor]
            else:
                factors = other_factors
        
        # 걸음 6: 남은 인자 곱하기
        if verbose:
            print(f"\nStep 6: Multiplying {len(factors)} remaining factors...")
        
        result_factor = factors[0]
        for f in factors[1:]:
            result_factor = result_factor.multiply(f)
        
        # 걸음 7: 고르게 하기
        result_factor = result_factor.normalize()
        
        if verbose:
            print(f"Step 7: Normalized result")
        
        # 인자를 사전으로 바꾸기
        result = {}
        for assignment in result_factor._enumerate_assignments(query_vars):
            key = tuple(assignment[var] for var in query_vars)
            result[key] = result_factor._get_value(assignment)
        
        return result


def demonstrate_variable_elimination():
    """단순한 망에서 변수 없애기를 보인다."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Variable Elimination")
    print("="*70)
    
    # 날씨 망 들여오고 쌓기
    import sys
    sys.path.append('..')
    from beginner.bayesian_networks_basics import build_weather_network
    
    bn = build_weather_network()
    
    # 추론 엔진 만들기
    ve = VariableElimination(bn)
    
    # 물음 1: 단순한 물음
    print("\n" + "="*70)
    print("Query 1: P(Rain | WetGrass=1)")
    print("="*70)
    
    result = ve.query(['Rain'], {'WetGrass': 1}, verbose=True)
    
    print(f"\nResult:")
    print(f"P(Rain=0 | WetGrass=1) = {result[(0,)]:.4f}")
    print(f"P(Rain=1 | WetGrass=1) = {result[(1,)]:.4f}")
    
    # 물음 2: 물음 변수가 여럿
    print("\n" + "="*70)
    print("Query 2: P(Rain, Sprinkler | WetGrass=1)")
    print("="*70)
    
    result = ve.query(['Rain', 'Sprinkler'], {'WetGrass': 1}, verbose=True)
    
    print(f"\nResult:")
    for (rain, sprinkler), prob in sorted(result.items()):
        print(f"P(Rain={rain}, Sprinkler={sprinkler} | WetGrass=1) = {prob:.4f}")


def compare_enumeration_vs_ve():
    """
    낱낱 세기와 변수 없애기의 효율을 견준다.
    """
    print("\n" + "="*70)
    print("COMPARISON: Enumeration vs Variable Elimination")
    print("="*70)
    
    import sys
    sys.path.append('..')
    from beginner.bayesian_networks_basics import build_weather_network
    from beginner.simple_inference import InferenceByEnumeration
    
    bn = build_weather_network()
    
    # 낱낱 세기
    print("\nEnumeration:")
    enum = InferenceByEnumeration(bn)
    start = time.time()
    enum_result = enum.query_marginal(['Rain'], {'WetGrass': 1}, use_cache=False)
    enum_time = time.time() - start
    print(f"Time: {enum_time:.6f} seconds")
    
    # 변수 없애기
    print("\nVariable Elimination:")
    ve = VariableElimination(bn)
    start = time.time()
    ve_result = ve.query(['Rain'], {'WetGrass': 1}, verbose=False)
    ve_time = time.time() - start
    print(f"Time: {ve_time:.6f} seconds")
    
    # 비교
    print(f"\nSpeedup: {enum_time/ve_time:.2f}x faster")
    print(f"\nNote: For small networks the difference is small,")
    print(f"but for larger networks VE can be orders of magnitude faster!")


def main():
    """주된 보여 주기."""
    print("\n" + "="*70)
    print("VARIABLE ELIMINATION ALGORITHM")
    print("="*70)
    
    print("\nVariable Elimination is the standard algorithm for exact")
    print("inference in Bayesian Networks. It exploits the factored")
    print("structure to avoid exponential blowup (when possible).")
    
    demonstrate_variable_elimination()
    compare_enumeration_vs_ve()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. VE is much more efficient than enumeration")
    print("\n2. Key operations:")
    print("   - Factor product (multiply factors)")
    print("   - Marginalization (sum out variables)")
    
    print("\n3. Elimination order matters!")
    print("   - Good order: polynomial time")
    print("   - Bad order: exponential time")
    print("   - Finding optimal order is NP-hard")
    
    print("\n4. Time complexity: O(n * d^(w+1))")
    print("   - w = induced width (depends on order)")
    print("   - For trees: w=1, very efficient!")
    print("   - For dense graphs: can be expensive")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()```

## 논의

변수 없애기는 CPT을 인자로 바꾼 뒤 숨은 변수를 되풀이해 없앤다. 없앨 변수마다 알고리즘은 그 변수를 담은 인자를 모두 모아 하나의 인자로 곱한 다음, 그 변수를 합으로 지운다(주변화한다). 남은 인자는 다시 무리에 넣는다.

핵심 연산은 인자 곱과 주변화이다. 인자 곱은 서로 어긋나지 않는 대입마다 값을 곱해 인자 둘을 합친다. 곧 $\phi_1(A,B)$과 $\phi_2(B,C)$이 있으면 $(\phi_1 \times \phi_2)(A,B,C) = \phi_1(A,B) \times \phi_2(B,C)$이다. 주변화는 변수를 합으로 지운다. 곧 $\phi'(X) = \sum_Y \phi(X, Y)$이다.

시간 복잡도는 $O(n \cdot d^{w+1})$이며, 여기서 $n$은 변수의 개수, $d$은 가장 큰 값의 가짓수, $w$은 (없애는 차례가 정하는) 이끌린 너비이다. 가장 좋은 없애기 차례를 찾는 일은 NP-어려움이지만, 이웃 최소나 채움 최소 같은 어림짐작이 실전에서 잘 듣는다. 나무 꼴 망에서는 $w = 1$이고 변수 없애기는 선형 시간에 돈다.

## 연습문제

**연습문제 1.**
날씨 망에서 물음 $P(\text{Rain} | \text{WetGrass}=1)$에 대해 변수 없애기 알고리즘을 걸음마다 따라가라. 걸음마다 중간 인자를 보여라.

??? success "연습문제 1 풀이"
    1. CPT에서 나온 첫 인자: $f_1(C)$, $f_2(C,S)$, $f_3(C,R)$, $f_4(S,R,W)$
2. 증거 $W=1$으로 줄이기: $f_4$은 $f_4'(S,R) = P(W=1|S,R)$이 된다
3. $C$ 없애기: $f_1(C), f_2(C,S), f_3(C,R)$을 모은다. 곱한다: $f_5(C,S,R) = f_1 \times f_2 \times f_3$. $C$을 주변화한다: $f_6(S,R) = \sum_C f_5(C,S,R)$
4. $S$ 없애기: $f_6(S,R), f_4'(S,R)$을 모은다. 곱한다: $f_7(S,R) = f_6 \times f_4'$. $S$을 주변화한다: $f_8(R) = \sum_S f_7(S,R)$
5. $f_8(R)$을 고르게 하여 $P(R|W=1)$을 얻는다

핵심 이득은 이것이다. 변수 4개 전체에 걸친 인자를 한꺼번에 만드는 일이 결코 없다.


---

**연습문제 2.**
없애는 차례가 효율에 왜 중요한지 설명하여라. 한 차례가 다른 차례보다 지수만큼 나쁜 보기를 들어라.

??? success "연습문제 2 풀이"
    없애는 차례가 중간 인자의 최대 크기를 정한다. $P(E)$을 구하려는 사슬 $A \to B \to C \to D \to E$을 생각해 보자.

좋은 차례(A, B, C, D): 없애는 걸음마다 많아야 변수 2개에 걸친 인자를 만든다. 최대 인자 크기: $d^2$.

다른 그래프에서의 나쁜 차례(잎 $L_1, \ldots, L_k$에 이어진 중심 H을 갖는 별 꼴): 잎을 먼저 없애면 괜찮다(저마다 변수 2개에 걸친 인자를 만든다). 그러나 $H$을 먼저 없애면 모든 잎 인자를 변수 $k$개에 걸친 거대한 인자 하나로 합쳐야 한다. 크기는 $d^k$, 곧 지수이다!

일반으로 없애는 차례의 이끌린 너비 $w$이 가장 큰 중간 인자를 정한다(항목 $d^{w+1}$개). 좋은 차례는 $w$을 가장 작게 하고, 나쁜 차례는 $w$을 $n-1$까지 키울 수 있다.


---

**연습문제 3.**
이진 변수를 갖는 사슬 망 $A \to B \to C \to D$에 대해 간추린 변수 없애기를 구현하고 $P(D | A = 1)$을 셈하여라.

??? success "연습문제 3 풀이"
    ```python
import numpy as np

# 사슬 A->B->C->D의 CPT
P_A = np.array([0.6, 0.4])
P_B_A = np.array([[0.7, 0.3], [0.3, 0.7]])
P_C_B = np.array([[0.8, 0.2], [0.4, 0.6]])
P_D_C = np.array([[0.9, 0.1], [0.2, 0.8]])

# 증거: A=1
# B 없애기: f(B) = P(B|A=1), 그다음 f(C) = sum_B P(C|B)*P(B|A=1)
f_B = P_B_A[1, :]  # [P(B=0|A=1), P(B=1|A=1)]
f_C = np.zeros(2)
for c in range(2):
    for b in range(2):
        f_C[c] += P_C_B[b, c] * f_B[b]

# C 없애기: f(D) = sum_C P(D|C)*f(C)
f_D = np.zeros(2)
for d in range(2):
    for c in range(2):
        f_D[d] += P_D_C[c, d] * f_C[c]

# 정규화
P_D_given_A1 = f_D / f_D.sum()
print(f'P(D=0|A=1) = {P_D_given_A1[0]:.4f}')
print(f'P(D=1|A=1) = {P_D_given_A1[1]:.4f}')
```

