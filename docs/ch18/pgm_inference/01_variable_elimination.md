# 변수 제거

변수 제거(variable elimination)는 인수분해된 분포에서 정확한 주변·조건부 확률을 구한다. 제거할 변수를 포함한 인자만 곱하고 그 변수를 합산하므로 전체 결합표를 만들 필요가 없을 수 있다. 효율은 제거 순서와 중간 인자의 크기에 달려 있으며 일반적인 지수 비용이 사라지는 것은 아니다.

## 1. 코드

??? note "코드 (533줄)"

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
    미리 볼 것: 첫스텝 수준의 01-04
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
        변수 배치에 걸친 인자(퍼텐셜 함수)를 나타낸다.

        인자 φ(X1, ..., Xk)은 (X1, ..., Xk)의 대입마다
        음이 아닌 수에 잇는 함수이다.

        인자는 변수 없애기의 근본 자료 짜임이다.
        """

        def __init__(self,
                     variables: List[str],
                     cardinalities: Dict[str, int],
                     values: Optional[np.ndarray] = None):
            """
            인자의 첫스텝을 잡는다.

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
            변수 없애기의 첫스텝을 잡는다.

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
                    # 남은 배치에서 이웃이 가장 적은 변수 찾기
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

            # 스텝 1: CPT에서 첫 인자 만들기
            factors = self._create_factors_from_cpts()

            if verbose:
                print(f"\nStep 1: Created {len(factors)} initial factors from CPTs")

            # 스텝 2: 증거로 인자 줄이기
            if evidence:
                factors = [f.reduce(evidence) for f in factors]
                if verbose:
                    print(f"Step 2: Reduced factors with evidence {evidence}")

            # 스텝 3: 없앨 변수 정하기
            all_vars = set(self.bn.graph.nodes())
            variables_to_eliminate = all_vars - set(query_vars) - set(evidence.keys())

            if verbose:
                print(f"Step 3: Variables to eliminate: {sorted(variables_to_eliminate)}")

            # 스텝 4: 없애는 차례 고르기
            if elimination_order is None:
                elimination_order = self._choose_elimination_order(variables_to_eliminate)

            if verbose:
                print(f"Step 4: Elimination order: {elimination_order}")

            # 스텝 5: 변수를 하나씩 없애기
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

            # 스텝 6: 남은 인자 곱하기
            if verbose:
                print(f"\nStep 6: Multiplying {len(factors)} remaining factors...")

            result_factor = factors[0]
            for f in factors[1:]:
                result_factor = result_factor.multiply(f)

            # 스텝 7: 고르게 하기
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
        main()
    ```


## 2. 논의

### 인자 곱과 합산

먼저 CPT를 인자로 바꾸고 관측 증거에 해당하는 축 값을 고정한다. 제거할 변수 Y를 포함한 모든 인자를 모아 곱한 뒤 Y를 합산하고 새 인자를 집합에 돌려놓는다. 마지막에는 질의 변수의 인자를 정규화한다.

$$[\phi_1\phi_2](A,B,C)=\phi_1(A,B)\phi_2(B,C),\qquad\phi'(X)=\sum_Y\phi(X,Y)$$

정규화 전 결과는 증거를 포함한 결합 질량이다. 증거 확률이 0이면 조건부분포가 정의되지 않는다.

### 제거 순서와 복잡도

인자 스코프에서 만든 무방향 상호작용 그래프에서 변수를 제거할 때 그 이웃을 서로 연결한다. 이 fill-in을 포함한 유도 폭 w와 최대 상태 수 d에 따라 대표적인 표형 연산 비용은 O(n d^(w+1))로 나타난다. 최적 순서 탐색은 일반적으로 어렵고 최소 차수·최소 fill-in 등의 휴리스틱을 사용한다.

무방향 쌍별 트리에서 적절히 잎부터 제거하면 w=1이다. 하지만 DAG의 뼈대가 트리라는 사실만으로 w=1이라고 할 수는 없다. 부모가 많은 CPT는 처음부터 큰 인자이며 도덕화 그래프에서 부모 사이가 연결된다. 질의가 여러 변수의 큰 결합표를 요구하면 출력 크기 자체의 비용도 남는다.

### 구현의 범위

Factor는 변수명을 정렬하므로 외부 values 배열도 그 축 순서로 제공해야 한다. 스칼라 인자의 예상 형태는 ()인데 일부 메서드는 길이 1 배열을 전달하고 values[0]을 사용하므로 빈 변수 인자를 만드는 경로는 일관되지 않다. 모든 변수가 증거로 고정되거나 마지막 변수를 합산하는 경우를 완성된 구현으로 가정하지 않는다.

자동 순서는 DAG의 무방향 뼈대에서 남은 제거 변수 이웃 수만 비교한다. 부모 간 도덕화·실제 인자 스코프·fill-in·유지할 질의 변수 이웃을 반영하지 않으므로 표준적인 동적 최소 차수 순서와 다르다. 정확한 인자 연산이 유지되면 순서의 차이는 결과보다 비용에 영향을 주지만, 사용자 순서에 숨은 변수가 빠지거나 질의 변수가 들어가면 올바른 질의를 보장하지 못한다. 질의와 증거는 겹치지 않게 하고 상태 값과 정규화 합을 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
날씨 네트워크에서 P(Rain|WetGrass=1)를 C, S 순서로 제거해 표현하라.

</div>

??? success "연습문제 1 풀이"

    초기 인자는 f₁(C)=P(C), f₂(C,S)=P(S|C), f₃(C,R)=P(R|C), f₄(S,R,W)=P(W|S,R)다. 증거로 f₄를 g(S,R)=P(W=1|S,R)로 줄인다. C를 제거해 h(S,R)=∑_C f₁f₂f₃를 만들고, S를 제거해 u(R)=∑_S h(S,R)g(S,R)를 얻는다. u의 합이 양수일 때 정규화한다. 이 순서에서는 최대 세 변수 인자만 필요하며 네 변수 전체 표는 만들지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
중심 H와 잎 L₁,…,L_k의 쌍별 인자가 있는 별 그래프에서 제거 순서의 영향을 비교하라.

</div>

??? success "연습문제 2 풀이"

    한 잎을 질의로 유지하고 다른 잎부터 제거하면 중간 인자가 작다. 반대로 H를 먼저 제거하면 H와 모든 잎을 포함한 곱 인자는 d^(k+1) 항목, H 합산 뒤 인자는 d^k 항목을 갖는다. 단일 질의에서도 중간 비용이 지수적으로 커질 수 있다. 모든 잎의 결합분포가 출력이면 그 출력 크기 자체도 d^k이므로 과제를 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 A→B→C→D 예제가 P(D|A=1)를 계산하는 인자 갱신을 설명하라.

</div>

??? success "연습문제 3 풀이"

    A=1에서 P(B|A=1)를 시작 인자로 사용한다. B를 합산해 C의 조건부분포를 얻고 C를 합산해 D의 분포를 얻는다. P(A=1)>0에서 공통 인자 P(A=1)는 정규화로 약분되므로 별도로 곱하지 않아도 된다. 이는 사슬 전용 배열 계산이며 위 범용 Factor 클래스의 스칼라·순서 처리를 검증하는 테스트는 아니다.

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


## 정리하며

변수 제거는 관련 인자의 곱과 합산을 반복해 정확한 추론을 수행한다. 유도 폭과 출력 크기가 비용을 결정하며, 좋은 순서도 모든 모형을 다항 시간으로 만들지는 않는다. 증거의 양의 확률, 인자 축 순서·스칼라 표현과 제거 순서의 완전성을 확인한다.
