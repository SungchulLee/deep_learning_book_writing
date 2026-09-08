# 단순한 추론

베이즈 망에서 추론이란 관측한 증거가 주어졌을 때 조건부 확률을 셈하는 일이다. 이 모듈은 가장 근본이 되는 길인 낱낱 세기 추론을 다루는데, 이는 숨은 변수의 모든 있을 수 있는 꼴에 걸쳐 합하여 정확한 답을 셈한다. 지수만큼 비싸지만 낱낱 세기는 더 효율적인 알고리즘을 이해하는 또렷한 바탕을 주고, 물리치기 표집 같은 어림 방법을 시험하는 잣대가 된다.

## 1. 코드

```python
"""
베이즈 망에서의 단순한 추론
======================================

이 모듈은 베이즈 망의 기본 추론 기법을 다루며
낱낱 세기로 하는 정확한 추론에 초점을 맞춘다.

학습 목표:
-------------------
1. 확률 모형에서 추론이 무엇인지 이해하기
2. 낱낱 세기 추론 배우기(마구잡이 길)
3. 주변 물음 던지기: P(X)
4. 조건부 물음 던지기: P(X|E)
5. 물리치기 표집과 그 한계 이해하기

수학의 바탕:
------------------------
주변 추론: P(X) = Σ_{Y} P(X, Y)
조건부 추론: P(X|E) = P(X, E) / P(E)
                              = Σ_{Y} P(X, E, Y) / Σ_{X,Y} P(X, E, Y)

지은이: 교육용 ML 팀
수준: 첫걸음
미리 볼 것: 01_pgm_fundamentals.py, 02_bayesian_networks_basics.py
"""

import numpy as np
from typing import Dict, List, Set, Optional
from itertools import product
import time
from collections import defaultdict

# ========================================================================
# 메인
# ========================================================================

# 앞선 모듈에서 들여오기
import sys
sys.path.append('..')
from beginner.bayesian_networks_basics import BayesianNetwork, build_weather_network


class InferenceByEnumeration:
    """
    있을 수 있는 모든 꼴을 낱낱이 세어 정확히 추론한다.
    
    이것이 가장 곧바른 추론 방법이다:
    1. 모든 변수의 있을 수 있는 대입을 낱낱이 센다
    2. 증거와 어긋나지 않는 대입의 확률을 합한다
    3. 고르게 하여 조건부 확률을 얻는다
    
    시간 복잡도: 이진 변수 n개에 대해 O(2^n)(지수이다!)
    공간 복잡도: O(2^n)
    
    효율은 낮지만 이 방법은 다음과 같다:
    - 정확하다(어림이 아니다)
    - 이해하기 쉽다
    - 작은 망에 쓸모 있다
    - 다른 알고리즘을 시험하는 좋은 잣대이다
    """
    
    def __init__(self, bn: BayesianNetwork):
        """
        추론 엔진의 첫걸음을 잡는다.
        
        인수:
            bn: 추론할 베이즈 망
        """
        self.bn = bn
        self.query_cache = {}  # 되풀이되는 물음을 위한 캐시
    
    def query_marginal(self, 
                      query_vars: List[str],
                      evidence: Optional[Dict[str, int]] = None,
                      use_cache: bool = True) -> Dict[tuple, float]:
        """
        낱낱 세기로 P(query_vars | evidence)을 셈한다.
        
        인수:
            query_vars: 확률 분포를 셈할 변수
            evidence: 관측한 변수 값의 사전(None일 수 있다)
            use_cache: 캐시한 결과를 쓸지 여부
        
        반환값:
            query_vars의 대입마다 확률에 잇는 사전
        
        보기:
            # P(Rain | Cloudy=1) 셈하기
            result = inference.query_marginal(
                query_vars=['Rain'],
                evidence={'Cloudy': 1}
            )
            # 되돌림: {(0,): 0.2, (1,): 0.8}
        """
        if evidence is None:
            evidence = {}
        
        # 캐시 열쇠 만들기
        query_key = (tuple(sorted(query_vars)), tuple(sorted(evidence.items())))
        if use_cache and query_key in self.query_cache:
            return self.query_cache[query_key]
        
        print(f"\nComputing P({', '.join(query_vars)}" + 
              (f" | {', '.join(f'{k}={v}' for k, v in evidence.items())})" if evidence else ")"))
        print("-" * 60)
        
        # 망의 모든 변수 얻기
        all_vars = list(self.bn.graph.nodes())
        
        # 변수를 물음, 증거, 숨은 것으로 가르기
        hidden_vars = [v for v in all_vars if v not in query_vars and v not in evidence]
        
        print(f"Query variables: {query_vars}")
        print(f"Evidence variables: {list(evidence.keys())}")
        print(f"Hidden variables: {hidden_vars}")
        
        # 결과 사전 첫걸음 잡기
        result = defaultdict(float)
        
        # 가짓수 얻기
        query_cards = [self.bn.cardinalities[v] for v in query_vars]
        hidden_cards = [self.bn.cardinalities[v] for v in hidden_vars]
        
        total_configurations = np.prod(query_cards) * np.prod(hidden_cards)
        print(f"\nEnumerating {int(total_configurations)} configurations...")
        
        # 물음 변수와 숨은 변수의 모든 대입 낱낱이 세기
        for query_values in product(*[range(c) for c in query_cards]):
            # 물음 대입마다 숨은 변수의 모든 대입에 걸쳐 합하기
            query_assignment = dict(zip(query_vars, query_values))
            
            for hidden_values in product(*[range(c) for c in hidden_cards]):
                hidden_assignment = dict(zip(hidden_vars, hidden_values))
                
                # 온전한 대입
                full_assignment = {**query_assignment, **hidden_assignment, **evidence}
                
                # 베이즈 망으로 P(온전한 대입) 셈하기
                prob = self.bn.compute_joint_probability(full_assignment)
                
                # 이 물음 대입의 결과에 더하기
                result[query_values] += prob
        
        # 조건부 확률 P(물음 | 증거)을 얻으려고 고르게 하기
        total = sum(result.values())
        
        if total > 0:
            result = {k: v/total for k, v in result.items()}
        else:
            print("Warning: Evidence has probability 0!")
            # 고른 분포
            result = {k: 1.0/len(result) for k in result.keys()}
        
        # 결과 캐시에 담기
        if use_cache:
            self.query_cache[query_key] = dict(result)
        
        return dict(result)
    
    def query_single_variable(self, 
                             variable: str,
                             evidence: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        변수 하나만 묻는 편한 메서드.
        
        인수:
            variable: 물을 변수
            evidence: 증거 사전
        
        반환값:
            변수의 값마다의 확률을 담은 NumPy 배열
        
        보기:
            # P(Rain | Cloudy=1) 셈하기
            probs = inference.query_single_variable('Rain', {'Cloudy': 1})
            print(f"P(Rain=0|Cloudy=1) = {probs[0]}")
            print(f"P(Rain=1|Cloudy=1) = {probs[1]}")
        """
        result_dict = self.query_marginal([variable], evidence)
        
        # 배열로 바꾸기
        cardinality = self.bn.cardinalities[variable]
        probs = np.zeros(cardinality)
        
        for value_tuple, prob in result_dict.items():
            probs[value_tuple[0]] = prob
        
        return probs
    
    def most_probable_explanation(self, 
                                  evidence: Dict[str, int]) -> Dict[str, int]:
        """
        증거가 주어졌을 때 가장 그럴듯한 온전한 대입을 찾는다.
        
        이를 MAP(최대 뒤확률) 추론이라고도 한다.
        argmax_X P(X | evidence)을 찾는다.
        
        인수:
            evidence: 관측한 변수
        
        반환값:
            증거가 아닌 모든 변수에 대한 가장 그럴듯한 대입
        
        보기:
            # 잔디가 젖었다면 가장 그럴듯한 설명은 무엇인가?
            mpe = inference.most_probable_explanation({'WetGrass': 1})
            # 되돌릴 수 있는 값: {'Cloudy': 1, 'Rain': 1, 'Sprinkler': 0}
        """
        print(f"\nFinding Most Probable Explanation given evidence:")
        print(f"Evidence: {evidence}")
        print("-" * 60)
        
        # 증거가 아닌 변수 모두 얻기
        all_vars = list(self.bn.graph.nodes())
        query_vars = [v for v in all_vars if v not in evidence]
        
        # 모든 대입 낱낱이 세기
        best_assignment = None
        best_prob = -1
        
        query_cards = [self.bn.cardinalities[v] for v in query_vars]
        
        for query_values in product(*[range(c) for c in query_cards]):
            assignment = dict(zip(query_vars, query_values))
            full_assignment = {**assignment, **evidence}
            
            prob = self.bn.compute_joint_probability(full_assignment)
            
            if prob > best_prob:
                best_prob = prob
                best_assignment = assignment
        
        print(f"\nMost probable assignment: {best_assignment}")
        print(f"Probability: {best_prob:.6f}")
        
        return best_assignment


class RejectionSampling:
    """
    물리치기 표집으로 하는 어림 추론.
    
    물리치기 표집:
    1. 앞확률 P(X)에서 표본을 만든다
    2. 증거와 맞지 않는 표본을 물리친다
    3. 받아들인 표본에서 통계량을 셈한다
    
    장점:
    - 구현이 단순하다
    - 치우침 없는 어림자이다
    
    단점:
    - 증거가 잘 안 일어나면 몹시 비효율적일 수 있다
    - 물리치는 비율이 아주 높을 수 있다
    """
    
    def __init__(self, bn: BayesianNetwork):
        """물리치기 표집의 첫걸음을 잡는다."""
        self.bn = bn
    
    def query(self,
             query_vars: List[str],
             evidence: Dict[str, int],
             num_samples: int = 10000) -> Dict[tuple, float]:
        """
        물리치기 표집으로 추론한다.
        
        인수:
            query_vars: 물을 변수
            evidence: 증거 사전
            num_samples: 만들 표본의 개수
        
        반환값:
            물음 변수에 걸친 어림 확률 분포
        """
        print(f"\nRejection Sampling: {num_samples} samples")
        print("-" * 60)
        
        accepted_samples = []
        rejected = 0
        
        # 표본 만들기
        for _ in range(num_samples):
            sample = self.bn.forward_sample()
            
            # 표본이 증거와 맞는지 살피기
            matches = all(sample[var] == val for var, val in evidence.items())
            
            if matches:
                accepted_samples.append(sample)
            else:
                rejected += 1
        
        print(f"Accepted: {len(accepted_samples)}")
        print(f"Rejected: {rejected}")
        print(f"Acceptance rate: {len(accepted_samples)/num_samples:.2%}")
        
        if len(accepted_samples) == 0:
            print("Warning: No samples accepted! Try more samples or check evidence.")
            return {}
        
        # 받아들인 표본에서 확률 셈하기
        result = defaultdict(int)
        
        for sample in accepted_samples:
            key = tuple(sample[var] for var in query_vars)
            result[key] += 1
        
        # 정규화
        total = sum(result.values())
        result = {k: v/total for k, v in result.items()}
        
        return dict(result)


def demonstrate_simple_queries():
    """
    날씨 망에 단순한 추론 물음을 던져 보인다.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Simple Inference Queries")
    print("="*70)
    
    # 신경망 만들기
    bn = build_weather_network()
    inference = InferenceByEnumeration(bn)
    
    # 물음 1: 주변 확률 P(Rain)
    print("\n" + "="*70)
    print("Query 1: What's the probability of rain?")
    print("="*70)
    
    rain_probs = inference.query_single_variable('Rain')
    print(f"\nP(Rain=0) = {rain_probs[0]:.4f}")
    print(f"P(Rain=1) = {rain_probs[1]:.4f}")
    
    # 물음 2: 조건부 확률 P(Rain | Cloudy=1)
    print("\n" + "="*70)
    print("Query 2: What's the probability of rain given it's cloudy?")
    print("="*70)
    
    rain_given_cloudy = inference.query_single_variable('Rain', {'Cloudy': 1})
    print(f"\nP(Rain=0 | Cloudy=1) = {rain_given_cloudy[0]:.4f}")
    print(f"P(Rain=1 | Cloudy=1) = {rain_given_cloudy[1]:.4f}")
    
    print("\nNote: Rain is more likely when it's cloudy (0.8 vs 0.2)")
    
    # 물음 3: 여러 변수 P(Rain, Sprinkler | WetGrass=1)
    print("\n" + "="*70)
    print("Query 3: Given wet grass, what caused it?")
    print("="*70)
    
    result = inference.query_marginal(['Rain', 'Sprinkler'], {'WetGrass': 1})
    
    print("\nP(Rain, Sprinkler | WetGrass=1):")
    for (rain, sprinkler), prob in sorted(result.items()):
        print(f"  Rain={rain}, Sprinkler={sprinkler}: {prob:.4f}")
    
    print("\nInterpretation:")
    print("- Most likely: Rain=1, Sprinkler=0 (rain caused it)")
    print("- Second: Rain=0, Sprinkler=1 (sprinkler caused it)")
    print("- Least likely: Rain=0, Sprinkler=0 (shouldn't happen but numerical errors)")


def demonstrate_mpe():
    """
    가장 그럴듯한 설명(MPE) 추론을 보인다.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Most Probable Explanation (MPE)")
    print("="*70)
    
    bn = build_weather_network()
    inference = InferenceByEnumeration(bn)
    
    # 잔디가 젖었을 때의 MPE
    print("\n" + "="*70)
    print("Scenario: You observe that the grass is wet")
    print("Question: What's the most likely complete explanation?")
    print("="*70)
    
    mpe = inference.most_probable_explanation({'WetGrass': 1})
    
    print("\nInterpretation:")
    if mpe.get('Rain') == 1:
        print("- It probably rained")
    if mpe.get('Cloudy') == 1:
        print("- It was probably cloudy")
    if mpe.get('Sprinkler') == 1:
        print("- The sprinkler was probably on")
    else:
        print("- The sprinkler was probably off")


def compare_exact_vs_approximate():
    """
    정확한 추론과 물리치기 표집을 견준다.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: Exact vs Approximate Inference")
    print("="*70)
    
    bn = build_weather_network()
    
    # 정확한 추론
    print("\n" + "-"*70)
    print("EXACT INFERENCE (Enumeration)")
    print("-"*70)
    
    exact_inference = InferenceByEnumeration(bn)
    
    start = time.time()
    exact_result = exact_inference.query_single_variable('Rain', {'WetGrass': 1})
    exact_time = time.time() - start
    
    print(f"\nP(Rain | WetGrass=1):")
    print(f"  Rain=0: {exact_result[0]:.4f}")
    print(f"  Rain=1: {exact_result[1]:.4f}")
    print(f"Time: {exact_time:.6f} seconds")
    
    # 어림 추론
    print("\n" + "-"*70)
    print("APPROXIMATE INFERENCE (Rejection Sampling)")
    print("-"*70)
    
    rejection_sampling = RejectionSampling(bn)
    
    start = time.time()
    approx_result = rejection_sampling.query(['Rain'], {'WetGrass': 1}, num_samples=10000)
    approx_time = time.time() - start
    
    print(f"\nP(Rain | WetGrass=1):")
    print(f"  Rain=0: {approx_result.get((0,), 0):.4f}")
    print(f"  Rain=1: {approx_result.get((1,), 0):.4f}")
    print(f"Time: {approx_time:.6f} seconds")
    
    # 비교
    print("\n" + "-"*70)
    print("COMPARISON")
    print("-"*70)
    print(f"\nError in P(Rain=1 | WetGrass=1):")
    error = abs(exact_result[1] - approx_result.get((1,), 0))
    print(f"  {error:.4f} ({error/exact_result[1]*100:.2f}% relative error)")
    
    print(f"\nNote: Rejection sampling is approximate but can be much faster")
    print(f"for large networks. However, it can be very inefficient if")
    print(f"evidence is unlikely (low acceptance rate).")


def main():
    """
    주된 보여 주기 함수.
    """
    print("\n" + "="*70)
    print("SIMPLE INFERENCE IN BAYESIAN NETWORKS")
    print("="*70)
    
    print("\nTopics covered:")
    print("1. Inference by enumeration (exact)")
    print("2. Marginal queries: P(X) and P(X|E)")
    print("3. Most Probable Explanation (MPE)")
    print("4. Rejection sampling (approximate)")
    
    # 시연 실행
    demonstrate_simple_queries()
    demonstrate_mpe()
    compare_exact_vs_approximate()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. Inference = computing P(Query | Evidence)")
    print("\n2. Enumeration is exact but exponentially expensive:")
    print("   - Time: O(2^n) for n binary variables")
    print("   - Practical only for small networks")
    
    print("\n3. Approximate methods (like rejection sampling):")
    print("   - Trade accuracy for speed")
    print("   - Useful for large networks")
    print("   - Accuracy improves with more samples")
    
    print("\n4. More efficient exact methods exist:")
    print("   - Variable elimination (next module!)")
    print("   - Junction tree algorithm")
    print("   - These exploit network structure")
    
    print("\n" + "="*70)
    print("Next: Learn about efficient inference algorithms!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()```

## 2. 논의

낱낱 세기 추론은 조건부 확률의 정의 $P(Q|E) = P(Q, E) / P(E)$을 써서 $P(\text{Query} | \text{Evidence})$ 꼴의 물음에 답한다. 분자와 분모 모두 숨은 변수의 모든 대입에 걸쳐 결합 확률을 합하여 셈한다. 이진 변수가 $n$개면 항 $O(2^n)$개의 값을 매겨야 한다.

코드는 어림 추론 방법인 물리치기 표집도 구현한다. 망에서 앞먹임 표본을 뽑고 증거와 어긋나는 표본은 버린다. 남은 표본이 뒤확률 분포를 어림한다. 단순하고 치우침이 없지만, 증거가 잘 일어나지 않는 일이면 물리치기 표집은 몹시 비효율적일 수 있다. 표본 대부분이 버려지기 때문이다.

가장 그럴듯한 설명(MPE) 물음은 $P(\text{assignment} | \text{evidence})$을 가장 크게 하는 온전한 대입을 찾는다. 이는 진단 추론에 쓸모 있다. 곧 관측한 증상이 주어졌을 때 모든 변수의 가장 그럴듯한 꼴은 무엇인가? 정확한 낱낱 세기와 물리치기 표집의 견줌은 정확함과 셈 값 사이의 주고받음을 도드라지게 하며, 이것이 변수 없애기 같은 더 효율적인 알고리즘을 부른다.

## 연습문제

**연습문제 1.**
날씨 망의 CPT을 써서 $P(\text{Cloudy}=1 | \text{WetGrass}=1)$을 손으로 셈하여라. 낱낱 세기의 모든 걸음을 보여라.

??? success "연습문제 1 풀이"
    $P(C=1|W=1) = P(C=1,W=1)/P(W=1)$이 필요하다.

$P(C=1,W=1) = \sum_{S,R} P(C=1)P(S|C=1)P(R|C=1)P(W=1|S,R)$

펼치면 다음과 같다.
- $S=0,R=0$: $0.5 \times 0.9 \times 0.2 \times 0.0 = 0$
- $S=0,R=1$: $0.5 \times 0.9 \times 0.8 \times 0.9 = 0.324$
- $S=1,R=0$: $0.5 \times 0.1 \times 0.2 \times 0.9 = 0.009$
- $S=1,R=1$: $0.5 \times 0.1 \times 0.8 \times 0.99 = 0.0396$

$P(C=1,W=1) = 0.3726$

마찬가지로 $P(C=0,W=1) = 0.1725$을 셈한다(C=0 CPT으로 S, R에 걸쳐 합한다).

$P(W=1) = 0.3726 + 0.1725 = 0.5451$

$P(C=1|W=1) = 0.3726/0.5451 \approx 0.6836$

잔디가 젖었다는 사실이 흐렸을 가능성을 더 높인다(앞확률은 0.5, 뒤확률은 약 0.68).


---

**연습문제 2.**
증거의 확률이 낮을 때 물리치기 표집이 왜 실전에서 못 쓰게 되는지 설명하여라. $P(\text{evidence})$으로 나타낸 받아들임 비율은 얼마인가?

??? success "연습문제 2 풀이"
    물리치기 표집에서는 앞확률 분포에서 온전한 표본을 만들고 증거와 맞을 때만 남긴다. 무작위 표본이 증거와 맞을 확률은 꼭 $P(\text{evidence})$이다. 그러므로 다음과 같다:

- 받아들임 비율 = $P(\text{evidence})$
- 받아들인 표본 하나를 얻는 데 필요한 기대 표본 수 = $1/P(\text{evidence})$

$P(\text{evidence}) = 0.001$(드문 증거)이면 받아들인 표본 하나를 얻는 데 표본이 1000개쯤 필요하다. $P(E) = 10^{-6}$처럼 아주 드문 증거라면 받아들인 표본 하나마다 백만 개가 필요해 이 방법을 쓸 수 없게 된다.

그래서 더 정교한 방법이 나온다. 늘 받아들이되 무게를 다시 주는 가능도 무게 주기, 증거에 곧바로 조건을 거는 깁스 표집, 그리고 표집을 아예 하지 않는 변수 없애기이다.


---

**연습문제 3.**
날씨 망에 대해 가능도 무게 주기를 구현하고, 물음 $P(\text{Rain}|\text{WetGrass}=1)$에서 물리치기 표집과 효율을 견주어라.

??? success "연습문제 3 풀이"
    ```python
import numpy as np

np.random.seed(42)
n_samples = 50000

# 물리치기 표집
accepted_rain = []
for _ in range(n_samples):
    c = np.random.binomial(1, 0.5)
    s = np.random.binomial(1, 0.1 if c else 0.5)
    r = np.random.binomial(1, 0.8 if c else 0.2)
    pw = [[0,0.9],[0.9,0.99]][s][r] if r or s else 0
    w = np.random.binomial(1, [[0,0.9],[0.9,0.99]][s][r])
    if w == 1:
        accepted_rain.append(r)

print(f'Rejection: P(R=1|W=1)={np.mean(accepted_rain):.4f}, '
      f'accepted={len(accepted_rain)}/{n_samples}')

# 가능도 무게 주기
weighted_rain = []
weights = []
for _ in range(n_samples):
    c = np.random.binomial(1, 0.5)
    s = np.random.binomial(1, 0.1 if c else 0.5)
    r = np.random.binomial(1, 0.8 if c else 0.2)
    w_prob = [[0,0.9],[0.9,0.99]][s][r]
    weighted_rain.append(r)
    weights.append(w_prob)

weights = np.array(weights)
weighted_rain = np.array(weighted_rain)
result = np.sum(weights * weighted_rain) / np.sum(weights)
print(f'Likelihood weighting: P(R=1|W=1)={result:.4f}, used all {n_samples} samples')
```
가능도 무게 주기는 증거 변수를 붙박아 두고 표집한 어버이가 주어졌을 때 증거의 확률로 표본마다 무게를 주어 모든 표본을 쓴다(물리치지 않는다). 특히 증거가 잘 일어나지 않는 일일 때 훨씬 효율적이다.

## 정리하며

**다룬 것** — 단순한 추론

낱낱 세기 추론은 조건부 확률의 정의 $P(Q|E) = P(Q, E) / P(E)$을 써서 $P(\text{Query} | \text{Evidence})$ 꼴의 물음에 답한다.

고갱이 갈래는 `InferenceByEnumeration`, `RejectionSampling`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
