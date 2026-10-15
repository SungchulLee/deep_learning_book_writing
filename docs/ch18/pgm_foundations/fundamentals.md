# 확률 그래프 모델의 기초

확률 그래프 모델은 변수의 조건부 독립을 그래프로 나타내고 결합분포를 국소 인자로 분해한다. 표현의 간결함, 확률 질의와 인과 해석의 차이를 먼저 이해한 뒤 방향·무방향 모형과 이산 분포 연산을 살펴본다.

---

## 1. 확률에서의 차원의 저주

각 변수가 k개의 상태를 갖는 이산 변수 n개의 제한 없는 결합분포에는 kⁿ개의 표 항목과 다음 자유 매개변수가 필요하다.

$$k^n-1$$

마지막 하나는 전체 합이 1이라는 조건으로 정해진다. 이진 변수 20개이면 1,048,575개, 50개이면 약 1.13×10¹⁵개다. 충분한 자료를 얻고 표를 저장하는 문제가 모두 어려워진다. 그래프 모델은 조건부 독립과 분포 형태의 가정을 통해 이 표현을 줄이지만 임의의 분포를 항상 희소하게 표현할 수는 없다.

## 2. 핵심 통찰: 조건부 독립

X와 Y가 Z를 조건으로 독립이라는 것은 양의 확률을 갖는 Z의 값에서 다음 분해가 성립한다는 뜻이다.

$$P(X,Y\mid Z)=P(X\mid Z)P(Y\mid Z)$$

적절히 정의된 조건부분포에서는 P(X|Y,Z)=P(X|Z)와 동치다. Z를 알고 나면 Y가 X의 조건부분포를 추가로 바꾸지 않는다. 연속 변수에서는 조건부 밀도와 거의 모든 값에서의 관계로 이해한다.

가상 공통원인 D와 두 결과 S₁·S₂에 $P(D,S_1,S_2)=P(D)P(S_1|D)P(S_2|D)$를 가정하면 S₁⊥S₂|D다. 결과 둘에 직접 간선이 없다는 말만으로 현실의 모든 숨은 의존성이 사라지는 것은 아니며, 전체 모형의 인수분해 가정이 중요하다.

## 3. PGM의 엄밀한 정의

PGM은 확률변수를 노드로 나타내고 그래프에 맞는 인수분해나 마르코프 독립 조건을 부여한 분포 모형이다. 방향 그래프는 부모 조건부분포를, 무방향 그래프는 클리크의 잠재함수와 전역 정규화 상수를 사용한다.

국소 인자가 작으면 저장·추정이 간단해질 수 있다. 그러나 일반적인 정확 추론은 여전히 어려우며 그래프의 트리폭과 변수 상태 수에 크게 좌우된다. 그래프에서 읽는 독립은 그 그래프의 모형을 따르는 분포에 대한 보장이다.

## 4. 두 큰 갈래

### 방향 그래프 모델(베이즈 네트워크)

```
    Cloudy
    /    \
   v      v
 Rain    Sprinkler
    \    /
     v  v
   WetGrass
```

베이즈 네트워크는 DAG를 사용한다. 노드마다 부모 조건부분포 P(X_i|Pa(X_i))를 지정하고 이산형이면 CPT로 저장할 수 있다. 화살표는 인수분해의 방향이며 추가적인 인과 가정 없이 개입 효과까지 의미하지는 않는다.

**인수분해:**

$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

### 무방향 그래프 모델(마르코프 랜덤 필드)

```
    A --- B
    |     |
    |     |
    C --- D
```

마르코프 랜덤 필드(MRF)는 무방향 그래프의 클리크에 음이 아닌 잠재함수 ψ_C를 둔다. 이 함수는 개별 조건부 확률일 필요가 없고, 곱을 유한한 양의 분배함수 Z로 나누어 정규화한다. 양의 분포에서 그래프의 마르코프 독립성과 클리크 인수분해의 동치는 Hammersley–Clifford 정리의 조건 아래 성립한다. 단순한 상관관계의 대칭성만으로 모형이 정해지는 것은 아니다.

**인수분해:**

$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

---

## 5. PGM의 근본 연산

### 주변화와 조건화

이산 변수에서 관심 없는 변수를 합산하면 주변분포를 얻는다.

$$P(X=x)=\sum_yP(X=x,Y=y)$$

양의 증거 확률 P(Y=y)>0이면 조건부분포는 다음과 같다.

$$P(X\mid Y=y)=\frac{P(X,Y=y)}{\sum_xP(X=x,Y=y)}$$

### 질의 구분

- **주변·조건부 확률**: 관심 변수의 분포를 계산하며 다른 미관측 변수는 합산한다.
- **Marginal MAP**: 관심 변수 Q에 대해 $\arg\max_q\sum_hp(q,h\mid e)$를 구한다.
- **MPE**: 증거 이외 모든 변수의 공동 상태 $(q,h)$를 최대화한다.

같은 변수 집합을 최대화하면 P(X,e)와 P(X|e)의 최대화 위치는 같지만, 일부 변수를 합산하는 marginal MAP와 모든 변수를 최대화하는 MPE는 일반적으로 다르다.

## 6. PyTorch 구현: 이산 분포

다음 클래스는 이산 결합표의 주변화·조건화·수치 독립성 확인을 구현한다. 입력 표의 형태·음이 아닌 값·양의 합과 축 순서를 검증해야 한다. 확률 0인 증거는 정규화할 수 없다. marginalize는 실제 축 순서를 재배열하지 않으므로 keep_variables를 원래 순서로 전달해야 한다. 모든 변수를 그대로 유지할 때 비어 있는 축 목록을 합산에 전달하는 경우의 동작도 별도로 확인한다.

이 클래스는 결합표 전체를 저장하므로 그래프 인수분해를 이용해 지수 크기를 줄인 구현은 아니다. 독립성 확인은 표의 근사 등식을 검사하며 표본 자료에 대한 유의성 검정과 다르다.

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class DiscreteDistribution:
    """
    여러 변수에 걸친 이산 확률 분포를 나타낸다.
    
    분포는 차원마다 확률 변수 하나에 해당하는
    여러 차원 텐서로 저장한다.
    """
    
    def __init__(self, 
                 variables: List[str],
                 cardinalities: Dict[str, int],
                 values: Optional[torch.Tensor] = None):
        """
        이산 확률 분포의 첫걸음을 잡는다.
        
        인수:
            variables: 차례대로 늘어놓은 변수 이름의 목록
            cardinalities: 변수 이름을 값의 개수에 잇는 사전
            values: 확률의 텐서(없어도 되며 기본값은 고른 분포)
        """
        self.variables = variables
        self.cardinalities = cardinalities
        self.shape = tuple(cardinalities[var] for var in variables)
        
        if values is None:
            self.values = torch.ones(self.shape) / torch.prod(
                torch.tensor(self.shape)
            ).float()
        else:
            self.values = values
            self.values = self.values / self.values.sum()
    
    def marginalize(self, keep_variables: List[str]) -> 'DiscreteDistribution':
        """
        keep_variables에 없는 변수를 주변화한다.
        
        P(X) = sum_Y P(X, Y)
        """
        sum_vars = [var for var in self.variables if var not in keep_variables]
        sum_axes = tuple(self.variables.index(var) for var in sum_vars)
        new_values = self.values.sum(dim=sum_axes)
        new_cards = {var: self.cardinalities[var] for var in keep_variables}
        return DiscreteDistribution(keep_variables, new_cards, new_values)
    
    def condition(self, evidence: Dict[str, int]) -> 'DiscreteDistribution':
        """
        관측 값에 조건을 건다.
        
        P(X | Y=y) = P(X, Y=y) / P(Y=y)
        """
        indices = []
        remaining_vars = []
        
        for var in self.variables:
            if var in evidence:
                indices.append(evidence[var])
            else:
                indices.append(slice(None))
                remaining_vars.append(var)
        
        conditioned = self.values[tuple(indices)]
        conditioned = conditioned / conditioned.sum()
        new_cards = {var: self.cardinalities[var] for var in remaining_vars}
        return DiscreteDistribution(remaining_vars, new_cards, conditioned)
    
    def is_independent(self, 
                       var1: str, 
                       var2: str, 
                       tol: float = 1e-6) -> bool:
        """
        변수 둘이 주변으로 독립인지 검정한다.
        
        P(X,Y) = P(X)P(Y)일 때 그리고 그때만 X _|_ Y
        """
        p_var1 = self.marginalize([var1])
        p_var2 = self.marginalize([var2])
        p_joint = self.marginalize([var1, var2])
        
        idx1 = p_joint.variables.index(var1)
        idx2 = p_joint.variables.index(var2)
        
        shape1 = [1, 1]
        shape1[idx1] = self.cardinalities[var1]
        shape2 = [1, 1]
        shape2[idx2] = self.cardinalities[var2]
        
        product = p_var1.values.view(*shape1) * p_var2.values.view(*shape2)
        diff = torch.abs(p_joint.values - product)
        return diff.max().item() < tol
    
    def is_conditionally_independent(self,
                                     var1: str,
                                     var2: str,
                                     given: List[str],
                                     tol: float = 1e-6) -> bool:
        """
        다른 변수가 주어졌을 때 변수 둘이 조건부 독립인지 검정한다.
        
        모든 z에 대해 P(X,Y|Z=z) = P(X|Z=z)P(Y|Z=z)일 때 그리고 그때만 X _|_ Y | Z
        """
        from itertools import product as cartesian_product
        
        given_cards = [self.cardinalities[var] for var in given]
        
        for assignment in cartesian_product(*[range(c) for c in given_cards]):
            evidence = dict(zip(given, assignment))
            conditioned = self.condition(evidence)
            if not conditioned.is_independent(var1, var2, tol):
                return False
        return True
    
    def entropy(self) -> torch.Tensor:
        """섀넌 엔트로피 H(X) = -sum P(x) log P(x)을 nat 단위로 셈한다."""
        probs = self.values.flatten()
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log(probs))
    
    def __repr__(self) -> str:
        return f"DiscreteDistribution({self.variables}, shape={self.shape})"

# --- 보여 주기 ---
if __name__ == "__main__":
    dist = DiscreteDistribution(
        variables=['X', 'Y', 'Z'],
        cardinalities={'X': 2, 'Y': 2, 'Z': 2},
        values=torch.tensor([
            [[0.1, 0.05], [0.1, 0.15]],   # X=0
            [[0.15, 0.1], [0.2, 0.15]]    # X=1
        ])
    )
    
    print(f"Joint distribution P(X, Y, Z), shape: {dist.shape}")
    print(f"Entropy: {dist.entropy():.4f} nats")
    
    p_x = dist.marginalize(['X'])
    print(f"\nP(X): {p_x.values}")
    
    p_xy_given_z = dist.condition({'Z': 1})
    print(f"\nP(X, Y | Z=1):\n{p_xy_given_z.values}")
    
    print(f"\nX _|_ Y? {dist.is_independent('X', 'Y')}")
    print(f"X _|_ Y | Z? {dist.is_conditionally_independent('X', 'Y', ['Z'])}")
```

**출력:**

```
Joint distribution P(X, Y, Z), shape: (2, 2, 2)
Entropy: 2.0162 nats

P(X): tensor([0.4000, 0.6000])

P(X, Y | Z=1):
tensor([[0.1111, 0.3333],
        [0.2222, 0.3333]])

X _|_ Y? False
X _|_ Y | Z? False
```

---

## 7. 근본이 되는 세 구조

기본 그래프 구조 셋을 이해하는 것이 방향 그래프에서 조건부 독립을 읽어 내는 키다. 이 구조들은 PGM 이론과 실전 곳곳에서 되풀이해 나타난다.

### 1. 사슬: X -> Z -> Y

```
X --> Z --> Y
```

사슬에서는 Z를 조건으로 두면 X와 Y가 독립이다. Z를 조건으로 두지 않을 때는 경로가 열려 종속일 수 있지만, 특정 CPT에서 추가 독립이 생길 수도 있다.

$$X \perp\!\!\!\perp Y \mid Z$$

### 2. 포크(공통원인): X <- Z -> Y

```
X <-- Z --> Y
```

포크에서는 공통 부모 Z를 조건으로 두면 X와 Y가 독립이다. Z를 합산한 혼합분포에서는 일반적으로 종속일 수 있다. 독립이 아닌 모든 분포가 반드시 이 공통원인 구조라는 뜻은 아니다.

$$X \perp\!\!\!\perp Y \mid Z$$

### 3. 충돌점(v-구조): X -> Z <- Y

```
X --> Z <-- Y
```

세 노드만 있는 충돌점 X→Z←Y에서는 X와 Y가 주변적으로 독립이다. Z 또는 그 자손에 조건을 두면 경로가 열려 종속이 생길 수 있다. 이는 특정 수치에서의 필연적 종속은 아니며 다음 식은 explaining away가 발생하는 경우를 나타낸다.

$$X \perp\!\!\!\perp Y \quad \text{(marginal)}, \qquad X \not\perp\!\!\!\perp Y \mid Z \quad \text{(explaining away)}$$

### explaining away 효과

아래 가상 경보 예제에서 X는 침입, Y는 지진, Z는 경보다. 두 원인은 독립이지만 경보를 관측한 뒤 지진을 알면 침입의 조건부 확률이 낮아진다. 지진이 경보를 설명할 수 있기 때문이다. 이런 감소 방향은 이 예제의 CPT에 따른 결과이며 임의의 충돌점에서 언제나 음의 연관이 생기는 것은 아니다.

```python
import torch

# 앞확률
p_burglary = 0.001
p_earthquake = 0.002

# P(Alarm | Burglary, Earthquake)
p_alarm_given_b_e = torch.tensor([
    [0.001, 0.29],    # Burglary=0: [Earthquake=0, Earthquake=1]
    [0.94, 0.95]      # Burglary=1: [Earthquake=0, Earthquake=1]
])

# P(Alarm=1) 셈하기
p_alarm = (
    p_alarm_given_b_e[0, 0] * (1 - p_burglary) * (1 - p_earthquake)
    + p_alarm_given_b_e[0, 1] * (1 - p_burglary) * p_earthquake
    + p_alarm_given_b_e[1, 0] * p_burglary * (1 - p_earthquake)
    + p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
)

# 베이즈 규칙으로 P(Burglary=1 | Alarm=1)
p_burglary_given_alarm = (
    p_alarm_given_b_e[1, 0] * p_burglary * (1 - p_earthquake)
    + p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
) / p_alarm

# P(Burglary=1 | Alarm=1, Earthquake=1)
p_alarm_and_earthquake = (
    p_alarm_given_b_e[0, 1] * (1 - p_burglary) * p_earthquake
    + p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
)
p_burglary_given_alarm_earthquake = (
    p_alarm_given_b_e[1, 1] * p_burglary * p_earthquake
) / p_alarm_and_earthquake

print(f"P(Burglary | Alarm)                = {p_burglary_given_alarm:.4f}")
print(f"P(Burglary | Alarm, Earthquake)    = {p_burglary_given_alarm_earthquake:.4f}")
print("\nExplaining away: learning about the earthquake decreased burglary probability.")
```

**출력:**

```
P(Burglary | Alarm)                = 0.3736
P(Burglary | Alarm, Earthquake)    = 0.0033

Explaining away: learning about the earthquake decreased burglary probability.
```

---

## 8. 계량 금융에서의 쓰임새: PGM으로 본 인자 모델

단일 요인 모형을 다음과 같이 정의할 수 있다.

$$R_i=\alpha_i+\beta_iF+\epsilon_i$$

오차들이 F를 조건으로 독립이면 R_i⊥R_j|F라는 포크 구조를 얻는다. 잔차 상관이 0이라는 조건만으로 독립이 보장되는 것은 아니며, 공동 정규분포 등의 추가 조건이 있어야 두 개념을 연결할 수 있다.

요인이 설명하지 못하는 산업·지역 효과나 잔차 의존성이 있으면 그래프를 확장해야 한다. 구조화한 공분산 추정이 가능하더라도 시간에 따른 계수 변화와 자료의 의존성을 별도 점검해야 한다. 요인 모형을 그렸다는 사실이 인과 효과나 투자 성과를 보장하지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
A→B→C의 조건부 독립을 이용해 결합분포를 인수분해하라.

</div>

??? success "연습문제 1 풀이"
    일반 연쇄법칙은 $p(A,B,C)=p(A)p(B|A)p(C|A,B)$다. 이 DAG의 마르코프 조건 C⊥A|B를 적용하면 마지막 항이 p(C|B)가 되어 $p(A)p(B|A)p(C|B)$를 얻는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
베이즈 네트워크와 MRF의 표현·정규화 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"
    베이즈 네트워크는 DAG와 부모 조건부분포를 사용하며 각 조건부분포를 정규화한다. MRF는 무방향 클리크 잠재함수의 곱을 전역 분배함수로 정규화한다. 생성 순서가 자연스러운 모형과 공간적 상호작용 모형에 각각 유용할 수 있지만 인과성이나 모든 독립 구조의 표현력이 자동으로 동일한 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
A→C←B에서 C를 조건으로 두면 반드시 종속이 되는가?

</div>

??? success "연습문제 3 풀이"
    주변화하면 $p(A,B)=p(A)p(B)\sum_Cp(C|A,B)=p(A)p(B)$다. C를 조건으로 두면 일반적으로 곱으로 분해되지 않을 수 있지만 CPT가 한 부모를 무시하는 등 특별한 경우 독립이 남을 수 있다. d-연결은 종속 가능성을 허용하며 모든 수치에서 종속임을 보장하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
베이즈 네트워크의 마르코프 블랭킷과 조건부 독립을 설명하라.

</div>

??? success "연습문제 4 풀이"
    블랭킷은 X의 부모·자식·자식의 다른 부모다. 양의 조건 확률에서 $p(X|\text{나머지})\propto p(X|Pa(X))\prod_{Y\in Ch(X)}p(Y|Pa(Y))$이며 X를 포함하지 않는 인자는 약분된다. 남은 인자의 변수는 X와 블랭킷뿐이므로 블랭킷을 알면 그 밖의 변수는 X의 조건부분포를 바꾸지 않는다. 확률 0인 조건은 별도 정의가 필요하다.

## 정리하며

PGM은 인수분해와 조건부 독립을 연결한다. DAG의 방향을 인과 효과로 읽으려면 추가 가정이 필요하며, MRF의 잠재함수는 전역 정규화가 필요하다. d-분리의 독립 보장과 수치에 따른 추가 독립, marginal MAP와 MPE, 결합표 연산과 그래프를 활용한 효율적 추론을 구분한다.
