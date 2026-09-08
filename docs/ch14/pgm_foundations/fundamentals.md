# PGM의 바탕

---

## 1. 확률에서의 차원의 저주

높은 차원의 확률 분포를 다룰 때 우리는 근본적인 어려움을 만난다. 바로 **매개변수의 지수 자람**이다. 저마다 값을 $k$가지 갖는 이산 확률 변수 $n$개의 결합 분포를 생각해 보자. 온전히 못 박으려면 다음이 필요하다:

$$\text{Parameters} = k^n - 1$$

이진 변수가 20개면 매개변수가 백만 개를 넘는다. 50개면 $10^{15}$을 넘어 저장하기도 어림하기도 추론하기도 도무지 감당할 수 없다.

---

## 2. 핵심 통찰: 조건부 독립

풀이의 열쇠는 실제 세상의 변수가 다른 모든 변수에 기대는 일이 드물다는 것을 알아채는 데 있다. **조건부 독립** 관계 덕분에 복잡한 분포를 더 단순한 항의 곱으로 쪼갤 수 있다.

### 정의: 조건부 독립

확률 변수 $X$과 $Y$이 $Z$을 조건으로 **조건부 독립**이라는 것은($X \perp\!\!\!\perp Y \mid Z$으로 쓴다) 다음을 뜻한다:

$$P(X, Y \mid Z) = P(X \mid Z) \cdot P(Y \mid Z)$$

같은 말로 다음과 같다.

$$P(X \mid Y, Z) = P(X \mid Z)$$

곧 $Z$을 알고 나면 $Y$을 더 알아도 $X$에 대해 새로 알게 되는 것이 없다는 뜻이다.

### 보기: 의료 진단

변수 셋을 생각하자. $D$ = 질병(있음/없음), $S_1$ = 증상 1, $S_2$ = 증상 2이다. 두 증상이 모두 질병에서 오되 서로 곧바로 영향을 주지 않는다면 다음과 같다:

$$S_1 \perp\!\!\!\perp S_2 \mid D$$

질병 상태를 알면 증상들은 서로 독립이 된다. 이것이 **함께 낳은 원인**(갈래) 짜임이다.

---

## 3. PGM의 엄밀한 정의

**확률 그래프 모형(PGM)**은 다음을 만족하는 짝 $(G, P)$이다:

1. **$G$**은 조건부 독립 관계를 담은 그래프 짜임이다
2. **$P$**은 $G$에 따라 쪼개지는 확률 분포이다

그래프는 다음을 준다:

- **간결한 표현**: 국소 확률 분포만 저장하면 된다
- **독립 짜임**: 조건부 독립을 그래프에서 곧바로 읽을 수 있다
- **효율적인 추론**: 그래프 짜임이 감당할 만한 셈을 가능하게 한다
- **알아보기 쉬움**: 변수 사이의 관계가 곧바로 그려진다

---

## 4. 두 큰 갈래

### 방향 그래프 모형(베이즈 망)

```
    Cloudy
    /    \
   v      v
 Rain    Sprinkler
    \    /
     v  v
   WetGrass
```

베이즈 망은 변이 인과 관계나 낳음 관계를 나타내는 방향 비순환 그래프(DAG)를 쓴다. 마디마다 조건부 확률표(CPT) $P(X_i \mid \text{Parents}(X_i))$을 갖는다.

**쪼개기:**

$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

### 방향 없는 그래프 모형(마르코프 무작위 마당)

```
    A --- B
    |     |
    |     |
    C --- D
```

마르코프 무작위 마당은 대칭 관계를 나타내는 무방향 변을 쓴다. 인자(퍼텐셜 함수)는 파벌 위에 정해지며 "어버이"나 "자식"이라는 개념이 없다.

**쪼개기:**

$$P(X_1, \ldots, X_n) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(X_C)$$

---

## 5. PGM의 근본 연산

### 주변화

결합 분포 $P(X, Y)$이 주어졌을 때 주변 분포 $P(X)$을 셈한다:

$$P(X = x) = \sum_y P(X = x, Y = y)$$

### 조건 걸기

결합 분포 $P(X, Y)$과 관측 $Y = y$이 주어졌을 때 뒤확률 $P(X \mid Y = y)$을 셈한다:

$$P(X \mid Y = y) = \frac{P(X, Y = y)}{P(Y = y)} = \frac{P(X, Y = y)}{\sum_x P(X = x, Y = y)}$$

### 추론 물음

PGM에 흔히 던지는 물음:

| 물음 갈래 | 설명 | 보기 |
|------------|-------------|---------|
| **주변** | $P(X)$ | 질병의 확률은 얼마인가? |
| **조건부** | $P(X \mid E)$ | 증상이 주어졌을 때 질병의 확률은? |
| **MAP** | $\arg\max_X P(X \mid E)$ | 증상이 주어졌을 때 가장 그럴듯한 진단은? |
| **MPE** | $\arg\max_X P(X, E)$ | 가장 그럴듯한 온전한 설명은? |

---

## 6. PyTorch 구현: 이산 분포

다음 클래스는 주변화, 조건 걸기, 독립 검정을 갖춘 이산 결합 분포를 구현한다:

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

---

## 7. 근본이 되는 세 짜임

기본 그래프 짜임 셋을 이해하는 것이 방향 그래프에서 조건부 독립을 읽어 내는 열쇠이다. 이 짜임들은 PGM 이론과 실전 곳곳에서 되풀이해 나타난다.

### 1. 사슬: X -> Z -> Y

```
X --> Z --> Y
```

$X$과 $Y$은 주변으로는 **기대고** 있으나 $Z$을 조건으로 두면 **독립**이다. 정보가 $Z$을 거쳐 $X$에서 $Y$으로 흐르는데, $Z$을 보면 이 흐름이 막힌다.

$$X \perp\!\!\!\perp Y \mid Z$$

### 2. 갈래(함께 낳은 원인): X <- Z -> Y

```
X <-- Z --> Y
```

$X$과 $Y$은 주변으로는 ($Z$ 때문에 뒤섞여) **기대고** 있으나 $Z$을 조건으로 두면 **독립**이다. 함께 낳은 원인 $Z$을 알고 나면 한쪽 결과를 알아도 다른 쪽에 대해 알려 주는 바가 없다.

$$X \perp\!\!\!\perp Y \mid Z$$

### 3. 충돌자(v-짜임): X -> Z <- Y

```
X --> Z <-- Y
```

$X$과 $Y$은 주변으로는 **독립**이나 $Z$을 조건으로 두면 **기대고** 있다. 함께 낳은 결과 $Z$을 보는 일이 서로 독립인 원인들 사이에 기댐을 만든다. 이것이 **설명해 치우기** 효과이다.

$$X \perp\!\!\!\perp Y \quad \text{(marginal)}, \qquad X \not\perp\!\!\!\perp Y \mid Z \quad \text{(explaining away)}$$

### 설명해 치우기 효과

$X$ = 도둑, $Y$ = 지진, $Z$ = 경보라고 하자. 경보가 울리고($Z = 1$) 지진이 있었음을 알게 되면($Y = 1$), 도둑이 들었다는 믿음($X = 1$)이 **줄어든다**. 지진이 경보를 "설명해 치워" 도둑이라는 설명이 덜 필요해진다.

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

---

## 8. 계량 금융에서의 쓰임새: PGM으로 본 인자 모형

계량 금융의 고전적인 인자 모형은 PGM으로 자연스럽게 나타난다. 시장 인자 $F$이 자산 수익률 $R_1, \ldots, R_n$을 이끄는 한 인자 모형을 생각해 보자:

$$R_i = \alpha_i + \beta_i F + \epsilon_i, \qquad \epsilon_i \perp\!\!\!\perp \epsilon_j \mid F$$

이것이 바로 **갈래** 짜임 $R_i \leftarrow F \rightarrow R_j$이다. 자산 수익률은 (함께 있는 시장 인자를 거쳐) 주변으로는 서로 얽혀 있으나 그 인자를 조건으로 두면 독립이다. 여러 인자 모형은 이를 함께 낳은 원인 여럿으로 넓히며, 그래프가 어느 인자가 어느 자산을 이끄는지를 담는다. 이 짜임을 알아보면 공분산을 효율적으로 어림하고 위험을 쪼갤 수 있다.

---

## 연습문제

**연습문제 1.**
변수 셋의 베이즈 망 $A \to B \to C$이 주어졌을 때, 확률의 사슬 규칙과 그래프가 뜻하는 조건부 독립을 써서 결합 분포 $p(A, B, C)$을 적어라.

??? success "연습문제 1 풀이"
    그래프는 $A \perp\!\!\perp C \mid B$을 담는다. 베이즈 망 쪼개기를 쓰면 다음과 같다:

    $$p(A, B, C) = p(A) \, p(B \mid A) \, p(C \mid B)$$

    조건부 독립 덕분에 $p(C \mid B, A) = p(C \mid B)$이며, 이것이 일반 사슬 규칙 $p(A)p(B|A)p(C|A,B)$을 간단하게 만든다.

---

**연습문제 2.**
베이즈 망과 마르코프 무작위 마당의 차이를 설명하여라. 어느 때 어느 쪽을 고르겠는가?

??? success "연습문제 2 풀이"
    **베이즈 망**(방향 그래프 모형)은 방향 변으로 조건부 기댐을 나타내고 결합 분포를 조건부 분포 $p(x_i \mid \text{parents}(x_i))$의 곱으로 쪼갠다. **마르코프 무작위 마당**(방향 없는 그래프 모형)은 무방향 변을 쓰고 결합 분포를 파벌 위 퍼텐셜 함수의 곱으로 쪼갠 뒤 나눔 함수 $Z$으로 고르게 한다. 인과나 낳음 짜임이 자연스러울 때(이를테면 의료 진단) 베이즈 망이 낫다. 관계가 대칭일 때(이를테면 그림 나누기, 공간 통계) 마르코프 무작위 마당이 낫다.

---

**연습문제 3.**
"설명해 치우기" 짜임 $A \to C \leftarrow B$을 생각하여라. $A$과 $B$이 주변으로는 독립이지만 $C$을 조건으로 두면 기대고 있음을 보여라.

??? success "연습문제 3 풀이"
    쪼개기 $p(A, B, C) = p(A)p(B)p(C \mid A, B)$에서 $C$에 걸쳐 주변화하면 다음과 같다:

    $$p(A, B) = \sum_C p(A)p(B)p(C \mid A, B) = p(A)p(B)$$

    그러므로 $A \perp\!\!\perp B$이다. 그러나 $p(A, B \mid C) = p(C \mid A, B) p(A) p(B) / p(C)$은 일반으로 $p(A \mid C) p(B \mid C)$으로 쪼개지지 않는다. $C$을 보는 일이 $A$과 $B$을 묶는다. 곧 함께 낳은 결과 $C$이 일어났음을 알면 원인 $A$과 $B$이 그것을 설명하려고 겨루게 된다.

---

**연습문제 4.**
베이즈 망에서 마디 $X$의 **마르코프 이불**은 그 어버이, 자식, 그리고 자식의 다른 어버이로 이루어진다. 마르코프 이불을 조건으로 두면 $X$이 다른 모든 마디와 조건부 독립임을 증명하여라.

??? success "연습문제 4 풀이"
    베이즈 망 쪼개기에 따라 $p(X \mid \text{all others}) \propto p(X \mid \text{parents}(X)) \prod_{Y \in \text{children}(X)} p(Y \mid \text{parents}(Y))$이다. 첫 인자에는 $X$과 그 어버이만 들어간다. 곱 안의 인자마다 $X$, 자식 $Y$, 그리고 $Y$의 다른 어버이($X$의 짝 어버이)가 들어간다. 다른 변수는 나타나지 않으므로 마르코프 이불을 조건으로 두면 $X$은 다른 모든 변수와 조건부 독립이다. $\square$

## 정리하며

| 짜임 | 주변 | 가운데에 조건 걸기 | 규칙 |
|-----------|----------|----------------------|------|
| 사슬: $A \to B \to C$ | 기댐 | 독립 | 막힘 |
| 갈래: $A \leftarrow B \to C$ | 기댐 | 독립 | 막힘 |
| 충돌자: $A \to B \leftarrow C$ | 독립 | 기댐 | 설명해 치우기 |

| 개념 | 설명 |
|---------|-------------|
| **PGM** | 그래프 + 분포 쪼개기 |
| **조건부 독립** | $X \perp\!\!\!\perp Y \mid Z$: $Z$을 알면 $X$과 $Y$이 독립이 된다 |
| **방향(베이즈 망)** | CPT을 갖는 DAG, 인과로 풀이 |
| **방향 없음(MRF)** | 대칭인 기댐, 퍼텐셜 함수 |
| **핵심 연산** | 주변화, 조건 걸기, 추론 |
