# 우도 기반 평가: NLL·BPD·퍼플렉서티

생성한 표본을 살펴보는 것과 실제 테스트 데이터에 부여한 확률을 평가하는 것은 서로 다른 질문이다. 우도 기반 평가는 후자를 다룬다. 이 페이지에서는 관측치별 음의 로그 우도(NLL)를 출발점으로 이미지의 차원당 비트 수(BPD), 언어 모델의 퍼플렉서티(PPL)를 연결하고, 코드가 가정하는 입력과 한계를 확인한다.

## 1. 로그 우도와 분포 적합도

모델 pθ와 데이터 분포 p에 대해 기대 NLL은 −Eₚlog pθ(X)다. 같은 표본 공간과 기준 측도 아래 적분이 잘 정의되면 교차 엔트로피와 KL 발산으로 분해된다.

$$-\mathbb E_p\log p_\theta(X)=H(p)+D_{\mathrm{KL}}(p\|p_\theta)$$

p를 고정하면 기대 NLL을 최소화하는 것은 KL을 최소화하는 것과 같다. 유한 테스트 표본에서 측정한 순위가 모집단 순위를 항상 보장하지는 않는다. 학습에 사용하지 않은 동일 데이터와 동일 관측 단위에서 비교해야 한다.

로그의 밑이 e이면 단위는 nat다. 이산 확률의 −log p(x)는 비음수지만 연속 밀도는 1보다 클 수 있어 NLL이 음수일 수도 있다. 참분포와 완전히 일치해도 기대 NLL은 엔트로피이며 일반적으로 0이 아니다. 이산 부호 길이와 연결할 때는 log₂를 사용하고 부호화 조건·오버헤드를 구분한다. 연속 밀도의 NLL을 그대로 정밀도와 무관한 비트 길이로 읽을 수는 없다.

## 2. 관측치별 NLL과 정규 근사 구간

N개의 테스트 관측치에 대해 다음 평균을 계산한다.

$$\widehat{\operatorname{NLL}}=-\frac1N\sum_{i=1}^N\log p_\theta(x_i)$$

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union

class NLLEvaluator:
    """
    만들어 내는 모델의 음의 로그 가능도 따지개.
    
    음의 로그 가능도는 모델이 실제 자료에 확률을 얼마나 매기는지 잰다.
    음의 로그 가능도가 낮을수록 모델이 자료 분포에 잘 맞는다.
    
    수학의 뜻매김:
        NLL = -E_{x~p_data}[log p_model(x)]
           = -(1/N) Σ log p_model(x_i)
    """
    
    @staticmethod
    def compute_nll(log_probs: torch.Tensor) -> float:
        """
        로그 확률에서 음의 로그 가능도를 셈한다.
        
        인수:
            log_probs: 표본마다의 로그 확률 [N]
                       이 값은 model.log_prob(x)에서 나온다
        
        반환값:
            음의 로그 가능도 값(스칼라이며 작을수록 좋다)
        """
        # 음의 로그 가능도는 평균 로그 확률의 음수이다
        nll = -torch.mean(log_probs)
        return nll.item()
    
    @staticmethod
    def compute_nll_with_ci(log_probs: torch.Tensor,
                           confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        믿음 구간과 함께 음의 로그 가능도를 셈한다.
        
        흐릿함을 어림하려 평균의 표준 오차를 쓴다.
        
        인수:
            log_probs: 로그 확률 [N]
            confidence: 믿음 수준(기본값 95%)
        
        반환값:
            (음의 로그 가능도, 아래 끝, 위 끝) 튜플
        """
        from scipy import stats
        
        n = len(log_probs)
        nll = -torch.mean(log_probs).item()
        
        # 표준 오차 = 표준 편차 / sqrt(n)
        std = torch.std(log_probs).item()
        se = std / np.sqrt(n)
        
        # 믿음 구간의 Z 점수
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha / 2)
        
        # 음의 로그 가능도의 믿음 구간
        # 참고: NLL = -mean(log_probs)이므로 음수를 취한다
        lower = nll - z * se
        upper = nll + z * se
        
        return nll, lower, upper
    
    @staticmethod
    def evaluate_model(model,
                      test_data: torch.Tensor,
                      batch_size: int = 64) -> dict:
        """
        음의 로그 가능도로 만들어 내는 모델을 따진다.
        
        인수:
            .log_prob() 방법을 지닌 만들어 내는 모델
            test_data: 시험 자료 [N, ...]
            batch_size: 따질 배치 크기
        
        반환값:
            음의 로그 가능도와 통계를 담은 사전
        """
        model.eval()
        all_log_probs = []
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i+batch_size]
                log_probs = model.log_prob(batch)
                all_log_probs.append(log_probs)
        
        log_probs = torch.cat(all_log_probs)
        
        nll, lower, upper = NLLEvaluator.compute_nll_with_ci(log_probs)
        
        return {
            'nll': nll,
            'nll_lower': lower,
            'nll_upper': upper,
            'mean_log_prob': -nll,
            'n_samples': len(test_data)
        }

# 보기: 정규 분포 모델
class GaussianModel(nn.Module):
    """보여 주기를 위한 단순한 정규 분포 모델."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        정규 분포에서 로그 확률을 셈한다.
        
        log N(x|μ,σ²) = -0.5 * [(x-μ)²/σ² + log(2πσ²)]
        """
        sigma = torch.exp(self.log_sigma)
        
        # 로그 확률 셈하기
        log_prob = -0.5 * (
            ((x - self.mu) / sigma) ** 2 +
            2 * self.log_sigma +
            np.log(2 * np.pi)
        )
        
        # 차원에 걸쳐 더한다, 꼴: [배치 크기]
        return log_prob.sum(dim=-1)

def demonstrate_nll():
    """음의 로그 가능도 셈하기를 보여 준다."""
    print("=" * 70)
    print("Negative Log-Likelihood Demonstration")
    print("=" * 70)
    
    # N(0, 1)에서 시험 자료를 만든다
    test_data = torch.randn(1000, 10)
    
    # 모델 1: 올바른 분포
    model_correct = GaussianModel(dim=10)
    model_correct.mu.data.fill_(0.0)
    model_correct.log_sigma.data.fill_(0.0)  # 시그마 = 1
    
    # 모델 2: 틀린 평균
    model_wrong_mean = GaussianModel(dim=10)
    model_wrong_mean.mu.data.fill_(2.0)
    model_wrong_mean.log_sigma.data.fill_(0.0)
    
    # 모델 3: 틀린 흩어짐
    model_wrong_var = GaussianModel(dim=10)
    model_wrong_var.mu.data.fill_(0.0)
    model_wrong_var.log_sigma.data.fill_(1.0)  # 시그마 = e ≈ 2.72
    
    evaluator = NLLEvaluator()
    
    print("\nTest data: 1000 samples from N(0, I)")
    print("-" * 50)
    
    for name, model in [("Correct N(0,1)", model_correct),
                        ("Wrong mean N(2,1)", model_wrong_mean),
                        ("Wrong var N(0,e²)", model_wrong_var)]:
        results = evaluator.evaluate_model(model, test_data)
        print(f"\n{name}:")
        print(f"  NLL: {results['nll']:.4f} [{results['nll_lower']:.4f}, {results['nll_upper']:.4f}]")
    
    print("\nNote: Lower NLL = Better fit to data")

demonstrate_nll()
```

**보존된 출력:**

```
======================================================================
Negative Log-Likelihood Demonstration
======================================================================

Test data: 1000 samples from N(0, I)
--------------------------------------------------

Correct N(0,1):
  NLL: 14.1577 [14.0163, 14.2991]

Wrong mean N(2,1):
  NLL: 34.1680 [33.7560, 34.5800]

Wrong var N(0,e²):
  NLL: 19.8618 [19.8426, 19.8809]

Note: Lower NLL = Better fit to data
```

compute_nll과 compute_nll_with_ci는 **각 관측치의 전체 log_prob를 담은 길이 N 벡터**를 가정한다. 독립 좌표의 로그확률을 N×D로 반환한다면 좌표 축을 먼저 합해야 한다. 원소 전체를 평균하면 관측치별 NLL과 다른 값이 되고, len만 N으로 쓰는 표준오차 계산도 일관되지 않는다.

신뢰구간 함수는 표본 표준편차/√N에 정규분포 분위수를 곱한다. 독립성, 유한 분산과 충분한 표본 등 근사 조건이 필요하다. N=1의 표준편차는 정의되지 않으며 시계열의 상관이나 같은 원본의 반복 관측을 무시한 구간은 부정확할 수 있다. 이 구간은 주어진 모델의 테스트 평균 변동만 다루고 모델 학습의 불확실성까지 포함하지 않는다.

데모는 10차원 표준정규 테스트 데이터를 만들고 고정 가우스 세 개를 비교한다. 학습된 확산 모델을 평가하지 않는다. 정답 모델의 기대 NLL은 5log(2πe)≈14.19이며 평균을 모든 좌표에서 2로 틀리면 기대 NLL이 20 늘어난다. log_sigma=1은 표준편차 e, 분산 e²를 뜻한다. 출력의 유한 표본 결과와 이론적 기댓값은 구분한다.

## 3. 차원당 비트 수(BPD)

같은 관측치별 NLL을 D개 스칼라 차원과 비트 단위로 정규화한다.

$$\operatorname{BPD}=\frac{\operatorname{NLL}}{D\ln2}$$

```python
class BPDCalculator:
    """
    고르게 맞춘 가능도 견줌을 위한 차원마다 비트 셈개.
    
    BPD = NLL / (D × ln(2))
    
    여기서 D은 자료의 차원이고 ln(2)이 내트를 비트로 바꾼다.
    
    왜 차원마다 비트인가?
    1. 자료 차원에 맞게 고르게 맞춘다
    2. 정보 이론의 풀이(화소마다의 비트 수)
    3. 자료 묶음에 걸쳐 공정한 견줌을 가능하게 한다
    """
    
    @staticmethod
    def nll_to_bpd(nll: float, dimensions: int) -> float:
        """
        음의 로그 가능도를 차원마다 비트로 바꾼다.
        
        인수:
            nll: 음의 로그 가능도(내트 단위)
            dimensions: 온 자료 차원
                       (보기: MNIST이면 28*28=784, CIFAR이면 32*32*3=3072)
        
        반환값:
            차원마다 비트 값
        """
        return nll / (dimensions * np.log(2))
    
    @staticmethod
    def bpd_to_nll(bpd: float, dimensions: int) -> float:
        """
        차원마다 비트를 음의 로그 가능도로 되돌린다.
        
        인수:
            bpd: 차원마다 비트
            dimensions: 자료의 차원
        
        반환값:
            음의 로그 가능도 값
        """
        return bpd * dimensions * np.log(2)
    
    @staticmethod
    def compute_bpd(log_probs: torch.Tensor, dimensions: int) -> float:
        """
        로그 확률에서 곧바로 차원마다 비트를 셈한다.
        
        인수:
            log_probs: 로그 확률 [N]
            dimensions: 자료의 차원
        
        반환값:
            차원마다 비트 값
        """
        nll = -torch.mean(log_probs).item()
        return nll / (dimensions * np.log(2))
    
    @staticmethod
    def evaluate_image_model(model,
                            images: torch.Tensor,
                            batch_size: int = 64) -> dict:
        """
        차원마다 비트로 그림 만들어 내는 모델을 따진다.
        
        인수:
            .log_prob() 방법을 지닌 만들어 내는 모델
            images: 시험 그림 [N, C, H, W]
            batch_size: 배치 크기
        
        반환값:
            차원마다 비트와 관련 잣대를 담은 사전
        """
        # 차원을 얻는다
        _, c, h, w = images.shape
        dimensions = c * h * w
        
        model.eval()
        all_log_probs = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                log_probs = model.log_prob(batch)
                all_log_probs.append(log_probs)
        
        log_probs = torch.cat(all_log_probs)
        
        nll = -torch.mean(log_probs).item()
        bpd = nll / (dimensions * np.log(2))
        
        return {
            'bpd': bpd,
            'nll': nll,
            'dimensions': dimensions,
            'interpretation': BPDCalculator.interpret_bpd(bpd)
        }
    
    @staticmethod
    def interpret_bpd(bpd: float) -> str:
        """
        자연 그림의 차원마다 비트 값을 풀이한다.
        
        인수:
            bpd: 차원마다 비트
        
        반환값:
            풀이 글자열
        """
        if bpd > 8.0:
            return "Worse than uniform (8-bit) - model is wrong"
        elif bpd > 5.0:
            return "Poor - basic compression only"
        elif bpd > 3.5:
            return "Moderate - captures some structure"
        elif bpd > 2.5:
            return "Good - captures significant structure"
        elif bpd > 1.5:
            return "Excellent - approaching optimal compression"
        else:
            return "Outstanding - near-optimal for natural images"

def demonstrate_bpd():
    """차원마다 비트 셈하기와 견주기를 보여 준다."""
    print("=" * 70)
    print("Bits Per Dimension Demonstration")
    print("=" * 70)
    
    # 여러 상황의 로그 확률을 흉내 낸다
    # 보여 주려 차원마다 비트 값이 뜻하는 바를 셈한다
    
    dimensions = {
        'MNIST': 28 * 28,
        'CIFAR-10': 32 * 32 * 3,
        'ImageNet-256': 256 * 256 * 3
    }
    
    print("\nComparison of dimensionalities:")
    print("-" * 50)
    
    for name, dim in dimensions.items():
        print(f"{name}: {dim:,} dimensions")
    
    print("\nWhat different BPD values mean:")
    print("-" * 50)
    
    bpd_values = [8.0, 5.0, 3.5, 3.0, 2.5]
    
    print(f"{'BPD':>6} | {'MNIST NLL':>12} | {'CIFAR-10 NLL':>14} | {'Interpretation'}")
    print("-" * 70)
    
    for bpd in bpd_values:
        nll_mnist = BPDCalculator.bpd_to_nll(bpd, dimensions['MNIST'])
        nll_cifar = BPDCalculator.bpd_to_nll(bpd, dimensions['CIFAR-10'])
        interp = BPDCalculator.interpret_bpd(bpd)
        print(f"{bpd:>6.1f} | {nll_mnist:>12.1f} | {nll_cifar:>14.1f} | {interp}")
    
    print("\nKey insight: BPD enables fair comparison across different image sizes!")

demonstrate_bpd()
```

**보존된 출력:**

```
======================================================================
Bits Per Dimension Demonstration
======================================================================

Comparison of dimensionalities:
--------------------------------------------------
MNIST: 784 dimensions
CIFAR-10: 3,072 dimensions
ImageNet-256: 196,608 dimensions

What different BPD values mean:
--------------------------------------------------
   BPD |    MNIST NLL |   CIFAR-10 NLL | Interpretation
----------------------------------------------------------------------
   8.0 |       4347.4 |        17034.8 | Poor - basic compression only
   5.0 |       2717.1 |        10646.7 | Moderate - captures some structure
   3.5 |       1902.0 |         7452.7 | Good - captures significant structure
   3.0 |       1630.3 |         6388.0 | Good - captures significant structure
   2.5 |       1358.6 |         5323.4 | Excellent - approaching optimal compression

Key insight: BPD enables fair comparison across different image sizes!
```

이미지의 D는 채널·높이·너비의 곱이다. 다만 BPD는 차원 수를 나눈다는 이유만으로 서로 다른 데이터셋·양자화·정규화의 난이도를 제거하지는 않는다. 코드의 interpret_bpd가 반환하는 고정된 “좋음” 등급과 자연 이미지 최적 압축 수준은 검증된 보편 기준이 아니다.

8비트 이산 픽셀의 균등모델은 차원당 8비트다. 이를 연속 밀도와 비교하려면 변환을 반영해야 한다. k∈{0,…,255}에 독립 균등 노이즈 u∈[0,1)을 더해 y=(k+u)/256으로 만든 경우, 연속 밀도 NLL을 Dln2로 나눈 값에 **8을 더한 양**이 통상 이산 코딩 길이 상한의 형태다. 다른 스케일에서는 해당 Jacobian·양자화 보정이 달라진다.

이 BPDCalculator는 그런 전처리 보정을 자동으로 하지 않는다. 이산 log_prob인지 연속 밀도 log_prob인지, 반환값이 정확한 우도인지 변분 한계인지 호출자가 알려야 한다. 이진 Bernoulli 이미지처럼 이미 이산 확률을 계산한 경우에 동일한 +8을 붙이지 않는다.

## 4. 토큰별 퍼플렉서티(PPL)

유효한 다음 토큰 T개의 조건부 확률을 모아 계산한다.

$$\operatorname{PPL}=\exp\left(-\frac1T\sum_{t=1}^T
\log p_\theta(w_t\mid w_{<t})\right)$$

```python
class PerplexityCalculator:
    """
    말 모델의 헷갈림도 셈개.
    
    Perplexity = exp(NLL per token)
               = exp(-1/T Σ log p(w_t | w_{<t}))
    
    직관: 자리마다 실제로 쓰이는 낱말 수.
    헷갈림도가 낮을수록 예측이 더 자신 있다.
    """
    
    @staticmethod
    def compute_perplexity(log_probs: torch.Tensor,
                          lengths: Optional[torch.Tensor] = None) -> float:
        """
        토큰 로그 확률에서 헷갈림도를 셈한다.
        
        인수:
            log_probs: 로그 확률 [batch, seq_len] 또는 [total_tokens]
            lengths: 길이가 다른 배치를 위한 차례 길이(있으면)
        
        반환값:
            헷갈림도 값
        """
        if lengths is not None:
            # 길이가 다른 차례
            total_log_prob = 0.0
            total_tokens = 0
            
            for i, length in enumerate(lengths):
                total_log_prob += log_probs[i, :length].sum().item()
                total_tokens += length.item()
            
            avg_nll = -total_log_prob / total_tokens
        else:
            # 붙박인 길이거나 펼친 것
            avg_nll = -torch.mean(log_probs).item()
        
        perplexity = np.exp(avg_nll)
        return perplexity
    
    @staticmethod
    def evaluate_language_model(model,
                               input_ids: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None,
                               batch_size: int = 16) -> dict:
        """
        헷갈림도로 말 모델을 따진다.
        
        인수:
            model: forward()이 로짓을 돌려주는 말 모델
            input_ids: 토큰 번호 [N, seq_len]
            attention_mask: 눈여겨보기 덮개 [N, seq_len]
            batch_size: 배치 크기
        
        반환값:
            헷갈림도와 통계를 담은 사전
        """
        model.eval()
        total_log_prob = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_ids = input_ids[i:i+batch_size]
                
                if attention_mask is not None:
                    batch_mask = attention_mask[i:i+batch_size]
                else:
                    batch_mask = torch.ones_like(batch_ids)
                
                # 순전파
                logits = model(batch_ids)  # [배치, 차례 길이, 낱말 수]
                
                # 인과 말 모델을 위해 옮긴다: 다음 토큰을 헤아린다
                shift_logits = logits[:, :-1, :]
                shift_labels = batch_ids[:, 1:]
                shift_mask = batch_mask[:, 1:]
                
                # 로그 확률을 셈한다
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                
                # 실제 토큰의 로그 확률을 모은다
                gathered = log_probs.gather(
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                
                # 가림막을 쓰고 쌓는다
                masked_log_probs = gathered * shift_mask
                total_log_prob += masked_log_probs.sum().item()
                total_tokens += shift_mask.sum().item()
        
        avg_nll = -total_log_prob / total_tokens
        perplexity = np.exp(avg_nll)
        
        return {
            'perplexity': perplexity,
            'nll_per_token': avg_nll,
            'total_tokens': total_tokens,
            'interpretation': PerplexityCalculator.interpret_perplexity(
                perplexity, vocab_size=model.config.vocab_size if hasattr(model, 'config') else 50000
            )
        }
    
    @staticmethod
    def interpret_perplexity(ppl: float, vocab_size: int = 50000) -> str:
        """
        헷갈림도 값을 풀이한다.
        
        인수:
            ppl: 헷갈림도 값
            vocab_size: 견줄 낱말 수
        
        반환값:
            풀이 글자열
        """
        if ppl >= vocab_size:
            return "Random baseline - model learns nothing"
        elif ppl > 200:
            return "Poor - basic patterns only"
        elif ppl > 50:
            return "Moderate - captures some language structure"
        elif ppl > 20:
            return "Good - strong language understanding"
        elif ppl > 10:
            return "Excellent - near state-of-the-art"
        else:
            return "Outstanding - highly confident predictions"

def demonstrate_perplexity():
    """헷갈림도 셈하기를 보여 준다."""
    print("=" * 70)
    print("Perplexity Demonstration")
    print("=" * 70)
    
    vocab_size = 10000
    seq_len = 100
    
    print(f"\nLanguage model with vocabulary size: {vocab_size}")
    print("-" * 50)
    
    # 상황 1: 아무 바탕
    print("\nScenario 1: Random Baseline (uniform predictions)")
    random_log_prob = np.log(1.0 / vocab_size)
    random_ppl = np.exp(-random_log_prob)
    print(f"  Log prob per token: {random_log_prob:.4f}")
    print(f"  Perplexity: {random_ppl:.1f}")
    print(f"  Interpretation: Effectively choosing from all {vocab_size} words")
    
    # 상황 2: 보통 모델
    print("\nScenario 2: Moderate Model (~1% probability per token)")
    moderate_log_prob = np.log(0.01)
    moderate_ppl = np.exp(-moderate_log_prob)
    print(f"  Log prob per token: {moderate_log_prob:.4f}")
    print(f"  Perplexity: {moderate_ppl:.1f}")
    print(f"  Interpretation: Effectively choosing from ~{int(moderate_ppl)} words")
    
    # 상황 3: 좋은 모델
    print("\nScenario 3: Good Model (~20% probability per token)")
    good_log_prob = np.log(0.2)
    good_ppl = np.exp(-good_log_prob)
    print(f"  Log prob per token: {good_log_prob:.4f}")
    print(f"  Perplexity: {good_ppl:.1f}")
    print(f"  Interpretation: Effectively choosing from ~{int(good_ppl)} words")
    
    print("\n" + "-" * 50)
    print("Key insight: Lower perplexity = More confident predictions")
    print("Perplexity = 'Effective vocabulary size' at each position")

demonstrate_perplexity()
```

**보존된 출력:**

```
======================================================================
Perplexity Demonstration
======================================================================

Language model with vocabulary size: 10000
--------------------------------------------------

Scenario 1: Random Baseline (uniform predictions)
  Log prob per token: -9.2103
  Perplexity: 10000.0
  Interpretation: Effectively choosing from all 10000 words

Scenario 2: Moderate Model (~1% probability per token)
  Log prob per token: -4.6052
  Perplexity: 100.0
  Interpretation: Effectively choosing from ~99 words

Scenario 3: Good Model (~20% probability per token)
  Log prob per token: -1.6094
  Perplexity: 5.0
  Interpretation: Effectively choosing from ~4 words

--------------------------------------------------
Key insight: Lower perplexity = More confident predictions
Perplexity = 'Effective vocabulary size' at each position
```

PPL은 관측된 정답 토큰의 역확률을 기하평균한 값이다. 모든 정답 확률이 q일 때 1/q이며, 매 위치에서 실제로 고르는 후보 수를 직접 세는 값은 아니다. 유효한 이산 확률에서 최소는 1이고, 어휘 크기 V의 균등모델은 V다. 정답에 극히 낮은 확률을 주면 PPL은 V보다 커질 수 있다.

비교할 때 토크나이저, 평가 텍스트, 문맥 길이와 마스크 조건을 맞춘다. 시퀀스별 PPL의 산술평균 대신 전체 유효 토큰의 로그확률 합을 토큰 수로 나눈다. 코드의 lengths 인수는 유효 토큰이 각 행의 앞에서 연속으로 놓였다고 가정한다. 유효 토큰이 0개이면 나눗셈이 정의되지 않는다.

evaluate_language_model은 model(batch_ids)가 곧 B×L×V logits 텐서를 반환하고, 모델 내부에서 인과 마스킹을 수행한다고 가정한다. 결과 객체의 .logits를 반환하는 일반적인 인터페이스에는 그대로 맞지 않는다. attention_mask는 손실 집계에만 쓰고 모델에 전달하지 않으므로 padding이 문맥 계산에 미치는 영향도 별도로 처리해야 한다. 낮은 PPL을 곧 강한 언어 이해나 특정 수준의 최신 성능이라고 등급화하지 않는다.

## 5. 우도와 표본의 시각적 품질

```python
def demonstrate_likelihood_sample_tradeoff():
    """
    가능도가 높다고 좋은 표본은 아님을 보인다.
    """
    print("=" * 70)
    print("Likelihood vs. Sample Quality Tradeoff")
    print("=" * 70)
    
    # 참 분포: 봉우리 둘
    # 봉우리 1: N(-3, 1), 무게 0.5
    # 봉우리 2: N(+3, 1), 무게 0.5
    
    print("\nTrue distribution: Mixture of two Gaussians")
    print("  Mode 1: N(-3, 1) with 50% weight")
    print("  Mode 2: N(+3, 1) with 50% weight")
    
    # 참 분포에서 시험 자료를 만든다
    n_samples = 1000
    test_data = np.concatenate([
        np.random.randn(n_samples // 2) - 3,
        np.random.randn(n_samples // 2) + 3
    ])
    
    # 모델 A: 정규 분포 하나(봉우리 무너짐)
    # 봉우리 하나만 담지만 정밀도는 높다
    print("\n" + "-" * 50)
    print("Model A: Single Gaussian N(-3, 1)")
    print("  - High quality samples (realistic)")
    print("  - LOW diversity (missing one mode)")
    
    mu_a, sigma_a = -3.0, 1.0
    log_probs_a = -0.5 * ((test_data - mu_a) / sigma_a)**2 - np.log(sigma_a) - 0.5 * np.log(2*np.pi)
    nll_a = -np.mean(log_probs_a)
    
    print(f"  NLL: {nll_a:.4f}")
    
    # 모델 B: 넓은 정규 분포(두 봉우리를 덮지만 흐릿하다)
    print("\n" + "-" * 50)
    print("Model B: Wide Gaussian N(0, 5)")
    print("  - LOW quality samples (blurry)")
    print("  - High coverage (includes both modes)")
    
    mu_b, sigma_b = 0.0, 5.0
    log_probs_b = -0.5 * ((test_data - mu_b) / sigma_b)**2 - np.log(sigma_b) - 0.5 * np.log(2*np.pi)
    nll_b = -np.mean(log_probs_b)
    
    print(f"  NLL: {nll_b:.4f}")
    
    # 모델 C: 참 섞기(가장 좋다)
    print("\n" + "-" * 50)
    print("Model C: True Mixture (ideal)")
    print("  - High quality samples")
    print("  - Full coverage")
    
    # 섞기에서의 로그 확률
    log_prob_mode1 = -0.5 * ((test_data + 3) ** 2) - 0.5 * np.log(2*np.pi)
    log_prob_mode2 = -0.5 * ((test_data - 3) ** 2) - 0.5 * np.log(2*np.pi)
    log_probs_c = np.logaddexp(log_prob_mode1 + np.log(0.5), log_prob_mode2 + np.log(0.5))
    nll_c = -np.mean(log_probs_c)
    
    print(f"  NLL: {nll_c:.4f}")
    
    # 요약
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"{'Model':<20} {'NLL':>10} {'Sample Quality':>20} {'Coverage':>15}")
    print("-" * 70)
    print(f"{'A (Mode Collapse)':<20} {nll_a:>10.4f} {'High':>20} {'Low':>15}")
    print(f"{'B (Wide/Blurry)':<20} {nll_b:>10.4f} {'Low':>20} {'High':>15}")
    print(f"{'C (True Mixture)':<20} {nll_c:>10.4f} {'High':>20} {'High':>15}")
    
    print("\n⚠️ Key Insight: Model B has BETTER likelihood than Model A,")
    print("   but Model A produces BETTER samples for mode -3!")
    print("\n→ Always combine likelihood metrics with sample-based metrics (FID, IS)")

demonstrate_likelihood_sample_tradeoff()
```

**보존된 출력:**

```
======================================================================
Likelihood vs. Sample Quality Tradeoff
======================================================================

True distribution: Mixture of two Gaussians
  Mode 1: N(-3, 1) with 50% weight
  Mode 2: N(+3, 1) with 50% weight

--------------------------------------------------
Model A: Single Gaussian N(-3, 1)
  - High quality samples (realistic)
  - LOW diversity (missing one mode)
  NLL: 10.1232

--------------------------------------------------
Model B: Wide Gaussian N(0, 5)
  - LOW quality samples (blurry)
  - High coverage (includes both modes)
  NLL: 2.7191

--------------------------------------------------
Model C: True Mixture (ideal)
  - High quality samples
  - Full coverage
  NLL: 2.1012

======================================================================
Summary:
======================================================================
Model                       NLL       Sample Quality        Coverage
----------------------------------------------------------------------
A (Mode Collapse)       10.1232                 High             Low
B (Wide/Blurry)          2.7191                  Low            High
C (True Mixture)         2.1012                 High            High

⚠️ Key Insight: Model B has BETTER likelihood than Model A,
   but Model A produces BETTER samples for mode -3!

→ Always combine likelihood metrics with sample-based metrics (FID, IS)
```

이 데모의 실제 분포는 N(−3,1)과 N(3,1)을 반씩 섞은 1차원 혼합이다. 모델 A는 왼쪽 모드만, 모델 B는 넓은 가우스 하나로 두 영역을 덮고, 모델 C는 참 혼합분포를 사용한다. B의 sigma_b=5는 **표준편차 5·분산 25**이며 출력의 N(0,5) 표기에서 두 번째 인수를 분산으로 읽으면 안 된다.

참 혼합 아래 기대 NLL은 A에서 0.5log(2π)+9.5≈10.42, B에서 log5+0.5log(2π)+0.2≈2.73이다. B는 A보다 두 모드를 덜 놓치지만 실제 데이터가 드문 가운데 영역에도 질량을 둔다. 정답 C는 기대 로그 우도 기준으로 최적이다. 이 비교는 제한된 잘못된 모델들 사이에서 우도와 표본 특성의 순위가 달라질 수 있음을 보여 준다.

다만 이 코드는 이미지를 만들지 않으며 출력의 “흐릿함”이나 “높은 품질”을 측정하지 않는다. 우도를 높이면 반드시 품질을 희생한다는 필연적인 법칙도 아니다. 통계적 적합도, 지각 품질과 조건 충실도가 어떤 질문에 답하는지 나누어 평가한다.

## 6. 확산 모델에서 보고할 우도의 종류

DDPM의 변분 목적에는 중간 KL 항 외에 마지막 단계의 재구성 항과 말단 prior 항 등이 포함된다. ELBO≤log pθ(x)이므로 −ELBO는 NLL의 상한이다. 단순 노이즈 MSE의 숫자만 변환해 정확한 NLL이나 BPD로 보고할 수는 없다.

확률 흐름 ODE는 연속 변수변환으로 자신이 정의하는 밀도를 평가할 수 있다. 수치 적분·발산 추정 오차를 고려해야 하며, 근사 스코어에서 그 ODE의 밀도가 이산 DDPM 생성기의 우도와 자동으로 같지는 않다. 어느 모델의 우도인지와 계산법을 명시한다.

자기회귀 모델과 정규화 흐름은 정의된 모델의 log_prob를 계산할 수 있는 구조다. VAE는 보통 한계·추정치를, 일반적인 GAN은 생성 샘플을 제공하지만 tractable한 밀도를 제공하지 않는다. 그렇다고 GAN의 확률분포 자체가 없는 것은 아니다. 에너지 모델도 정규화 상수 계산·추정 가능성에 따라 우도 평가가 달라진다.

이 페이지의 NLLEvaluator는 model.log_prob 인터페이스를 호출할 뿐 확산 ELBO나 ODE 우도를 구현하지 않는다. 확산 예제에 바로 적용할 수 있는 완성된 우도 평가기로 설명하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
참분포와 같은 모델이면 NLL은 항상 0인가? 연속 NLL이 음수일 수 있는가?

</div>

??? success "연습문제 1 풀이"

    참분포와 같을 때 기대 NLL은 그 분포의 엔트로피이며 일반적으로 0이 아니다. 연속 밀도는 1을 넘을 수 있으므로 개별 NLL과 기대 NLL이 음수일 수 있다. 이산 확률의 비음수 NLL과 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
모델이 N×D의 독립 좌표 로그확률을 반환할 때 NLLEvaluator에 전달하기 전 무엇을 해야 하는가?

</div>

??? success "연습문제 2 풀이"

    좌표 축 D를 합해 관측치별 전체 로그확률 벡터 N개를 만든다. 이 벡터의 평균과 표준편차로 관측치별 NLL·표준오차를 계산한다. 유효한 근사 구간에는 독립성·표본 수 등의 조건도 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
8비트 픽셀을 (k+u)/256으로 dequantization했을 때 연속 BPD와 이산 코딩 길이의 관계를 설명하라.

</div>

??? success "연습문제 3 풀이"

    연속 밀도 NLL/(Dln2)에 8을 더한 형태가 통상 이산 코딩 길이 상한이다. 연속 단위 구간을 픽셀별 폭 1/256인 셀로 연결하는 보정이다. 다른 변환은 다른 보정이 필요하고, 이미 이산 확률을 계산한 모델에는 이 +8을 그대로 적용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
단순 노이즈 MSE와 −ELBO, ODE 우도를 같은 값으로 보고할 수 없는 이유는 무엇인가?

</div>

??? success "연습문제 4 풀이"

    단순 MSE는 변분 목적의 가중치와 경계 항을 모두 포함하지 않는다. −ELBO는 NLL 상한이고 정확한 NLL과 다르다. ODE는 자신이 정의하는 밀도를 평가하며 수치 오차와 근사 스코어에서의 모델 차이도 있다. 어떤 양을 계산했는지 명시해야 한다.



## 정리하며

NLL은 관측 단위와 밀도·확률의 기준을 정해 해석한다. BPD는 양자화·스케일 보정과 데이터 조건을, PPL은 유효 토큰과 문맥 조건을 맞추어 비교한다. 확산의 단순 MSE, 변분 한계와 ODE 우도는 서로 다른 양이다.

**참고 문헌**

1. Theis, L., van den Oord, A., & Bethge, M. (2016). "A Note on the Evaluation of Generative Models." *ICLR*.

2. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning." Springer.

3. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.

4. Kingma, D. P., & Dhariwal, P. (2018). "Glow: Generative Flow with Invertible 1×1 Convolutions." *NeurIPS*.
