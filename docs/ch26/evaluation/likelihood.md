# 가능도에 바탕한 따지기
## 개요

가능도에 바탕한 따지기는 만들어 내는 모델이 실제 자료에 얼마나 잘 확률을 매기는지 잰다. 표본에 바탕한 잣대(FID, 인셉션 점수)와 달리 가능도 잣대는 원칙 있는 앎 이론의 따지기를 준다. 이 마디는 음의 로그 가능도, 차원마다 비트, 헷갈림도를 다룬다.

!!! info "배움 목표"
    이 절을 마치면 다음을 할 수 있게 된다.
    
    - 가능도 잣대의 수학 바탕을 이해한다
    - 음의 로그 가능도, 차원마다 비트, 헷갈림도 셈하기를 PyTorch으로 짠다
    - 가능도 잣대를 풀이하고 그 한계를 이해한다
    - 만들어 내는 모델마다 알맞은 잣대를 고른다
    - 가능도와 표본 품질의 맞바꿈을 안다

## 수학적 바탕

### 맞음의 잣대로서의 가능도

만들어 내는 모델 $p_\theta(x)$과 자료 분포 $p_{\text{data}}(x)$에 대해 **로그 가능도**는 모델이 자료에 확률 무게를 얼마나 매기는지 잰다.

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

가능도가 클수록 자료 분포에 더 잘 맞는다.

### 어긋 엔트로피와의 이음

음의 로그 가능도는 자료와 모델 사이의 어긋 엔트로피와 같다.

$$
\text{NLL} = -\mathcal{L}(\theta) = H(p_{\text{data}}, p_\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

이는 다음으로 나뉜다.

$$
H(p_{\text{data}}, p_\theta) = H(p_{\text{data}}) + D_{\text{KL}}(p_{\text{data}} \| p_\theta)
$$

$H(p_{\text{data}})$은 상수이므로 음의 로그 가능도를 가장 작게 하는 것은 쿨백-라이블러 벌어짐을 가장 작게 하는 것과 같다.

### 가장 좋은 누르기와의 이음

앎 이론에 따라 $p_\theta$에 맞춘 부호로 $p_{\text{data}}$의 자료를 담을 때 기댓값 부호 길이는 다음과 같다.

$$
\mathbb{E}[\text{code length}] = H(p_{\text{data}}, p_\theta)
$$

**풀이**: 음의 로그 가능도는 모델을 누르기 얼개로 삼아 실제 자료를 담는 데 평균 몇 비트가 필요한지 잰다.

## 음의 로그 가능도(NLL)

### 정의

자료 묶음 $\mathcal{D} = \{x_1, ..., x_N\}$이 주어질 때:

$$
\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i)
$$

**성질:**

- 음의 로그 가능도가 낮을수록 모델이 잘 맞는다
- 올바른 확률 분포에서는 음의 로그 가능도 ≥ 0이다
- 이론으로는 모델이 자료와 완벽히 맞을 때만 음의 로그 가능도 = 0이다(실제로는 불가능하다)

### PyTorch 구현

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
            batch_size: 따질 묶음 크기
        
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
        
        # 차원에 걸쳐 더한다, 꼴: [묶음 크기]
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

## 차원마다 비트(BPD)

### 왜 고르게 맞추는가?

날 음의 로그 가능도 값은 자료 차원에 매인다.

- MNIST(28×28×1 = 784차원): 음의 로그 가능도 ≈ 1000
- CIFAR-10(32×32×3 = 3072차원): 음의 로그 가능도 ≈ 4000
- ImageNet(256×256×3 = 196608차원): 음의 로그 가능도 ≈ 300000

**차원마다 비트는 자료 크기가 달라도 공정히 견줄 수 있게 고르게 맞춘다.**

### 정의

$$
\text{BPD} = \frac{\text{NLL}}{D \cdot \ln(2)}
$$

여기서 각 기호는 다음과 같다.

- $D$은 자료의 온 차원이다
- $\ln(2) \approx 0.693$이 내트를 비트로 바꾼다

**풀이**: 차원 하나를 담는 데 필요한 평균 비트 수.

### 그림에서의 흔한 값

| 모델 갈래 | 자료 묶음 | 차원마다 비트 |
|------------|---------|-----|
| 고른 분포(8비트) | 아무거나 | 8.0 |
| PixelCNN++ | CIFAR-10 | 약 2.9 |
| Glow | CIFAR-10 | 약 3.3 |
| DDPM | CIFAR-10 | 약 3.7 |
| 실제 그림 | 자연 | 약 1-2(어림) |

### PyTorch 구현

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
            batch_size: 묶음 크기
        
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

## 헷갈림도

### 말 모델에서의 뜻매김

헷갈림도는 말 모델의 여느 잣대이다.

$$
\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p(w_t | w_{<t})\right) = \exp(\text{NLL per token})
$$

여기서 $T$은 차례의 길이이다.

### 직관적인 해석

헷갈림도는 자리마다 **실제로 쓰이는 낱말 수**를 나타낸다.

- 헷갈림도 = 100: 모델이 똑같이 그럴듯한 낱말 100개에서 고르는 만큼 헷갈려 한다
- 헷갈림도 = 10: 모델이 그럴듯한 낱말 약 10개로 좁혔다
- 헷갈림도 = 1: 모델이 온전히 확신한다(이론으로만 이룰 수 있다)

### 흔한 값

| 모델 | 자료 묶음 | 헷갈림도 |
|-------|---------|------------|
| 마구잡이 | 아무거나 | 낱말 수 |
| N-그램 | PTB | 약 150 |
| 긴 짧은 기억 | PTB | 약 60 |
| 변환기 | PTB | 약 25 |
| GPT-2 | WikiText-103 | 약 18 |
| GPT-3 | 여러 가지 | 약 15 |

### PyTorch 구현

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
            lengths: 길이가 다른 묶음을 위한 차례 길이(있으면)
        
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
            batch_size: 묶음 크기
        
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
                logits = model(batch_ids)  # [묶음, 차례 길이, 낱말 수]
                
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

## 가능도와 표본 품질의 맞바꿈

### 결정적인 한계

!!! warning "중요"
    **가능도가 높다고 좋은 표본이 보장되지는 않는다!**

이는 만들어 내는 모델 따지기에서 가장 중요한 통찰 가운데 하나이다.

### 맞바꿈이 있는 까닭

**경우 1: 높은 가능도, 나쁜 표본**

모델은 다음으로 높은 가능도를 이룰 수 있다.

- 모든 봉우리를 덮는다(그럴듯하지 않은 것까지)
- 흩어짐과 흐릿함이 크다
- 흐릿한 헤아림으로 "안전하게 간다"

**경우 2: 낮은 가능도, 좋은 표본**

모델은 다음이면서도 좋은 표본을 만들 수 있다.

- 봉우리 몇을 놓친다(봉우리 무너짐)
- 지나치게 자신 있다
- 드물지만 올바른 자료 점을 무시한다

### 보여 주기

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

## 언제 가능도 잣대를 쓸까

### 가능도를 다룰 수 있는 모델

| 모델 갈래 | 가능도를 셈할 수 있는가? | 권하는 잣대 |
|------------|------------------------|-------------------|
| 변분 자기 부호기 | 증거 하한(아래 한계) | 증거 하한, 되짓기 음의 로그 가능도 |
| 고르게 하는 흐름 | 정확함 | 음의 로그 가능도, 차원마다 비트 |
| 자기 되돌이 | 정확함 | 음의 로그 가능도, 차원마다 비트, 헷갈림도 |
| 퍼짐 | 어림(증거 하한으로) | 차원마다 비트, FID |
| 맞겨루기 만들개 | **아니다** | FID과 인셉션 점수만 |
| 에너지 바탕 | 다룰 수 없다 | FID과 그 밖의 잣대 |

### 실제의 길잡이

1. **그림에는 차원마다 비트를 쓰라**: 해상도에 걸쳐 견줄 수 있다
2. **글에는 헷갈림도를 쓰라**: 말 모델의 여느 잣대이다
3. **믿음 구간을 알려라**: 흐릿함이 중요하다
4. **표본 잣대와 아울러라**: FID, 인셉션 점수, 정밀도와 재현율

## 요약

!!! success "핵심 간추리기"
    
    1. **음의 로그 가능도는 맞음을 잰다**: 낮을수록 모델이 자료에 더 큰 확률을 매긴다
    
    2. **차원마다 비트는 차원에 맞게 고르게 맞춘다**: 자료 묶음에 걸쳐 공정히 견줄 수 있다
    
    3. **말에는 헷갈림도**: "실제로 쓰이는 낱말 수"를 나타낸다
    
    4. **결정적인 한계**: 높은 가능도 ≠ 좋은 표본
    
    5. **가장 좋은 방식**: 늘 표본 바탕 잣대(FID, 인셉션 점수)와 아울러라

## 참고 문헌

1. Theis, L., van den Oord, A., & Bethge, M. (2016). "A Note on the Evaluation of Generative Models." *ICLR*.

2. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning." Springer.

3. Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.

4. Kingma, D. P., & Dhariwal, P. (2018). "Glow: Generative Flow with Invertible 1×1 Convolutions." *NeurIPS*.

## 연습문제

**연습문제 1.**
로그 가능도가 퍼짐 모형을 따지는 데 쓸모 있는 자인 까닭을 밝혀라. 그 한계는 무엇인가?

??? success "연습문제 1 풀이"
    로그 가능도는 남겨 둔 시험 자료에 모형이 확률을 얼마나 잘 매기는지 잰다. $\mathcal{L} = \frac{1}{N}\sum_i \log p_\theta(x_i)$이다. 쓸모 있는 까닭은 이렇다. (1) 올바른 점수 규칙이다(참 분포에서 가장 커진다). (2) 모형을 견줄 수 있는 수 하나를 준다. (3) 표본이 나쁜 것과 최빈값 무너짐을 모두 벌한다. **한계**는 이렇다. (1) 가능도가 높아도 표본이 나쁠 수 있다(참되지 않은 자리에 무게를 두는 섞음 따위). (2) 퍼짐 모형에서는 딱 맞게 셈하기가 흔히 어렵다(ELBO이나 값비싼 상미분 방정식 따짐이 든다). (3) 느낌의 좋음을 곧바로 재지는 않는다.

---

**연습문제 2.**
만들개 모형을 따지는 자로서 FID, 인셉션 점수, 로그 가능도를 견주어라.

??? success "연습문제 2 풀이"
    | 자 | 재는 것 | 참 자료가 드는가 | 최빈값 무너짐을 알아내는가 | 느낌의 좋음 |
    |--------|---------|-------------------|----------------------|-------------------|
    | **FID** | 분포의 닮음 | 그렇다 | 그렇다 | 좋음 |
    | **인셉션 점수** | 품질 + 다양함 | 아니다 | 일부 | 보통 |
    | **로그 가능도** | 밀도의 정확함 | 그렇다(시험 묶음) | 그렇다 | 여림 |

    FID은 사람의 판단과 잘 이어지고 품질과 다양함을 모두 담아 가장 널리 쓰인다. 인셉션 점수는 만든 표본만 따진다. 로그 가능도는 이론으로 원칙이 있지만 느낌의 품질과 어긋날 수 있다. 가장 좋은 방식은 셋 다 알리는 것이다.

---

**연습문제 3.**
차원마다 비트(BPD) 잣대란 무엇인가? 퍼짐 모델에서 어떻게 셈하는가?

??? success "연습문제 3 풀이"
    차원마다 비트는 음의 로그 가능도를 자료 차원으로 고르게 맞추고 비트로 바꾼다. 곧 $\text{BPD} = -\frac{\log_2 p(x)}{d}$이며 $d$은 차원의 수이다(예컨대 CIFAR-10은 $3 \times 32 \times 32 = 3072$). 퍼짐 모델에서 로그 가능도는 증거 하한으로 가둬진다. 곧 $\log p(x) \geq \text{ELBO} = -\sum_t L_t$이며 $L_t$은 쿨백-라이블러 벌어짐 항이다. 정확한 셈은 확률 흐름 상미분 방정식과 순간 변수 바꿈 공식을 쓴다. 차원마다 비트가 낮을수록 좋은 모델이다. 최고 수준의 퍼짐 모델은 CIFAR-10에서 약 2.5을 이룬다.

---

**연습문제 4.**
FID이 뛰어난 만들어 내는 모델이 왜 실제 쓰임새에서 실패할 수 있는가? 어떤 따지기를 더 해야 하는가?

??? success "연습문제 4 풀이"
    FID은 평균 분포 품질을 재지만 다음을 놓친다. (1) **꼬리 움직임**: 드물지만 중요한 잘못됨(흠, 불쾌한 내용)이 평균에 묻힌다. (2) **조건 충실함**: FID을 흔히 조건 없이 셈하는데 갈래나 글 조건 FID은 다를 수 있다. (3) **외우기**: 익히기 자료를 외운 모델은 FID이 낮지만 만들어 내기에는 쓸모없다. (4) **조건 안의 다양함**: 비슷한 채근에 같은 그림을 내도 FID이 낮을 수 있다. 더 따질 것: 정밀도와 재현율 곡선, 갈래마다 FID, 외우기 알아내기(익히기 묶음까지의 가장 가까운 이웃 거리), 품질과 다양함에 대한 사람 따지기, 쓰임새에 맞춘 잣대(예컨대 글에서 그림으로 모델의 글과 그림 맞음).
