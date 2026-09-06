# 나아간 에너지 바탕 모델 주제

에너지 바탕 모델은 퍼짐 모델, 흐름 바탕 방법, 숨은 변수 모델을 아우르는 요즘 만들어 내기 방식과 깊이 이어진다. 이 이음을 이해하면 통계 역학의 고전 생각이 오늘날 가장 힘센 만들어 내기 틀을 어떻게 떠받치는지 드러난다. 이 단원은 최신 발전과, 겉보기에 다른 방식들을 가로질러 에너지 함수가 주는 아우르는 관점을 살핀다.

## 코드

```python
"""
나아간 에너지 바탕 모델 주제: 퍼짐 모델 및 요즘 연구와의 이음
======================================================================

최신 발전과 다른 만들어 내는 모델과의 이음을 살핀다.

걸리는 시간: 90~120분
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def ebm_diffusion_connection():
    """에너지 바탕 모델과 퍼짐 모델의 이음을 이해한다."""
    print("\nEBMs and Diffusion Models")
    print("-" * 50)
    print("Score-based generative models are EBMs where:")
    print("  E(x,t) defines the score at noise level t")
    print("  ∇ₓ log p(x,t) = -∇ₓ E(x,t)")
    print("\nDiffusion process:")
    print("  Forward: Add noise gradually x → x_T")
    print("  Reverse: Denoise using learned score")
    print("\n✓ EBMs provide theoretical foundation for diffusion")

def flow_based_ebms():
    """다룰 만한 가능도를 위한 흐름 바탕 에너지 바탕 모델."""
    print("\nFlow-Based Energy Models")
    print("-" * 50)
    print("Combine flows with EBMs:")
    print("  - Flows provide tractable Z")
    print("  - EBM refines the distribution")
    print("✓ Best of both worlds")

def latent_variable_ebms():
    """숨은 변수를 가진 에너지 바탕 모델."""
    print("\nLatent Variable EBMs")
    print("-" * 50)
    print("E(x,z) with latent z:")
    print("  - More expressive models")
    print("  - Hierarchical representations")
    print("✓ Combines EBMs with VAE-like structure")

def modern_research_directions():
    """에너지 바탕 모델의 지금 연구."""
    print("\nModern Research Directions")
    print("-" * 50)
    print("1. Improved sampling (HMC, ULA, MALA)")
    print("2. Better architectures (transformers, diffusion UNets)")
    print("3. Theoretical understanding (convergence, capacity)")
    print("4. Applications (video, 3D, multimodal)")
    print("5. Connections to physics and causality")

def main():
    print("="*70)
    print("ADVANCED EBM TOPICS")
    print("="*70)
    
    ebm_diffusion_connection()
    print()
    flow_based_ebms()
    print()
    latent_variable_ebms()
    print()
    modern_research_directions()
    
    print("\n" + "="*70)
    print("CURRICULUM COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
```

## 논의

에너지 바탕 모델과 퍼짐 모델의 이음은 요즘 만들어 내기에서 가장 중요한 통찰 가운데 하나이다. 점수 바탕 만들어 내는 모델은 잡음 수준 $t$마다 점수 함수 $\nabla_x \log p_t(x)$을 배우는데 이는 잡음 수준으로 이름표를 단 에너지 함수 무리를 배우는 것과 같다. 앞 퍼짐 과정은 자료를 잡음으로 차츰 망가뜨리고 뒤 과정은 배운 점수로 잡음을 없앤다. 이는 바로 에너지 바탕 모델이 뽑기에 쓰는 랑주뱅 움직임이다. 이 관점은 퍼짐 모델이 서로 다른 잡음 잣수에서 익힌 에너지 바탕 모델들을 조심스레 짠 배움 차례임을 드러낸다.

흐름 바탕 에너지 바탕 모델은 고르게 맞추는 흐름과 에너지 함수를 합쳐 두 쪽의 좋은 점을 얻는다. 고르게 맞추는 흐름은 드러난 밀도를 가진 다룰 만한 바탕 분포를 주고, 에너지 고침 항은 분포를 다듬어 잘게 나뉜 얼개를 담는다. 그렇게 얻은 모델은 $p(x) \propto p_{\text{flow}}(x) \cdot \exp(-E_\theta(x))$ 꼴이며 흐름이 분포의 큰 덩어리를 맡고 에너지 바탕 모델이 남은 얼개를 담는다. 이 방식은 다룰 만한 가능도 어림과 에너지 바탕 모델의 너그러움을 함께 가진 모델을 준다.

숨은 변수 에너지 바탕 모델은 본 자료 $x$과 숨은 변수 $z$ 모두에 대해 에너지 함수 $E(x, z)$을 뜻매김한다. $z$을 가장자리로 몰아내면 $x$만의 은근한 에너지 모델을 얻는다: $p(x) \propto \int \exp(-E(x, z))\, dz$. 이 틀은 에너지 바탕 모델과 변분 자기 부호기류 모델을 하나로 묶고 켜진 나타냄을 가능하게 한다. 지금의 연구 최전선에는 에너지 바탕 모델을 영상 만들어 내기, 3차원 모양 나타내기, 여러 갈래 배움에 쓰는 일과, 뽑기 모임과 모델 담이에 대한 이론 보장을 세우는 일이 있다.

## 연습문제

**연습문제 1.**
잡음 수준 $\sigma$에서 퍼짐 모델의 점수 함수가 잡음 없애기 함수와 $\nabla_x \log p_\sigma(x) = (D_\theta(x, \sigma) - x) / \sigma^2$으로 이어짐을 보여라. 여기서 $D_\theta(x, \sigma)$은 잡음 낀 들임에서 깨끗한 자료를 어림하는 잡음 없애개이다.

??? success "연습문제 1 풀이"
    잡음 낀 분포는 $p_\sigma(x) = \int p_{\text{data}}(y) \mathcal{N}(x; y, \sigma^2 I)\, dy$이다. 트위디 공식에 따라 사후 평균은 다음을 만족한다:
    
    $$
    \mathbb{E}[y | x] = x + \sigma^2 \nabla_x \log p_\sigma(x)
    $$
    
    가장 좋은 잡음 없애개는 사후 평균을 헤아린다: $D_\theta(x, \sigma) = \mathbb{E}[y | x]$. 고쳐 쓰면:
    
    $$
    \nabla_x \log p_\sigma(x) = \frac{D_\theta(x, \sigma) - x}{\sigma^2}
    $$
    
    이는 잡음 없애개를 배우는 것이 점수 함수를 배우는 것과 같음을 보인다. 퍼짐 모델은 보통 잡음 헤아림 $\epsilon_\theta(x, \sigma)$이나 잡음 없애개 $D_\theta(x, \sigma)$ 가운데 하나를 매개변수로 삼는데 둘 다 점수와 곧바로 이어져 있다.

---

**연습문제 2.**
흐름 바탕 에너지 바탕 모델이 다룰 수 없는 나눔 함수를 담고도 정확한 가능도를 셈하는 데 쓰일 수 있는 까닭을 적어라. 여기서 중요도 뽑기의 몫은 무엇인가?

??? success "연습문제 2 풀이"
    흐름 바탕 에너지 바탕 모델에서 $p(x) = p_{\text{flow}}(x) \cdot \exp(-E_\theta(x)) / Z$이고 $Z = \mathbb{E}_{p_{\text{flow}}}[\exp(-E_\theta(x))]$이다. 흐름이 중요도 뽑기의 다룰 만한 제안 분포를 준다:
    
    $$
    Z = \int p_{\text{flow}}(x) \exp(-E_\theta(x))\, dx = \mathbb{E}_{x \sim p_{\text{flow}}}[\exp(-E_\theta(x))]
    $$
    
    이 기댓값은 흐름에서 뽑은 몬테카를로 표본으로 어림할 수 있다: $x_i \sim p_{\text{flow}}$일 때 $\hat{Z} = \frac{1}{N} \sum_{i=1}^N \exp(-E_\theta(x_i))$이다. 그러면 자료 점의 로그 가능도는 다음과 같다:
    
    $$
    \log p(x) = \log p_{\text{flow}}(x) - E_\theta(x) - \log Z
    $$
    
    $Z$의 중요도 뽑기 어림은 치우치지 않았으며 흐름이 에너지 바탕 모델 분포를 더 잘 어림할수록 흩어짐이 줄어든다.

---

**연습문제 3.**
에너지 함수가 $E(x, z) = E_{\text{recon}}(x, z) + E_{\text{prior}}(z)$으로 나뉘고 $E_{\text{recon}}$이 되짓기 품질을 재며 $E_{\text{prior}}$이 숨은 공간에 규칙을 세우는 숨은 변수 에너지 바탕 모델을 짜라. 이를 여느 변분 자기 부호기와 견주고 핵심 차이를 짚어라.

??? success "연습문제 3 풀이"
    ```python
    class LatentVariableEBM(nn.Module):
        def __init__(self, x_dim, z_dim, hidden_dim=256):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, x_dim)
            )
            self.encoder = nn.Sequential(
                nn.Linear(x_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, z_dim)
            )
        
        def energy(self, x, z):
            x_recon = self.decoder(z)
            E_recon = 0.5 * ((x - x_recon)**2).sum(dim=1)
            E_prior = 0.5 * (z**2).sum(dim=1)
            return E_recon + E_prior
    ```
    
    변분 자기 부호기와의 핵심 차이: (1) 숨은 변수 에너지 바탕 모델은 특정한 부호기 분포를 가정하지 않고 $p(x, z) \propto \exp(-E(x, z))$을 뜻매김하지만 변분 자기 부호기는 정규 부호기 $q_\phi(z|x)$을 쓴다. (2) 숨은 에너지 바탕 모델을 익히려면 결합 $(x, z)$ 공간에서 마르코프 사슬 몬테카를로 뽑기가 필요하지만 변분 자기 부호기는 매개변수 다시 쓰기 재주를 쓴다. (3) 숨은 에너지 바탕 모델은 봉우리 여럿인 사후 분포 $p(z|x)$을 자연스럽게 나타낼 수 있지만 여느 변분 자기 부호기는 봉우리 하나인 정규 사후 분포에 갇힌다. (4) 숨은 에너지 바탕 모델은 셈 비용(마르코프 사슬 몬테카를로)을 나타냄의 너그러움과 맞바꾼다.
