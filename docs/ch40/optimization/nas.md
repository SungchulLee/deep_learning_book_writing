# 신경 얼개 찾기
## 두루 보기

신경 얼개 찾기(NAS)은 신경 그물 얼개를 꾸미는 일을 저절로 하게 하여, 손으로 하던 얼개 다듬기를 알고리즘의 가장 좋게 하기로 갈음한다. NAS은 사람이 꾸민 그물보다 잘 드는 얼개를 찾아내는 일이 잦아, 모형 눌러 담기와 내놓기 다듬기의 든든한 연장이 된다.

## 왜 하는가

손으로 꾸민 얼개에는 켜의 수, 켜의 너비, 알갱이 크기, 건너뛰는 이음, 살림 함수, 잣대 잡기 꾀 같은 헤아릴 수 없이 많은 고름이 든다. NAS은 이 밭을 짜임새 있게 둘러보며 맞음, 늦음, 모형 크기 같은 목표에 맞게 다듬은 얼개를 찾는다.

## 찾을 밭 꾸미기

### 흔한 찾을 밭

```python
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import random

# 켜마다 고를 셈의 후보를 매긴다
OPERATIONS = {
    'conv_3x3': lambda C: nn.Conv2d(C, C, 3, padding=1),
    'conv_5x5': lambda C: nn.Conv2d(C, C, 5, padding=2),
    'sep_conv_3x3': lambda C: nn.Sequential(
        nn.Conv2d(C, C, 3, padding=1, groups=C),
        nn.Conv2d(C, C, 1)
    ),
    'dil_conv_3x3': lambda C: nn.Conv2d(C, C, 3, padding=2, dilation=2),
    'max_pool_3x3': lambda C: nn.MaxPool2d(3, stride=1, padding=1),
    'avg_pool_3x3': lambda C: nn.AvgPool2d(3, stride=1, padding=1),
    'skip_connect': lambda C: nn.Identity(),
    'none': lambda C: Zero(C),
}

class Zero(nn.Module):
    """NAS의 0 셈(0을 낸다)."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    
    def forward(self, x):
        return torch.zeros_like(x)
```

### 짐을 나눠 쓰는 한 번 찾기 NAS

```python
class SearchCell(nn.Module):
    """미분할 수 있는 얼개 매개변수를 지닌 NAS 칸."""
    
    def __init__(self, channels: int, operations: Dict):
        super().__init__()
        self.ops = nn.ModuleDict({
            name: op_fn(channels) for name, op_fn in operations.items()
        })
        # 얼개 매개변수(배울 수 있다)
        self.alphas = nn.Parameter(
            torch.randn(len(operations)) * 0.01
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 얼개 매개변수에 소프트맥스
        weights = torch.softmax(self.alphas, dim=0)
        
        # 모든 셈의 짐 준 합
        output = sum(
            w * op(x) for w, (name, op) in zip(weights, self.ops.items())
        )
        return output
    
    def get_best_op(self) -> str:
        """짐이 가장 큰 셈을 돌려준다."""
        idx = self.alphas.argmax().item()
        return list(self.ops.keys())[idx]


class DARTSNetwork(nn.Module):
    """DARTS 결의 미분할 수 있는 NAS."""
    
    def __init__(self, num_cells: int = 8, channels: int = 16,
                 num_classes: int = 10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.cells = nn.ModuleList([
            SearchCell(channels, OPERATIONS) for _ in range(num_cells)
        ])
        
        self.classifier = nn.Linear(channels, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
    
    def architecture_parameters(self):
        """따로 가장 좋게 할 얼개 매개변수를 돌려준다."""
        return [cell.alphas for cell in self.cells]
    
    def weight_parameters(self):
        """짐 매개변수를 돌려준다(얼개 매개변수는 뺀다)."""
        arch_params = set(id(p) for p in self.architecture_parameters())
        return [p for p in self.parameters() if id(p) not in arch_params]
    
    def derive_architecture(self) -> List[str]:
        """찾아낸 얼개를 뽑아낸다."""
        return [cell.get_best_op() for cell in self.cells]
```

## 쇠 붙임새를 아는 NAS

쇠 붙임새의 옭아맴(늦음, 기억)을 찾기 목표에 넣는다.

```python
class LatencyPredictor:
    """후보 얼개의 미루어 봄 늦음을 미리 본다."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.latency_cache = {}
    
    def measure_latency(self, module: nn.Module, input_shape: Tuple,
                       num_runs: int = 100) -> float:
        """참 미루어 봄 늦음을 잰다."""
        import time
        x = torch.randn(1, *input_shape).to(self.device)
        module = module.to(self.device).eval()
        
        # 몸풀기
        with torch.no_grad():
            for _ in range(10):
                module(x)
        
        # 잰다
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                module(x)
                times.append(time.perf_counter() - start)
        
        return sum(times) / len(times) * 1000  # ms


def hardware_aware_loss(logits, targets, architecture_params,
                       latency_predictor, target_latency_ms=10.0,
                       lambda_latency=0.1):
    """아우른 잃음: 맞음 + 늦음 벌."""
    import torch.nn.functional as F
    
    # 일 잃음
    task_loss = F.cross_entropy(logits, targets)
    
    # 늦음 벌(미분할 수 있는 어림)
    predicted_latency = sum(
        torch.softmax(alpha, dim=0).sum() for alpha in architecture_params
    )
    latency_penalty = torch.relu(predicted_latency - target_latency_ms)
    
    return task_loss + lambda_latency * latency_penalty
```

## 얼개 다듬기로서의 낮은 자리 쪼개기

낮은 자리 쪼개기는 큰 켜를 같은 일을 하는 작은 켜로 갈음하는 얼개 바꿈으로 볼 수 있다. 이로써 NAS과 행렬 쪼개기가 이어진다.

# 낮은 자리 쪼개기

## 두루 보기

낮은 자리 쪼개기는 짐 행렬을 더 작은 행렬들의 곱으로 갈라 신경 그물을 눌러 담는다. 익힌 짐 행렬은 참으로 쓰이는 자리가 낮은 일이 잦다는 살핌을 쓴 것이다. 이로써 모형이 드러내는 힘은 거의 지키면서 담는 자리와 셈 값을 함께 줄인다.

## 수학 밑바탕

### 행렬의 자리와 어림

행렬 $\mathbf{W} \in \mathbb{R}^{m \times n}$은 낮은 자리 쪼개기로 어림할 수 있다.

$$\mathbf{W} \approx \mathbf{U}\mathbf{V}^T$$

여기서

- $\mathbf{U} \in \mathbb{R}^{m \times r}$
- $\mathbf{V} \in \mathbb{R}^{n \times r}$
- $r \ll \min(m, n)$은 자리

**매개변수 줄어듦:**

- 본디: 매개변수 $m \times n$개
- 쪼갠 뒤: 매개변수 $m \times r + n \times r = r(m + n)$개
- 줄어드는 값: $\frac{mn}{r(m+n)}$

### 특잇값 쪼개기(SVD)

자리가 $r$인 짐 행렬 $\mathbf{W} \in \mathbb{R}^{m \times n}$에서 특잇값 쪼개기(SVD)는 다음을 준다.

$$\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

여기서

- $\mathbf{U} \in \mathbb{R}^{m \times m}$은 왼쪽 특이 벡터를 담는다(서로 곧고 길이가 1)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$은 특잇값 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$을 담는다
- $\mathbf{V} \in \mathbb{R}^{n \times n}$은 오른쪽 특이 벡터를 담는다(서로 곧고 길이가 1)

### 잘라 낸 SVD(낮은 자리 어림)

(프로베니우스 노름에서) 가장 좋은 자리 $k$의 어림은 으뜸 특잇값 $k$개만 남겨 얻는다.

$$\mathbf{W}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

여기서 $\mathbf{U}_k \in \mathbb{R}^{m \times k}$, $\mathbf{\Sigma}_k \in \mathbb{R}^{k \times k}$, $\mathbf{V}_k \in \mathbb{R}^{n \times k}$이다.

**에카르트-영-미르스키 정리:**

$$\mathbf{W}_k = \arg\min_{\text{rank}(\mathbf{A}) \leq k} \|\mathbf{W} - \mathbf{A}\|_F$$

**어림 어긋남:**

$$\|\mathbf{W} - \mathbf{W}_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

### 눌러 담은 견줌

**본디 자리:** 매개변수 $mn$개

**쪼갠 뒤 자리:** $mk + k + kn = k(m + n + 1) \approx k(m + n)$

**눌러 담은 견줌:**

$$\rho = \frac{mn}{k(m + n)}$$

눌러 담으려면 $k < \frac{mn}{m + n}$이어야 한다.

**보기:** $\mathbf{W} \in \mathbb{R}^{1024 \times 1024}$에서

- 본디: 매개변수 1,048,576개
- $k = 64$이면: 매개변수 131,136개(8배 눌러 담음)
- $k = 128$이면: 매개변수 262,272개(4배 눌러 담음)

## 선형 켜 쪼개기

### 밑바탕 SVD 쪼개기

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict, List


def svd_decomposition(weight: torch.Tensor, 
                      rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    잘라 낸 SVD로 짐 행렬을 쪼갠다.
    
    Args:
        weight: 쪼갤 짐 행렬 (m x n)
        rank: 쪼갤 과녁 자리
        
    Returns:
        U, V: weight ≈ U @ V.T이 되는 쪼갠 행렬
    """
    # SVD를 한다
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    
    # 과녁 자리로 잘라 낸다
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank, :]
    
    # 특잇값을 U에 녹여 넣는다
    U_scaled = U_r * S_r.unsqueeze(0)
    
    # 되살림: weight ≈ U_scaled @ V_r
    return U_scaled, V_r.T


def compute_reconstruction_error(original: torch.Tensor, 
                                 reconstructed: torch.Tensor) -> float:
    """견준 되살림 어긋남을 셈한다."""
    error = torch.norm(original - reconstructed) / torch.norm(original)
    return error.item()


# 보기: 눌러 담음과 어긋남의 맞바꿈을 살핀다
def analyze_rank_tradeoff(W: torch.Tensor):
    """자리마다 어긋남과 눌러 담음을 살핀다."""
    print("자리 살피기:")
    print("-" * 60)
    
    for rank in [16, 32, 64, 128]:
        U, V = svd_decomposition(W, rank)
        W_approx = U @ V.T
        
        error = compute_reconstruction_error(W, W_approx)
        original_params = W.numel()
        factored_params = U.numel() + V.numel()
        compression = original_params / factored_params
        
        print(f"자리 {rank:3d}: 어긋남={error:.4f}, "
              f"눌러 담음={compression:.2f}배")
```

### FactorizedLinear 켜

```python
class FactorizedLinear(nn.Module):
    """
    더 작은 켜 둘로 쪼갠 선형 켜.
    
    본디: y = Wx + b  (W은 m x n)
    쪼갠 뒤: y = U(Vx) + b  (U은 m x r, V은 r x n)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # 쪼갠 켜
        self.V = nn.Linear(in_features, rank, bias=False)
        self.U = nn.Linear(rank, out_features, bias=bias)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """잣대를 줄인 제 자리 옮김에 가깝게 첫자리를 잡는다."""
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.U.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.V(x))
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, rank: int) -> 'FactorizedLinear':
        """
        이미 있는 선형 켜에서 쪼갠 켜를 만든다.
        
        SVD로 가장 좋은 쪼갬을 찾는다.
        """
        W = linear_layer.weight.data
        b = linear_layer.bias.data if linear_layer.bias is not None else None
        
        # SVD 쪼개기
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # 잘라 내고 특잇값을 고르게 나눈다
        U_r = U[:, :rank] * S[:rank].sqrt().unsqueeze(0)
        V_r = Vh[:rank, :] * S[:rank].sqrt().unsqueeze(1)
        
        # 쪼갠 켜를 만든다
        factorized = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            bias=linear_layer.bias is not None
        )
        
        factorized.U.weight.data = U_r
        factorized.V.weight.data = V_r
        if b is not None:
            factorized.U.bias.data = b
        
        return factorized


def svd_decompose_linear(layer: nn.Linear,
                         rank: int) -> Tuple[nn.Linear, nn.Linear]:
    """
    잘라 낸 SVD로 선형 켜를 쪼갠다.
    
    W ≈ U_k * Σ_k * V_k^T = A * B
    
    본디: y = Wx + b
    쪼갠 뒤: y = A(Bx) + b
    
    Args:
        layer: 쪼갤 선형 켜
        rank: 어림할 과녁 자리
        
    Returns:
        output = second(first(x))이 되는 선형 켜 둘(first, second)
    """
    W = layer.weight.data  # (out_features, in_features)
    b = layer.bias.data if layer.bias is not None else None
    
    # SVD 쪼개기
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # 자리 k으로 잘라 낸다
    U_k = U[:, :rank]  # (out_features, rank)
    S_k = S[:rank]     # (rank,)
    V_k = Vh[:rank, :]  # (rank, in_features)
    
    # 쪼갠 켜를 만든다
    # 첫 켜: x -> 가운데(자리 차수)
    first = nn.Linear(layer.in_features, rank, bias=False)
    first.weight.data = V_k  # (rank, in_features)
    
    # 둘째 켜: 가운데 -> 날임
    second = nn.Linear(rank, layer.out_features, bias=b is not None)
    second.weight.data = U_k @ torch.diag(S_k)  # (out_features, rank)
    if b is not None:
        second.bias.data = b
    
    return first, second
```

## 엮음 켜 쪼개기

### 자리로 쪼개기(가를 수 있는 엮음)

$k \times k$ 엮음을 $1 \times k$과 $k \times 1$ 엮음으로 쪼갠다.

```python
class SeparableConv2d(nn.Module):
    """
    자리로 가를 수 있는 엮음.
    
    k×k 엮음을 1×k 다음 k×1으로 갈음한다.
    
    본디: O(C_in × C_out × k × k × H × W)
    가른 뒤: O(C_in × C_out × 2k × H × W)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        
        # 1×k 엮음
        self.conv_h = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            bias=False
        )
        
        # k×1 엮음
        self.conv_v = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_h(x)
        x = self.conv_v(x)
        return x
```

### 깊이별로 가른 엮음(MobileNet 결)

자리 섞기와 갈래 섞기를 나눈다.

```python
class DepthwiseSeparableConv2d(nn.Module):
    """
    깊이별로 가른 엮음(MobileNet 결).
    
    1. 깊이별: 들임 갈래마다 k×k 거르개 하나
    2. 점별: 갈래를 섞는 1×1 엮음
    
    셈 줄어듦: C_out이 크면 k²/(k² + C_out) ≈ 1/k²
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        
        # 깊이별 엮음(groups=in_channels)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # 고갱이: 갈래마다 따로 거르개
            bias=False
        )
        
        # 점별 엮음(1×1)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# 매개변수를 견준다
def compare_conv_variants():
    in_ch, out_ch, kernel = 64, 128, 3
    
    standard = nn.Conv2d(in_ch, out_ch, kernel, padding=1)
    separable = DepthwiseSeparableConv2d(in_ch, out_ch, kernel, padding=1)
    
    std_params = sum(p.numel() for p in standard.parameters())
    sep_params = sum(p.numel() for p in separable.parameters())
    
    print(f"여느 엮음: 매개변수 {std_params:,}개")
    print(f"깊이별로 가른 엮음: 매개변수 {sep_params:,}개")
    print(f"줄어듦: {std_params/sep_params:.1f}배")
```

### 갈래별 SVD 쪼개기

```python
def decompose_conv_channel(conv: nn.Conv2d,
                           rank: int) -> nn.Sequential:
    """
    갈래별 쪼개기로 Conv2d을 쪼갠다.
    
    본디: k×k 알갱이로 C_in -> C_out
    쪼갠 뒤:
    1. k×k 알갱이로 C_in -> rank
    2. 1×1 알갱이로 rank -> C_out(점별)
    """
    W = conv.weight.data  # (C_out, C_in, k_h, k_w)
    
    # (C_out, C_in * k_h * k_w) 꼴로 바꾼다
    W_mat = W.view(conv.out_channels, -1)
    
    # SVD를 건다
    U, S, Vh = torch.linalg.svd(W_mat, full_matrices=False)
    
    # 잘라 낸다
    U_k = U[:, :rank]  # (C_out, rank)
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]  # (rank, C_in * k_h * k_w)
    
    # 쪼갠 엮음을 만든다
    conv1 = nn.Conv2d(
        conv.in_channels, rank,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False
    )
    conv1.weight.data = (Vh_k @ torch.diag(S_k ** 0.5)).view(
        rank, conv.in_channels, *conv.kernel_size
    )
    
    conv2 = nn.Conv2d(rank, conv.out_channels, kernel_size=1,
                      bias=conv.bias is not None)
    conv2.weight.data = (U_k @ torch.diag(S_k ** 0.5)).view(
        conv.out_channels, rank, 1, 1
    )
    if conv.bias is not None:
        conv2.bias.data = conv.bias.data
    
    return nn.Sequential(conv1, conv2)
```

### 터커 쪼개기

차수가 높은 텐서에서는 터커 쪼개기가 두루 쓰이는 낮은 자리 어림을 준다.

$$\mathcal{K} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)} \times_4 \mathbf{U}^{(4)}$$

```python
class TuckerConv2d(nn.Module):
    """
    엮음 켜의 터커 쪼개기.
    
    C_in × C_out × k × k을 다음으로 쪼갠다:
    1. 1×1 엮음: C_in → r_in(들임 갈래를 눌러 담는다)
    2. r_in × r_out × k × k 엮음(더 작은 알맹이)
    3. 1×1 엮음: r_out → C_out(날임 갈래를 넓힌다)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 rank_in: int, rank_out: int, stride: int = 1, 
                 padding: int = 0, bias: bool = True):
        super().__init__()
        
        # 들임 갈래를 눌러 담는다
        self.compress = nn.Conv2d(in_channels, rank_in, 1, bias=False)
        
        # 알맹이 엮음(자리를 줄임)
        self.core = nn.Conv2d(
            rank_in, rank_out, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        
        # 날임 갈래를 넓힌다
        self.expand = nn.Conv2d(rank_out, out_channels, 1, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        x = self.core(x)
        x = self.expand(x)
        return x


def tucker_decompose_conv(conv: nn.Conv2d,
                          ranks: Tuple[int, int]) -> nn.Sequential:
    """
    tensorly으로 하는 Conv2d 켜의 터커 쪼개기.
    
    엮음 셋으로 쪼갠다:
    1. 1x1 엮음: C_in -> rank[1]
    2. kxk 엮음: rank[1] -> rank[0]
    3. 1x1 엮음: rank[0] -> C_out
    """
    try:
        import tensorly as tl
        from tensorly.decomposition import partial_tucker
        tl.set_backend('pytorch')
    except ImportError:
        raise ImportError("터커 쪼개기에는 tensorly이 있어야 한다: pip install tensorly")
    
    W = conv.weight.data
    core, factors = partial_tucker(W, modes=[0, 1], rank=ranks, init='svd')
    
    conv_input = nn.Conv2d(conv.in_channels, ranks[1], kernel_size=1, bias=False)
    conv_input.weight.data = factors[1].t().unsqueeze(-1).unsqueeze(-1)
    
    conv_spatial = nn.Conv2d(ranks[1], ranks[0], kernel_size=conv.kernel_size,
                             stride=conv.stride, padding=conv.padding, bias=False)
    conv_spatial.weight.data = core
    
    conv_output = nn.Conv2d(ranks[0], conv.out_channels, kernel_size=1,
                            bias=conv.bias is not None)
    conv_output.weight.data = factors[0].unsqueeze(-1).unsqueeze(-1)
    if conv.bias is not None:
        conv_output.bias.data = conv.bias.data
    
    return nn.Sequential(conv_input, conv_spatial, conv_output)
```

## 자리 절로 고르기

### 힘에 기댄 고르기

```python
def select_rank_by_energy(weight: torch.Tensor,
                          energy_threshold: float = 0.95) -> int:
    """
    주어진 몫의 힘(프로베니우스 노름 제곱)을 지키는 자리를 고른다.
    
    Args:
        weight: 짐 행렬
        energy_threshold: 지킬 힘의 몫(0.95 = 95%)
        
    Returns:
        가장 좋은 자리
    """
    _, S, _ = torch.linalg.svd(weight.view(weight.size(0), -1), 
                               full_matrices=False)
    
    # 쌓인 힘을 셈한다
    total_energy = (S ** 2).sum()
    cumulative_energy = (S ** 2).cumsum(0) / total_energy
    
    # 문턱을 이루는 자리를 찾는다
    rank = (cumulative_energy < energy_threshold).sum().item() + 1
    
    return rank


def analyze_singular_values(weight: torch.Tensor) -> Dict:
    """
    특잇값의 분포를 살펴 가장 좋은 자리를 정한다.
    """
    _, S, _ = torch.linalg.svd(weight.view(weight.size(0), -1), 
                               full_matrices=False)
    S = S.cpu().numpy()
    
    # 쌓인 힘(설명한 흩어짐)
    total_energy = (S ** 2).sum()
    cumulative_energy = (S ** 2).cumsum() / total_energy
    
    # 힘 문턱마다의 자리를 찾는다
    thresholds = [0.9, 0.95, 0.99, 0.999]
    ranks_for_threshold = {}
    for thresh in thresholds:
        rank = (cumulative_energy < thresh).sum() + 1
        ranks_for_threshold[f'{thresh:.1%}_energy'] = int(rank)
    
    return {
        'singular_values': S,
        'cumulative_energy': cumulative_energy,
        'ranks_for_threshold': ranks_for_threshold,
        'effective_rank': int((S > S[0] * 0.01).sum()),
        'condition_number': S[0] / S[-1] if S[-1] > 0 else float('inf')
    }


def analyze_layer_ranks(model: nn.Module) -> List[Dict]:
    """
    모형에 든 켜 모두의 참으로 쓰이는 자리를 살핀다.
    """
    results = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            
            # 참으로 쓰이는 자리의 자를 셈한다
            total_energy = (S ** 2).sum()
            cumulative = (S ** 2).cumsum(0) / total_energy
            
            rank_95 = (cumulative < 0.95).sum().item() + 1
            rank_99 = (cumulative < 0.99).sum().item() + 1
            full_rank = min(W.shape)
            
            results.append({
                'layer': name,
                'shape': tuple(W.shape),
                'full_rank': full_rank,
                'rank_95': rank_95,
                'rank_99': rank_99,
                'compression_95': full_rank / rank_95
            })
    
    return results
```

## 모형 켜에서의 쪼개기

### 모형 통째로 쪼개기

```python
class LowRankModel(nn.Module):
    """
    미리 익힌 모형에 낮은 자리 쪼개기를 건다.
    """
    
    def __init__(self, model: nn.Module, rank_ratio: float = 0.5,
                 min_rank: int = 8):
        super().__init__()
        self.model = self._factorize_model(model, rank_ratio, min_rank)
    
    def _factorize_model(self, model: nn.Module, rank_ratio: float,
                         min_rank: int) -> nn.Module:
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                max_rank = min(module.in_features, module.out_features)
                rank = max(min_rank, int(max_rank * rank_ratio))
                if rank < max_rank:
                    first, second = svd_decompose_linear(module, rank)
                    setattr(model, name, nn.Sequential(first, second))
            elif isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
                max_rank = min(module.in_channels, module.out_channels)
                rank = max(min_rank, int(max_rank * rank_ratio))
                if rank < max_rank:
                    decomposed = decompose_conv_channel(module, rank)
                    setattr(model, name, decomposed)
            else:
                self._factorize_model(module, rank_ratio, min_rank)
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def factorize_model(model: nn.Module, rank_ratio: float = 0.5, 
                    min_rank: int = 8) -> nn.Module:
    """
    선형 켜 모두에 낮은 자리 쪼개기를 건다.
    
    Args:
        model: PyTorch 모형
        rank_ratio: 쓸 온 자리의 몫(0.5 = 50%)
        min_rank: 쓸 가장 작은 자리
        
    Returns:
        쪼갠 모형
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 과녁 자리를 셈한다
            full_rank = min(module.weight.shape)
            target_rank = max(min_rank, int(full_rank * rank_ratio))
            
            # 쪼갠 것을 만든다
            factorized = FactorizedLinear.from_linear(module, target_rank)
            setattr(model, name, factorized)
            
            print(f"{name} 쪼갬: {module.weight.shape} → 자리 {target_rank}")
        
        elif len(list(module.children())) > 0:
            # 안에 든 묶음을 되돌아 다룬다
            factorize_model(module, rank_ratio, min_rank)
    
    return model
```

### 쪼갠 뒤 곱게 맞추기

```python
def finetune_factorized_model(model: nn.Module,
                              train_loader: torch.utils.data.DataLoader,
                              val_loader: torch.utils.data.DataLoader,
                              epochs: int = 10,
                              lr: float = 1e-4,
                              device: str = 'cpu') -> Tuple[nn.Module, float]:
    """
    쪼갠 뒤 맞음을 되찾으려 모형을 곱게 맞춘다.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        # 익힌다
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 따진다
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        best_accuracy = max(best_accuracy, accuracy)
        
        print(f"{epoch+1}판: 잃음={total_loss/len(train_loader):.4f}, "
              f"맞음={accuracy*100:.2f}%")
    
    return model, best_accuracy
```

## LoRA: 낮은 자리로 맞추기

LoRA(낮은 자리로 맞추기)은 얼려 둔 미리 익힌 짐에 배울 수 있는 낮은 자리 행렬을 더해 큰 모형을 잘 드는 값으로 곱게 맞추게 한다.

$$y = Wx + BAx$$

여기서 $W$은 얼려 두고 $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$은 배운다.

```python
class LoRALinear(nn.Module):
    """
    LoRA(낮은 자리로 맞추기) 켜.
    
    본디 짐은 얼려 두고 배울 수 있는 낮은 자리 고침을 더한다.
    y = Wx + (BA)x
    
    여기서 W은 얼려 두고 B, A은 배우는 낮은 자리 행렬이다.
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha  # 잣대 값
        
        # 본디 짐을 얼린다
        for param in self.original.parameters():
            param.requires_grad = False
        
        # 낮은 자리로 맞추는 행렬
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # A은 작은 아무 값으로, B은 0으로 첫자리를 잡는다
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 본디 날임
        original_out = self.original(x)
        
        # LoRA이 보태는 몫: x @ A.T @ B.T
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        
        return original_out + self.scaling * lora_out
    
    def merge_weights(self):
        """미루어 보려고 LoRA 짐을 본디 짐에 녹여 넣는다."""
        with torch.no_grad():
            delta_W = self.scaling * (self.lora_B @ self.lora_A)
            self.original.weight.data += delta_W
    
    def get_trainable_params(self) -> int:
        """배울 수 있는 매개변수의 수를 돌려준다."""
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16,
               target_modules: List[str] = None) -> nn.Module:
    """
    모형의 선형 켜에 LoRA을 건다.
    
    Args:
        model: 고칠 모형
        rank: LoRA 자리
        alpha: 잣대 값
        target_modules: 과녁으로 삼을 묶음 이름의 목록(None = 모든 Linear)
        
    Returns:
        LoRA을 건 모형
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if target_modules is None or name in target_modules:
                setattr(model, name, LoRALinear(module, rank, alpha))
        elif len(list(module.children())) > 0:
            apply_lora(module, rank, alpha, target_modules)
    return model


def count_lora_params(model: nn.Module) -> Dict[str, int]:
    """LoRA 모형에서 배우는 매개변수와 얼린 매개변수를 센다."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return {
        'trainable': trainable,
        'frozen': frozen,
        'total': trainable + frozen,
        'trainable_ratio': trainable / (trainable + frozen)
    }
```

## 맞바꿈과 한계

### 눌러 담음 대 맞음

| 눌러 담은 견줌 | 흔한 맞음 떨어짐 | 되찾기 어려움 |
|-------------------|----------------------|---------------------|
| 2배 | 0.5% 미만 | 쉬움(잠깐 곱게 맞추면 된다) |
| 4배 | 0.5~2% | 가운데 |
| 8배 | 2~5% | 만만치 않음 |
| 16배 | 5% 넘음 | 아주 만만치 않음 |

### 낮은 자리 쪼개기를 쓸 때

**잘 맞는 자리:**

- 매개변수가 많은 큰 온통 이은 켜
- 갈래가 많은 엮음 켜
- 짐 분포가 본디 자리가 낮은 켜
- 곱게 맞추는 자리(LoRA)

**잘 맞지 않는 자리:**

- 작은 켜(덤으로 드는 값이 아낀 것보다 클 수 있다)
- 1×1 엮음(이미 잘 든다)
- 첫 켜와 마지막 켜(맞음에 걸리는 일이 잦다)

### 덤으로 드는 셈

쪼개기는 셈을 더 들인다.

- 행렬 곱 하나가 잇따른 곱 둘이 된다
- 기억을 짚는 결이 덜 잘 들 수 있다
- 가운데 자리가 작으면 GPU을 덜 쓰게 될 수 있다

## 간추림

낮은 자리 쪼개기는 모형 크기와 셈을 줄인다.

1. **SVD 쪼개기**: 가장 좋은 어림이며 익힘 뒤 눌러 담기에 좋다
2. **자리로 가르기**: 엮음을 자리로 쪼갠다(1×k과 k×1)
3. **깊이별로 가르기**: 잘 드는 얼개의 여느 길이다(MobileNet)
4. **터커 쪼개기**: 갈래와 자리를 함께 줄인다
5. **LoRA**: 미리 익힌 큰 모형을 잘 드는 값으로 곱게 맞춘다

고갱이로 즐겨 쓸 길:

- 눌러 담을 견줌을 정하기 앞서 켜의 자리를 살핀다
- 힘에 기댄 자리 고르기를 쓴다(힘의 95~99%을 지킨다)
- 쪼갠 뒤 곱게 맞추어 맞음을 되찾는다
- 가장 크게 눌러 담으려면 다른 재주(수 줄이기, 쳐내기)와 아우른다
- 큰 모형을 잘 드는 값으로 맞추려면 LoRA을 헤아린다

## 살펴볼 거리

1. Denton, E., et al. "Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation." NeurIPS 2014.
2. Jaderberg, M., et al. "Speeding up Convolutional Neural Networks with Low Rank Expansions." BMVC 2014.
3. Kim, Y., et al. "Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications." ICLR 2016.
4. Howard, A., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv 2017.
5. Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
6. Lebedev, V., et al. "Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition." ICLR 2015.


---

## 아우른 눌러 담기 흐름

NAS은 다른 눌러 담기 재주와 하나의 흐름으로 아우를 때 가장 잘 듣는다.

# 아우른 눌러 담기 흐름

## 두루 보기

모형을 가장 크게 눌러 담으려면 쳐내기, 수 줄이기, 앎 옮기기, 낮은 자리 쪼개기를 아울러야 한다. 이 방법들은 얽혀 서로 주고받으므로 가장 좋은 차례와 차림을 정하는 것이 어려운 대목이다.

## 눌러 담기 흐름 꾸미기

### 재주끼리 주고받음

눌러 담는 방법끼리는 서로 돕기도 하고 어긋나기도 한다.

| 짝 | 주고받음 | 붙임말 |
|-------------|-------------|-------|
| 쳐내기 → 수 줄이기 | 서로 돕는다 | 성긴 짐은 분포가 좁다 |
| 수 줄이기 → 쳐내기 | 그저 그렇거나 나쁘다 | 수 줄인 짐은 크기로 줄 세우기 어렵다 |
| 앎 옮기기 → 쳐내기 | 서로 돕는다 | 제자 얼개를 성김에 맞게 꾸민다 |
| 앎 옮기기 → 수 줄이기 | 서로 돕는다 | 앎 옮기기가 수 줄이기의 잃음을 메운다 |
| 낮은 자리 → 수 줄이기 | 서로 돕는다 | 쪼갠 켜는 분포가 더 단순한 일이 잦다 |

### 가장 좋은 차례

겪어 본 결과에 따라 즐겨 쓰는 차례는 이렇다.

```
1. 앎 옮기기(스승이 있으면, 골라 쓴다)
       ↓
2. 쳐내기(빨라지려면 짜임새 있는 쳐내기가 낫다)
       ↓  
3. 곱게 맞추기(쳐낸 뒤 되찾기)
       ↓
4. 낮은 자리 쪼개기(골라 쓴다)
       ↓
5. 수 줄이기를 아는 익힘 또는 익힘 뒤 수 줄이기
       ↓
6. 마지막 눈금 맞추기와 내보내기
```

**까닭:**

- 앎 옮기기를 먼저: 제자 얼개가 뒤이을 눌러 담기에 맞게 다듬어진다
- 수 줄이기 앞에 쳐내기: 짐 분포가 좁을수록 수 줄이기가 잘 된다
- 걸음 사이의 곱게 맞추기: 도막마다 맞음을 되찾는다
- 수 줄이기를 마지막에: 다듬어진 짐 분포의 덕을 본다

## PyTorch로 짜기

### 온전한 눌러 담기 흐름

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant
import copy
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CompressionConfig:
    """눌러 담기 흐름의 차림."""
    # 앎 옮기기
    use_distillation: bool = True
    temperature: float = 4.0
    alpha: float = 0.5
    distillation_epochs: int = 20
    
    # 쳐내기
    target_sparsity: float = 0.7
    pruning_method: str = 'structured'  # 'unstructured', 'structured'
    pruning_epochs: int = 10
    
    # 낮은 자리
    use_low_rank: bool = False
    rank_ratio: float = 0.5
    
    # 수 줄이기
    quantization_method: str = 'qat'  # 'ptq', 'qat'
    qat_epochs: int = 10
    
    # 두루
    learning_rate: float = 1e-3
    device: str = 'cpu'


class CompressionPipeline:
    """
    재주 여럿을 아우른 하나의 눌러 담기 흐름.
    
    받쳐 주는 것:
    - 앎 옮기기(스승에게서)
    - 짜임새 있는/없는 쳐내기
    - 낮은 자리 쪼개기
    - 익힘 뒤 수 줄이기와 수 줄이기를 아는 익힘
    """
    
    def __init__(self,
                 student: nn.Module,
                 teacher: Optional[nn.Module] = None,
                 config: Optional[CompressionConfig] = None):
        """
        Args:
            student: 눌러 담을 모형
            teacher: 앎 옮기기에 쓸 스승 모형(골라 씀)
            config: 눌러 담기 차림
        """
        self.student = student
        self.teacher = teacher
        self.config = config or CompressionConfig()
        
        # 눌러 담기 도막을 좇는다
        self.history = {
            'stage': [],
            'accuracy': [],
            'size_mb': [],
            'sparsity': []
        }
    
    def compress(self,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 calibration_loader: Optional[torch.utils.data.DataLoader] = None
                 ) -> nn.Module:
        """
        온 눌러 담기 흐름을 돌린다.
        
        Args:
            train_loader: 익힘 자료
            test_loader: 따질 시험 자료
            calibration_loader: 익힘 뒤 수 줄이기에 쓸 눈금 맞추기 자료(None이면 train_loader을 쓴다)
            
        Returns:
            눌러 담은 모형
        """
        device = self.config.device
        model = self.student.to(device)
        
        # 처음 상태를 적는다
        self._log_state('initial', model, test_loader)
        
        # 1도막: 앎 옮기기
        if self.config.use_distillation and self.teacher is not None:
            print("\n" + "="*60)
            print("1도막: 앎 옮기기")
            print("="*60)
            model = self._distillation_stage(model, train_loader, test_loader)
            self._log_state('after_distillation', model, test_loader)
        
        # 2도막: 쳐내기
        print("\n" + "="*60)
        print("2도막: 쳐내기")
        print("="*60)
        model = self._pruning_stage(model, train_loader, test_loader)
        self._log_state('after_pruning', model, test_loader)
        
        # 3도막: 낮은 자리 쪼개기(골라 씀)
        if self.config.use_low_rank:
            print("\n" + "="*60)
            print("3도막: 낮은 자리 쪼개기")
            print("="*60)
            model = self._low_rank_stage(model, train_loader, test_loader)
            self._log_state('after_low_rank', model, test_loader)
        
        # 4도막: 수 줄이기
        print("\n" + "="*60)
        print("4도막: 수 줄이기")
        print("="*60)
        
        cal_loader = calibration_loader or train_loader
        
        if self.config.quantization_method == 'qat':
            model = self._qat_stage(model, train_loader, test_loader)
        else:
            model = self._ptq_stage(model, cal_loader)
        
        self._log_state('final', model, test_loader)
        
        # 간추림을 찍는다
        self._print_summary()
        
        return model
    
    def _distillation_stage(self,
                            model: nn.Module,
                            train_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """앎 옮기기 익힘을 돌린다."""
        device = self.config.device
        model = model.to(device)
        teacher = self.teacher.to(device)
        teacher.eval()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.distillation_epochs
        )
        
        T = self.config.temperature
        alpha = self.config.alpha
        
        for epoch in range(self.config.distillation_epochs):
            model.train()
            epoch_loss = 0.0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                with torch.no_grad():
                    teacher_logits = teacher(data)
                
                optimizer.zero_grad()
                student_logits = model(data)
                
                # 앎 옮기기 잃음
                hard_loss = F.cross_entropy(student_logits, target)
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T ** 2)
                
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                acc = self._evaluate(model, test_loader)
                print(f"앎 옮기기 {epoch+1}/{self.config.distillation_epochs}판, "
                      f"잃음: {epoch_loss/len(train_loader):.4f}, 맞음: {acc*100:.2f}%")
        
        return model
    
    def _pruning_stage(self,
                       model: nn.Module,
                       train_loader: torch.utils.data.DataLoader,
                       test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """곱게 맞추기를 곁들여 쳐낸다."""
        device = self.config.device
        model = model.to(device)
        
        if self.config.pruning_method == 'structured':
            model = self._structured_pruning(model, train_loader, test_loader)
        else:
            model = self._unstructured_pruning(model, train_loader, test_loader)
        
        return model
    
    def _structured_pruning(self,
                            model: nn.Module,
                            train_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """짜임새 있는(거르개) 쳐내기를 건다."""
        import torch.nn.utils.prune as prune
        
        # 과녁 성김을 이룰 몫을 셈한다
        amount = self.config.target_sparsity
        
        # 엮음 켜에 짜임새 있는 쳐내기를 건다
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
        
        # 곱게 맞춘다
        model = self._fine_tune(model, train_loader, test_loader, 
                               self.config.pruning_epochs)
        
        # 쳐낸 것을 굳힌다
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass
        
        return model
    
    def _unstructured_pruning(self,
                              model: nn.Module,
                              train_loader: torch.utils.data.DataLoader,
                              test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """짜임새 없는(짐) 쳐내기를 건다."""
        import torch.nn.utils.prune as prune
        
        amount = self.config.target_sparsity
        
        # 두루 걸친 짜임새 없는 쳐내기
        parameters_to_prune = [
            (module, 'weight') 
            for module in model.modules() 
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        # 가리개를 지키며 곱게 맞춘다
        masks = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight_mask'):
                    masks[name] = module.weight_mask.clone()
        
        model = self._fine_tune(model, train_loader, test_loader,
                               self.config.pruning_epochs, masks=masks)
        
        # 쳐낸 것을 굳힌다
        for module, _ in parameters_to_prune:
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
        
        return model
    
    def _low_rank_stage(self,
                        model: nn.Module,
                        train_loader: torch.utils.data.DataLoader,
                        test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """낮은 자리 쪼개기를 건다."""
        # 큰 선형 켜를 쪼갠 것으로 갈음한다
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                max_rank = min(module.in_features, module.out_features)
                rank = max(8, int(max_rank * self.config.rank_ratio))
                
                if rank < max_rank and module.in_features > 64:
                    # 쪼갠다
                    first, second = self._svd_factorize_linear(module, rank)
                    
                    # 모형 안에서 갈음한다
                    parts = name.split('.')
                    parent = model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], nn.Sequential(first, second))
        
        # 곱게 맞춘다
        model = self._fine_tune(model, train_loader, test_loader, 5)
        
        return model
    
    def _svd_factorize_linear(self, layer: nn.Linear, rank: int) -> Tuple[nn.Linear, nn.Linear]:
        """SVD로 선형 켜를 쪼갠다."""
        W = layer.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        U_k = U[:, :rank]
        S_k = S[:rank]
        V_k = Vh[:rank, :]
        
        first = nn.Linear(layer.in_features, rank, bias=False)
        first.weight.data = V_k
        
        second = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
        second.weight.data = U_k @ torch.diag(S_k)
        if layer.bias is not None:
            second.bias.data = layer.bias.data
        
        return first, second
    
    def _qat_stage(self,
                   model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader) -> nn.Module:
        """수 줄이기를 아는 익힘."""
        device = self.config.device
        model = model.to(device)
        model.train()
        
        # QAT을 마련한다
        model.qconfig = quant.get_default_qat_qconfig('fbgemm')
        quant.prepare_qat(model, inplace=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.1)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config.qat_epochs):
            model.train()
            
            # 판의 절반이 지나면 살피개를 얼린다
            if epoch >= self.config.qat_epochs // 2:
                model.apply(quant.disable_observer)
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                acc = self._evaluate(model, test_loader)
                print(f"QAT {epoch+1}/{self.config.qat_epochs}판, 맞음: {acc*100:.2f}%")
        
        # 수 줄인 모형으로 바꾼다
        model.eval()
        model_quantized = quant.convert(model.cpu(), inplace=False)
        
        return model_quantized
    
    def _ptq_stage(self,
                   model: nn.Module,
                   calibration_loader: torch.utils.data.DataLoader) -> nn.Module:
        """익힘 뒤 수 줄이기."""
        model.eval()
        
        # 움직이는 수 줄이기(더 단순하고 거의 다 잘 듣는다)
        model_quantized = quant.quantize_dynamic(
            model.cpu(),
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        return model_quantized
    
    def _fine_tune(self,
                   model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   epochs: int,
                   masks: Optional[Dict] = None) -> nn.Module:
        """가리개를 지키며(골라 씀) 모형을 곱게 맞춘다."""
        device = self.config.device
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.1)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # 가리개가 주어지면 지킨다
                if masks:
                    with torch.no_grad():
                        for name, module in model.named_modules():
                            if name in masks and hasattr(module, 'weight'):
                                module.weight.data *= masks[name].to(device)
            
            if (epoch + 1) % 2 == 0:
                acc = self._evaluate(model, test_loader)
                print(f"곱게 맞추기 {epoch+1}/{epochs}판, 맞음: {acc*100:.2f}%")
        
        return model
    
    def _evaluate(self, model: nn.Module, 
                  test_loader: torch.utils.data.DataLoader) -> float:
        """모형의 맞음을 따진다."""
        device = self.config.device
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def _get_model_size(self, model: nn.Module) -> float:
        """모형 크기를 MB으로 얻는다."""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def _get_sparsity(self, model: nn.Module) -> float:
        """모형의 성김을 셈한다."""
        total, zeros = 0, 0
        for param in model.parameters():
            if param.dim() > 1:
                total += param.numel()
                zeros += (param.data.abs() < 1e-8).sum().item()
        return zeros / total if total > 0 else 0.0
    
    def _log_state(self, stage: str, model: nn.Module,
                   test_loader: torch.utils.data.DataLoader):
        """눌러 담기 상태를 적는다."""
        acc = self._evaluate(model, test_loader)
        size = self._get_model_size(model)
        sparsity = self._get_sparsity(model)
        
        self.history['stage'].append(stage)
        self.history['accuracy'].append(acc)
        self.history['size_mb'].append(size)
        self.history['sparsity'].append(sparsity)
        
        print(f"\n[{stage}] 맞음: {acc*100:.2f}%, "
              f"크기: {size:.2f} MB, 성김: {sparsity*100:.1f}%")
    
    def _print_summary(self):
        """눌러 담기 간추림을 찍는다."""
        print("\n" + "="*70)
        print("눌러 담기 흐름 간추림")
        print("="*70)
        print(f"\n{'도막':<25} {'맞음':<12} {'크기 (MB)':<12} {'성김':<12}")
        print("-" * 70)
        
        for i, stage in enumerate(self.history['stage']):
            print(f"{stage:<25} {self.history['accuracy'][i]*100:>8.2f}%   "
                  f"{self.history['size_mb'][i]:>8.2f}     "
                  f"{self.history['sparsity'][i]*100:>8.1f}%")
        
        # 통틀어 눌러 담은 정도를 셈한다
        initial_size = self.history['size_mb'][0]
        final_size = self.history['size_mb'][-1]
        initial_acc = self.history['accuracy'][0]
        final_acc = self.history['accuracy'][-1]
        
        print("\n" + "="*70)
        print("통틀어 본 자")
        print("="*70)
        print(f"눌러 담은 견줌:    {initial_size/final_size:.1f}배")
        print(f"크기 줄어듦:       {(1 - final_size/initial_size)*100:.1f}%")
        print(f"맞음 바뀜:      {(final_acc - initial_acc)*100:+.2f}%")
        print("="*70)
```

## 쓰는 보기

```python
# 모형을 만든다
teacher = LargeTeacherModel()  # 미리 익힌 스승
student = SmallStudentModel()   # 눌러 담을 제자

# 미리 익힌 스승을 얹는다
teacher.load_state_dict(torch.load('teacher.pth'))

# 눌러 담기를 차린다
config = CompressionConfig(
    use_distillation=True,
    temperature=4.0,
    alpha=0.5,
    distillation_epochs=20,
    target_sparsity=0.7,
    pruning_method='structured',
    pruning_epochs=10,
    use_low_rank=False,
    quantization_method='qat',
    qat_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 흐름을 만들어 돌린다
pipeline = CompressionPipeline(student, teacher, config)
compressed_model = pipeline.compress(train_loader, test_loader)

# 내보낸다
torch.save(compressed_model.state_dict(), 'compressed_model.pth')
```

## 좋은 버릇

### 차림 길잡이

| 형편 | 앎 옮기기 | 쳐내기 | 낮은 자리 | 수 줄이기 |
|----------|-------------|---------|----------|--------------|
| 가장 크게 눌러 담기 | ✓ | 짜임새 있게 80% | ✓ | QAT INT8 |
| 고르게 | ✓ | 짜임새 있게 50% | ✗ | QAT INT8 |
| 빨리 내놓기 | ✗ | ✗ | ✗ | 익힘 뒤 수 줄이기 INT8 |
| 손전화/끝단 | ✓ | 짜임새 있게 70% | ✗ | QAT INT8/INT4 |
| 서비스개 | ✓ | 짜임새 없이 90% | ✓ | FP16 |

### 흐름의 탈 벌레잡기

1. **한 도막에서 맞음이 너무 떨어짐**: 그 도막을 덜 세게 한다
2. **수 줄이기가 어그러짐**: 받쳐 주지 않는 셈이 있는지 살피고 움직이는 수 줄이기를 쓴다
3. **쳐내도 빨라지지 않음**: 짜임새 있는 쳐내기로 바꾼다
4. **앎 옮기기가 듣지 않음**: 스승의 맞음을 따지고 온도를 손본다

## 살펴볼 거리

1. Polino, A., et al. "Model Compression via Distillation and Quantization." ICLR 2018.
2. Han, S., et al. "Deep Compression." ICLR 2016.
3. Cheng, Y., et al. "A Survey of Model Compression and Acceleration." IEEE Signal Processing 2020.

## 익힘 문제

**익힘 1.**
이 마디에서 다룬 다듬기 재주들을 맞음 잃음, 미루어 봄 빨라짐, 짜기의 번거로움으로 견주어 맞바꿈을 밝혀라.

??? success "익힘 1 풀이"
    재주마다 맞바꿈의 결이 다르다. 수 줄이기(INT8)은 흔히 2~4배 빨라지면서 맞음 잃음이 1% 미만이고, 틀이 받쳐 주므로 짜는 품이 가운데쯤이다. 쳐내기는 성김의 결에 따라 빨라짐이 들쭉날쭉하며(짜임새 있는 쳐내기가 쇠 붙임새에 더 맞다) 맞음 잃음은 1~3%이다. 앎 옮기기는 얼개 자체의 미루어 봄 값은 그대로 두되 더 작은 제자를 써서 2~10배로 눌러 담고 맞음 잃음은 1~5%이다. 신경 얼개 찾기는 가장 좋은 얼개를 찾아 주지만 찾는 데 엄청난 셈이 든다(GPU 수천 시간). 금융 쓰임에서는 받아들일 수 있는 맞음 잃음이 어긋남의 값에 매인다. $\square$

---

**익힘 2.**
단순한 앞먹임 그물에 익힘 뒤 수 줄이기(INT8)을 짜 넣고, 잣대 자료 꾸러미에서 맞음이 얼마나 떨어지고 미루어 봄이 얼마나 빨라지는지 재어라.

??? success "익힘 2 풀이"
    PyTorch의 수 줄이기 API을 쓴다. (1) float32 모형을 밑금 맞음까지 익힌다. (2) 움직이는 수 줄이기에는 `torch.quantization.quantize_dynamic`을 쓰고, 붙박인 수 줄이기에는 본보기 자료로 눈금을 맞춘다. (3) 미루어 보는 때(묶음 1000개의 평균)와 시험 꾸러미의 맞음을 잰다. 흔한 결과: CPU에서 1.5~3배 빨라지고, 움직이는 수 줄이기는 맞음이 0.5% 미만, 눈금 맞춘 붙박인 수 줄이기는 0.2% 미만 떨어진다. 모형 크기는 약 4배 줄어든다(FP32에서 INT8으로). 고갱이: 붙박인 수 줄이기에는 내놓을 자리의 자료를 잘 드러내는 눈금 맞추기 꾸러미가 있어야 한다. $\square$

---

**익힘 3.**
내놓은 모형의 자료 옮겨감, 뜻 옮겨감, 됨됨이 떨어짐을 짚어내는 서비스 지켜보기 얼개를 꾸며라. 자와 알림 문턱을 밝혀라.

??? success "익힘 3 풀이"
    세 켜를 지켜본다. (1) 자료 옮겨감: KS 시험이나 PSI(무리 든든함 지수)으로 들임 결의 분포를 좇는다. 어떤 결이든 PSI > 0.2이면 알린다. (2) 뜻 옮겨감: 미루어 봄 분포의 옮겨감과 (얻을 수 있으면) 참 이름표 분포를 좇는다. 미루어 봄의 평균이 밑금 동안에서 잣대 어긋남 2배 넘게 옮겨가면 알린다. (3) 모형 떨어짐: 굴러가는 창으로 살아 있는 맞음과 잃음을 좇는다. 맞음이 밑금보다 3% 넘게 떨어지거나 늦음이 약속을 넘으면(p99 > 50ms 따위) 알린다. Grafana으로 판을 만들고, Prometheus에 자를 담고, PagerDuty으로 알림을 보낸다. $\square$

---

**익힘 4.**
금융 거래 얼개의 늦음 요건이 웹 서비스와 밑바탕부터 다른 까닭을 밝혀라. 이것이 내놓기 다듬기 꾀에 어떻게 걸리는가?

??? success "익힘 4 풀이"
    웹 서비스는 100~500ms의 늦음과 이따금의 치솟음을 받아 준다. 거래 얼개는 붙박이로 1밀리초 아래(고빈도 거래에서는 흔히 100마이크로초 미만)여야 한다. 그래서 다듬는 꾀가 달라진다. (1) 쓰레기 치우기의 멈춤을 없앤다(파이썬 대신 C++ 미루어 봄). (2) 기억을 미리 다 잡아 둔다(그때그때 잡지 않는다). (3) 실을 알맹이에 붙박는다(자리 바꿈을 없앤다). (4) 늦음이 가장 걸리는 길목에는 FPGA이나 ASIC을 쓴다. (5) 수 줄이기는 있어야 하되 붙박이지 않은 반올림을 들여서는 안 된다. 묶음 미루어 봄은 쓸 수 없다(판단 하나하나가 늦음에 걸린다). 내놓기 더미는 나름보다 가장 나쁜 자리의 늦음(p99.9)을 앞세운다. $\square$
