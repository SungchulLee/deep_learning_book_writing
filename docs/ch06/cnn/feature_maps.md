# 특징 맵

**특징 맵**은 입력에 합성곱 필터를 적용한 결과로, 어떤 무늬(특징)가 어디에서 얼마나 강하게 잡혔는지를 나타내는 활성값의 공간 격자이다. 특징 맵은 CNN의 근본적인 중간 표현이며, 데이터가 신경망을 지날수록 점점 더 추상적인 정보를 담는다.

특징 맵을 이해하는 일은 다음에 꼭 필요하다.

1. **구조 설계**: 층마다 알맞은 채널 차원과 공간 해상도 고르기
2. **자원 어림**: 주어진 구조에 필요한 메모리와 매개변수 계산하기
3. **벌레잡이와 해석**: 단계마다 신경망이 무엇을 배웠는지 그려 보기
4. **성능 최적화**: 층에 걸친 계산 병목 찾아내기

---

## 1. 특징 맵의 기하

### 텐서 모양의 관행

PyTorch에서 특징 맵 텐서의 모양은 $(N, C, H, W)$이다.

| 차원 | 기호 | 뜻 |
|-----------|--------|---------|
| 배치 | $N$ | 한꺼번에 처리하는 표본의 수 |
| 채널 | $C$ | 특징 맵의 수 (깊이) |
| 높이 | $H$ | 공간적 높이 |
| 너비 | $W$ | 공간적 너비 |

```python
import torch
import torch.nn as nn

# 크기가 224×224인 RGB 이미지 8장의 배치
x = torch.randn(8, 3, 224, 224)

# 첫 합성곱 층 뒤: 크기가 224×224인 특징 맵 64개
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
feat1 = conv1(x)
print(f"Input:  N={x.shape[0]}, C={x.shape[1]}, H={x.shape[2]}, W={x.shape[3]}")
print(f"After conv1: N={feat1.shape[0]}, C={feat1.shape[1]}, H={feat1.shape[2]}, W={feat1.shape[3]}")
# 입력:  N=8, C=3, H=224, W=224
# conv1 뒤: N=8, C=64, H=224, W=224
```

**출력:**

```
Input:  N=8, C=3, H=224, W=224
After conv1: N=8, C=64, H=224, W=224
```

### 공간 차원과 채널 차원

특징 맵은 서로 다른 두 종류의 정보를 담는다.

- **공간 차원** ($H \times W$): 특징이 이미지의 *어디*에 있는가
- **채널 차원** ($C$): 자리마다 *어떤* 특징이 잡혔는가

채널마다 서로 다른 학습된 필터에 대응하여 저마다 다른 무늬를 잡는다. 공간 격자는 잡아낸 특징들의 상대적인 자리를 지켜 준다.

```
Feature Map Stack (C×H×W):

Channel 0 (e.g., horizontal edges):    Channel 1 (e.g., vertical edges):
┌─────────────────┐                    ┌─────────────────┐
│ 0.0  0.0  0.0   │                    │ 0.0  0.8  0.0   │
│ 0.9  0.8  0.7   │                    │ 0.0  0.9  0.0   │
│ 0.0  0.0  0.0   │                    │ 0.0  0.7  0.0   │
└─────────────────┘                    └─────────────────┘

At each spatial position (i,j), the vector across channels
forms a local feature descriptor.
```

---

## 2. 신경망을 지나며 특징 맵이 달라지는 방식

### 특징의 위계

데이터가 CNN을 지나면 특징 맵이 특유의 변화를 겪는다.

```
Input Image (3×224×224)
        │
        ▼
    Conv Block 1 ──→ 64 × 224 × 224   (edges, colors, simple textures)
        │
        ▼ (downsample)
    Conv Block 2 ──→ 128 × 112 × 112  (corners, texture combinations)
        │
        ▼ (downsample)
    Conv Block 3 ──→ 256 × 56 × 56    (parts: eyes, wheels, windows)
        │
        ▼ (downsample)
    Conv Block 4 ──→ 512 × 28 × 28    (objects, semantic concepts)
        │
        ▼ (downsample)
    Conv Block 5 ──→ 512 × 14 × 14    (scene-level, abstract)
        │
        ▼ (global average pooling)
    Feature Vector ──→ 512 × 1 × 1
```

그 방식은 한결같다. **공간 해상도는 낮아지고** **채널 수는 늘어난다**. 이는 세밀한 공간 정보에서 풍부한 의미적 추상으로 옮겨 가는 흐름을 드러낸다.

### 해상도와 의미 사이의 맞바꿈

| 성질 | 앞쪽 층 | 깊은 층 |
|----------|-------------|-------------|
| 공간 해상도 | 높음 ($224 \times 224$) | 낮음 ($7 \times 7$) |
| 채널 수 | 적음 (64) | 많음 (512 이상) |
| 특징의 종류 | 저수준 (모서리, 질감) | 고수준 (물체, 장면) |
| 평행 이동에 대한 민감도 | 높음 (자리를 정확히 짚음) | 낮음 (위치에 무관) |
| 수용 영역 | 작음 (지역 맥락) | 큼 (전역 맥락) |

---

## 3. 매개변수와 메모리 분석

### 층마다의 매개변수 수

핵 크기가 $K$인 합성곱 층에 대해 다음과 같다.

$$\text{Parameters} = C_{out} \times C_{in} \times K^2 + C_{out}$$

### 층마다의 활성값 메모리

특징 맵 하나를 저장하는 데 드는 메모리는 다음과 같다.

$$\text{Memory (elements)} = N \times C \times H \times W$$

$$\text{Memory (bytes)} = N \times C \times H \times W \times \text{bytes per element}$$

float32(원소당 4바이트)에서는 $64 \times 224 \times 224$ 특징 맵 하나가 약 $64 \times 224 \times 224 \times 4 \approx 12.3$MB를 쓴다.

### 층별 분석

```python
import torch
import torch.nn as nn

def analyze_feature_maps(model, input_shape=(1, 3, 224, 224)):
    """
    모델을 지나며 특징 맵의 모양과 매개변수와 메모리를 분석한다.
    """
    x = torch.randn(*input_shape)
    
    print(f"{'Layer':<30} {'Output Shape':<25} {'Params':>12} {'Memory (MB)':>12}")
    print("-" * 82)
    print(f"{'Input':<30} {str(list(x.shape)):<25} {'—':>12} {x.numel()*4/1e6:>12.2f}")
    
    total_params = 0
    total_memory = x.numel() * 4  # 입력 메모리
    
    for name, layer in model.named_children():
        x = layer(x)
        params = sum(p.numel() for p in layer.parameters())
        mem_mb = x.numel() * 4 / 1e6
        total_params += params
        total_memory += x.numel() * 4
        
        print(f"{name:<30} {str(list(x.shape)):<25} {params:>12,} {mem_mb:>12.2f}")
    
    print("-" * 82)
    print(f"{'Total':<30} {'':25} {total_params:>12,} {total_memory/1e6:>12.2f}")
    
    return x

# 예: VGG 방식 특징 추출기
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),    # 블록 1
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    
    nn.Conv2d(64, 128, 3, padding=1),  # 블록 2
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    
    nn.Conv2d(128, 256, 3, padding=1), # 블록 3
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
)

# 출력을 깔끔하게 하려고 이름 붙인 Sequential 사용
named_model = nn.Sequential()
names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3']
for name, layer in zip(names, model):
    named_model.add_module(name, layer)

analyze_feature_maps(named_model)
```

**출력:**

```
Layer                          Output Shape                    Params  Memory (MB)
----------------------------------------------------------------------------------
Input                          [1, 3, 224, 224]                     —         0.60
conv1_1                        [1, 64, 224, 224]                1,792        12.85
relu1_1                        [1, 64, 224, 224]                    0        12.85
conv1_2                        [1, 64, 224, 224]               36,928        12.85
relu1_2                        [1, 64, 224, 224]                    0        12.85
pool1                          [1, 64, 112, 112]                    0         3.21
conv2_1                        [1, 128, 112, 112]              73,856         6.42
relu2_1                        [1, 128, 112, 112]                   0         6.42
conv2_2                        [1, 128, 112, 112]             147,584         6.42
relu2_2                        [1, 128, 112, 112]                   0         6.42
pool2                          [1, 128, 56, 56]                     0         1.61
conv3_1                        [1, 256, 56, 56]               295,168         3.21
relu3_1                        [1, 256, 56, 56]                     0         3.21
conv3_2                        [1, 256, 56, 56]               590,080         3.21
relu3_2                        [1, 256, 56, 56]                     0         3.21
pool3                          [1, 256, 28, 28]                     0         0.80
----------------------------------------------------------------------------------
Total                                                       1,145,408        96.14
```

---

## 4. "채널은 두 배, 해상도는 절반" 방식

요즘 CNN 구조는 한결같은 설계 방식을 따른다. (보폭 2 합성곱이나 풀링으로) 공간 해상도가 절반이 되면 채널 수를 두 배로 늘린다. 그러면 층마다의 계산 비용이 대체로 일정하게 유지된다.

$$\text{FLOPs} \propto C \times C \times K^2 \times H \times W$$

$C$을 두 배로 하고 $H$과 $W$을 각각 절반으로 하면 다음과 같다.

$$2C \times 2C \times K^2 \times \frac{H}{2} \times \frac{W}{2} = C^2 K^2 HW$$

부동소수점 연산 수가 거의 일정하게 남아 신경망 전체에 걸쳐 계산량이 고르게 퍼진다.

```python
import torch.nn as nn

# ResNet 방식의 채널 변화
stages = [
    ("Stage 1", 64,  56, 56),   # 줄기 뒤
    ("Stage 2", 128, 28, 28),   # 채널 2배, 해상도 0.5배
    ("Stage 3", 256, 14, 14),   # 채널 4배, 해상도 0.25배
    ("Stage 4", 512, 7,  7),    # 채널 8배, 해상도 0.125배
]

print(f"{'Stage':<12} {'Channels':>8} {'Resolution':>12} {'Elements':>12} {'Relative':>10}")
print("-" * 60)
base_elements = None
for name, c, h, w in stages:
    elements = c * h * w
    if base_elements is None:
        base_elements = elements
    print(f"{name:<12} {c:>8} {f'{h}×{w}':>12} {elements:>12,} {elements/base_elements:>10.2f}×")
```

**출력:**

```
Stage        Channels   Resolution     Elements   Relative
------------------------------------------------------------
Stage 1            64        56×56      200,704       1.00×
Stage 2           128        28×28      100,352       0.50×
Stage 3           256        14×14       50,176       0.25×
Stage 4           512          7×7       25,088       0.12×
```

---

## 5. 특징 맵 시각화

### 학습된 특징 그려 보기

특징 맵을 살펴보면 층마다 무엇을 잡아내는지 알 수 있다.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_maps(model, image, layer_name, max_channels=16):
    """
    특정 층의 특징 맵을 그린다.
    
    인수:
        model: CNN 모델
        image: 입력 텐서 (1, C, H, W)
        layer_name: 그릴 층의 이름
        max_channels: 보일 채널의 최대 개수
    """
    # 특징 맵을 붙잡으려고 훅 등록
    features = {}
    def hook(module, input, output):
        features['output'] = output.detach()
    
    # 대상 층을 찾아 훅 걸기
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break
    
    # 순전파
    model.eval()
    with torch.no_grad():
        _ = model(image)
    
    handle.remove()
    
    # 시각화한다
    feat = features['output'].squeeze(0)  # 배치 차원 없애기
    num_channels = min(feat.shape[0], max_channels)
    
    rows = int(np.ceil(num_channels / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
    
    for i in range(num_channels):
        axes[i].imshow(feat[i].cpu().numpy(), cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
    
    # 쓰지 않는 부분 그림 감추기
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps at {layer_name} ({feat.shape[0]} channels, '
                 f'{feat.shape[1]}×{feat.shape[2]})')
    plt.tight_layout()
    plt.savefig(f'feature_maps_{layer_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 채널의 통계량

```python
def feature_map_statistics(model, loader, layer_name, num_batches=10):
    """
    데이터셋 전체에 대해 한 층의 활성값 통계를 계산한다.
    죽은 뉴런, 포화, 분포 문제를 살피는 데 쓸모가 있다.
    """
    features_list = []
    
    def hook(module, input, output):
        features_list.append(output.detach())
    
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook)
            break
    
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            _ = model(images)
    
    handle.remove()
    
    # 이어 붙인 뒤 분석
    all_features = torch.cat(features_list, dim=0)
    
    # 채널별 통계량
    channel_means = all_features.mean(dim=(0, 2, 3))
    channel_stds = all_features.std(dim=(0, 2, 3))
    channel_sparsity = (all_features == 0).float().mean(dim=(0, 2, 3))
    
    print(f"Layer: {layer_name}")
    print(f"  Feature map shape: {list(all_features.shape[1:])}")
    print(f"  Mean activation: {channel_means.mean():.4f} ± {channel_means.std():.4f}")
    print(f"  Std activation: {channel_stds.mean():.4f}")
    print(f"  Sparsity (% zeros): {channel_sparsity.mean():.1%}")
    print(f"  Dead channels (>99% zero): {(channel_sparsity > 0.99).sum().item()}")
    
    return channel_means, channel_stds, channel_sparsity
```

---

## 6. 다채널 합성곱에서의 특징 맵

### 합성곱 하나가 특징 맵을 만드는 방식

출력 채널마다 모든 입력 채널에 걸친 서로 다른 3차원 핵이 그것을 만든다. 핵이 공간 차원 위를 미끄러지며 자리마다 내적을 계산한다.

$$\text{Feature Map}[o, i, j] = \sum_{c=0}^{C_{in}-1} \sum_{m,n} X[c, i+m, j+n] \cdot W[o, c, m, n] + b[o]$$

$C_{out}$개의 핵은 저마다 모든 입력 채널에 걸친 서로 다른 공간 무늬를 한꺼번에 잡는 법을 배운다.

### 채널을 섞는 1×1 합성곱

$1 \times 1$ 합성곱은 공간적인 거르기를 전혀 하지 않고 공간 위치마다 채널 차원에만 작용한다.

$$Y[o, i, j] = \sum_{c=0}^{C_{in}-1} X[c, i, j] \cdot W[o, c] + b[o]$$

이는 공간 위치마다 공유된 완전 연결층을 따로 적용하는 것과 같다. 쓰임새는 다음과 같다.

- **채널 줄이기**: $C_{in}$을 더 작은 $C_{out}$으로 줄인다 (병목)
- **채널 늘리기**: 채널 차원을 키운다
- **채널 사이의 학습**: 특징 맵에 걸친 정보를 엮는다
- **비선형 더하기**: 뒤에 활성화 함수를 붙일 때

```python
# 1×1 합성곱: 채널만 섞기
channel_mixer = nn.Conv2d(256, 64, kernel_size=1)  # 채널 256 → 64개

x = torch.randn(1, 256, 14, 14)
out = channel_mixer(x)
print(f"Channel reduction: {x.shape} → {out.shape}")
# 채널 줄이기: [1, 256, 14, 14] → [1, 64, 14, 14]

params = sum(p.numel() for p in channel_mixer.parameters())
print(f"Parameters: {params:,}")  # 256×64 + 64 = 16,448
```

**출력:**

```
Channel reduction: torch.Size([1, 256, 14, 14]) → torch.Size([1, 64, 14, 14])
Parameters: 16,448
```

---

## 7. 특징 맵 다시 쓰기: 건너뛰기 연결

건너뛰기(잔차) 연결은 앞쪽 층의 특징 맵이 중간 층을 건너뛰어 뒤쪽 특징 맵에 더해지거나 이어 붙게 해 준다.

### 더하는 건너뛰기 (ResNet)

$$\mathbf{Y} = F(\mathbf{X}) + \mathbf{X}$$

공간 차원과 채널 수가 맞아야 한다.

### 이어 붙이는 건너뛰기 (DenseNet, U-Net)

$$\mathbf{Y} = [\mathbf{X}, F(\mathbf{X})]$$

채널 차원을 따라 이어 붙여 갈수록 풍부한 특징 표현을 만든다.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """더하는 건너뛰기: 특징 맵의 차원을 지킨다."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)  # 특징 맵 다시 쓰기

class DenseBlock(nn.Module):
    """이어 붙이는 건너뛰기: 특징 맵이 쌓인다."""
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=False),
            ))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)  # 모든 특징 맵을 이어 붙임
```

---

## 8. 핵심 정리

1. **특징 맵은 3차원 텐서**($C \times H \times W$)이며 채널은 *무엇*을, 공간 차원은 *어디*를 담는다
2. **점진적인 추상화**: 신경망은 해상도가 높고 채널이 적은 입력을 해상도가 낮고 채널이 많은 표현으로 바꾼다
3. **채널 두 배, 해상도 절반 방식**은 단계에 걸쳐 계산량을 고르게 지킨다
4. **$1 \times 1$ 합성곱**은 공간 계산 없이 채널 차원을 효율적으로 다루게 해 준다
5. **활성값 메모리가 매개변수 메모리를 넘어설 때가 많다.** 특히 해상도가 높은 입력에서 그러하며, GPU 메모리를 가늠할 때 중요한 문제이다
6. 훅으로 하는 **특징 맵 시각화**는 벌레잡이와 해석에 꼭 필요한 도구이다

---

## 연습문제

**연습문제 1.**
CNN의 앞쪽 층과 깊은 층의 특징 맵이 나타내는 바가 어떻게 다른지 설명하라.

??? success "연습문제 1 풀이"
    앞쪽 층은 모서리, 꼭짓점, 색, 질감 같은 저수준 특징을 담는다. 중간 층은 저수준 특징이 모여 이루어진 부분(눈, 바퀴)을 담는다. 깊은 층은 얼굴이나 자동차 같은 고수준 의미 개념을 담는다. 이런 위계적인 특징 추출이 깊은 CNN의 핵심 강점이다.

---

**연습문제 2.**
모양이 $(3, 224, 224)$인 입력이 필터 64개, 핵 $7\times7$, 보폭 2, 덧대기 3인 합성곱 층을 지난 뒤의 출력 특징 맵 차원을 계산하라.

??? success "연습문제 2 풀이"
    출력: $(64, \lfloor(224+6-7)/2\rfloor+1, \lfloor(224+6-7)/2\rfloor+1) = (64, 112, 112)$.

---

**연습문제 3.**
순전파 훅을 써서 미리 학습된 CNN의 특징 맵 시각화를 구현하라.

??? success "연습문제 3 풀이"
    ```python
    activations = {}
    def hook(name):
        def fn(module, input, output):
            activations[name] = output.detach()
        return fn
    model.layer1.register_forward_hook(hook('layer1'))
    model(img)
    plt.imshow(activations['layer1'][0, 0].numpy())
    ```

---

**연습문제 4.**
$1 \times 1$ 합성곱(점별 합성곱)은 특징 맵을 어떻게 바꾸는가? 그 쓰임새는 무엇인가?

??? success "연습문제 4 풀이"
    $1\times1$ 합성곱은 공간 차원을 바꾸지 않고 공간 위치마다 채널을 섞는다. (1) 채널 차원 줄이기(ResNet의 병목 층), (2) 채널 늘리기, (3) (활성화와 함께) 비선형 더하기에 쓰인다. 공간 위치마다 공유된 MLP를 따로 적용하는 것과 같다.

## 정리하며

| 개념 | 설명 |
|---------|-------------|
| **모양** | $(N, C, H, W)$: 배치, 채널, 높이, 너비 |
| **채널** | 채널 하나 = 학습된 특징 검출기 하나의 공간적 반응 |
| **위계** | 앞쪽: 모서리와 질감 → 깊은 쪽: 물체와 의미 |
| **설계 방식** | 해상도를 절반으로 할 때 채널을 두 배로 |
| **$1 \times 1$ 합성곱** | 채널만 섞고 공간적 거르기는 하지 않음 |
| **건너뛰기 연결** | 앞쪽 특징 맵을 다시 씀 (더하거나 이어 붙임) |
| **시각화** | 훅으로 뽑아내면 학습된 표현이 드러남 |

**참고 문헌**

1. Zeiler, M. D., & Fergus, R. (2014). "Visualizing and Understanding Convolutional Networks." *ECCV*.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.

3. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). "Densely Connected Convolutional Networks." *CVPR*.

4. Lin, M., Chen, Q., & Yan, S. (2014). "Network In Network." *ICLR*. (1×1 합성곱을 처음 소개한 논문)

5. Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." *ICLR*.
