# 수용 영역

합성곱 신경망에서 뉴런의 **수용 영역**은 그 뉴런의 활성값에 영향을 주는 입력 이미지의 영역이다. 수용 영역을 이해하는 일은 CNN 설계의 바탕인데, 다음을 정하기 때문이다.

1. **쓸 수 있는 맥락**: 특징 검출기마다 얼마만큼의 공간 맥락에 닿을 수 있는가
2. **특징의 규모**: 신경망이 어느 크기의 무늬를 알아볼 수 있는가
3. **구조 결정**: 알맞은 핵 크기, 신경망의 깊이, 팽창률의 선택
4. **과제 적합성**: 신경망이 필요한 공간 관계를 붙잡을 수 있는가

이 절은 수용 영역의 계산을 엄밀하게 다루고, 실제로 쓸 수 있는 계산 도구와 설계에 주는 함의를 살핀다.

---

## 1. 정의

### 지역 수용 영역

뉴런의 **지역 수용 영역**은 그 뉴런에 곧바로 이어진 앞 층 뉴런들의 모임이다. 핵 크기가 $K$인 합성곱에서 지역 수용 영역은 $K \times K$이다.

```
Layer 1 output neuron sees a 3×3 region of input:

Input:              Layer 1:
■ ■ ■ · ·           ○ · ·
■ ■ ■ · ·     →     · · ·
■ ■ ■ · ·           · · ·
· · · · ·
· · · · ·

Local receptive field = 3×3
```

### 전역 수용 영역

**전역 수용 영역**(또는 그냥 "수용 영역")은 뉴런의 출력에 영향을 줄 수 있는 **원래 입력**의 영역이다. 이는 여러 층을 지나며 쌓인다.

```
Input:                Layer 1:              Layer 2:
■ ■ ■ ■ ■             ■ ■ ■                 ○
■ ■ ■ ■ ■    3×3      ■ ■ ■      3×3        
■ ■ ■ ■ ■    →        ■ ■ ■      →          
■ ■ ■ ■ ■                                   
■ ■ ■ ■ ■                                   

Layer 1 RF = 3×3     Layer 2 RF = 5×5
```

### 실효 수용 영역 (ERF)

**이론적인** 수용 영역과 **실효** 수용 영역은 같지 않다. 실제로 이론적인 수용 영역 안의 화소가 모두 뉴런의 활성값에 똑같이 이바지하지는 않는다.

- **가운데 화소**는 영향이 크다 (신경망을 지나는 길이 더 많다)
- **가장자리 화소**는 영향이 작다 (이바지하는 길이 더 적다)

실효 수용 영역은 대개 뉴런을 중심으로 하는 가우스 분포를 따르며, 이론적인 최댓값보다 훨씬 작을 때가 많다.

---

## 2. 수학적 정식화

### 한 층의 수용 영역

다음과 같은 합성곱 층 하나에 대해 살펴보자.

- 핵 크기: $K$
- 보폭: $s$
- 팽창률: $d$

(팽창을 감안한) 실효 핵 크기는 다음과 같다.

$$K_{\text{eff}} = d \cdot (K - 1) + 1$$

표준 합성곱($d = 1$)에서는 $K_{\text{eff}} = K$이다.

### 여러 층의 수용 영역

합성곱 층을 쌓으면 수용 영역이 다음에 따라 자란다.

$$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

여기서 각 기호는 다음과 같다.

- $r_l$: $l$번째 층 뒤의 수용 영역
- $r_0 = 1$: 처음의 수용 영역 (화소 하나)
- $K_l$: $l$번째 층의 핵 크기
- $d_l$: $l$번째 층의 팽창률
- $s_i$: $i$번째 층의 보폭
- $\prod_{i=1}^{l-1} s_i$: 누적 보폭 ("뜀"이라고도 한다)

### 누적 보폭 인수

인수 $\prod_{i=1}^{l-1} s_i$(누적 보폭 또는 "뜀")은 수용 영역이 자라는 방식을 이해하는 데 매우 중요하다.

1. 앞선 보폭 하나하나가 뒤따르는 층의 수용 영역 증가를 **키운다**
2. 보폭 2인 층은 그 뒤 모든 층의 "걸음 크기"를 두 배로 만든다
3. 그래서 앞쪽에서 하향 표본화를 하면 깊은 층의 수용 영역이 크게 넓어진다

### 간단히 한 식

**한결같은 구조**(핵 $k$, 보폭 $s$이 처음부터 끝까지 같음)에서는 다음과 같다.

$$r_L = 1 + \sum_{l=1}^{L} (k - 1) \cdot d_l \cdot s^{l-1}$$

**모두 보폭 1인 합성곱**(하향 표본화 없음)에서는 다음과 같다.

$$r_L = 1 + L \cdot (k - 1)$$

이는 보폭이 1일 때 수용 영역이 깊이에 따라 선형으로 자람을 보여 준다.

---

## 3. 수용 영역 계산하기

### 핵심 파이썬 구현

```python
def compute_receptive_field(layers):
    """
    합성곱과 풀링 층의 나열에 대해 수용 영역을 계산한다.
    
    인수:
        layers: 'kernel', 'stride', 'dilation' 열쇠를 가진 사전들의 리스트
    
    반환값:
        수용 영역의 크기와 뜀(누적 보폭)
    """
    rf = 1  # 화소 하나에서 시작
    jump = 1  # 누적 보폭
    
    for layer in layers:
        k = layer.get('kernel', 1)
        s = layer.get('stride', 1)
        d = layer.get('dilation', 1)
        
        # 팽창을 감안한 실효 핵 크기
        k_eff = d * (k - 1) + 1
        
        # 수용 영역 갱신
        rf = rf + (k_eff - 1) * jump
        
        # 뜀(누적 보폭) 갱신
        jump = jump * s
    
    return rf, jump

# 예: VGG 방식 신경망
vgg_layers = [
    {'kernel': 3, 'stride': 1},  # 3×3 합성곱
    {'kernel': 3, 'stride': 1},  # 3×3 합성곱
    {'kernel': 2, 'stride': 2},  # 2×2 최댓값 풀링
    {'kernel': 3, 'stride': 1},  # 3×3 합성곱
    {'kernel': 3, 'stride': 1},  # 3×3 합성곱
    {'kernel': 2, 'stride': 2},  # 2×2 최댓값 풀링
]

rf, jump = compute_receptive_field(vgg_layers)
print(f"Receptive field: {rf}×{rf}")  # 22×22
print(f"Jump (output stride): {jump}")  # 4
```

### 위치까지 좇는 층별 분석

정확한 위치를 잡으려면 다음도 함께 좇는다.

- **뜀**: 특징 맵에서 한 칸 움직이는 것이 입력 화소 몇 개에 해당하는가
- **시작**: 입력 좌표로 나타낸 첫 특징의 중심

```python
def analyze_receptive_field(layers, layer_names=None):
    """
    층마다의 자세한 수용 영역 분석.
    
    수용 영역의 크기와 뜀과 중심 자리를 좇는다.
    """
    rf = 1
    jump = 1
    start = 0.5  # 첫 특징의 중심 (0부터 세는 색인)
    
    print(f"{'Layer':<20} {'Kernel':<8} {'Stride':<8} {'Dilation':<8} {'RF':<8} {'Jump':<8}")
    print("-" * 68)
    print(f"{'Input':<20} {'-':<8} {'-':<8} {'-':<8} {rf:<8} {jump:<8}")
    
    for i, layer in enumerate(layers):
        k = layer.get('kernel', 1)
        s = layer.get('stride', 1)
        d = layer.get('dilation', 1)
        
        # 실효 핵 크기
        k_eff = d * (k - 1) + 1
        
        # 수용 영역 갱신
        rf = rf + (k_eff - 1) * jump
        
        # 시작 자리 갱신
        start = start + ((k_eff - 1) / 2) * jump
        
        # 뜀 갱신
        jump = jump * s
        
        name = layer_names[i] if layer_names else f"Layer {i+1}"
        print(f"{name:<20} {k:<8} {s:<8} {d:<8} {rf:<8} {jump:<8}")
    
    return rf, jump

# ResNet 방식의 앞쪽 몇 층 분석
resnet_layers = [
    {'kernel': 7, 'stride': 2},   # Conv1
    {'kernel': 3, 'stride': 2},   # 최댓값 풀링
    {'kernel': 3, 'stride': 1},   # 블록1 Conv1
    {'kernel': 3, 'stride': 1},   # 블록1 Conv2
    {'kernel': 3, 'stride': 2},   # 블록2 Conv1 (보폭)
    {'kernel': 3, 'stride': 1},   # 블록2 Conv2
]

names = ['Conv1 7×7/2', 'MaxPool 3×3/2', 'Block1 3×3', 'Block1 3×3',
         'Block2 3×3/2', 'Block2 3×3']

analyze_receptive_field(resnet_layers, names)
```

**출력:**
```
Layer                Kernel   Stride   Dilation RF       Jump    
--------------------------------------------------------------------
Input                -        -        -        1        1       
Conv1 7×7/2          7        2        1        7        2       
MaxPool 3×3/2        3        2        1        11       4       
Block1 3×3           3        1        1        19       4       
Block1 3×3           3        1        1        27       4       
Block2 3×3/2         3        2        1        35       8       
Block2 3×3           3        1        1        51       8       
```

---

## 4. 수용 영역을 넓히는 전략

### 전략 1: 핵 크기 키우기

**장점**: 수용 영역이 곧바로 넓어진다
**단점**: 매개변수가 제곱으로 늘고 계산 비용이 든다

$$\text{Parameters} \propto K^2$$

### 전략 2: 깊이 늘리기 (권장)

**장점**: 매개변수가 선형으로 늘고 특징이 층층이 조합된다
**단점**: 기울기 소실, 최적화의 어려움

3×3 층 두 개는 5×5 층 하나와 수용 영역이 같다.

$$r_{\text{two } 3\times3} = 1 + (3-1) + (3-1) = 5$$

$$r_{\text{one } 5\times5} = 1 + (5-1) = 5$$

그런데 **매개변수는 더 적다**.

- 3×3 층 두 개: 채널마다 가중치 $2 \times 3^2 = 18$개
- 5×5 층 하나: 채널마다 가중치 $5^2 = 25$개

게다가 **비선형이 더 많다**. ReLU가 하나가 아니라 둘이다.

### 전략 3: 보폭 늘리기

**장점**: 수용 영역이 빨리 넓어지고 계산이 줄어든다
**단점**: 정보가 사라지고 공간 해상도가 낮아진다

### 전략 4: 팽창 쓰기 (효율적)

**장점**: 매개변수는 선형으로 늘면서 수용 영역은 지수로 넓어진다
**단점**: 꼼꼼히 설계하지 않으면 격자 무늬 흠이 생긴다

```python
def compare_receptive_fields():
    """표준 합성곱과 팽창 합성곱을 견준다."""
    
    # 표준 3×3 합성곱 (5층)
    standard = [{'kernel': 3, 'stride': 1} for _ in range(5)]
    
    # 팽창률을 키워 가는 팽창 합성곱
    dilated = [
        {'kernel': 3, 'stride': 1, 'dilation': 1},
        {'kernel': 3, 'stride': 1, 'dilation': 2},
        {'kernel': 3, 'stride': 1, 'dilation': 4},
        {'kernel': 3, 'stride': 1, 'dilation': 8},
        {'kernel': 3, 'stride': 1, 'dilation': 16},
    ]
    
    rf_standard, _ = compute_receptive_field(standard)
    rf_dilated, _ = compute_receptive_field(dilated)
    
    print(f"Standard 5 layers (3×3): RF = {rf_standard}")   # 11
    print(f"Dilated 5 layers (d=1,2,4,8,16): RF = {rf_dilated}")  # 63
    print(f"Ratio: {rf_dilated / rf_standard:.1f}x larger with same parameters!")

compare_receptive_fields()
```

### WaveNet 방식의 지수 팽창

```python
def wavenet_receptive_field(num_blocks, layers_per_block, kernel_size=2):
    """
    WaveNet 방식 구조의 수용 영역을 계산한다.
    
    팽창 방식: 블록마다 1, 2, 4, 8, …을 되풀이한다.
    """
    layers = []
    for block in range(num_blocks):
        for i in range(layers_per_block):
            dilation = 2 ** i
            layers.append({
                'kernel': kernel_size,
                'stride': 1,
                'dilation': dilation
            })
    
    rf, _ = compute_receptive_field(layers)
    return rf, len(layers)

# 층 10개짜리 블록 3개로 이루어진 WaveNet
rf, total_layers = wavenet_receptive_field(3, 10, kernel_size=2)
print(f"WaveNet (3 blocks × 10 layers): {total_layers} layers, RF = {rf}")
# RF = 3 × (2^10 - 1) + 1 = 3069
```

---

## 5. 실효 수용 영역 (ERF)

### 이론과 실제의 간격

Luo 등(2016)은 실효 수용 영역이 가우스 분포를 따름을 보였다.

$$\text{ERF}(i, j) \propto \exp\left(-\frac{(i - c_i)^2 + (j - c_j)^2}{2\sigma^2}\right)$$

여기서 $(c_i, c_j)$은 중심이고 $\sigma$은 신경망의 깊이에 달려 있다.

### 실험에서 드러난 핵심 사실

1. **실효 수용 영역이 이론적 수용 영역보다 훨씬 작다**: 실효 수용 영역은 이론적인 최댓값보다 훨씬 작을 때가 많다
2. **실효 수용 영역은 $\sqrt{\text{깊이}}$에 비례해 자란다**: 깊이에 선형으로 자라지 않는다
3. **학습이 실효 수용 영역을 넓힌다**: 학습이 이어질수록 신경망이 더 넓은 맥락을 쓰게 된다
4. **건너뛰기 연결이 돕는다**: 잔차 연결이 실효 수용 영역을 크게 넓힌다
5. **배치 정규화**: 특징의 통계량을 고르게 하여 실효 수용 영역을 넓힐 수 있다

### 실효 수용 영역에 영향을 주는 요인

| 요인 | 실효 수용 영역에 대한 영향 |
|--------|--------------|
| ReLU 활성화 | 좁힌다 (죽은 뉴런이 기울기의 흐름을 막는다) |
| 배치 정규화 | 넓힐 수 있다 |
| 건너뛰기 연결 | 넓힌다 (기울기가 곧바로 흐르는 길) |
| 어텐션 장치 | 맥락에 따라 그때그때 달라진다 |
| 초기화 | 학습 초반의 실효 수용 영역에 영향을 준다 |
| 학습 기간 | 학습이 이어지면 대개 넓어진다 |

### 실효 수용 영역 재기

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def measure_effective_receptive_field(model, input_size=224, target_layer=None):
    """
    기울기 분석으로 CNN의 실효 수용 영역을 잰다.
    
    가운데 출력 단위의 모든 입력에 대한 기울기를 계산한다.
    이 기울기의 크기가 입력 화소마다의 영향을 나타낸다.
    """
    model.eval()
    
    # 기울기를 좇는 입력 만들기
    x = torch.zeros(1, 3, input_size, input_size, requires_grad=True)
    
    # 순전파
    if target_layer:
        # 훅으로 특정 층의 활성값 얻기
        features = {}
        def hook(module, input, output):
            features['out'] = output
        handle = target_layer.register_forward_hook(hook)
        _ = model(x)
        output = features['out']
        handle.remove()
    else:
        output = model(x)
    
    # 출력의 가운데 자리 얻기
    if output.dim() == 4:  # 특징 맵
        h, w = output.shape[2], output.shape[3]
        center_h, center_w = h // 2, w // 2
        
        # 가운데 뉴런에서 역전파 (단일 채널)
        grad_output = torch.zeros_like(output)
        grad_output[0, 0, center_h, center_w] = 1.0
        output.backward(grad_output)
    else:  # 펼친 출력
        grad_output = torch.zeros_like(output)
        grad_output[0, output.shape[1] // 2] = 1.0
        output.backward(grad_output)
    
    # 실효 수용 영역은 입력에 대한 기울기의 절댓값이다 (채널에 대해 합)
    erf = x.grad.abs().sum(dim=1).squeeze().detach().numpy()
    
    return erf

def visualize_erf_concept():
    """실효 수용 영역과 이론적 수용 영역의 개념을 보인다."""
    
    size = 51
    center = size // 2
    
    # 이론적 수용 영역 (고른 상자)
    theoretical = np.zeros((size, size))
    theoretical[5:46, 5:46] = 1.0  # 이론적 수용 영역 41×41
    
    # 실효 수용 영역 (가우스 모양)
    y, x = np.ogrid[-center:size-center, -center:size-center]
    sigma = 8
    effective = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].imshow(theoretical, cmap='Blues')
    axes[0].set_title('Theoretical Receptive Field\n(all pixels equal weight)')
    axes[0].axis('off')
    
    axes[1].imshow(effective, cmap='hot')
    axes[1].set_title('Effective Receptive Field\n(center-weighted, Gaussian)')
    axes[1].axis('off')
    
    # 단면 견주기
    axes[2].plot(theoretical[center, :], 'b-', linewidth=2, label='Theoretical')
    axes[2].plot(effective[center, :], 'r-', linewidth=2, label='Effective')
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Weight')
    axes[2].set_title('Cross-Section Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('erf_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'erf_comparison.png'")
    
    return fig

visualize_erf_concept()
```

---

## 6. 구조 분석

### 흔한 구조와 그 수용 영역

| 구조 | 수용 영역 전략 | 최종 수용 영역 (어림) |
|--------------|-------------|---------------------|
| AlexNet | 큰 핵 (11×11, 5×5) | 195×195 |
| VGG-16 | 작은 핵 (3×3), 깊음 | 212×212 |
| ResNet-50 | 건너뛰기 연결, 병목 | 483×483 |
| Inception | 여러 규모의 병렬 가지 | 가지마다 다름 |
| U-Net | 건너뛰기가 있는 부호기-복호기 | 전체 해상도 |
| DeepLab | 팽창 합성곱 (ASPP) | 매우 큼 |
| EfficientNet | 복합 규모 조정 | 모델에 따라 커짐 |

### 과제별로 필요한 수용 영역

| 과제 | 필요한 수용 영역 | 근거 |
|------|----------------|-----------|
| 모서리 검출 | 작음 (3×3 ~ 7×7) | 지역적인 기울기만 있으면 된다 |
| 질감 분류 | 보통 (30×30 ~ 100×100) | 질감의 무늬 |
| 물체 탐지 | 큼 (100×100 초과) | 물체 전체의 맥락 |
| 장면 이해 | 매우 큼 (200×200 초과) | 전역적인 관계 |
| 의미 분할 | 이미지 전체의 맥락 | 조밀 예측에는 맥락이 필요하다 |

### ResNet-50 자세히 뜯어보기

```python
# ResNet-50 수용 영역 계산 (병목 블록)
def analyze_resnet50():
    layers = [
        # 줄기
        {'kernel': 7, 'stride': 2, 'name': 'Conv1'},
        {'kernel': 3, 'stride': 2, 'name': 'MaxPool'},
    ]
    
    # 2단계 (블록 3개, 채널 256개)
    for i in range(3):
        layers.extend([
            {'kernel': 1, 'stride': 1, 'name': f'Stage2.Block{i+1}.1x1'},
            {'kernel': 3, 'stride': 1, 'name': f'Stage2.Block{i+1}.3x3'},
            {'kernel': 1, 'stride': 1, 'name': f'Stage2.Block{i+1}.1x1'},
        ])
    
    # 3단계 (블록 4개, 채널 512개, 첫 블록은 보폭 2)
    for i in range(4):
        stride = 2 if i == 0 else 1
        layers.extend([
            {'kernel': 1, 'stride': 1, 'name': f'Stage3.Block{i+1}.1x1'},
            {'kernel': 3, 'stride': stride, 'name': f'Stage3.Block{i+1}.3x3'},
            {'kernel': 1, 'stride': 1, 'name': f'Stage3.Block{i+1}.1x1'},
        ])
    
    # 4단계 (블록 6개, 채널 1024개)
    for i in range(6):
        stride = 2 if i == 0 else 1
        layers.extend([
            {'kernel': 1, 'stride': 1, 'name': f'Stage4.Block{i+1}.1x1'},
            {'kernel': 3, 'stride': stride, 'name': f'Stage4.Block{i+1}.3x3'},
            {'kernel': 1, 'stride': 1, 'name': f'Stage4.Block{i+1}.1x1'},
        ])
    
    # 5단계 (블록 3개, 채널 2048개)
    for i in range(3):
        stride = 2 if i == 0 else 1
        layers.extend([
            {'kernel': 1, 'stride': 1, 'name': f'Stage5.Block{i+1}.1x1'},
            {'kernel': 3, 'stride': stride, 'name': f'Stage5.Block{i+1}.3x3'},
            {'kernel': 1, 'stride': 1, 'name': f'Stage5.Block{i+1}.1x1'},
        ])
    
    rf, jump = compute_receptive_field(layers)
    print(f"ResNet-50 final receptive field: {rf}×{rf} pixels")
    print(f"Final jump (output stride): {jump}")
    print(f"Note: RF ({rf}) > typical input size (224)!")
    
    return rf

analyze_resnet50()
```

### 구조 견주기

```python
def compare_architectures():
    """여러 구조 방식의 수용 영역을 견준다."""
    
    architectures = {
        'AlexNet-like': [
            {'kernel': 11, 'stride': 4},  # Conv1
            {'kernel': 3, 'stride': 2},   # Pool1
            {'kernel': 5, 'stride': 1},   # Conv2
            {'kernel': 3, 'stride': 2},   # Pool2
            {'kernel': 3, 'stride': 1},   # Conv3
            {'kernel': 3, 'stride': 1},   # Conv4
            {'kernel': 3, 'stride': 1},   # Conv5
        ],
        
        'VGG-like (16 layers)': [
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
            {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 3, 'stride': 1}, {'kernel': 2, 'stride': 2},
        ],
        
        'Dilated (5 layers)': [
            {'kernel': 3, 'stride': 1, 'dilation': 1},
            {'kernel': 3, 'stride': 1, 'dilation': 2},
            {'kernel': 3, 'stride': 1, 'dilation': 4},
            {'kernel': 3, 'stride': 1, 'dilation': 8},
            {'kernel': 3, 'stride': 1, 'dilation': 16},
        ],
    }
    
    print("Architecture Receptive Field Comparison")
    print("=" * 50)
    print(f"{'Architecture':<25} {'RF':>8} {'Layers':>8} {'Params':>10}")
    print("-" * 50)
    
    for name, layers in architectures.items():
        rf, _ = compute_receptive_field(layers)
        num_layers = len(layers)
        # 매개변수의 대용치 (k^2의 합)
        params_proxy = sum(l.get('kernel', 1)**2 for l in layers)
        print(f"{name:<25} {rf:>8} {num_layers:>8} {params_proxy:>10}")

compare_architectures()
```

---

## 7. 완전한 예제: 수용 영역을 고려한 신경망 설계

```python
import torch
import torch.nn as nn

class RFAwareNetwork(nn.Module):
    """
    수용 영역 목표를 정해 놓고 설계한 신경망.
    
    목표: 화소 100×100까지의 물체를 알아보기 위한 약 128×128의 수용 영역.
    
    설계 근거:
    - 세밀한 특징을 위해 표준 합성곱으로 시작한다
    - 조절된 하향 표본화를 위해 풀링을 쓴다
    - 해상도를 잃지 않고 수용 영역을 빨리 넓히려고 팽창 합성곱으로 바꾼다
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 층별 수용 영역 계산:
        # 1층: RF = 1 + (3-1)*1 = 3, 뜀 = 1
        # 2층: RF = 3 + (3-1)*1 = 5, 뜀 = 1
        # 풀링 1:  RF = 5 + (2-1)*1 = 6, 뜀 = 2
        # 3층: RF = 6 + (3-1)*2 = 10, 뜀 = 2
        # 4층: RF = 10 + (3-1)*2 = 14, 뜀 = 2
        # 풀링 2:  RF = 14 + (2-1)*2 = 16, 뜀 = 4
        # 5층 (d=2): RF = 16 + (5-1)*4 = 32, 뜀 = 4  [k_eff=5]
        # 6층 (d=4): RF = 32 + (9-1)*4 = 64, 뜀 = 4  [k_eff=9]
        # 7층 (d=8): RF = 64 + (17-1)*4 = 128, 뜀 = 4 [k_eff=17]
        # 최종 수용 영역: 128×128 ✓
        
        self.features = nn.Sequential(
            # 블록 1: 표준 합성곱 (세밀한 특징)
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 2: 표준 합성곱
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 블록 3: 팽창 합성곱 (수용 영역이 빠르게 넓어짐)
            nn.Conv2d(128, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=8, dilation=8),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 전역 평균 풀링
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def verify_network_design():
    """수용 영역 계산을 확인하고 신경망을 시험한다."""
    
    layers = [
        {'kernel': 3, 'stride': 1},  # Conv1
        {'kernel': 3, 'stride': 1},  # Conv2
        {'kernel': 2, 'stride': 2},  # Pool1
        {'kernel': 3, 'stride': 1},  # Conv3
        {'kernel': 3, 'stride': 1},  # Conv4
        {'kernel': 2, 'stride': 2},  # Pool2
        {'kernel': 3, 'stride': 1, 'dilation': 2},  # Conv5 d=2
        {'kernel': 3, 'stride': 1, 'dilation': 4},  # Conv6 d=4
        {'kernel': 3, 'stride': 1, 'dilation': 8},  # Conv7 d=8
    ]
    
    names = ['Conv1', 'Conv2', 'Pool1', 'Conv3', 'Conv4', 
             'Pool2', 'Conv5(d=2)', 'Conv6(d=4)', 'Conv7(d=8)']
    
    print("RF-Aware Network Design Verification")
    print("=" * 60)
    rf, jump = analyze_receptive_field(layers, names)
    print(f"\nFinal receptive field: {rf}×{rf} (target: ~128×128) ✓")
    print(f"Output stride: {jump}")
    
    # 신경망 시험
    model = RFAwareNetwork(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    
    print(f"\nNetwork test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    verify_network_design()
```

---

## 8. PyTorch 도구: 수용 영역 자동 추적기

```python
import torch
import torch.nn as nn

class ReceptiveFieldTracker(nn.Module):
    """
    어떤 Sequential 방식 모델의 수용 영역이든 저절로 분석해 주는
    감싸개 모듈.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.rf_info = self._analyze_receptive_field()
    
    def _analyze_receptive_field(self):
        """합성곱과 풀링 층마다의 수용 영역을 분석한다."""
        info = []
        rf, jump = 1, 1
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                # 핵 크기 뽑기
                k = module.kernel_size
                k = k[0] if isinstance(k, tuple) else k
                
                # 보폭 뽑기
                s = module.stride
                s = s[0] if isinstance(s, tuple) else s
                
                # 팽창률 뽑기 (풀링에서는 기본값 1)
                d = getattr(module, 'dilation', 1)
                d = d[0] if isinstance(d, tuple) else d
                
                # 실효 핵을 계산하고 수용 영역 갱신
                k_eff = d * (k - 1) + 1
                rf = rf + (k_eff - 1) * jump
                jump = jump * s
                
                info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'kernel': k,
                    'stride': s,
                    'dilation': d,
                    'receptive_field': rf,
                    'jump': jump
                })
        
        return info
    
    def print_receptive_field(self):
        """수용 영역 분석을 보기 좋게 출력한다."""
        print(f"{'Layer':<30} {'Type':<12} {'K':>3} {'S':>3} {'D':>3} {'RF':>6} {'Jump':>6}")
        print("-" * 75)
        for layer in self.rf_info:
            print(f"{layer['name']:<30} {layer['type']:<12} "
                  f"{layer['kernel']:>3} {layer['stride']:>3} {layer['dilation']:>3} "
                  f"{layer['receptive_field']:>6} {layer['jump']:>6}")
        
        if self.rf_info:
            final = self.rf_info[-1]
            print("-" * 75)
            print(f"Final: RF = {final['receptive_field']}×{final['receptive_field']}, "
                  f"Output stride = {final['jump']}")
    
    def forward(self, x):
        return self.model(x)

# 사용 예
model = nn.Sequential(
    nn.Conv2d(3, 64, 7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(3, stride=2, padding=1),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 256, 3, stride=2, padding=1),
    nn.ReLU(),
)

tracker = ReceptiveFieldTracker(model)
tracker.print_receptive_field()
```

---

## 9. 핵심 정리

1. **정의**: 수용 영역은 뉴런의 활성값에 영향을 주는 입력 영역이다

2. **핵심 식**:

   $$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

3. **누적 보폭이 증가를 키운다**: 앞쪽에서 하향 표본화를 하면 뒤쪽 층의 수용 영역이 크게 넓어진다

4. **넓히는 전략 네 가지** (효율 순):
   - 팽창 합성곱 (수용 영역을 크게 하는 데 가장 효율적)
   - 작은 핵을 쓰는 더 깊은 신경망 (학습에는 이쪽이 낫다)
   - 보폭을 준 합성곱 (거칠고 해상도를 잃는다)
   - 더 큰 핵 (비싸고 거의 쓰지 않는다)

5. **실효 수용 영역 < 이론적 수용 영역**: 영향의 분포가 가우스 모양이기 때문이다

6. **실효 수용 영역은 $\sqrt{\text{깊이}}$에 비례해 자란다**: 선형이 아니므로 아주 깊은 신경망에서는 수확이 줄어들 수 있다

7. **과제에 맞추기**: 다루려는 물체나 무늬의 크기에 맞추어 수용 영역을 설계하라

8. **건너뛰기 연결이 실효 수용 영역을 넓힌다**: 잔차 연결이 기울기가 멀리 있는 입력까지 흐르도록 돕는다

---

## 연습문제

**연습문제 1.**
보폭이 1인 $3 \times 3$ 합성곱을 다섯 층 쌓았을 때의 이론적인 수용 영역을 계산하라.

??? success "연습문제 1 풀이"
    핵 크기가 $k$이고 보폭이 1인 합성곱에서는 층마다 수용 영역에 $k-1$이 더해진다. 수용 영역 1에서 시작하면 다음과 같다.

    1층 뒤: $1 + (3-1) = 3$. 2층 뒤: $3 + 2 = 5$. 3층 뒤: $5 + 2 = 7$. 4층 뒤: $7 + 2 = 9$. 5층 뒤: $9 + 2 = 11$.

    일반식은 $\text{RF} = 1 + L(k-1) = 1 + 5(2) = 11$이며 여기서 $L$은 층의 수이다.

---

**연습문제 2.**
보폭이 1인 $3 \times 3$ 합성곱만 써서 수용 영역이 정확히 $101 \times 101$인 신경망을 설계하라.

??? success "연습문제 2 풀이"
    $\text{RF} = 1 + L \times 2 = 101$에서 $3 \times 3$ 합성곱 $L = 50$층이 필요하다.

    아니면 팽창 합성곱을 써서 더 적은 층으로 101에 이를 수 있다. 팽창률이 $1, 2, 4, 8, 16, 32$일 때, 팽창률이 $d$인 $3 \times 3$ 팽창 합성곱 하나가 수용 영역에 $2d$을 더한다.

    $\text{RF} = 1 + 2(1 + 2 + 4 + 8 + 16 + 32) = 1 + 2(63) = 127 > 101$이다. 팽창률 $1, 2, 4, 8, 16$인 다섯 층은 $1 + 2(31) = 63$을 준다. 표준 합성곱과 팽창 합성곱을 섞으면 정확히 맞출 수 있다.

---

**연습문제 3.**
수용 영역이 자라는 정도를 견주어라. (가) 표준 $3 \times 3$ 층 10개, (나) 팽창률이 1, 2, 4, 8, 16인 팽창 층 5개.

??? success "연습문제 3 풀이"
    (가) 표준: $\text{RF} = 1 + 10 \times 2 = 21$.

    (나) 팽창: 팽창률이 $d$인 $3 \times 3$ 팽창 합성곱마다 수용 영역에 $2d$이 더해진다.

    $\text{RF} = 1 + 2(1) + 2(2) + 2(4) + 2(8) + 2(16) = 1 + 2 + 4 + 8 + 16 + 32 = 63$.

    팽창 합성곱은 층은 절반이고 층당 매개변수는 같으면서 수용 영역을 $3\times$ 넓힌다. 대신 성긴 표본화 방식에서 오는 "격자 무늬 흠"이 생길 수 있다.

---

**연습문제 4.**
가운데 화소에 대해 $\frac{\partial \text{output}_{c,h,w}}{\partial \text{input}}$을 계산하고 기울기의 크기를 그려서 실효 수용 영역(ERF) 시각화를 구현하라.

??? success "연습문제 4 풀이"
    ```python
    import torch
    import torch.nn as nn

    model = nn.Sequential(*[nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                           nn.ReLU()) for _ in range(10)])
    model[0][0] = nn.Conv2d(3, 64, 3, padding=1)

    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    out = model(x)
    # 가운데 화소에서 역전파
    target = torch.zeros_like(out)
    target[0, 0, 32, 32] = 1.0
    out.backward(target)
    erf = x.grad[0].abs().sum(dim=0)  # 채널에 대해 합
    # erf는 실효 수용 영역을 보이는 64x64 지도이다
    # 대체로 가우스 모양이며 이론적 수용 영역보다 훨씬 작다
    ```

## 정리하며

이 마당은 정의、수학적 정식화、수용 영역 계산하기、수용 영역을 넓히는 전략을 차례로 짚었다.

**참고 문헌**

1. Luo, W., Li, Y., Urtasun, R., & Zemel, R. (2016). Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. *NeurIPS*.

2. Yu, F., & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. *ICLR*.

3. Araujo, A., Norberg, W., Hooker, S., & Weinberger, K. (2019). Computing Receptive Fields of Convolutional Neural Networks. *Distill*.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

5. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.
