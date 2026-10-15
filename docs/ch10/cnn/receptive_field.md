# 수용 영역

수용 영역(Receptive Field)은 한 출력이 구조적으로 참조할 수 있는 입력 범위를 나타낸다. 커널 크기·보폭·팽창률을 함께 계산하면 모델이 연결하는 공간 맥락과 출력 격자의 간격을 분석할 수 있다.

구조가 허용하는 이론적 범위와 특정 입력에서 실제로 강하게 작용하는 범위는 다르다. 이 절은 두 개념을 구분하고 순차적인 합성곱·풀링 경로의 계산법을 설명한다.

## 1. 정의

### 국소 수용 영역

한 층의 국소 연결은 앞 층에서 직접 참조하는 위치이다. 팽창률 1인 K×K 합성곱은 채널별로 K²개 위치를 참조한다. 팽창률이 d이면 위치 수는 같아도 경계 범위는 $[d(K-1)+1]^2$로 넓어진다.

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

### 유효 수용 영역(ERF)

이론적 수용 영역은 연결 가능한 최대 범위이다. 유효 수용 영역은 출력에 대한 입력의 기울기 크기 등으로 영향의 분포를 정량화한 개념이며, 어떤 임계값이나 질량 비율을 기준으로 폭을 정했는지도 밝혀야 한다. 입력·가중치·활성화와 학습 상태에 따라 달라지므로 항상 같은 모양이나 크기를 갖지 않는다.

---

## 2. 수학적 정식화

### 한 층의 수용 영역

다음과 같은 합성곱 층 하나에 대해 살펴보자.

- 커널 크기: $K$
- 보폭: $s$
- 팽창률: $d$

(팽창을 감안한) 유효 커널 크기는 다음과 같다.

$$K_{\text{eff}} = d \cdot (K - 1) + 1$$

표준 합성곱($d = 1$)에서는 $K_{\text{eff}} = K$이다.

### 여러 층의 수용 영역

합성곱 층을 쌓으면 수용 영역이 다음에 따라 자란다.

$$r_l = r_{l-1} + (K_l - 1) \cdot d_l \cdot \prod_{i=1}^{l-1} s_i$$

여기서 각 기호는 다음과 같다.

- $r_l$: $l$번째 층 뒤의 수용 영역
- $r_0 = 1$: 처음의 수용 영역 (픽셀 하나)
- $K_l$: $l$번째 층의 커널 크기
- $d_l$: $l$번째 층의 팽창률
- $s_i$: $i$번째 층의 보폭
- $\prod_{i=1}^{l-1} s_i$: 누적 보폭 ("뜀"이라고도 한다)

### 누적 보폭 인수

인수 $\prod_{i=1}^{l-1} s_i$(누적 보폭 또는 "뜀")은 수용 영역이 자라는 방식을 이해하는 데 매우 중요하다.

1. 앞선 보폭 하나하나가 뒤따르는 층의 수용 영역 증가를 **키운다**
2. 보폭 2인 층은 그 뒤 모든 층의 "걸음 크기"를 두 배로 만든다
3. 그래서 앞쪽에서 다운샘플링를 하면 깊은 층의 수용 영역이 크게 넓어진다

### 간단히 한 식

**한결같은 구조**(커널 $k$, 보폭 $s$이 처음부터 끝까지 같음)에서는 다음과 같다.

$$r_L = 1 + \sum_{l=1}^{L} (k - 1) \cdot d_l \cdot s^{l-1}$$

**같은 커널 k, 보폭 1, 팽창률 1인 합성곱**에서는 다음과 같다.

$$r_L = 1 + L \cdot (k - 1)$$

이 조건에서는 깊이에 선형으로 자란다. 팽창률이 달라지면 그 합을 반영해야 한다.

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
print(f"Receptive field: {rf}×{rf}")  # 16×16
print(f"Jump (output stride): {jump}")  # 4
```

**출력:**

```
Receptive field: 16×16
Jump (output stride): 4
```

### 위치까지 좇는 층별 분석

정확한 위치를 잡으려면 다음도 함께 좇는다.

- **뜀**: 특징 맵에서 한 칸 움직이는 것이 입력 픽셀 몇 개에 해당하는가
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

### 전략 1: 커널 크기 키우기

**장점**: 수용 영역이 곧바로 넓어진다
**단점**: 매개변수가 제곱으로 늘고 계산 비용이 든다

$$\text{Parameters} \propto K^2$$

### 전략 2: 깊이 늘리기

**장점**: 매개변수가 선형으로 늘고 특징이 층층이 조합된다
**단점**: 기울기 소실, 최적화의 어려움

3×3 층 두 개는 5×5 층 하나와 수용 영역이 같다.

$$r_{\text{two } 3\times3} = 1 + (3-1) + (3-1) = 5$$

$$r_{\text{one } 5\times5} = 1 + (5-1) = 5$$

입력·중간·출력 채널이 모두 C이고 편향을 제외하면 두 3×3 층은 18C²개, 한 5×5 층은 25C²개의 가중치를 갖는다. 채널 수가 달라지면 이 비교도 다시 계산해야 한다. 수용 영역이 같아도 두 구조의 함수 표현과 실행 시간은 같지 않다.

위 비교는 단일 채널 또는 동일한 채널 수를 유지한 경우이며 층 사이 활성화의 비용은 별도로 고려한다.

게다가 **비선형이 더 많다**. ReLU가 하나가 아니라 둘이다.

### 전략 3: 보폭 늘리기

**장점**: 수용 영역이 빨리 넓어지고 계산이 줄어든다
**단점**: 정보가 사라지고 공간 해상도가 낮아진다

### 전략 4: 팽창률 조절하기

**장점**: 팽창률을 키워 가중치 수를 유지하면서 연결 범위를 넓힐 수 있다. 지수적 증가는 지수적으로 증가하는 팽창률을 사용한 경우이다.
**단점**: 꼼꼼히 설계하지 않으면 격자 아티팩트이 생긴다

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

**출력:**

```
Standard 5 layers (3×3): RF = 11
Dilated 5 layers (d=1,2,4,8,16): RF = 63
Ratio: 5.7x larger with same parameters!
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
# RF = 3 × (2^10 - 1) + 1 = 3 × 1023 + 1 = 3070
```

**출력:**

```
WaveNet (3 blocks × 10 layers): 30 layers, RF = 3070
```

---

## 5. 유효 수용 영역 (ERF)

### 영향 분포와 구조의 차이

[Luo 등(2016)](https://proceedings.neurips.cc/paper/2016/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf)은 특정 합성곱 모델의 영향 분포가 중심에 집중되는 현상을 분석했다. 단순화한 깊은 합성곱 사슬의 가우스 근사에서는 폭이 깊이의 제곱근에 비례할 수 있다. 이것은 모든 CNN의 정확한 분포나 모든 구조에서의 법칙은 아니다.

학습과 비선형성, 잔차 경로는 영향 분포를 바꾼다. 잔차 연결이 최적화를 돕는다는 사실만으로 유효 수용 영역을 항상 넓힌다고 결론 내릴 수 없다. 짧은 경로의 영향이 커져 중심에 더 집중할 수도 있다. 학습이 진행될 때 범위가 단조롭게 넓어진다는 보장도 없다.

BatchNorm은 모드를 구분해야 한다. 평가 모드에서 고정된 이동 통계를 쓰면 위치별 변환이지만, 학습 중에는 배치·공간 통계를 공유하므로 한 위치가 다른 위치와 표본에 의존할 수 있다. 따라서 아래 국소 합성곱 점화식을 적용할 때는 통계 공유 같은 비국소 연산을 제외하거나 그 의존성을 별도로 설명해야 한다.

### 유효 수용 영역 재기

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

**출력:**

```
Saved visualization to 'erf_comparison.png'
```

---

## 6. 구조 분석

### 구조별 분석 범위

VGG처럼 순차적인 구조는 합성곱과 풀링의 경로를 차례로 계산할 수 있다. ResNet·Inception·U-Net처럼 여러 경로가 합쳐지는 구조에서는 각 경로의 수용 영역과 좌표 정렬을 추적해야 한다. 전역 평균 풀링이나 어텐션이 들어가면 출력의 참조 범위도 별도로 반영한다.

AlexNet이나 ResNet-50이라는 이름만으로 모든 출력의 수용 영역이 하나의 고정값으로 정해지지는 않는다. 어느 층을 대상으로 하는지와 보폭의 위치·패딩·구현 변형을 명시해야 한다. 아래 표와 코드의 값도 명시한 층 배열의 결과이며 모든 동명 모델을 대표하지 않는다.

필요한 문맥의 크기는 과제와 입력 해상도, 물체 크기에 따라 달라진다. 특정 과제에 100픽셀이나 200픽셀 같은 보편적인 최소 기준을 두기보다, 배포 데이터에 필요한 문맥과 위치 정밀도를 기준으로 설계한다.

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

**출력:**

```
ResNet-50 final receptive field: 427×427 pixels
Final jump (output stride): 32
Note: RF (427) > typical input size (224)!
```

### 구조 비교

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

**출력:**

```
Architecture Receptive Field Comparison
==================================================
Architecture                    RF   Layers     Params
--------------------------------------------------
AlexNet-like                   163        7        191
VGG-like (16 layers)           212       18        137
Dilated (5 layers)              63        5         45
```

---

## 7. 예제: 수용 영역을 고려한 신경망 설계

코드의 128×128은 합성곱·풀링 경로에서 마지막 특징 한 위치의 이론적 범위이다. 그 뒤 AdaptiveAvgPool2d가 모든 공간 위치를 결합하므로 최종 분류 로짓의 참조 범위와는 다르다. 학습 모드의 BatchNorm에 의한 통계 공유도 이 계산에 포함되지 않는다.

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

**출력:**

```
RF-Aware Network Design Verification
============================================================
Layer                Kernel   Stride   Dilation RF       Jump    
--------------------------------------------------------------------
Input                -        -        -        1        1       
Conv1                3        1        1        3        1       
Conv2                3        1        1        5        1       
Pool1                2        2        1        6        2       
Conv3                3        1        1        10       2       
Conv4                3        1        1        14       2       
Pool2                2        2        1        16       4       
Conv5(d=2)           3        1        2        32       4       
Conv6(d=4)           3        1        4        64       4       
Conv7(d=8)           3        1        8        128      4       

Final receptive field: 128×128 (target: ~128×128) ✓
Output stride: 4

Network test:
  Input shape: torch.Size([2, 3, 224, 224])
  Output shape: torch.Size([2, 10])
  Total parameters: 1,740,362
```

---

## 8. PyTorch 도구: 수용 영역 계산기

아래 클래스는 등록된 Conv2d·풀링 모듈의 순서를 하나의 사슬로 간주한다. forward의 실제 실행 그래프를 추적하지 않아 분기·잔차·재사용 모듈에 일반적으로 적용할 수 없다. 함수형 풀링과 AdaptiveAvgPool2d도 반영하지 않고, 튜플의 첫 축만 사용하므로 비정사각 입력·커널은 축별 계산이 필요하다.

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

**출력:**

```
Layer                          Type           K   S   D     RF   Jump
---------------------------------------------------------------------------
0                              Conv2d         7   2   1      7      2
2                              MaxPool2d      3   2   1     11      4
3                              Conv2d         3   1   1     19      4
5                              Conv2d         3   1   1     27      4
7                              Conv2d         3   2   1     35      8
---------------------------------------------------------------------------
Final: RF = 35×35, Output stride = 8
```

---

## 9. 핵심 정리

1. 이론적 수용 영역은 구조적 연결 범위이고 유효 수용 영역은 영향의 분포를 반영한다.
2. 단일 순차 경로의 경계 범위는 $r_l=r_{l-1}+(K_l-1)d_l\prod_{i<l}s_i$로 계산한다.
3. 출력 간격과 첫 출력의 중심까지 알려면 누적 보폭과 패딩도 추적한다.
4. 분기·전역 풀링·통계 공유는 단순한 합성곱 사슬과 구분한다.
5. 커널·깊이·보폭·팽창률은 해상도와 계산 비용에 서로 다른 영향을 준다. 모든 과제에 공통인 효율 순위는 없다.
6. 한 입력의 기울기 지도가 구조적으로 가능한 모든 연결을 나타내는 것은 아니다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
보폭이 1인 $3 \times 3$ 합성곱을 다섯 층 쌓았을 때의 이론적인 수용 영역을 계산하라.

</div>

??? success "연습문제 1 풀이"
    커널 크기가 $k$이고 보폭이 1인 합성곱에서는 층마다 수용 영역에 $k-1$이 더해진다. 수용 영역 1에서 시작하면 다음과 같다.

    1층 뒤: $1 + (3-1) = 3$. 2층 뒤: $3 + 2 = 5$. 3층 뒤: $5 + 2 = 7$. 4층 뒤: $7 + 2 = 9$. 5층 뒤: $9 + 2 = 11$.

    일반식은 $\text{RF} = 1 + L(k-1) = 1 + 5(2) = 11$이며 여기서 $L$은 층의 수이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
보폭 1인 3×3 합성곱으로 이론적 수용 영역 101×101을 만드는 두 가지 설계를 제시하라.

</div>

??? success "연습문제 2 풀이"
    팽창률 1인 층만 쓰면 $1+2L=101$에서 L=50이다. 팽창률을 허용하면 1,2,4,8,16,19의 여섯 층으로 합이 50이므로 $1+2\times50=101$이다. 이 값은 경계 범위이며 실제 입력의 가장자리와 패딩, 채널 연결을 별도로 고려한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
보폭 1에서 표준 3×3 층 10개와 팽창률 1,2,4,8,16인 3×3 층 5개의 이론적 수용 영역을 비교하라.

</div>

??? success "연습문제 3 풀이"
    표준은 $1+2\times10=21$, 팽창은 $1+2(1+2+4+8+16)=63$이다. 같은 채널 수와 편향 설정이면 후자의 층 수는 절반이다. 이 팽창률 조합에서 연결은 조밀하게 이어질 수 있지만 일반적인 팽창률 조합에서는 빈 위치가 생길 수 있으므로 수용 영역의 너비와 연결 밀도를 구분한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
가운데 픽셀에 대해 $\frac{\partial \text{output}_{c,h,w}}{\partial \text{input}}$을 계산하고 기울기의 크기를 그려서 유효 수용 영역(ERF) 시각화를 구현하라.

</div>

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

수용 영역 계산은 구조가 연결할 수 있는 입력의 범위를 분석한다. 단순 사슬의 점화식과 실제 실행 그래프의 경로를 구분하고, 이론적 경계·입력 패딩·전역 연산과 특정 입력의 기울기 분포를 함께 해석해야 한다.

**참고 문헌**

1. Luo, W., Li, Y., Urtasun, R., & Zemel, R. (2016). Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. *NeurIPS*.

2. Yu, F., & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. *ICLR*.

3. Araujo, A., Norberg, W., Hooker, S., & Weinberger, K. (2019). Computing Receptive Fields of Convolutional Neural Networks. *Distill*.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

5. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.
