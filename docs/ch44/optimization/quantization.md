# 양자화

---

## 1. 개관

양자화는 뉴럴 네트워크의 웨이트와 활성화를 부동소수점(흔히 32비트)에서 더 적은 비트(8비트, 4비트, 나아가 이진값)으로 정밀도를 낮춘다. 이로써 메모리와 계산 비용이 크게 줄고, 이를 받쳐 주는 하드웨어에서는 추론이 크게 빨라진다.

---

## 2. 양자화 기초

### 왜 양자화하는가

뉴럴 네트워크의 웨이트와 활성화는 흔히 32비트 부동소수점 수로 담긴다.

```
FP32: ±[1.18e-38, 3.4e+38], 열 자리로 약 7자리 촘촘함
      부호 (1) | 지수 (8) | 가수 (23)
```

뉴럴 네트워크의 계산에는 이만한 정밀도가 거의 필요하지 않다. 양자화는 값을 더 낮은 정밀도로 옮긴다.

```
INT8: [-128, 127] 또는 [0, 255]
      모두 8비트

FP16: ±[6.1e-5, 65504], 열 자리로 약 3자리
      부호 (1) | 지수 (5) | 가수 (10)
```

### 양자화의 이점

| 지표 | FP32 → INT8 | FP32 → FP16 |
|--------|-------------|-------------|
| 모델 크기 | 4배 줄어듦 | 2배 줄어듦 |
| 메모리 대역폭 | 4배 줄어듦 | 2배 줄어듦 |
| 계산 속도 | 2~4배 빠름* | 1.5~2배 빠름* |
| 정확도 손실 | 흔히 0~2% | 흔히 0.5% 미만 |

*속도 향상은 하드웨어 지원 여부에 달려 있다

---

## 3. 수학 기초

### 양자화 매핑

양자화는 연속적인 부동소수점 값을 이산 정수 집합으로 옮긴다.

$$x_{\text{float}} \in \mathbb{R} \rightarrow x_{\text{quant}} \in \{0, 1, \ldots, 2^b - 1\}$$

여기서 $b$는 비트 너비다.

### 균일 아핀 양자화

가장 흔한 양자화 구조는 스케일과 0점을 지닌 선형 매핑을 쓴다.

**양자화:**

$$x_q = \text{round}\left(\frac{x - z}{s}\right) = \text{round}\left(\frac{x}{s}\right) - z_q$$

**역양자화:**

$$\hat{x} = s \cdot (x_q + z_q) = s \cdot x_q + z$$

여기서:

- $s$는 스케일 값
- $z$는 0점(바이어스)
- $z_q$는 양자화된 0점

### 스케일과 0점 계산

자리가 $[x_{\min}, x_{\max}]$인 텐서에서

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

$$z = q_{\min} - \text{round}\left(\frac{x_{\min}}{s}\right)$$

부호 없는 INT8에서는 $q_{\min} = 0$, $q_{\max} = 255$

부호 있는 INT8에서는 $q_{\min} = -128$, $q_{\max} = 127$

### 대칭 대 비대칭 양자화

**대칭 양자화**($z = 0$):

$$x_q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x_{\min}|, |x_{\max}|)}{q_{\max}}$$

더 단순하나 한쪽으로 쏠린 분포에서는 표현 범위를 낭비한다.

**비대칭 양자화**($z \neq 0$):
위의 온전한 식을 쓴다. 한쪽으로 쏠린 분포(ReLU 뒤의 활성화에서 흔하다)에서 양자화 범위를 더 잘 쓴다.

---

## 4. PyTorch로 구현하기

### 기초 양자화 계산

```python
import torch
import torch.nn as nn
import torch.quantization
from typing import Tuple, Dict, List
import time

def compute_quantization_params(tensor: torch.Tensor, 
                                num_bits: int = 8, 
                                symmetric: bool = True) -> Tuple[float, int]:
    """
    수 줄이기에 쓸 잣대와 0점을 셈한다.
    
    Args:
        tensor: 수를 줄일 들임 텐서
        num_bits: 수 줄이기에 쓸 비트 수
        symmetric: 대칭 수 줄이기를 쓸지
        
    Returns:
        잣대, 0점
    """
    if symmetric:
        # 대칭: 자리는 [-max_abs, max_abs], zero_point = 0
        max_abs = tensor.abs().max()
        qmax = 2 ** (num_bits - 1) - 1
        scale = max_abs / qmax
        zero_point = 0
    else:
        # 어긋남: 온 자리를 쓴다
        min_val = tensor.min()
        max_val = tensor.max()
        qmin = 0
        qmax = 2 ** num_bits - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(-min_val / scale))
    
    return scale.item(), zero_point

def quantize(tensor: torch.Tensor, scale: float, zero_point: int, 
             num_bits: int = 8) -> torch.Tensor:
    """텐서를 정수로 수 줄인다."""
    qmin = 0 if zero_point != 0 else -(2 ** (num_bits - 1))
    qmax = 2 ** num_bits - 1 if zero_point != 0 else 2 ** (num_bits - 1) - 1
    
    q = torch.round(tensor / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int8)

def dequantize(q_tensor: torch.Tensor, scale: float, 
               zero_point: int) -> torch.Tensor:
    """정수 텐서를 뜨는 소수로 되돌린다."""
    return scale * (q_tensor.float() - zero_point)
```

---

## 5. 양자화 오차 살피기

### 양자화 잡음

균일 양자화의 오차는 거의 균등 분포를 따른다.

$$\epsilon = x - \hat{x} \sim \mathcal{U}\left(-\frac{s}{2}, \frac{s}{2}\right)$$

**평균 제곱 오차:**

$$\text{MSE} = \mathbb{E}[\epsilon^2] = \frac{s^2}{12}$$

**신호 대 양자화 잡음비(SQNR):**

$$\text{SQNR} = 10 \log_{10}\left(\frac{\sigma_x^2}{\text{MSE}}\right) \approx 6.02b + 4.77 \text{ dB}$$

비트를 하나 더할 때마다 SQNR이 약 6 dB 오른다.

### 오차 누적

다층 네트워크에서는 양자화 오차가 쌓인다.

$$\epsilon_{\text{output}} = f(x + \epsilon_x, W + \epsilon_W) - f(x, W)$$

선형 레이어에서는 오차가 레이어의 너비와 깊이에 따라 커지므로 레이어마다 다른 양자화 기법이 있어야 한다.

---

## 6. 양자화의 유형

### 1. 동적 양자화

웨이트는 미리 양자화하고, 활성화는 실행 중에 그때그때 양자화한다.

**잘 맞는 자리**: 입력 크기가 바뀌는 모델, RNN, LSTM, 트랜스포머.

```python
class LinearModel(nn.Module):
    """움직이는 수 줄이기의 보기 모형."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                 num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def apply_dynamic_quantization(model: nn.Module,
                               dtype: torch.dtype = torch.qint8) -> nn.Module:
    """
    모형에 움직이는 수 줄이기를 건다.
    
    움직이는 수 줄이기:
    - 짐: 미리 줄인다(INT8)
    - 살림: 미루어 볼 때 그때그때 줄인다
    
    Args:
        model: 미리 익힌 FP32 모형
        dtype: 수 줄이기 클래스(torch.qint8 또는 torch.float16)
        
    Returns:
        움직이는 수 줄이기를 건 모형
    """
    model.eval()
    
    # 앞선 크기를 잰다
    size_before = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 어느 켜 클래스를 줄일지 밝힌다
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.LSTM, nn.GRU},
        dtype=dtype
    )
    
    # 뒤의 크기를 잰다
    size_after = sum(
        p.numel() * p.element_size() 
        for p in quantized_model.parameters()
    )
    
    print(f"앞선 크기: {size_before / 1e6:.2f} MB")
    print(f"뒤의 크기: {size_after / 1e6:.2f} MB")
    print(f"눌러 담음: {size_before / size_after:.2f}배")
    
    return quantized_model
```

### 2. 정적 양자화

웨이트를 미리 양자화하고, 캘리브레이션 데이터로 활성화의 스케일과 0점까지 미리 정해 둔다.

**잘 맞는 자리**: 입력 크기가 고정된 곳, CNN, 비전 모델.

```python
class QuantizableModel(nn.Module):
    """
    붙박인 수 줄이기에 맞게 마련한 모형.
    
    핵심 고침:
    1. 들임/날임에 QuantStub/DeQuantStub을 더한다
    2. 함수 셈을 같은 일을 하는 배치로 바꾼다
    3. 될 수 있으면 켜를 녹여 붙인다
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # 수 줄이기 꼭지
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # 모형의 켜
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()  # F.relu이 아니라 배치를 쓴다
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)  # 들임의 수를 줄인다
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        x = self.dequant(x)  # 날임의 수를 되돌린다
        return x
    
    def fuse_model(self):
        """
        수 줄이기가 잘 되도록 Conv-BN-ReLU 이음을 녹여 붙인다.
        
        녹여 붙이면 가운데의 수 줄이기/되돌리기 스텝을 건너뛰어
        수 줄이기 어긋남이 줄어든다.
        """
        torch.quantization.fuse_modules(
            self,
            [['conv1', 'bn1', 'relu1'],
             ['conv2', 'bn2', 'relu2']],
            inplace=True
        )

def apply_static_quantization(model: nn.Module,
                              calibration_loader: torch.utils.data.DataLoader,
                              backend: str = 'fbgemm') -> nn.Module:
    """
    눈금을 맞추어 붙박인 수 줄이기를 건다.
    
    붙박인 수 줄이기:
    - 짐: 미리 줄인다
    - 살림: 눈금 맞추기에서 미리 셈한 잣대/0점으로 줄인다
    
    살림의 자리를 정하려면 잘 드러내는 눈금 맞추기 자료가 있어야 한다.
    
    Args:
        model: QuantizableModel(수 줄이기 꼭지가 있어야 한다)
        calibration_loader: 본보기 보기를 담은 DataLoader
        backend: 'fbgemm'(x86) 또는 'qnnpack'(ARM)
        
    Returns:
        붙박인 수 줄이기를 건 모형
    """
    model.eval()
    
    # 방법이 있으면 켜를 녹여 붙인다
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # 수 줄이기 차림을 잡는다
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # 살피개를 끼운다
    torch.quantization.prepare(model, inplace=True)
    
    # 눈금 맞추기: 본보기 자료를 모형에 흘린다
    print("눈금 맞추는 중...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_loader):
            model(data)
            if batch_idx >= 100:  # 눈금 맞추기에 배치 100개를 쓴다
                break
    
    # 수 줄인 모형으로 바꾼다
    torch.quantization.convert(model, inplace=True)
    
    return model
```

### 3. 양자화 인식 트레인(QAT)

트레인하는 동안 양자화를 흉내 내어 네트워크가 양자화 오차에 강건해지도록 배우게 한다.

**잘 맞는 자리**: 정적 양자화로 정확도가 받아들일 수 없을 만큼 떨어질 때.

**순전파:**

$$\hat{x} = \text{FakeQuant}(x) = s \cdot \text{round}\left(\frac{\text{clamp}(x, x_{\min}, x_{\max})}{s}\right)$$

**역전파(직통 추정기(STE)):**

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \hat{x}} \cdot \mathbf{1}[x_{\min} \leq x \leq x_{\max}]$$

그래디언트는 (클리핑 범위 안에서) 양자화를 항등 함수인 양 지나간다.

```python
def train_with_qat(model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   epochs: int = 10,
                   lr: float = 1e-3,
                   backend: str = 'fbgemm',
                   device: str = 'cpu') -> nn.Module:
    """
    수 줄이기를 아는 익힘(QAT).
    
    익히는 동안 거짓 수 줄이기로 흉내를 내어
    모형이 수 줄이기 어긋남에 든든한 짐을 배우게 한다.
    
    QAT은 흔히 익힘 뒤 수 줄이기보다 맞음이 1~2% 낫다.
    
    Args:
        model: 익힐 QuantizableModel
        train_loader: 익힘 자료
        test_loader: 따질 시험 자료
        epochs: 익힘 판 수
        lr: 러닝 비율
        backend: 수 줄이기 뒷단
        device: 익힐 장치
        
    Returns:
        수 줄인 모형
    """
    model = model.to(device)
    model.train()
    
    # 켜를 녹여 붙인다
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    
    # QAT 차림을 잡는다
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    
    # QAT을 마련한다(거짓 수 줄이기 배치를 끼운다)
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 익힘 차림
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("수 줄이기를 아는 익힘을 비롯한다...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 몸풀기 뒤 배치 잣대 잡기의 자를 얼린다
        if epoch >= epochs // 2:
            model.apply(torch.quantization.disable_observer)
        if epoch >= epochs * 3 // 4:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # 따진다
        if (epoch + 1) % 2 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, pred = output.max(1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            print(f"{epoch+1}/{epochs}판, "
                  f"잃음: {train_loss/len(train_loader):.4f}, "
                  f"맞음: {100*correct/total:.2f}%")
    
    # 온통 수 줄인 모형으로 바꾼다
    model.eval()
    model_quantized = torch.quantization.convert(model.cpu(), inplace=False)
    
    return model_quantized
```

---

## 7. 혼합 정밀도(FP16)

요즘 GPU에서 더 빠르게 계산하려고 반정밀도를 쓴다.

```python
def mixed_precision_inference(model: nn.Module, 
                              data: torch.Tensor) -> torch.Tensor:
    """
    절로 섞이는 촘촘함(AMP)으로 미루어 본다.
    
    AMP은 걱정 없는 자리에는 FP16을, 있어야 할 자리에는 FP32을 절로 쓴다.
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model(data)
    
    return output

def benchmark_precision(model: nn.Module, data: torch.Tensor, 
                        iterations: int = 100) -> Dict[str, float]:
    """FP32과 FP16의 됨됨이를 견준다."""
    device = torch.device('cuda')
    model = model.to(device)
    data = data.to(device)
    
    # FP32 잣대 재기
    model_fp32 = model.float()
    data_fp32 = data.float()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model_fp32(data_fp32)
    torch.cuda.synchronize()
    fp32_time = time.perf_counter() - start
    
    # FP16 잣대 재기
    model_fp16 = model.half()
    data_fp16 = data.half()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model_fp16(data_fp16)
    torch.cuda.synchronize()
    fp16_time = time.perf_counter() - start
    
    print(f"FP32: {fp32_time*1000:.2f} ms")
    print(f"FP16: {fp16_time*1000:.2f} ms")
    print(f"빨라짐: {fp32_time/fp16_time:.2f}배")
    
    return {
        'fp32_time_ms': fp32_time * 1000,
        'fp16_time_ms': fp16_time * 1000,
        'speedup': fp32_time / fp16_time
    }
```

---

## 8. 캘리브레이션 기법

### 최솟값-최댓값 캘리브레이션

```python
class MinMaxObserver:
    """눈금 맞추기에 쓸 가장 작은/큰 값을 좇는다."""
    
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, tensor: torch.Tensor):
        self.min_val = min(self.min_val, tensor.min().item())
        self.max_val = max(self.max_val, tensor.max().item())
    
    def compute_params(self, num_bits: int = 8) -> Tuple[float, int]:
        qmin, qmax = 0, 2**num_bits - 1
        scale = (self.max_val - self.min_val) / (qmax - qmin)
        zero_point = int(round(-self.min_val / scale))
        return scale, zero_point
```

### 히스토그램 캘리브레이션

```python
import numpy as np

class HistogramObserver:
    """눈금을 더 잘 맞추려고 잦기 그림을 좇는다."""
    
    def __init__(self, num_bins: int = 2048):
        self.num_bins = num_bins
        self.histogram = None
        self.bin_edges = None
    
    def update(self, tensor: torch.Tensor):
        values = tensor.flatten().cpu().numpy()
        
        if self.histogram is None:
            self.histogram, self.bin_edges = np.histogram(
                values, bins=self.num_bins
            )
        else:
            hist, _ = np.histogram(values, bins=self.bin_edges)
            self.histogram += hist
    
    def compute_params_entropy(self, num_bits: int = 8):
        """
        KL 갈림으로 가장 좋은 임계를 셈한다(엔트로피 눈금 맞추기).
        
        수를 줄일 때 소식 잃음을 가장 작게 하는 임계를 찾는다.
        """
        # 서비스에서는 PyTorch에 든 HistogramObserver을 쓴다
        pass
```

### 백분위 캘리브레이션

```python
class PercentileObserver:
    """튀는 값에 든든하도록 백분위를 쓴다."""
    
    def __init__(self, percentile: float = 99.99):
        self.percentile = percentile
        self.values = []
    
    def update(self, tensor: torch.Tensor):
        self.values.extend(tensor.flatten().tolist())
    
    def compute_params(self, num_bits: int = 8) -> Tuple[float, int]:
        values = np.array(self.values)
        min_val = np.percentile(values, 100 - self.percentile)
        max_val = np.percentile(values, self.percentile)
        
        qmin, qmax = 0, 2**num_bits - 1
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(round(-min_val / scale))
        
        return scale, zero_point
```

---

## 9. 채널별 대 텐서별 양자화

### 텐서별 양자화

텐서 전체에 스케일/0점 하나:

- 구현이 더 단순하다
- 채널마다 웨이트 분포가 다르면 정밀도를 잃을 수 있다

### 채널별 양자화

출력 채널마다 따로 스케일/0점을 둔다.

- 웨이트 분포를 더 잘 지킨다
- 컨볼루션 레이어에서는 사실상 필수다
- 요즘 양자화의 표준 방법이다

```python
def analyze_weight_distribution(conv: nn.Conv2d) -> Dict:
    """
    날임 클래스마다 짐 분포를 살핀다.
    
    클래스마다 짐 자리가 크게 다를 때
    클래스마다의 수 줄이기가 도움이 된다.
    """
    weights = conv.weight.data  # (out_ch, in_ch, kH, kW)
    
    channel_stats = []
    for ch in range(weights.size(0)):
        ch_weights = weights[ch].flatten()
        channel_stats.append({
            'channel': ch,
            'min': ch_weights.min().item(),
            'max': ch_weights.max().item(),
            'mean': ch_weights.mean().item(),
            'std': ch_weights.std().item(),
            'range': (ch_weights.max() - ch_weights.min()).item()
        })
    
    # 자리의 바뀜을 셈한다
    ranges = [s['range'] for s in channel_stats]
    range_ratio = max(ranges) / (min(ranges) + 1e-8)
    
    return {
        'channel_stats': channel_stats,
        'range_ratio': range_ratio,
        'recommendation': 'per-channel' if range_ratio > 2.0 else 'per-tensor'
    }
```

---

## 10. 혼합 정밀도 양자화

레이어마다 양자화에 예민한 정도가 다르다. 혼합 정밀도는 레이어마다 비트 너비를 달리 준다.

```python
def mixed_precision_sensitivity(model: nn.Module,
                                test_loader: torch.utils.data.DataLoader,
                                bit_widths: List[int] = [4, 8, 16],
                                device: str = 'cpu') -> Dict:
    """
    비트 너비마다 켜가 얼마나 예민한지 살핀다.
    
    Args:
        model: 살필 모형
        test_loader: 시험 자료
        bit_widths: 해 볼 비트 너비
        device: 미루어 볼 장치
        
    Returns:
        켜마다 비트 너비마다의 예민함 점수
    """
    import copy
    
    baseline_acc = evaluate_model(model, test_loader, device)
    results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            results[name] = {}
            
            for bits in bit_widths:
                # 이 켜만 수 줄이기를 흉내 낸다
                model_copy = copy.deepcopy(model)
                
                for n, m in model_copy.named_modules():
                    if n == name:
                        # 짐의 수를 줄인다
                        weight = m.weight.data
                        q_min = -(2 ** (bits - 1))
                        q_max = 2 ** (bits - 1) - 1
                        scale = (weight.max() - weight.min()) / (q_max - q_min)
                        
                        quantized = torch.clamp(
                            torch.round(weight / scale),
                            q_min, q_max
                        )
                        m.weight.data = quantized * scale
                        break
                
                acc = evaluate_model(model_copy, test_loader, device)
                acc_drop = baseline_acc - acc
                
                results[name][bits] = {
                    'accuracy': acc,
                    'accuracy_drop': acc_drop
                }
    
    return results

def evaluate_model(model: nn.Module, 
                   test_loader: torch.utils.data.DataLoader,
                   device: str = 'cpu') -> float:
    """모형의 맞음을 따진다."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total
```

---

## 11. 하드웨어 고려 사항

### 지원되는 연산

| 연산 | INT8 CPU | INT8 GPU | FP16 GPU |
|-----------|----------|----------|----------|
| Conv2d | ✓ | ✓ | ✓ |
| 선형 | ✓ | ✓ | ✓ |
| 배치 정규화 | 융합 | 융합 | ✓ |
| ReLU | 융합 | 융합 | ✓ |
| 더하기 | ✓ | ✓ | ✓ |
| 이어 붙이기 | ✓ | ✓ | ✓ |
| 소프트맥스 | FP32로 대체 | FP32로 대체 | ✓ |

### 백엔드 고르기

```python
def select_quantization_backend(target_device: str) -> str:
    """
    타깃 쇠 붙임새에 맞는 수 줄이기 뒷단을 고른다.
    
    Args:
        target_device: 'x86_cpu', 'arm_cpu', 'nvidia_gpu', 'apple_neural_engine'
        
    Returns:
        PyTorch 수 줄이기의 뒷단 이름
    """
    backends = {
        'x86_cpu': 'fbgemm',      # 인텔/AMD CPU
        'arm_cpu': 'qnnpack',      # ARM CPU(손전화)
        'nvidia_gpu': 'tensorrt',  # NVIDIA GPU(TensorRT이 있어야 한다)
        'apple_neural_engine': 'coreml'  # 애플 장치(coremltools이 있어야 한다)
    }
    
    return backends.get(target_device, 'fbgemm')
```

---

## 12. 모범 사례

### 레이어의 예민함

첫 레이어와 마지막 레이어가 양자화에 가장 예민한 것이 보통이다.

```python
def apply_mixed_precision_strategy(model: nn.Module,
                                   sensitive_layers: List[str] = None) -> nn.Module:
    """
    예민한 켜에는 더 높은 촘촘함을 쓴다.
    
    흔한 꾀:
    - 첫 컨볼루션 켜: FP32이나 FP16(들임 소식을 지킨다)
    - 마지막 선형 켜: FP32이나 FP16(날임의 촘촘함을 지킨다)
    - 가운데 켜: INT8
    """
    if sensitive_layers is None:
        # 기본값: 첫 켜와 마지막 켜
        layer_names = [name for name, _ in model.named_modules() 
                      if isinstance(_, (nn.Conv2d, nn.Linear))]
        sensitive_layers = [layer_names[0], layer_names[-1]] if layer_names else []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name in sensitive_layers:
                module.qconfig = None  # FP32으로 둔다
            else:
                module.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    return model
```

### 배치 정규화 접어 넣기

추론 효율을 높이려고 배치 정규화를 앞의 컨볼루션/선형 레이어에 접어 넣는다.

```python
def fold_batchnorm(model: nn.Module) -> nn.Module:
    """
    배치 잣대 잡기 켜를 앞의 컨볼루션 켜에 접어 넣는다.
    
    Conv-BN 이음에서:
    y = γ * (Wx + b - μ) / σ + β
      = (γ/σ) * Wx + (γ(b-μ)/σ + β)
      = W' * x + b'
    
    여기서:
    W' = W * γ/σ
    b' = γ(b-μ)/σ + β
    """
    # PyTorch가 수 줄이기 동안 절로 해 준다
    torch.quantization.fuse_modules(
        model,
        # 녹여 붙일 결을 밝힌다
        [['conv', 'bn', 'relu']],  # 보기 결
        inplace=True
    )
    return model
```

### 캘리브레이션 데이터 지침

```python
def prepare_calibration_data(dataset, num_samples: int = 1000) -> List[torch.Tensor]:
    """
    잘 드러내는 눈금 맞추기 자료를 마련한다.
    
    길잡이:
    - 보기 500~2000개를 쓴다
    - 여러 결의 보기를 담는다
    - 서비스 자료의 분포와 맞춘다
    - 익힘 자료 불리기는 쓰지 않는다
    """
    indices = torch.randperm(len(dataset))[:num_samples]
    
    calibration_samples = []
    for idx in indices:
        sample, _ = dataset[idx]
        calibration_samples.append(sample.unsqueeze(0))
    
    return calibration_samples
```

---

## 13. 양자화 오차 디버깅

```python
def debug_quantization_error(original: nn.Module,
                             quantized: nn.Module,
                             sample_input: torch.Tensor) -> Dict:
    """
    수 줄이기 어긋남이 어디서 오는지 살핀다.
    
    FP32 모형과 수 줄인 모형의 가운데 살림을 견준다.
    """
    original.eval()
    quantized.eval()
    
    activations_fp32 = {}
    activations_quant = {}
    
    # 살림을 잡으려 갈고리를 건다
    def make_hook(storage, name):
        def hook(module, input, output):
            storage[name] = output.detach()
        return hook
    
    hooks = []
    for name, module in original.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            hooks.append(module.register_forward_hook(
                make_hook(activations_fp32, name)
            ))
    
    for name, module in quantized.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
            hooks.append(module.register_forward_hook(
                make_hook(activations_quant, name)
            ))
    
    # 앞으로 스텝
    with torch.no_grad():
        _ = original(sample_input)
        _ = quantized(sample_input)
    
    # 갈고리를 뗀다
    for hook in hooks:
        hook.remove()
    
    # 살림을 견준다
    errors = {}
    for name in activations_fp32:
        if name in activations_quant:
            fp32_act = activations_fp32[name].float()
            quant_act = activations_quant[name].float()
            
            mse = ((fp32_act - quant_act) ** 2).mean().item()
            rel_error = (mse / (fp32_act ** 2).mean().item()) ** 0.5
            
            errors[name] = {
                'mse': mse,
                'relative_error': rel_error,
                'max_error': (fp32_act - quant_act).abs().max().item()
            }
    
    return errors

def validate_quantization(original_model: nn.Module, 
                          quantized_model: nn.Module, 
                          test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """
    본디 모형과 수 줄인 모형의 맞음을 견준다.
    """
    original_model.eval()
    quantized_model.eval()
    
    original_correct = 0
    quantized_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # 본디 모형
            out_orig = original_model(data)
            pred_orig = out_orig.argmax(dim=1)
            original_correct += (pred_orig == target).sum().item()
            
            # 수 줄인 모형
            out_quant = quantized_model(data)
            pred_quant = out_quant.argmax(dim=1)
            quantized_correct += (pred_quant == target).sum().item()
            
            total += target.size(0)
    
    orig_acc = original_correct / total
    quant_acc = quantized_correct / total
    
    print(f"본디 맞음: {orig_acc*100:.2f}%")
    print(f"수 줄인 맞음: {quant_acc*100:.2f}%")
    print(f"맞음 떨어짐: {(orig_acc - quant_acc)*100:.2f}%")
    
    return orig_acc, quant_acc

def measure_quantization_impact(original: nn.Module,
                                quantized: nn.Module,
                                test_loader: torch.utils.data.DataLoader,
                                device: str = 'cpu') -> Dict:
    """
    본디 모형과 수 줄인 모형의 됨됨이를 견준다.
    """
    # 모형 크기
    def get_size_mb(model):
        param_size = sum(p.nelement() * p.element_size() 
                        for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() 
                         for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    orig_size = get_size_mb(original)
    quant_size = get_size_mb(quantized)
    
    # 미루어 보는 때
    sample_input = next(iter(test_loader))[0][:1].to(device)
    
    # 몸풀기
    for _ in range(10):
        with torch.no_grad():
            _ = original(sample_input)
            _ = quantized(sample_input.cpu())
    
    # 때 재기
    n_runs = 100
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = original(sample_input)
    orig_time = (time.time() - start) / n_runs
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = quantized(sample_input.cpu())
    quant_time = (time.time() - start) / n_runs
    
    return {
        'original_size_mb': orig_size,
        'quantized_size_mb': quant_size,
        'compression_ratio': orig_size / quant_size,
        'original_latency_ms': orig_time * 1000,
        'quantized_latency_ms': quant_time * 1000,
        'speedup': orig_time / quant_time
    }
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
45장이 다루는 최적화 기법(양자화·가지치기·지식 증류·아키텍처 탐색)을 정확도 손실, 추론 속도 향상, 구현의 번거로움으로 비교하여 트레이드오프를 설명하여라.

</div>

??? success "풀이"
    기법마다 트레이드오프의 양상이 다르다. 양자화(INT8)는 흔히 2~4배 빨라지면서 정확도 손실이 1% 미만이고, 프레임워크가 지원하므로 구현 부담이 중간쯤이다. 가지치기는 희소성의 형태에 따라 속도 향상이 들쭉날쭉하며(구조적 가지치기가 하드웨어에 더 맞다) 정확도 손실은 1~3%이다. 지식 증류는 추론 방식 자체는 그대로 두되 더 작은 학생을 써서 2~10배로 압축하고 정확도 손실은 1~5%이다. 뉴럴 네트워크 아키텍처 탐색은 가장 좋은 구조를 찾아 주지만 탐색에 엄청난 계산이 든다(GPU 수천 시간). 금융 응용에서는 받아들일 수 있는 정확도 손실이 오류의 비용에 달려 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
단순한 순방향 네트워크에 트레인 후 양자화(INT8)를 구현하고, 표준 데이터셋에서 정확도가 얼마나 떨어지고 추론이 얼마나 빨라지는지 재어라.

</div>

??? success "풀이"
    PyTorch의 양자화 API를 쓴다. (1) float32 모델을 베이스라인 정확도까지 트레인한다. (2) 동적 양자화에는 `torch.quantization.quantize_dynamic`을 쓰고, 정적 양자화에는 대표 데이터로 캘리브레이션한다. (3) 추론 지연(배치 1000개의 평균)과 테스트 데이터의 정확도를 잰다. 흔한 결과: CPU에서 1.5~3배 빨라지고, 동적 양자화는 정확도가 0.5% 미만, 캘리브레이션한 정적 양자화는 0.2% 미만 떨어진다. 모델 크기는 약 4배 줄어든다(FP32에서 INT8로). 핵심: 정적 양자화에는 배포 환경의 데이터를 잘 대표하는 캘리브레이션 데이터가 있어야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
배포된 모델의 데이터 드리프트, 개념 드리프트, 성능 저하를 탐지하는 프로덕션 모니터링 시스템을 설계하여라. 지표와 알림 임계를 밝혀라.

</div>

??? success "풀이"
    세 층위를 모니터링한다. (1) 데이터 드리프트: KS 검정이나 PSI(모집단 안정성 지수)로 입력 피처의 분포를 추적한다. 어느 피처든 PSI > 0.2이면 알린다. (2) 개념 드리프트: 예측 분포의 이동과 (얻을 수 있으면) 실제 레이블 분포를 추적한다. 예측의 평균이 베이스라인 기간 대비 표준편차 2배 넘게 이동하면 알린다. (3) 모델 성능 저하: 슬라이딩 윈도로 실시간 정확도와 로스를 추적한다. 정확도가 베이스라인보다 3% 넘게 떨어지거나 지연이 SLA를 넘으면(p99 > 50ms 등) 알린다. Grafana로 대시보드를 만들고, Prometheus에 지표를 저장하고, PagerDuty로 알림을 보낸다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 트레이딩 시스템의 지연 요건이 웹 서비스와 근본적으로 다른 까닭을 설명하여라. 이것이 배포 최적화 전략에 어떤 영향을 주는가?

</div>

??? success "풀이"
    웹 서비스는 100~500ms의 지연과 이따금의 지연 급증을 허용한다. 트레이딩 시스템은 결정적으로(deterministic) 1밀리초 아래(고빈도 거래에서는 흔히 100마이크로초 미만)여야 한다. 그래서 최적화 전략이 달라진다. (1) 가비지 컬렉션 멈춤을 없앤다(파이썬 대신 C++ 추론). (2) 메모리를 미리 다 할당해 둔다(동적으로 할당하지 않는다). (3) 스레드를 코어에 고정한다(컨텍스트 스위칭을 없앤다). (4) 지연이 가장 중요한 경로에는 FPGA나 ASIC을 쓴다. (5) 양자화는 필요하되 비결정적인 반올림을 들여서는 안 된다. 배치 추론은 쓸 수 없다(판단 하나하나가 지연에 민감하다). 배포 스택은 처리량보다 최악의 경우 지연(p99.9)을 우선한다. $\square$

## 정리하며

양자화는 효율적인 배포의 핵심 도구다.

1. **동적 양자화**: 적용하기 쉽고 RNN/트랜스포머에 좋다
2. **정적 양자화**: 압축 효과가 가장 크나 캘리브레이션이 있어야 한다
3. **QAT**: 정확도가 가장 좋으나 트레인 인프라가 있어야 한다
4. **FP16**: 단순하고 GPU에서 잘 빨라지며 정확도 손실이 적다

| 상황 | 권장 방법 |
|----------|-------------------|
| 빨리 배포 | 동적 양자화 |
| 가장 좋은 정확도 | QAT |
| CNN/비전 | 정적 양자화 |
| RNN/트랜스포머 | 동적 양자화 |
| GPU에 배포 | FP16 / INT8 TensorRT |

핵심 권장 사항:

- 손쉬운 이득을 얻으려면 동적 양자화에서 시작한다
- CNN/비전 모델에는 정적 양자화를 쓴다
- 정확도 하락을 받아들일 수 없으면 QAT를 적용한다
- 양자화한 뒤에는 늘 정확도를 확인한다
- 배포 데이터를 잘 대표하는 캘리브레이션 데이터를 쓴다
- 필요하면 첫 레이어와 마지막 레이어를 더 높은 정밀도로 둔다

**살펴볼 거리**

1. Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR 2018.
2. Banner, R., et al. "Post Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment." NeurIPS 2019.
3. Nagel, M., et al. "Data-Free Quantization Through Weight Equalization and Bias Correction." ICCV 2019.
4. Gholami, A., et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." arXiv 2021.
5. Krishnamoorthi, R. "Quantizing Deep Convolutional Networks for Efficient Inference." arXiv 2018.
