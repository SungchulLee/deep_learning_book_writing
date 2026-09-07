# 수 줄이기
## 두루 보기

수 줄이기는 신경 그물의 짐과 살림을 뜨는 소수점(흔히 32비트)에서 더 적은 비트(8비트, 4비트, 나아가 두 값)으로 촘촘함을 낮춘다. 이로써 기억 자리와 셈 값이 크게 줄고, 이를 받쳐 주는 쇠 붙임새에서는 미루어 봄이 크게 빨라진다.

## 수 줄이기 밑바탕

### 왜 수를 줄이는가

신경 그물의 짐과 살림은 흔히 32비트 뜨는 소수로 담긴다.

```
FP32: ±[1.18e-38, 3.4e+38], 열 자리로 약 7자리 촘촘함
      부호 (1) | 지수 (8) | 가수 (23)
```

신경 그물의 셈은 거의 이만한 촘촘함이 있어야 하지 않다. 수 줄이기는 값을 더 낮은 촘촘함으로 옮긴다.

```
INT8: [-128, 127] 또는 [0, 255]
      모두 8비트

FP16: ±[6.1e-5, 65504], 열 자리로 약 3자리
      부호 (1) | 지수 (5) | 가수 (10)
```

### 수 줄이기의 나은 점

| 자 | FP32 → INT8 | FP32 → FP16 |
|--------|-------------|-------------|
| 모형 크기 | 4배 줄어듦 | 2배 줄어듦 |
| 기억 너비 | 4배 줄어듦 | 2배 줄어듦 |
| 셈 빠르기 | 2~4배 빠름* | 1.5~2배 빠름* |
| 맞음 잃음 | 흔히 0~2% | 흔히 0.5% 미만 |

*빨라짐은 쇠 붙임새가 받쳐 주는지에 매인다

## 수학 밑바탕

### 수 줄이기 옮김

수 줄이기는 이어지는 뜨는 소수 값을 띄엄한 정수 모임으로 옮긴다.

$$x_{\text{float}} \in \mathbb{R} \rightarrow x_{\text{quant}} \in \{0, 1, \ldots, 2^b - 1\}$$

여기서 $b$은 비트 너비다.

### 고른 아핀 수 줄이기

가장 흔한 수 줄이기 얼개는 잣대와 0점을 지닌 선형 옮김을 쓴다.

**수 줄이기:**

$$x_q = \text{round}\left(\frac{x - z}{s}\right) = \text{round}\left(\frac{x}{s}\right) - z_q$$

**수 되돌리기:**

$$\hat{x} = s \cdot (x_q + z_q) = s \cdot x_q + z$$

여기서:

- $s$은 잣대 값
- $z$은 0점(치우침)
- $z_q$은 수 줄인 0점

### 잣대와 0점 셈하기

자리가 $[x_{\min}, x_{\max}]$인 텐서에서

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

$$z = q_{\min} - \text{round}\left(\frac{x_{\min}}{s}\right)$$

부호 없는 INT8에서는 $q_{\min} = 0$, $q_{\max} = 255$

부호 있는 INT8에서는 $q_{\min} = -128$, $q_{\max} = 127$

### 맞바꿈 대칭 대 어긋난 수 줄이기

**대칭 수 줄이기**($z = 0$):

$$x_q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x_{\min}|, |x_{\max}|)}{q_{\max}}$$

더 단순하나 한쪽으로 쏠린 분포에서는 쓸 자리를 버린다.

**어긋난 수 줄이기**($z \neq 0$):
위의 온전한 식을 쓴다. 한쪽으로 쏠린 분포(ReLU 뒤의 살림에서 흔하다)에서 수 줄이기 자리를 더 잘 쓴다.

## PyTorch로 짜기

### 밑바탕 수 줄이기 셈

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

## 수 줄이기 어긋남 살피기

### 수 줄이기 잡음

고른 수 줄이기의 어긋남은 거의 고른 분포를 따른다.

$$\epsilon = x - \hat{x} \sim \mathcal{U}\left(-\frac{s}{2}, \frac{s}{2}\right)$$

**평균 제곱 어긋남:**

$$\text{MSE} = \mathbb{E}[\epsilon^2] = \frac{s^2}{12}$$

**신호 대 수 줄이기 잡음 견줌(SQNR):**

$$\text{SQNR} = 10 \log_{10}\left(\frac{\sigma_x^2}{\text{MSE}}\right) \approx 6.02b + 4.77 \text{ dB}$$

비트를 하나 더할 때마다 SQNR이 약 6 dB 오른다.

### 어긋남 퍼짐

여러 켜 그물에서는 수 줄이기 어긋남이 쌓인다.

$$\epsilon_{\text{output}} = f(x + \epsilon_x, W + \epsilon_W) - f(x, W)$$

선형 켜에서는 어긋남이 켜의 너비와 깊이에 따라 커지므로 켜마다 다른 수 줄이기 꾀가 있어야 한다.

## 수 줄이기의 갈래

### 1. 움직이는 수 줄이기

짐은 미리 줄이고, 살림은 돌아갈 때 그때그때 줄인다.

**잘 맞는 자리**: 들임 크기가 바뀌는 모형, RNN, LSTM, 변환기.

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
        dtype: 수 줄이기 갈래(torch.qint8 또는 torch.float16)
        
    Returns:
        움직이는 수 줄이기를 건 모형
    """
    model.eval()
    
    # 앞선 크기를 잰다
    size_before = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 어느 켜 갈래를 줄일지 밝힌다
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

### 2. 붙박인 수 줄이기

눈금 맞추기 자료를 써서 짐과 살림을 모두 미리 줄인다.

**잘 맞는 자리**: 들임 크기가 붙박인 곳, CNN, 보기 모형.

```python
class QuantizableModel(nn.Module):
    """
    붙박인 수 줄이기에 맞게 마련한 모형.
    
    고갱이 고침:
    1. 들임/날임에 QuantStub/DeQuantStub을 더한다
    2. 함수 셈을 같은 일을 하는 묶음으로 바꾼다
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
        self.relu1 = nn.ReLU()  # F.relu이 아니라 묶음을 쓴다
        
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
        
        녹여 붙이면 가운데의 수 줄이기/되돌리기 걸음을 건너뛰어
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
            if batch_idx >= 100:  # 눈금 맞추기에 묶음 100개를 쓴다
                break
    
    # 수 줄인 모형으로 바꾼다
    torch.quantization.convert(model, inplace=True)
    
    return model
```

### 3. 수 줄이기를 아는 익힘(QAT)

익히는 동안 수 줄이기를 흉내 내어 그물이 수 줄이기 어긋남에 든든해지도록 배우게 한다.

**잘 맞는 자리**: 붙박인 수 줄이기로 맞음이 받아들일 수 없을 만큼 떨어질 때.

**앞으로 걸음:**

$$\hat{x} = \text{FakeQuant}(x) = s \cdot \text{round}\left(\frac{\text{clamp}(x, x_{\min}, x_{\max})}{s}\right)$$

**되돌아 걸음(곧장 지나가는 어림개):**

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \hat{x}} \cdot \mathbf{1}[x_{\min} \leq x \leq x_{\max}]$$

기울기는 (자르는 자리 안에서) 수 줄이기를 제 자리 함수인 양 지나간다.

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
        lr: 배움 비율
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
    
    # QAT을 마련한다(거짓 수 줄이기 묶음을 끼운다)
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 익힘 차림
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("수 줄이기를 아는 익힘을 비롯한다...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 몸풀기 뒤 묶음 잣대 잡기의 자를 얼린다
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

## 섞인 촘촘함(FP16)

요즘 GPU에서 더 빠르게 셈하려고 반 촘촘함을 쓴다.

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

## 눈금 맞추기 꾀

### 가장 작은 값-가장 큰 값 눈금 맞추기

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

### 잦기 그림 눈금 맞추기

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
        KL 갈림으로 가장 좋은 문턱을 셈한다(엔트로피 눈금 맞추기).
        
        수를 줄일 때 소식 잃음을 가장 작게 하는 문턱을 찾는다.
        """
        # 서비스에서는 PyTorch에 든 HistogramObserver을 쓴다
        pass
```

### 백분위 눈금 맞추기

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

## 갈래마다 대 텐서마다의 수 줄이기

### 텐서마다의 수 줄이기

온 텐서에 잣대/0점 하나:

- 짜기가 더 단순하다
- 갈래마다 짐 분포가 다르면 촘촘함을 잃을 수 있다

### 갈래마다의 수 줄이기

날임 갈래마다 따로 잣대/0점을 둔다.

- 짐 분포를 더 잘 지킨다
- 엮음 켜에는 꼭 있어야 한다
- 요즘 수 줄이기의 여느 길이다

```python
def analyze_weight_distribution(conv: nn.Conv2d) -> Dict:
    """
    날임 갈래마다 짐 분포를 살핀다.
    
    갈래마다 짐 자리가 크게 다를 때
    갈래마다의 수 줄이기가 도움이 된다.
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

## 섞인 촘촘함 수 줄이기

켜마다 수 줄이기에 예민한 정도가 다르다. 섞인 촘촘함은 켜마다 비트 너비를 달리 준다.

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

## 쇠 붙임새에서 헤아릴 것

### 받쳐 주는 셈

| 셈 | INT8 CPU | INT8 GPU | FP16 GPU |
|-----------|----------|----------|----------|
| Conv2d | ✓ | ✓ | ✓ |
| 선형 | ✓ | ✓ | ✓ |
| 묶음 잣대 잡기 | 녹여 붙임 | 녹여 붙임 | ✓ |
| ReLU | 녹여 붙임 | 녹여 붙임 | ✓ |
| 더하기 | ✓ | ✓ | ✓ |
| 이어 붙이기 | ✓ | ✓ | ✓ |
| 소프트맥스 | FP32으로 물러남 | FP32으로 물러남 | ✓ |

### 뒷단 고르기

```python
def select_quantization_backend(target_device: str) -> str:
    """
    과녁 쇠 붙임새에 맞는 수 줄이기 뒷단을 고른다.
    
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

## 좋은 버릇

### 켜의 예민함

첫 켜와 마지막 켜가 수 줄이기에 가장 예민한 것이 보통이다.

```python
def apply_mixed_precision_strategy(model: nn.Module,
                                   sensitive_layers: List[str] = None) -> nn.Module:
    """
    예민한 켜에는 더 높은 촘촘함을 쓴다.
    
    흔한 꾀:
    - 첫 엮음 켜: FP32이나 FP16(들임 소식을 지킨다)
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

### 묶음 잣대 잡기 접어 넣기

잘 들도록 묶음 잣대 잡기를 앞의 엮음/선형 켜에 접어 넣는다.

```python
def fold_batchnorm(model: nn.Module) -> nn.Module:
    """
    묶음 잣대 잡기 켜를 앞의 엮음 켜에 접어 넣는다.
    
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

### 눈금 맞추기 자료 길잡이

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

## 수 줄이기 어긋남 벌레잡기

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
    
    # 앞으로 걸음
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

## 간추림

잘 드는 내놓기에는 수 줄이기가 있어야 한다.

1. **움직이는 수 줄이기**: 걸기 쉽고 RNN/변환기에 좋다
2. **붙박인 수 줄이기**: 가장 잘 눌러 담으나 눈금 맞추기가 있어야 한다
3. **QAT**: 맞음이 가장 좋으나 익힘 바탕이 있어야 한다
4. **FP16**: 단순하고 GPU에서 잘 빨라지며 맞음 잃음이 적다

| 형편 | 즐겨 쓸 방법 |
|----------|-------------------|
| 빨리 내놓기 | 움직이는 수 줄이기 |
| 가장 좋은 맞음 | QAT |
| CNN/보기 | 붙박인 수 줄이기 |
| RNN/변환기 | 움직이는 수 줄이기 |
| GPU에 내놓기 | FP16 / INT8 TensorRT |

고갱이로 즐겨 쓸 길:

- 손쉬운 보람을 얻으려면 움직이는 수 줄이기에서 비롯한다
- CNN/보기 모형에는 붙박인 수 줄이기를 쓴다
- 맞음 떨어짐을 받아들일 수 없으면 QAT을 건다
- 수를 줄인 뒤에는 늘 맞음을 따진다
- 잘 드러내는 눈금 맞추기 자료를 쓴다
- 있어야 하면 첫 켜와 마지막 켜를 더 높은 촘촘함으로 둔다

## 살펴볼 거리

1. Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR 2018.
2. Banner, R., et al. "Post Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment." NeurIPS 2019.
3. Nagel, M., et al. "Data-Free Quantization Through Weight Equalization and Bias Correction." ICCV 2019.
4. Gholami, A., et al. "A Survey of Quantization Methods for Efficient Neural Network Inference." arXiv 2021.
5. Krishnamoorthi, R. "Quantizing Deep Convolutional Networks for Efficient Inference." arXiv 2018.

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
