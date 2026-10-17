# 트랜스포머의 학습 최적화

트랜스포머 학습의 러닝레이트 스케줄, 정규화와 메모리 절감 방법을 살펴본다. 기법의 원리와 예제의 실제 적용 범위를 구분하며, 특정 설정이나 속도·메모리 배율이 모든 모델에서 성립한다고 가정하지 않는다.

## 1. 러닝레이트 일정

### 워밍업 문제

워밍업은 초기 러닝레이트를 낮게 시작해 점차 높이는 방법이다. 초기화·정규화 위치·옵티마이저와 큰 러닝레이트의 상호작용을 완화할 수 있다. 모든 트랜스포머가 같은 원인으로 불안정하거나 반드시 워밍업을 필요로 하는 것은 아니다.

### 원래 트랜스포머의 일정

아래 사용자 정의 스케줄러는 생성 시 러닝레이트를 설정하지 않는다. step 호출 순서에 따라 첫 갱신이 옵티마이저의 초기 러닝레이트로 실행될 수 있다. 갱신 번호와 사용할 러닝레이트의 대응을 먼저 정하고 양의 warmup_steps를 확인한다.

"Attention Is All You Need" 논문은 워밍업 동안 선형으로 늘었다가 단계 수의 제곱근의 역수에 비례해 잦아드는 일정을 쓴다.

$$
lr = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step}^{-0.5}, \; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)
$$

이 일정에는 두 국면이 있다.

1. **워밍업 국면**(단계 $\leq$ warmup\_steps): 러닝레이트가 0에서 꼭대기 값 $d_{\text{model}}^{-0.5} \cdot \text{warmup\_steps}^{-0.5}$까지 선형으로 는다.
2. **잦아듦 국면**(단계 $>$ warmup\_steps): 러닝레이트가 $\text{step}^{-0.5}$으로 잦아든다.

```python
import torch
import math

class TransformerLRScheduler:
    """
    예열을 갖춘 본디 트랜스포머의 러닝레이트 일정.
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        """러닝레이트를 고치고 단계 세개를 하나 올린다."""
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _compute_lr(self) -> float:
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
```

### 워밍업을 곁들인 코사인 감쇠

널리 쓰이는 다른 방법은 선형 워밍업과 코사인 감쇠를 함께 쓴다.

$$
lr(t) = \begin{cases}
lr_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\[6pt]
lr_{\min} + \frac{lr_{\max} - lr_{\min}}{2}\left(1 + \cos\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}}\right)\right) & t > T_{\text{warmup}}
\end{cases}
$$

```python
class CosineWarmupScheduler:
    """선형 예열을 곁들인 코사인 담금질."""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float = 1e-4,
        lr_min: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _compute_lr(self) -> float:
        if self.step_num <= self.warmup_steps:
            # 선형 워밍업
            return self.lr_max * self.step_num / self.warmup_steps
        else:
            # 코사인 감쇠
            progress = (self.step_num - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )
```

### 일정 비교

CosineWarmupScheduler는 total_steps를 넘은 progress를 제한하지 않아 이후 러닝레이트가 다시 오를 수 있다. 0≤warmup_steps<total_steps와 사용 범위를 확인해야 한다.

| 일정 | 꼭대기 러닝레이트 | 워밍업 | 감쇠 | 쓰는 곳 |
|----------|---------|--------|-------|---------|
| 제곱근의 역수 | $\propto d_{\text{model}}^{-0.5}$ | 선형 | $t^{-0.5}$ | 원래 트랜스포머 |
| 코사인 워밍업 | 조정 가능 | 선형 | 코사인 | GPT-3, LLaMA |
| 선형 워밍업과 선형 감쇠 | 조정 가능 | 선형 | 선형 | BERT |
| 워밍업 뒤 일정 | 조정 가능 | 선형 | 없음 | 미세 조정 |

---

## 2. 규제 기법

### 경사 클리핑

경사 클리핑는 파라메터를 갱신하기 전에 그래디언트의 노름에 뚜껑을 씌워 그래디언트가 터지는 것을 막는다.

$$
\hat{g} = \begin{cases}
g & \text{if } \|g\| \leq \tau \\
\tau \cdot \frac{g}{\|g\|} & \text{if } \|g\| > \tau
\end{cases}
$$

여기서 $\tau$은 자르는 임계값이다(트랜스포머에서는 대개 1.0).

```python
import torch.nn as nn

def train_step_with_clipping(model, optimizer, criterion, src, tgt, max_norm=1.0):
    """그래디언트 자르기를 쓰는 학습 단계."""
    model.train()
    optimizer.zero_grad()
    
    output = model(src, tgt)
    loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
    loss.backward()
    
    # 전역 노름으로 그래디언트를 자른다
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    
    optimizer.step()
    return loss.item(), grad_norm.item()
```

### 드롭아웃

트랜스포머는 구조의 여러 곳에 드롭아웃을 적용한다.

1. **위치 인코딩을 더한 뒤**: 합쳐진 임베딩의 성분을 무작위로 떨어뜨린다
2. **어텐션 웨이트 뒤**(멀티헤드 어텐션 안에서): 어텐션 연결을 무작위로 떨어뜨린다
3. **아래 레이어마다 그 뒤**(잔차를 더하기 전에): 아래 레이어의 출력을 떨어뜨린다
4. **순전파 뉴럴 넷트웍 안에서**: 중간 활성을 떨어뜨린다

드롭아웃 확률은 데이터·모델·학습 예산에 따라 선택한다. 큰 모델 0.1, 작은 모델 0.3 같은 고정 규칙은 없다. 평가에서는 끄며, 어텐션 드롭아웃 뒤 웨이트 행 합이 매번 1인 것도 아니다.

```python
class TransformerBlockWithDropout(nn.Module):
    """드롭아웃을 적용하는 자리를 모두 보인다."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout,           # 주의 웨이트에 드롭아웃
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),       # 순전파 뉴럴 넷트웍 안의 드롭아웃
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)  # 주의 아래 레이어 뒤의 드롭아웃
        self.dropout2 = nn.Dropout(dropout)  # 순전파 아래 레이어 뒤의 드롭아웃
    
    def forward(self, x, mask=None):
        # 잔차를 곁들인 자기 주의
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # 잔차를 곁들인 순전파
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x
```

### 웨이트 감쇠

L2 패널티는 로스에 웨이트 제곱합을 더해 경사에 lambda×theta 항을 만든다. 아래 첫 식은 SGD 형태에서의 관계다. 적응 옵티마이저에서는 이 항을 경사에 합치는 방식과 계수를 직접 감쇠하는 방식이 일반적으로 다르다.

$$
\theta_{t+1} = \theta_t - \eta \left(\nabla_\theta \mathcal{L} + \lambda \theta_t\right)
$$

AdamW는 적응 경사 갱신과 별도로 계수를 감쇠한다. 널리 쓰이는 선택이지만 유일한 표준이나 최적해를 보장하는 방법은 아니다. 아래 configure_optimizer는 이름에 norm·bias 등이 있는지로 분류하므로 ln_1.weight 같은 이름의 정규화 계수를 놓칠 수 있다. 실제 모듈·파라메터 그룹을 확인한다.

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
$$

```python
import torch.optim as optim

def configure_optimizer(model, lr=1e-4, weight_decay=0.01):
    """
    웨이트 감쇠를 웨이트 행렬에만 적용하고 바이어스나 레이어 정규화
    파라메터에는 적용하지 않도록 AdamW를 설정한다.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 바이어스와 레이어 정규화 파라메터를 웨이트 감쇠에서 뺀다
        if 'bias' in name or 'norm' in name or 'layernorm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-9)
```

---

## 3. 메모리와 계산의 어려움

### 이차인 어텐션 비용

셀프 어텐션은 시간 복잡도가 $O(n^2 \cdot d)$, 메모리 복잡도가 $O(n^2)$이며 여기서 $n$은 시퀀스 길이이다. $d_{\text{model}} = 1024$이고 $n = 2048$인 모델에서는 다음과 같다.

$$
\text{Attention matrix size per head} = n^2 = 2048^2 = 4{,}194{,}304 \text{ elements}
$$

배치 1개·헤드 16개에서 FP32 점수 한 벌은 256MiB다. 배치·레이어·여러 중간 행렬·경사·웨이트는 별도다. 메모리 효율 커널은 점수 전체를 저장하지 않을 수 있다.

### 활성 체크포인팅

활성 체크포인팅은 일부 중간 활성 대신 재계산에 필요한 입력을 저장해 역전파에서 해당 구간을 다시 계산한다. 경사 자체를 저장하는 체크포인트나 학습 재시작 파일과는 다르다.

- 저장·재계산 범위와 구간 길이에 따라 절감량이 달라진다.
- 아래처럼 모든 블록의 입력을 저장하면 경계 활성은 여전히 레이어 수에 비례할 수 있다.
- 메모리가 항상 O(N+L)이거나 학습이 정확히 33% 느려진다고 단정할 수 없다.
- 재계산 시 난수·모드·상태 변경의 일관성과 실제 메모리·시간을 확인한다.

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedEncoder(nn.Module):
    """기억을 아끼려고 그래디언트 검문점을 쓰는 인코더."""
    
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockWithDropout(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.use_checkpoint = True
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # 역전파 때 활성을 다시 셈한다
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        return x
```

### 혼합 정밀도 학습

혼합 정밀도는 지원되는 연산에 FP16·BF16 등을 사용하면서 민감한 계산의 정밀도를 유지한다. AMP가 모델 전체를 일괄 half로 변환하는 것은 아니다. FP16의 작은 경사에는 GradScaler가 도움이 되며, BF16은 지수 범위가 달라 같은 스케일링 요구를 갖지는 않는다. 장치·자료형 지원을 확인해야 한다.

```python
from torch.amp import autocast, GradScaler

def train_step_mixed_precision(
    model, optimizer, criterion, src, tgt, scaler, device
):
    """자동 섞인 정밀도를 쓰는 학습 단계."""
    model.train()
    optimizer.zero_grad()
    
    with autocast(device_type=device.type, dtype=torch.float16):
        output = model(src, tgt)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
    
    # float16 그래디언트가 밑으로 넘치지 않게 로스의 크기를 조정한다
    scaler.scale(loss).backward()
    
    # 자르기 전에 그래디언트의 크기를 되돌린다
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

# 사용법
scaler = GradScaler()
loss = train_step_mixed_precision(model, optimizer, criterion, src, tgt, scaler, device)
```

**혼합 정밀도로 절감하는 메모리**:

| 저장 요소 | 확인할 점 |
|---|---|
| 웨이트 | AMP에서 원래 FP32 계수를 유지할 수 있음 |
| 경사 | 파라메터 dtype·학습 구현에 따라 다름 |
| 활성 | 일부 저정밀 연산에서 원소당 저장을 줄일 수 있음 |
| Adam 상태 | FP32의 m·v만 해도 계수당 8바이트, master weight 등은 별도 |

전체 메모리가 일괄 반으로 줄거나 속도가 두 배가 된다는 보장은 없다. 실제 활성·작업 공간·캐시·옵티마이저를 포함해 측정한다.

### 효율적인 어텐션 구조

긴 시퀀스에서는 $O(n^2)$의 어텐션 비용을 줄이는 여러 대안이 있다.

| 방법 | 줄이는 비용 | 조건 |
|---|---|---|
| FlashAttention | 전체 점수의 명시적 저장·메모리 접근 | 정확한 어텐션의 산술 비용은 여전히 길이에 이차 |
| 희소 어텐션 | 허용 연결만 계산 | 패턴·전용 커널에 따라 복잡도 결정 |
| 선형 어텐션 | 커널 피처의 결합 순서 | 피처 차원·근사 조건을 함께 확인 |
| 슬라이딩 윈도 | 위치별 연결 범위 | 고정 윈도와 실제 국소 커널이 필요 |

---

## 4. 학습 파이프라인 조각

이 함수는 앞서 정의한 configure_optimizer와 CosineWarmupScheduler, 호환되는 model과 배치를 가정한다. tgt는 한 토큰 이동하지만 원문·목표·메모리 패딩 마스크를 전달하지 않아 criterion의 ignore_index만으로 PAD가 어텐션에 끼치는 영향을 막지는 못한다.

첫 optimizer 갱신은 초기 lr=1e-4로 수행한 뒤 scheduler가 첫 워밍업 값을 설정한다. AMP에서 scaler가 비유한 경사로 갱신을 건너뛰어도 scheduler.step은 진행한다. 따라서 스케줄 단계와 실제 갱신 횟수를 구분해야 한다. warmup_steps가 총 단계보다 길면 감쇠 구간에 들어가지 않는다.

학습 로스는 AMP, 검증은 기본 정밀도에서 계산하고 두 로스 모두 배치 평균을 단순 평균한다. 검증 최고 웨이트는 파일에 저장하지만 복원·반환하지 않으며 옵티마이저·스케줄러·스케일러·난수 상태도 저장하지 않는다. 완전한 재시작 파이프라인은 아니다.

앞의 clipping·mixed_precision 단일 단계 조각은 model(src,tgt) 출력에 같은 tgt를 정답으로 주고 인과 마스크도 넘기지 않는다. 목표 시프트와 마스크를 처리하는 외부 모델을 가정하지 않으면 올바른 Seq2Seq 다음 토큰 학습이 아니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    d_model: int = 512,
    warmup_steps: int = 4000,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    use_amp: bool = True,
    device: str = "cuda"
):
    """
    표준 트랜스포머 최적화를 갖춘 온전한 학습 파이프라인.
    
    AdamW, 예열 일정, 그래디언트 자르기, 섞인 정밀도,
    이름표 매끄럽게 하기, 검증 평가를 담는다.
    """
    device = torch.device(device)
    model = model.to(device)
    
    # 웨이트 감쇠를 갈라 둔 최적화기
    optimizer = configure_optimizer(model, lr=1e-4, weight_decay=weight_decay)
    
    # 러닝레이트 스케줄러
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=warmup_steps,
        total_steps=total_steps, lr_max=1e-4
    )
    
    # 이름표 매끄럽게 하기를 곁들인 로스
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,                                # 채움을 무시한다
        label_smoothing=label_smoothing
    )
    
    # 섞인 정밀도
    scaler = GradScaler(enabled=use_amp)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # --- 학습 ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            with autocast(device_type=device.type, enabled=use_amp):
                tgt_mask = torch.triu(
                    torch.ones(tgt_input.size(1), tgt_input.size(1), device=device),
                    diagonal=1
                ).bool()
                
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_labels.reshape(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 검증 ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]
                
                tgt_mask = torch.triu(
                    torch.ones(tgt_input.size(1), tgt_input.size(1), device=device),
                    diagonal=1
                ).bool()
                
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_labels.reshape(-1)
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 가장 좋은 모형을 저장한다
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_transformer.pt")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
트랜스포머 최적화의 설정을 어떤 기준으로 정하고 기록해야 하는가?

</div>

??? success "풀이"
    러닝레이트·스케줄·유효 배치 토큰 수·Adam 계수·감쇠·드롭아웃·클리핑을 모델 초기화와 데이터에 맞춰 선택한다. 경사 누적·분산 평균·AMP로 실제 갱신 단위가 달라지는지도 기록한다. 원 논문의 4000 워밍업 단계를 모든 모델의 기본값으로 일반화하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
워밍업의 역할과 LayerNorm 통계에 대한 흔한 오해를 설명하라.

</div>

??? success "풀이"
    워밍업은 초기의 큰 갱신을 줄이고 초기화·정규화·옵티마이저와의 상호작용을 완화할 수 있다. LayerNorm은 입력마다 통계를 계산하며 BatchNorm 같은 이동 통계를 학습하는 레이어가 아니다. “통계가 자리 잡을 때까지 기다린다”는 설명은 부정확하고 워밍업 필요성도 설정별로 검증해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
AMP·GradScaler·클리핑의 순서와 메모리 해석을 설명하라.

</div>

??? success "풀이"
    지원 장치에서 autocast로 순전파·로스를 계산하고 스케일된 로스로 역전파한다. scaler.unscale_ 후 원래 경사에 클리핑하고 scaler.step·update를 적용한다. AMP는 계수·Adam 상태 전체를 절반 크기로 만드는 기능이 아니며 속도·메모리·정밀도는 실제 모델에서 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
실효 배치 2048개, 장치 8개, 장치당 마이크로배치 32개면 필요한 누적 단계는 얼마인가?

</div>

??? success "풀이"
    8단계다. 각 갱신이 같은 크기의 8×32×8 표본을 반영한다는 가정이다. 유효 토큰 수가 다르면 로스 웨이트를 조정해야 하며 마지막 미완성 누적 그룹과 분산 동기화도 처리한다. BatchNorm 등 배치 의존 연산이 있으면 큰 단일 배치와 완전히 같지는 않다.

## 정리하며

러닝레이트의 갱신 시점, 목표·마스크와 PAD 로스 처리부터 일치시킨다. AMP·클리핑·체크포인팅은 계산·메모리의 절충이며 보편적인 배율이나 안정성을 보장하지 않는다. 검증 모델 선택과 학습 재시작에 필요한 상태도 구분한다.

**참고 문헌**

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." ICLR. (AdamW)
3. Micikevicius, P., et al. (2018). "Mixed Precision Training." ICLR.
4. Chen, T., et al. (2016). "Training Deep Nets with Sublinear Memory Cost." (Gradient checkpointing)
5. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
