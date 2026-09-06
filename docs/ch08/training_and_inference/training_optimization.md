# 트랜스포머의 학습 최적화
## 개요

트랜스포머를 잘 학습시키려면 그 고유한 성질, 곧 깊은 잔차 신경망, 기억이 제곱으로 느는 주의 기반 계산, 초매개변수에 민감함에 맞는 최적화 방법이 필요하다. 이 절에서는 학습률 일정, 규제 기법, 기억과 계산 비용을 다스리는 방법을 다룬다.

## 학습률 일정

### 예열 문제

트랜스포머는 학습 초기의 학습률에 민감하다. 무작위로 초기화된 주의 가중치가 분산이 큰 기울기를 내므로 처음 학습률이 크면 갱신이 흔들린다. 본디 트랜스포머 논문은 이를 풀려고 예열 일정을 들여왔다.

### 본디 트랜스포머의 일정

"Attention Is All You Need" 논문은 예열 동안 선형으로 늘었다가 단계 수의 제곱근의 역수에 비례해 잦아드는 일정을 쓴다.

$$
lr = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step}^{-0.5}, \; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)
$$

이 일정에는 두 국면이 있다.

1. **예열 국면**(단계 $\leq$ warmup\_steps): 학습률이 0에서 꼭대기 값 $d_{\text{model}}^{-0.5} \cdot \text{warmup\_steps}^{-0.5}$까지 선형으로 는다.
2. **잦아듦 국면**(단계 $>$ warmup\_steps): 학습률이 $\text{step}^{-0.5}$으로 잦아든다.

```python
import torch
import math

class TransformerLRScheduler:
    """
    예열을 갖춘 본디 트랜스포머의 학습률 일정.
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        """학습률을 고치고 단계 세개를 하나 올린다."""
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

### 예열을 곁들인 코사인 담금질

널리 쓰이는 다른 방법은 선형 예열과 코사인 감쇠를 함께 쓴다.

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

### 일정 견주기

| 일정 | 꼭대기 학습률 | 예열 | 감쇠 | 쓰는 곳 |
|----------|---------|--------|-------|---------|
| 제곱근의 역수 | $\propto d_{\text{model}}^{-0.5}$ | 선형 | $t^{-0.5}$ | 본디 트랜스포머 |
| 코사인 예열 | 조정 가능 | 선형 | 코사인 | GPT-3, LLaMA |
| 선형 예열과 선형 감쇠 | 조정 가능 | 선형 | 선형 | BERT |
| 예열 뒤 일정 | 조정 가능 | 선형 | 없음 | 미세 조정 |

## 규제 기법

### 기울기 자르기

기울기 자르기는 매개변수를 갱신하기 전에 기울기의 노름에 뚜껑을 씌워 기울기가 터지는 것을 막는다.

$$
\hat{g} = \begin{cases}
g & \text{if } \|g\| \leq \tau \\
\tau \cdot \frac{g}{\|g\|} & \text{if } \|g\| > \tau
\end{cases}
$$

여기서 $\tau$은 자르는 문턱값이다(트랜스포머에서는 대개 1.0).

```python
import torch.nn as nn

def train_step_with_clipping(model, optimizer, criterion, src, tgt, max_norm=1.0):
    """기울기 자르기를 쓰는 학습 단계."""
    model.train()
    optimizer.zero_grad()
    
    output = model(src, tgt)
    loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
    loss.backward()
    
    # 전역 노름으로 기울기를 자른다
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    
    optimizer.step()
    return loss.item(), grad_norm.item()
```

### 드롭아웃

트랜스포머는 구조의 여러 곳에 드롭아웃을 적용한다.

1. **위치 인코딩을 더한 뒤**: 합쳐진 임베딩의 성분을 무작위로 떨어뜨린다
2. **주의 가중치 뒤**(다중 머리 주의 안에서): 주의 연결을 무작위로 떨어뜨린다
3. **아래 층마다 그 뒤**(잔차를 더하기 전에): 아래 층의 출력을 떨어뜨린다
4. **순전파 신경망 안에서**: 중간 활성을 떨어뜨린다

흔한 드롭아웃 비율은 큰 모형에서 0.1, 작은 모형에서 0.3이다.

```python
class TransformerBlockWithDropout(nn.Module):
    """드롭아웃을 적용하는 자리를 모두 보인다."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout,           # 주의 가중치에 드롭아웃
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),       # 순전파 신경망 안의 드롭아웃
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)  # 주의 아래 층 뒤의 드롭아웃
        self.dropout2 = nn.Dropout(dropout)  # 순전파 아래 층 뒤의 드롭아웃
    
    def forward(self, x, mask=None):
        # 잔차를 곁들인 자기 주의
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # 잔차를 곁들인 순전파
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x
```

### 가중치 감쇠

가중치 감쇠(L2 규제)는 가중치 크기의 제곱에 비례하는 벌점을 더해 매개변수 값이 지나치게 커지지 않게 한다.

$$
\theta_{t+1} = \theta_t - \eta \left(\nabla_\theta \mathcal{L} + \lambda \theta_t\right)
$$

(트랜스포머의 표준 최적화기인) AdamW에서는 가중치 감쇠를 기울기를 거치지 않고 매개변수에 곧바로 적용하는데, 적응 학습률과 올바로 어울리려면 이것이 중요하다.

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
$$

```python
import torch.optim as optim

def configure_optimizer(model, lr=1e-4, weight_decay=0.01):
    """
    가중치 감쇠를 가중치 행렬에만 적용하고 편향이나 층 정규화
    매개변수에는 적용하지 않도록 AdamW를 설정한다.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 편향과 층 정규화 매개변수를 가중치 감쇠에서 뺀다
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

## 기억과 계산의 어려움

### 이차인 주의 비용

자기 주의는 시간 복잡도가 $O(n^2 \cdot d)$, 기억 복잡도가 $O(n^2)$이며 여기서 $n$은 수열 길이이다. $d_{\text{model}} = 1024$이고 $n = 2048$인 모형에서는 다음과 같다.

$$
\text{Attention matrix size per head} = n^2 = 2048^2 = 4{,}194{,}304 \text{ elements}
$$

float32으로 머리가 16개이면 주의 층 하나가 학습 중에 주의 행렬을 약 256MB 담는다.

### 기울기 검문점

기울기 검문점은 중간 활성을 담아 두는 대신 역전파 때 다시 셈하여 계산으로 기억을 산다.

- **검문점이 없으면**: 중간 활성을 모두 담아 둔다 → 기억이 $O(N \cdot L)$이며 여기서 $N$은 수열 길이, $L$은 층 수이다
- **검문점이 있으면**: 층의 입력만 담아 두고 활성은 다시 셈한다 → 기억이 $O(N + L)$이지만 학습이 약 33% 느리다

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedEncoder(nn.Module):
    """기억을 아끼려고 기울기 검문점을 쓰는 인코더."""
    
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

### 섞인 정밀도 학습

섞인 정밀도는 계산 대부분에 float16을 쓰고 수치에 민감한 연산(손실 크기 조정, 정규화, 소프트맥스)에는 float32을 지킨다.

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
    
    # float16 기울기가 밑으로 넘치지 않게 손실의 크기를 조정한다
    scaler.scale(loss).backward()
    
    # 자르기 전에 기울기의 크기를 되돌린다
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

# 사용법
scaler = GradScaler()
loss = train_step_mixed_precision(model, optimizer, criterion, src, tgt, scaler, device)
```

**섞인 정밀도로 아끼는 기억**:

| 부품 | float32 | float16 | 아낌 |
|-----------|---------|---------|---------|
| 모형 매개변수 | 매개변수당 4바이트 | 매개변수당 2바이트 | 50% |
| 활성 | 성분당 4바이트 | 성분당 2바이트 | 50% |
| 최적화기 상태 (Adam) | 매개변수당 12바이트 | 매개변수당 8바이트 | 33% |

### 효율적인 주의 얼개

긴 수열에서는 $O(n^2)$의 주의 비용을 줄이는 여러 대안이 있다.

| 방법 | 복잡도 | 방식 |
|--------|-----------|----------|
| 플래시 주의 | 시간 $O(n^2)$, 기억 $O(n)$ | 입출력을 고려한 타일 나누기 |
| 성긴 주의 | $O(n \sqrt{n})$ | 자리의 일부에만 주의한다 |
| 선형 주의 | $O(n)$ | 핵 어림 |
| 미끄러지는 창 | $O(n \cdot w)$ | 국소 주의 창 |

## 온전히 다듬은 학습 파이프라인

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
    
    AdamW, 예열 일정, 기울기 자르기, 섞인 정밀도,
    이름표 매끄럽게 하기, 검증 평가를 담는다.
    """
    device = torch.device(device)
    model = model.to(device)
    
    # 가중치 감쇠를 갈라 둔 최적화기
    optimizer = configure_optimizer(model, lr=1e-4, weight_decay=weight_decay)
    
    # 학습률 스케줄러
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=warmup_steps,
        total_steps=total_steps, lr_max=1e-4
    )
    
    # 이름표 매끄럽게 하기를 곁들인 손실
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

## 요약

잘 통하는 트랜스포머 학습은 몇 가지 핵심 방법을 아우른다.

1. **학습률 예열**은 무작위 초기화에서 오는 학습의 흔들림을 막는다. 본디의 제곱근 역수 일정과 코사인 예열이 모두 널리 쓰인다.
2. **기울기 자르기**(대개 노름 1.0 이하)는 기울기가 터지는 것을 막아 준다.
3. **가중치 감쇠**(AdamW와 함께)는 적응 학습률과 올바로 어울리면서 규제를 준다. 편향과 정규화 매개변수는 뺀다.
4. **드롭아웃**을 구조의 여러 곳에 적용한다.
5. **이름표 매끄럽게 하기**는 지나친 자신을 막고 일반화를 낫게 한다.
6. **기울기 검문점**과 **섞인 정밀도**는 기억의 한계 안에서 큰 모형을 학습시키는 데 꼭 필요하다.

## 참고 문헌

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." ICLR. (AdamW)
3. Micikevicius, P., et al. (2018). "Mixed Precision Training." ICLR.
4. Chen, T., et al. (2016). "Training Deep Nets with Sublinear Memory Cost." (Gradient checkpointing)
5. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.

## 연습문제

**연습문제 1.**
트랜스포머 학습의 핵심 초매개변수와 흔한 값을 들어라.

??? success "연습문제 1 풀이"
    학습률은 1e-4에서 5e-4, 예열은 4000단계(또는 전체의 1~5%), 배치 크기는 토큰 256~2048개, 가중치 감쇠는 0.01~0.1, 드롭아웃은 0.1, 기울기 자르기는 1.0, 최적화기는 AdamW($\beta_1=0.9, \beta_2=0.98$), 학습률 일정은 예열 뒤 코사인이나 선형 감쇠이다.

---

**연습문제 2.**
트랜스포머 학습에서 학습률 예열이 왜 중요한지 설명하라.

??? success "연습문제 2 풀이"
    트랜스포머가 처음의 큰 갱신에 민감한 것은 (1) 주의 가중치가 무작위여서 불안정하고, (2) 층 정규화 통계가 아직 자리 잡지 않았으며, (3) Adam의 이차 적률 어림이 초기에 치우쳐 있기 때문이다. 예열은 큰 학습률을 쓰기 전에 모형이 자리 잡게 해 준다. 예열이 없으면 학습이 갈라지기 일쑤이다.

---

**연습문제 3.**
섞인 정밀도 학습과 그것이 트랜스포머에 주는 이점을 설명하라.

??? success "연습문제 3 풀이"
    섞인 정밀도는 앞먹임과 역전파에 float16을(기억이 2배 줄고 행렬 곱이 빠르다), 최적화기 상태와 손실 크기 조정에 float32을 쓴다. 손실 크기 조정은 float16에서 기울기가 밑으로 넘치는 것을 막는다. 이점은 약 2배 빨라지고 기억이 약 2배 줄며 정확도에 미치는 영향은 무시할 만하다는 것이다. 파이토치에서는 `torch.cuda.amp.autocast()`와 `GradScaler`를 쓴다.

---

**연습문제 4.**
기울기 모으기란 무엇이며 트랜스포머 학습에서 언제 필요한가?

??? success "연습문제 4 풀이"
    기울기 모으기는 갱신하기 전에 여러 작은 배치에 걸쳐 기울기를 모아 큰 배치를 흉내 낸다. 바라는 배치 크기가 GPU 기억을 넘을 때 필요하다. 이를테면 배치 크기 2048에 GPU 8개, 작은 배치 32라면 $2048/(8 \times 32) = 8$단계에 걸쳐 모은다.
