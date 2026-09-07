# 기울기 관리

기울기 폭발과 소실은 딥러닝의 두 가지 근본 문제이다. 기울기 자르기는 기울기의 노름이나 값을 제한하여 불안정한 갱신을 막고, 기울기 누적은 GPU 메모리가 적어도 큰 배치를 흉내 낸다. 학습 중에 기울기 통계를 살펴보는 것은 학습의 문제를 진단하는 데 꼭 필요하다.

## 코드

```python
"""
================================================================================
중급 03: 기울기 다루기와 자르기
================================================================================

배울 내용:
- 기울기 폭발과 소실 이해하기
- 기울기 자르기 기법
- 큰 배치를 위한 기울기 누적
- 학습 중 기울기 살펴보기
- 기울기 관리의 좋은 관행

선수 지식:
- 입문자용 튜토리얼을 모두 마친다
- 역전파의 기본을 이해한다

소요 시간: 약 20분
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 80)
print("GRADIENT MANIPULATION AND CLIPPING")
print("=" * 80)

# ============================================================================
# 1절: 기울기 문제 이해하기
# ============================================================================
print("\n" + "-" * 80)
print("GRADIENT PROBLEMS IN DEEP LEARNING")
print("-" * 80)

print("""
TWO MAJOR GRADIENT PROBLEMS:

1. GRADIENT EXPLOSION:
   • Gradients become very large (>1000)
   • Causes unstable training
   • Weights update by huge amounts
   • Loss becomes NaN or Inf
   • Common in RNNs and very deep networks

2. GRADIENT VANISHING:
   • Gradients become very small (<0.001)
   • Early layers don't learn
   • Training is extremely slow
   • Common in deep networks without proper activation/normalization

SOLUTIONS:
   • Gradient clipping (for explosion)
   • Better architectures (ResNet, BatchNorm)
   • Better activations (ReLU instead of sigmoid)
   • Careful initialization
""")

# ============================================================================
# 2절: 기울기 폭발 모의실험
# ============================================================================
print("\n" + "-" * 80)
print("SIMULATING GRADIENT EXPLOSION")
print("-" * 80)

class ProblematicModel(nn.Module):
    """
    기울기 폭발이 일어나기 쉬운 모델
    (초기화가 나쁘고 정규화가 없다)
    """
    def __init__(self):
        super(ProblematicModel, self).__init__()
        # 처음 가중치가 크면 → 폭발할 수 있다
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        
        # 큰 가중치로 초기화 (나쁜 관행이다!)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0, std=2.0)  # 너무 크다!
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 문제가 있는 모델 만들기
bad_model = ProblematicModel()
optimizer = optim.SGD(bad_model.parameters(), lr=0.1)  # 학습률이 높으면 더 나빠진다
criterion = nn.MSELoss()

print("Training model WITHOUT gradient clipping:\n")

# 몇 단계 학습
for step in range(5):
    # 무작위 데이터 생성
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # 순전파
    outputs = bad_model(inputs)
    loss = criterion(outputs, targets)
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    
    # 기울기의 노름 확인
    total_norm = 0
    for p in bad_model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"Step {step+1}: Loss = {loss.item():.4f}, Gradient Norm = {total_norm:.4f}")
    
    # 갱신 (폭발할 수 있다!)
    optimizer.step()
    
    # 손실이 NaN이 되었는지 확인
    if torch.isnan(loss):
        print("\n⚠️  Training collapsed! Loss became NaN due to gradient explosion")
        break

# ============================================================================
# 3절: 노름으로 기울기 자르기
# ============================================================================
print("\n" + "-" * 80)
print("GRADIENT CLIPPING BY NORM")
print("-" * 80)

print("""
GRADIENT CLIPPING BY NORM:
  • Rescale gradients if their norm exceeds a threshold
  • Formula: g_clipped = (max_norm / norm(g)) × g  if norm(g) > max_norm
  • Preserves gradient direction
  • Most common clipping method
  
torch.nn.utils.clip_grad_norm_(parameters, max_norm)
""")

# 같은 구조의 새 모델 만들기
model = ProblematicModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)
max_norm = 1.0  # 기울기를 최대 노름 1.0으로 자른다

print(f"Training model WITH gradient clipping (max_norm={max_norm}):\n")

for step in range(5):
    # 무작위 데이터 생성
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # 순전파
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    
    # 자르기 전의 기울기 노름 계산
    total_norm_before = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** 0.5
    
    # 기울기 자르기
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # 자른 뒤의 기울기 노름 계산
    total_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5
    
    print(f"Step {step+1}: Loss = {loss.item():.4f}, "
          f"Grad Norm: {total_norm_before:.4f} → {total_norm_after:.4f}")
    
    # 매개변수 갱신
    optimizer.step()

print("\n✓ Training is stable! Gradients are clipped to max_norm")

# ============================================================================
# 4절: 값으로 기울기 자르기
# ============================================================================
print("\n" + "-" * 80)
print("GRADIENT CLIPPING BY VALUE")
print("-" * 80)

print("""
GRADIENT CLIPPING BY VALUE:
  • Clip each gradient element to [-clip_value, clip_value]
  • Simpler but can distort gradient direction
  • Less common than clipping by norm
  
torch.nn.utils.clip_grad_value_(parameters, clip_value)
""")

# 새 모델 만들기
model_value_clip = ProblematicModel()
optimizer_value = optim.SGD(model_value_clip.parameters(), lr=0.1)
clip_value = 0.5

print(f"Training with VALUE clipping (clip_value={clip_value}):\n")

for step in range(3):
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    outputs = model_value_clip(inputs)
    loss = criterion(outputs, targets)
    
    optimizer_value.zero_grad()
    loss.backward()
    
    # 자르기 전의 기울기 통계 보이기
    max_grad = max(p.grad.abs().max().item() 
                   for p in model_value_clip.parameters() if p.grad is not None)
    
    # 값으로 기울기 자르기
    torch.nn.utils.clip_grad_value_(model_value_clip.parameters(), clip_value)
    
    # 자른 뒤의 기울기 통계 보이기
    max_grad_after = max(p.grad.abs().max().item() 
                         for p in model_value_clip.parameters() if p.grad is not None)
    
    print(f"Step {step+1}: Max gradient: {max_grad:.4f} → {max_grad_after:.4f}")
    
    optimizer_value.step()

# ============================================================================
# 5절: 기울기 누적
# ============================================================================
print("\n" + "-" * 80)
print("GRADIENT ACCUMULATION")
print("-" * 80)

print("""
GRADIENT ACCUMULATION:
  • Simulate large batch sizes with limited memory
  • Accumulate gradients over multiple forward passes
  • Update parameters after N accumulation steps
  • Effective batch size = batch_size × accumulation_steps
  
쓰임새:
  • GPU memory is limited
  • Want to train with large effective batch sizes
  • Training very large models
""")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

model_accum = SimpleModel()
optimizer_accum = optim.SGD(model_accum.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 설정
batch_size = 8
accumulation_steps = 4  # 실효 배치 크기 = 8 × 4 = 32

print(f"Batch size: {batch_size}")
print(f"Accumulation steps: {accumulation_steps}")
print(f"Effective batch size: {batch_size * accumulation_steps}\n")

print("Training with gradient accumulation:")

for epoch in range(2):
    print(f"\nEpoch {epoch + 1}:")
    
    # 미니배치 4개 모의실험
    for step in range(4):
        # 작은 배치 생성
        inputs = torch.randn(batch_size, 10)
        targets = torch.randn(batch_size, 1)
        
        # 순전파
        outputs = model_accum(inputs)
        loss = criterion(outputs, targets)
        
        # 누적 횟수로 손실을 나눈다
        loss = loss / accumulation_steps
        
        # 역전파 (기울기 누적)
        loss.backward()
        
        print(f"  Step {step+1}: Loss = {loss.item() * accumulation_steps:.4f}")
        
        # accumulation_steps마다 매개변수 갱신
        if (step + 1) % accumulation_steps == 0:
            optimizer_accum.step()
            optimizer_accum.zero_grad()
            print("  → Parameters updated and gradients cleared")

print("\n✓ Gradient accumulation allows training with larger effective batch size")

# ============================================================================
# 6절: 기울기 살펴보기
# ============================================================================
print("\n" + "-" * 80)
print("MONITORING GRADIENTS DURING TRAINING")
print("-" * 80)

def compute_gradient_stats(model):
    """
    모델의 기울기에 대한 통계를 계산한다
    """
    stats = {
        'max': 0.0,
        'min': float('inf'),
        'mean': 0.0,
        'norm': 0.0,
        'num_params': 0
    }
    
    total_sum = 0.0
    total_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            # 통계 갱신
            stats['max'] = max(stats['max'], grad.abs().max().item())
            stats['min'] = min(stats['min'], grad.abs().min().item())
            total_sum += grad.sum().item()
            total_count += grad.numel()
            
            # 노름에 대한 기여 계산
            stats['norm'] += grad.norm().item() ** 2
            stats['num_params'] += 1
    
    if total_count > 0:
        stats['mean'] = total_sum / total_count
        stats['norm'] = stats['norm'] ** 0.5
    
    return stats

# 기울기 감시 시연
model_monitor = SimpleModel()
optimizer_monitor = optim.SGD(model_monitor.parameters(), lr=0.01)

print("Example gradient statistics:\n")

for step in range(3):
    inputs = torch.randn(16, 10)
    targets = torch.randn(16, 1)
    
    outputs = model_monitor(inputs)
    loss = criterion(outputs, targets)
    
    optimizer_monitor.zero_grad()
    loss.backward()
    
    # 기울기 통계를 계산하여 보이기
    stats = compute_gradient_stats(model_monitor)
    
    print(f"Step {step+1}:")
    print(f"  Gradient norm: {stats['norm']:.6f}")
    print(f"  Max gradient: {stats['max']:.6f}")
    print(f"  Min gradient: {stats['min']:.6f}")
    print(f"  Mean gradient: {stats['mean']:.6f}\n")
    
    optimizer_monitor.step()

# ============================================================================
# 7절: 좋은 관행
# ============================================================================
print("\n" + "-" * 80)
print("BEST PRACTICES FOR GRADIENT MANAGEMENT")
print("-" * 80)

print("""
✓ ALWAYS DO:

1. MONITOR GRADIENTS:
   • Log gradient norms during training
   • Watch for explosion (>10) or vanishing (<0.0001)
   • Use tensorboard or wandb for visualization

2. USE GRADIENT CLIPPING:
   • Especially for RNNs and transformers
   • Typical max_norm: 0.5 - 5.0
   • Start with 1.0 and adjust

3. PROPER INITIALIZATION:
   • Use Xavier/Kaiming initialization
   • Avoid large initial weights
   • PyTorch does this by default

4. BATCH NORMALIZATION:
   • Helps stabilize gradients
   • Reduces need for aggressive clipping

5. APPROPRIATE LEARNING RATE:
   • Too high → explosion
   • Too low → vanishing
   • Use learning rate schedulers

✗ AVOID:

1. IGNORING GRADIENT PROBLEMS:
   • If loss becomes NaN, investigate gradients
   • Don't just restart training

2. VERY DEEP NETWORKS WITHOUT SKIP CONNECTIONS:
   • Use ResNet-style architectures
   • Add skip connections

3. SIGMOID/TANH IN DEEP NETWORKS:
   • Causes vanishing gradients
   • Use ReLU or variants

4. ACCUMULATING GRADIENTS UNINTENTIONALLY:
   • Always call optimizer.zero_grad()
   • Before loss.backward()
""")

# ============================================================================
# 8절: 기울기 관리를 갖춘 완전한 학습 루프
# ============================================================================
print("\n" + "-" * 80)
print("COMPLETE EXAMPLE: Training Loop with Gradient Management")
print("-" * 80)

print("""
def train_with_gradient_management(model, train_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    max_norm = 1.0  # 기울기 자르기 문턱값
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # 순전파
            output = model(data)
            loss = criterion(output, target)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            
            # 기울기 자르기 (선택 사항이지만 권장한다)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # 기울기 살펴보기 (N 배치마다)
            if batch_idx % 100 == 0:
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                
                print(f'Epoch {epoch}, Batch {batch_idx}: '
                      f'Loss={loss.item():.4f}, '
                      f'Grad Norm={grad_norm:.4f}')
            
            # 매개변수 갱신
            optimizer.step()
""")

# ============================================================================
# 요약
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. GRADIENT PROBLEMS:
   • Explosion: Gradients too large → unstable training
   • Vanishing: Gradients too small → slow/no learning

2. GRADIENT CLIPPING:
   • By norm: torch.nn.utils.clip_grad_norm_(params, max_norm)
     → Preserves direction, most common
   • By value: torch.nn.utils.clip_grad_value_(params, clip_value)
     → Simpler but can distort direction

3. GRADIENT ACCUMULATION:
   • Simulate large batch sizes
   • Accumulate over multiple forward passes
   • Useful when GPU memory is limited

4. MONITORING:
   • Always track gradient norms
   • Warning signs: norm > 10 or norm < 0.0001
   • Log and visualize during training

5. IMPLEMENTATION:
   • Clip after backward(), before step()
   • Typical max_norm: 0.5 - 5.0 (start with 1.0)
   • Combine with proper initialization and architecture

6. WHEN TO USE CLIPPING:
   • RNNs/LSTMs: Almost always
   • Transformers: Very common
   • CNNs: Sometimes, especially if very deep
   • Simple networks: Rarely needed

다음 단계:
→ Add gradient monitoring to your training loops
→ Experiment with different clipping thresholds
→ Use gradient accumulation for larger effective batch sizes
→ Study advanced techniques (gradient centralization, etc.)
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

노름으로 자르기는 기울기 벡터의 L2 노름이 문턱값을 넘으면 벡터 전체의 배율을 줄여, 매개변수별 기울기의 상대적인 크기와 방향은 그대로 지킨다. 시간 단계나 층을 거치며 행렬 곱이 되풀이되어 기울기 폭발이 흔한 RNN과 트랜스포머의 표준 방식이다.

기울기 누적은 GPU 메모리를 그만큼 더 쓰지 않고도 실효 배치를 크게 하여 학습하게 해 준다. `optimizer.step()` 전에 `loss.backward()`을 여러 번 부르면 작은 배치들에 걸쳐 기울기가 쌓인다. 손실을 누적 횟수로 나누면 기울기의 크기가 큰 배치 하나를 쓴 것과 같아진다.

학습 중에 기울기의 통계(노름, 평균, 최솟값, 최댓값)를 살피면 문제를 미리 알아챌 수 있다. 기울기의 노름이 시간이 갈수록 커지면 폭발이 다가온다는 신호이고, 줄어들면 소실을 뜻할 수 있다. 이 통계를 TensorBoard나 Weights & Biases에 남기면 기울기의 건강 상태를 눈으로 살펴볼 수 있다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

