# 기울기 관리

기울기 폭발과 소실은 딥러닝의 두 가지 근본 문제이다. 기울기 자르기는 기울기의 노름이나 값을 제한하여 불안정한 갱신을 막고, 기울기 누적은 GPU 메모리가 적어도 큰 배치를 흉내 낸다. 학습 중에 기울기 통계를 살펴보는 것은 학습의 문제를 진단하는 데 꼭 필요하다.

## 1. 코드

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
기울기의 두 가지 큰 문제:

1. 기울기 폭발:
   • 기울기가 매우 커진다(1000 초과)
   • 학습이 불안정해진다
   • 가중치가 엄청나게 크게 갱신된다
   • 손실이 NaN이나 Inf가 된다
   • RNN과 아주 깊은 신경망에서 흔하다

2. 기울기 소실:
   • 기울기가 매우 작아진다(0.001 미만)
   • 앞쪽 층이 학습하지 못한다
   • 학습이 매우 느리다
   • 활성화나 정규화가 알맞지 않은 깊은 신경망에서 흔하다

SOLUTIONS:
   • 기울기 절단(폭발에 대비)
   • 더 나은 구조(ResNet, 배치 정규화)
   • 더 나은 활성화(시그모이드 대신 ReLU)
   • 신중한 초기화
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
노름 기준 기울기 절단:
  • 노름이 임계값을 넘으면 기울기의 크기를 다시 맞춘다
  • 식: norm(g) > max_norm이면 g_clipped = (max_norm / norm(g)) × g
  • 기울기 방향을 보존한다
  • 가장 흔한 절단 방법이다
  
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
값 기준 기울기 절단:
  • 기울기 원소마다 [-clip_value, clip_value]로 자른다
  • 더 단순하지만 기울기 방향을 일그러뜨릴 수 있다
  • 노름 기준 절단보다 덜 쓴다
  
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
기울기 누적:
  • 적은 메모리로 큰 배치 크기를 흉내 낸다
  • 여러 번의 순전파에 걸쳐 기울기를 쌓는다
  • N번 쌓은 뒤 매개변수를 갱신한다
  • 유효 배치 크기 = batch_size × accumulation_steps
  
쓰임새:
  • GPU 메모리가 부족하다
  • 유효 배치 크기를 크게 하여 학습하고 싶다
  • 아주 큰 모델을 학습한다
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

1. 기울기를 살펴라:
   • 학습 중 기울기 노름을 기록한다
   • 폭발(10 초과)이나 소실(0.0001 미만)을 살피라
   • 시각화에는 텐서보드나 wandb를 쓴다

2. 기울기 절단을 써라:
   • 특히 RNN과 트랜스포머에서
   • 흔한 max_norm 값: 0.5~5.0
   • 1.0으로 시작하여 조정한다

3. 알맞은 초기화:
   • 자비에르나 카이밍 초기화를 쓴다
   • 초기 가중치를 크게 두지 않는다
   • PyTorch가 기본으로 이렇게 한다

4. 배치 정규화:
   • 기울기를 안정시키는 데 도움이 된다
   • 과감한 절단의 필요를 줄인다

5. 알맞은 학습률:
   • 너무 크면 → 폭발
   • 너무 작으면 → 소실
   • 학습률 스케줄러를 쓴다

✗ AVOID:

1. 기울기 문제를 무시하기:
   • 손실이 NaN이 되면 기울기를 조사하라
   • 그냥 학습을 다시 시작하지 마라

2. 건너뛰기 연결 없는 아주 깊은 신경망:
   • ResNet 방식 구조를 쓴다
   • 건너뛰기 연결을 더한다

3. 깊은 신경망에서 시그모이드나 tanh 쓰기:
   • 기울기 소실을 일으킨다
   • ReLU나 그 변형을 쓴다

4. 뜻하지 않게 기울기가 쌓이기:
   • 늘 optimizer.zero_grad()를 부르라
   • loss.backward() 앞에서
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
1. 기울기 문제:
   • 폭발: 기울기가 너무 크다 → 불안정한 학습
   • 소실: 기울기가 너무 작다 → 학습이 느리거나 멈춘다

2. 기울기 절단:
   • 노름 기준: torch.nn.utils.clip_grad_norm_(params, max_norm)
     → 방향을 보존하며 가장 흔하다
   • 값 기준: torch.nn.utils.clip_grad_value_(params, clip_value)
     → 더 단순하지만 방향을 일그러뜨릴 수 있다

3. 기울기 누적:
   • 큰 배치 크기를 흉내 낸다
   • 여러 번의 순전파에 걸쳐 쌓는다
   • GPU 메모리가 부족할 때 쓸모 있다

4. MONITORING:
   • 늘 기울기 노름을 추적하라
   • 경고 신호: 노름이 10을 넘거나 0.0001보다 작다
   • 학습 중에 기록하고 시각화하라

5. IMPLEMENTATION:
   • backward() 뒤, step() 앞에서 자른다
   • 흔한 max_norm: 0.5~5.0(1.0으로 시작하라)
   • 알맞은 초기화와 구조를 함께 쓰라

6. 언제 절단을 쓰는가:
   • RNN과 LSTM: 거의 언제나
   • 트랜스포머: 매우 흔하다
   • CNN: 때때로, 특히 아주 깊을 때
   • 단순한 신경망: 거의 필요 없다

다음 단계:
→ 학습 루프에 기울기 감시를 더하라
→ 절단 임계값을 달리하여 실험해 보라
→ 유효 배치 크기를 키우려면 기울기 누적을 쓰라
→ 고급 기법(기울기 중심화 등)을 살펴보라
""")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 2. 논의

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

## 정리하며

**다룬 것** — 기울기 관리

노름으로 자르기는 기울기 벡터의 L2 노름이 문턱값을 넘으면 벡터 전체의 배율을 줄여, 매개변수별 기울기의 상대적인 크기와 방향은 그대로 지킨다.

핵심 클래스는 `ProblematicModel`, `SimpleModel`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
