# 혼합 정밀도 학습 - 더 빠르게, 더 적은 메모리로

혼합 정밀도는 연산마다 적절한 자료형을 사용해 속도와 메모리 비용을 조절한다. 이 예시는 CUDA의 autocast와 그래디언트 스케일링 흐름을 보여 주며, 이득은 장치·모델·배치 크기에 따라 달라진다.

## 1. 코드

```python
"""튜토리얼 24: 섞인 촘촘함 익히기 - 기억 자리를 덜 쓰며 더 빠르게"""
import torch
import torch.nn as nn
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. What is Mixed Precision?")
    print("""
    섞인 촘촘함 익히기:
    - 앞으로/역전파에는 float16(반 촘촘함)을 쓴다
    - 파라메터를 고칠 때는 float32(온 촘촘함)을 쓴다
    - 좋은 점: 2~3배 빨라지고 기억 자리가 50% 준다
    - 필요한 것: 텐서 코어를 지닌 GPU(V100, A100, RTX 20/30 계열)
    """)
    
    header("2. Float16 vs Float32")
    x_32 = torch.tensor([1.0], dtype=torch.float32)
    x_16 = torch.tensor([1.0], dtype=torch.float16)
    print(f"Float32: {x_32.dtype}, size: {x_32.element_size()} bytes")
    print(f"Float16: {x_16.dtype}, size: {x_16.element_size()} bytes")
    print(f"Memory savings: {100 * (1 - x_16.element_size()/x_32.element_size()):.0f}%")
    
    header("3. Automatic Mixed Precision (AMP)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()  # Prevents underflow
    
    print("Using autocast for mixed precision:")
    x = torch.randn(32, 100, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    # AMP를 쓰는 학습 단계
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():  # Automatic mixed precision
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
    
    scaler.scale(loss).backward()  # Scale loss to prevent underflow
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Loss computed in mixed precision: {loss.item():.4f}")
    
    header("4. Manual Mixed Precision")
    model_fp16 = model.half()  # Convert to float16
    print("Model converted to float16:")
    for name, param in model_fp16.named_parameters():
        print(f"  {name}: {param.dtype}")
    
    header("5. When to Use Mixed Precision")
    print("""
    다음 때에 섞인 촘촘함을 쓴다.
    ✓ 큰 모델을 익힐 때
    ✓ GPU 기억 자리가 넉넉하지 않을 때
    ✓ 맞는 GPU(텐서 코어)이 있을 때
    ✓ 배치 크기가 목이 될 때
    
    조심할 것:
    ✗ 아주 작은 그래디언트(GradScaler을 써라)
    ✗ 맞춤 셈(fp16을 받치지 않을 수 있다)
    ✗ 수치가 흔들리는 탈
    """)
    
    header("6. Complete Training Example")
    print("""
    # 표준 학습
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
    # 혼합 정밀도 학습
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    """)

if __name__ == "__main__":
    main()
```

**출력:**

```

======================================================================
1. What is Mixed Precision?
======================================================================

    섞인 촘촘함 익히기:
    - 앞으로/역전파에는 float16(반 촘촘함)을 쓴다
    - 파라메터를 고칠 때는 float32(온 촘촘함)을 쓴다
    - 좋은 점: 2~3배 빨라지고 기억 자리가 50% 준다
    - 필요한 것: 텐서 코어를 지닌 GPU(V100, A100, RTX 20/30 계열)
    

======================================================================
2. Float16 vs Float32
======================================================================
Float32: torch.float32, size: 4 bytes
Float16: torch.float16, size: 2 bytes
Memory savings: 50%

======================================================================
3. Automatic Mixed Precision (AMP)
======================================================================
Using autocast for mixed precision:
Loss computed in mixed precision: 2.3975

======================================================================
4. Manual Mixed Precision
======================================================================
Model converted to float16:
  0.weight: torch.float16
  0.bias: torch.float16
  2.weight: torch.float16
  2.bias: torch.float16

======================================================================
5. When to Use Mixed Precision
======================================================================

    다음 때에 섞인 촘촘함을 쓴다.
    ✓ 큰 모델을 익힐 때
    ✓ GPU 기억 자리가 넉넉하지 않을 때
    ✓ 맞는 GPU(텐서 코어)이 있을 때
    ✓ 배치 크기가 목이 될 때
    
    조심할 것:
    ✗ 아주 작은 그래디언트(GradScaler을 써라)
    ✗ 맞춤 셈(fp16을 받치지 않을 수 있다)
    ✗ 수치가 흔들리는 탈
    

======================================================================
6. Complete Training Example
======================================================================

    # 표준 학습
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
    # 혼합 정밀도 학습
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
```

## 2. 논의

`autocast`는 지원되는 연산의 입력·연산 정밀도를 선택한다. 모델 전체를 무조건 float16으로 바꾸는 것과 다르며, 수치적으로 민감한 연산은 더 높은 정밀도를 사용할 수 있다.

float16의 작은 기울기가 0으로 사라지는 것을 줄이기 위해 로스를 확대해 역전파하고, 갱신 전에 그래디언트를 다시 축소한다. 스케일러는 비유한 그래디언트를 감지하면 갱신을 건너뛸 수 있다. 클리핑은 스케일을 해제한 그래디언트에 적용해야 한다.

기록된 코드는 기존 `torch.cuda.amp` API를 사용한다. 현재 API는 `torch.amp.autocast`·`torch.amp.GradScaler`이며 이 페이지의 코드는 원래 형태를 유지했다. bfloat16은 float16보다 지수 범위가 넓어 그래디언트 스케일링의 필요성이 다를 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SGD 대신 Adam 옵티마이저를 쓰도록 코드를 수정하라. 100 에포크에 걸친 수렴 속도를 비교하라.

</div>

??? success "풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 러닝레이트와 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 러닝레이트는
    # 보통 SGD보다 작다.
    ```

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 로스에 미치는 영향을 설명하라.

</div>

??? success "풀이"
    업데이트 사이에 경사를 초기화하지 않으면 서로 다른 파라미터에서 계산한 이전 경사까지 누적된다. 따라서 현재 로스의 경사만 사용하는 학습 규칙과 달라지고, 진동이나 발산을 일으킬 수 있다. 경사끼리 상쇄될 수도 있으므로 크기가 항상 증가하거나 반드시 발산하는 것은 아니다. 의도적인 경사 누적에서는 정해진 여러 마이크로배치를 처리한 뒤 한 번 업데이트하고 경사를 초기화한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
옵티마이저에 L2 정규화(웨이트 감쇠)를 추가하고 그것이 최종 파라메터 값에 어떤 영향을 주는지 관찰하라.

</div>

??? success "풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 로스에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 웨이트를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 웨이트의 크기가 조금 더 작아진다.
    ```

## 정리하며

혼합 정밀도는 연산마다 적절한 자료형을 사용해 속도와 메모리 비용을 조절한다. 이 예시는 CUDA의 autocast와 그래디언트 스케일링 흐름을 보여 주며, 이득은 장치·모델·배치 크기에 따라 달라진다.
