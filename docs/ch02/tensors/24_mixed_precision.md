# 혼합 정밀도 학습 - 더 빠르게, 더 적은 메모리로

이 스크립트는 혼합 정밀도 학습으로 더 빠르게 더 적은 메모리를 쓰는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 1. 코드

```python
"""튜토리얼 24: 섞인 촘촘함 익히기 - 기억 자리를 덜 쓰며 더 빠르게"""
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. What is Mixed Precision?")
    print("""
    섞인 촘촘함 익히기:
    - 앞으로/역전파에는 float16(반 촘촘함)을 쓴다
    - 매개변수를 고칠 때는 float32(온 촘촘함)을 쓴다
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
    ✗ 아주 작은 기울기(GradScaler을 써라)
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

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사 초기화, 역전파, 매개변수 갱신이다. 각 구성 요소가 결정적인 역할을 한다. 최적화기는 갱신 규칙(SGD, Adam 등)을 캡슐화하고 학습률과 모멘텀 상태를 내부에서 관리한다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

## 연습문제

**연습문제 1.**
SGD 대신 Adam 최적화기를 쓰도록 코드를 수정하라. 100 에폭에 걸친 수렴 속도를 비교하라.

??? success "연습문제 1 풀이"
    ```python
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Adam은 적응적 학습률과 모멘텀 덕분에 보통 SGD보다
    # 빠르게 수렴한다. 다만 Adam의 최적 학습률은
    # 보통 SGD보다 작다.
    ```

---


**연습문제 2.**
학습 루프에서 `optimizer.zero_grad()`를 없애면 어떤 일이 생기는가? 실험해 보고 학습 손실에 미치는 영향을 설명하라.

??? success "연습문제 2 풀이"
    `optimizer.zero_grad()`가 없으면 경사가 반복에 걸쳐 누적된다. 실효 경사가 매 단계 커져서 매개변수 갱신이 점점 커진다. 학습이 불안정해지고 손실은 대개 발산한다. PyTorch가 경사 누적 패턴을 지원하기 위해 기본적으로 경사를 누적하기 때문이다.

---


**연습문제 3.**
최적화기에 L2 정칙화(가중치 감쇠)를 추가하고 그것이 최종 매개변수 값에 어떤 영향을 주는지 관찰하라.

??? success "연습문제 3 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 손실에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 가중치를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 가중치의 크기가 조금 더 작아진다.
    ```

## 정리하며

**다룬 것** — 혼합 정밀도 학습 - 더 빠르게, 더 적은 메모리로

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 3개로 직접 확인할 수 있다.
