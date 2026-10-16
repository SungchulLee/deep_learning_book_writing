# 모델 체크포인팅 - 모델 저장과 불러오기

학습을 이어 가거나 추론 모델을 옮기려면 어떤 상태를 저장할지 먼저 정해야 한다. 웨이트만 복원하는 경우와 최적화 상태까지 복원하는 체크포인트를 구별한다.

## 1. 코드

```python
"""튜토리얼 27: 모델 되짚음 저장 - 모델 저장하고 불러오기"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

# ========================================================================
# 메인
# ========================================================================

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def main():
    # 체크포인트용 임시 디렉터리 만들기
    os.makedirs('checkpoints', exist_ok=True)
    
    header("1. Saving Model State Dict")
    model = SimpleModel()
    print("Model created:")
    print(model)
    
    torch.save(model.state_dict(), 'model_weights.pth')
    print("\nModel weights saved to 'model_weights.pth'")
    print("This saves only the parameters, not the architecture!")
    
    header("2. Loading Model State Dict")
    new_model = SimpleModel()  # Must create architecture first
    new_model.load_state_dict(torch.load('model_weights.pth'))
    print("Weights loaded into new model")
    
    # 웨이트가 일치하는지 확인
    param1 = list(model.parameters())[0]
    param2 = list(new_model.parameters())[0]
    print(f"Weights match: {torch.equal(param1, param2)}")
    
    header("3. Saving Entire Model")
    torch.save(model, 'full_model.pth')
    print("Full model saved (architecture + weights)")
    
    loaded_model = torch.load('full_model.pth')
    print("Full model loaded")
    print("Note: This requires the model class definition to be available!")
    
    header("4. Saving Training Checkpoint")
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    epoch = 10
    loss = 0.123
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, 'training_checkpoint.pth')
    print("Training checkpoint saved with:")
    print(f"  - Epoch: {epoch}")
    print(f"  - Model weights")
    print(f"  - Optimizer state")
    print(f"  - Loss: {loss}")
    
    header("5. Resuming Training")
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    
    checkpoint = torch.load('training_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Last loss: {last_loss}")
    print("Ready to continue training!")
    
    header("6. Save Best Model Only")
    best_loss = float('inf')
    current_loss = 0.1
    
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved! Loss: {best_loss:.4f}")
    
    header("7. Model Versioning")
    epoch = 5
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }, f'model_epoch_{epoch}.pth')
    print(f"Checkpoint saved: model_epoch_{epoch}.pth")
    print("Useful for comparing different training stages!")
    
    header("8. Saving for Inference Only")
    model.eval()  # Set to evaluation mode
    torch.save(model.state_dict(), 'inference_model.pth')
    print("Inference-only model saved")
    print("Remember to call model.eval() before inference!")
    
    header("9. Cross-Platform Compatibility")
    print("""
    가장 널리 맞물리게 하려면
    
    # 저장
    torch.save(model.state_dict(), 'model.pth', _use_new_zipfile_serialization=True)
    
    # 장치 대응과 함께 불러오기
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    
    # 그런 다음 원하는 장치로 옮긴다
    model = model.to(device)
    """)
    
    header("10. Best Practices")
    print("""
    모델 되짚음 저장을 잘 하는 버릇:
    
    1. 너그럽게 쓰려면 온 모델이 아니라 state_dict을 저장하라
    2. 학습 상태(판, 최적화기, 로스)를 저장하라
    3. 자리를 아끼려면 가장 좋은 것 N개만 남겨라
    4. 알아보기 쉬운 이름을 써라(판, 자, 날짜)
    5. 에폭마다 또는 N 스텝마다 저장하라
    6. 저장한 뒤 제대로 불러와지는지 따져라
    7. 다른 기기에서 불러올 때는 map_location='cpu'을 써라
    8. 곁들인 정보(초파라메터 따위)도 저장하라
    9. 불러오는 코드를 자주 시험하라
    10. 추론 전용 저장을 따로 두어라
    
    이름 보기: model_epoch50_loss0.123_acc0.95.pth
    """)
    
    # 정리
    import shutil
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    print("\nCheckpoint files cleaned up.")

if __name__ == "__main__":
    main()
```

??? note "전체 출력 (96줄)"

    ```

    ======================================================================
    1. Saving Model State Dict
    ======================================================================
    Model created:
    SimpleModel(
      (fc1): Linear(in_features=10, out_features=20, bias=True)
      (fc2): Linear(in_features=20, out_features=1, bias=True)
    )

    Model weights saved to 'model_weights.pth'
    This saves only the parameters, not the architecture!

    ======================================================================
    2. Loading Model State Dict
    ======================================================================
    Weights loaded into new model
    Weights match: True

    ======================================================================
    3. Saving Entire Model
    ======================================================================
    Full model saved (architecture + weights)
    Full model loaded
    Note: This requires the model class definition to be available!

    ======================================================================
    4. Saving Training Checkpoint
    ======================================================================
    Training checkpoint saved with:
      - Epoch: 10
      - Model weights
      - Optimizer state
      - Loss: 0.123

    ======================================================================
    5. Resuming Training
    ======================================================================
    Resuming from epoch 10
    Last loss: 0.123
    Ready to continue training!

    ======================================================================
    6. Save Best Model Only
    ======================================================================
    New best model saved! Loss: 0.1000

    ======================================================================
    7. Model Versioning
    ======================================================================
    Checkpoint saved: model_epoch_5.pth
    Useful for comparing different training stages!

    ======================================================================
    8. Saving for Inference Only
    ======================================================================
    Inference-only model saved
    Remember to call model.eval() before inference!

    ======================================================================
    9. Cross-Platform Compatibility
    ======================================================================

        가장 널리 맞물리게 하려면
        
        # 저장
        torch.save(model.state_dict(), 'model.pth', _use_new_zipfile_serialization=True)
        
        # 장치 대응과 함께 불러오기
        model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        
        # 그런 다음 원하는 장치로 옮긴다
        model = model.to(device)
        

    ======================================================================
    10. Best Practices
    ======================================================================

        모델 되짚음 저장을 잘 하는 버릇:
        
        1. 너그럽게 쓰려면 온 모델이 아니라 state_dict을 저장하라
        2. 학습 상태(판, 최적화기, 로스)를 저장하라
        3. 자리를 아끼려면 가장 좋은 것 N개만 남겨라
        4. 알아보기 쉬운 이름을 써라(판, 자, 날짜)
        5. 에폭마다 또는 N 스텝마다 저장하라
        6. 저장한 뒤 제대로 불러와지는지 따져라
        7. 다른 기기에서 불러올 때는 map_location='cpu'을 써라
        8. 곁들인 정보(초파라메터 따위)도 저장하라
        9. 불러오는 코드를 자주 시험하라
        10. 추론 전용 저장을 따로 두어라
        
        이름 보기: model_epoch50_loss0.123_acc0.95.pth
        

    Checkpoint files cleaned up.
    ```


## 2. 논의

`state_dict`는 파라메터뿐 아니라 등록된 영속 버퍼도 담는다. 불러올 때는 이름과 모양이 호환되는 모델 구조가 필요하다. 전체 객체를 저장하는 방식보다 클래스 경로 의존성이 작지만 구조 정의가 불필요해지는 것은 아니다.

학습 재개에는 모델 외에 옵티마이저·스케줄러·AMP 스케일러의 상태와 진행 단계가 필요할 수 있다. 정확한 재현을 원하면 난수 상태와 데이터 순서도 고려해야 한다. 다른 장치로 읽을 때는 `map_location`을 사용하고, 추론 시에는 평가 모드와 그래디언트 추적 중단을 별도로 설정한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
SGD 대신 Adam 옵티마이저를 쓰도록 코드를 수정하라. 100 에포크에 걸친 수렴 속도를 비교하라.

</div>

??? success "연습문제 1 풀이"
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

??? success "연습문제 2 풀이"
    업데이트 사이에 경사를 초기화하지 않으면 서로 다른 파라미터에서 계산한 이전 경사까지 누적된다. 따라서 현재 로스의 경사만 사용하는 학습 규칙과 달라지고, 진동이나 발산을 일으킬 수 있다. 경사끼리 상쇄될 수도 있으므로 크기가 항상 증가하거나 반드시 발산하는 것은 아니다. 의도적인 경사 누적에서는 정해진 여러 마이크로배치를 처리한 뒤 한 번 업데이트하고 경사를 초기화한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
옵티마이저에 L2 정규화(웨이트 감쇠)를 추가하고 그것이 최종 파라메터 값에 어떤 영향을 주는지 관찰하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    # weight_decay는 로스에 L2 벌점항 lambda * ||w||^2을 더한다.
    # 이는 웨이트를 작게 유도하여 과적합을 막을 수 있다.
    # 최종 웨이트의 크기가 조금 더 작아진다.
    ```

## 정리하며

학습을 이어 가거나 추론 모델을 옮기려면 어떤 상태를 저장할지 먼저 정해야 한다. 웨이트만 복원하는 경우와 최적화 상태까지 복원하는 체크포인트를 구별한다.
