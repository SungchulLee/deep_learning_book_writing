# 모델 체크포인팅 - 모델 저장과 불러오기

이 스크립트는 모델 체크포인팅, 즉 모델을 저장하고 불러오는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""튜토리얼 27: 모델 되짚음 저장 - 모델 저장하고 불러오기"""
import torch
import torch.nn as nn
import torch.optim as optim
import os

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
    os.makedirs('/home/claude/checkpoints', exist_ok=True)
    
    header("1. Saving Model State Dict")
    model = SimpleModel()
    print("Model created:")
    print(model)
    
    torch.save(model.state_dict(), '/home/claude/checkpoints/model_weights.pth')
    print("\nModel weights saved to 'model_weights.pth'")
    print("This saves only the parameters, not the architecture!")
    
    header("2. Loading Model State Dict")
    new_model = SimpleModel()  # Must create architecture first
    new_model.load_state_dict(torch.load('/home/claude/checkpoints/model_weights.pth'))
    print("Weights loaded into new model")
    
    # 가중치가 일치하는지 확인
    param1 = list(model.parameters())[0]
    param2 = list(new_model.parameters())[0]
    print(f"Weights match: {torch.equal(param1, param2)}")
    
    header("3. Saving Entire Model")
    torch.save(model, '/home/claude/checkpoints/full_model.pth')
    print("Full model saved (architecture + weights)")
    
    loaded_model = torch.load('/home/claude/checkpoints/full_model.pth')
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
    torch.save(checkpoint, '/home/claude/checkpoints/training_checkpoint.pth')
    print("Training checkpoint saved with:")
    print(f"  - Epoch: {epoch}")
    print(f"  - Model weights")
    print(f"  - Optimizer state")
    print(f"  - Loss: {loss}")
    
    header("5. Resuming Training")
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())
    
    checkpoint = torch.load('/home/claude/checkpoints/training_checkpoint.pth')
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
        torch.save(model.state_dict(), '/home/claude/checkpoints/best_model.pth')
        print(f"New best model saved! Loss: {best_loss:.4f}")
    
    header("7. Model Versioning")
    epoch = 5
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }, f'/home/claude/checkpoints/model_epoch_{epoch}.pth')
    print(f"Checkpoint saved: model_epoch_{epoch}.pth")
    print("Useful for comparing different training stages!")
    
    header("8. Saving for Inference Only")
    model.eval()  # Set to evaluation mode
    torch.save(model.state_dict(), '/home/claude/checkpoints/inference_model.pth')
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
    2. 학습 상태(판, 최적화기, 손실)를 저장하라
    3. 자리를 아끼려면 가장 좋은 것 N개만 남겨라
    4. 알아보기 쉬운 이름을 써라(판, 자, 날짜)
    5. 에폭마다 또는 N 걸음마다 저장하라
    6. 저장한 뒤 제대로 불러와지는지 따져라
    7. 다른 기기에서 불러올 때는 map_location='cpu'을 써라
    8. 곁들인 정보(초매개변수 따위)도 저장하라
    9. 불러오는 코드를 자주 시험하라
    10. 추론 전용 저장를 따로 두어라
    
    이름 보기: model_epoch50_loss0.123_acc0.95.pth
    """)
    
    # 정리
    import shutil
    if os.path.exists('/home/claude/checkpoints'):
        shutil.rmtree('/home/claude/checkpoints')
    print("\nCheckpoint files cleaned up.")

if __name__ == "__main__":
    main()```

## 논의

PyTorch의 `nn.Module`은 신경망 구조를 정의하는 체계적인 방법을 제공한다. 각 모듈이 자신의 매개변수와 하위 모듈을 관리하므로 모델을 살펴보고, 저장하고, 장치 사이에 옮기기가 간편하다.

모델 체크포인팅은 학습 진행 상황을 디스크에 저장하여 중단으로부터의 복구와 모델 배포를 가능하게 한다. 모델 전체를 피클로 저장하는 것보다 (매개변수만 담은) `state_dict`를 저장하는 편이 낫다. 이식성이 좋고 정확한 클래스 정의 경로에 의존하지 않기 때문이다.

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
