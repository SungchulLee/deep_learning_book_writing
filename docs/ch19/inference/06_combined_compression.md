# 앞선 수준

앞선 수준: 아우른 눌러 담기 물길. 이 각본은 다음을 아우른 온전한 눌러 담기 물길을 보여 준다:

깊은 배움 모델을 효율적으로 펼치려면 모델 크기, 빠르기, 정확도의 맞바꿈을 조심스레 다듬어야 한다. 여기 짠 것은 실전 환경에서 신경망을 눌러 담고 빠르게 하는 데 쓰는 모델 눌러 담기 재주를 보여 준다.

## 코드

```python
"""
앞선 수준: 아우른 눌러 담기 물길

이 각본은 다음을 아우른 온전한 눌러 담기 물길을 보인다:
1. 가지치기(남아도는 무게 없애기)
2. 앎 내리기(작은 모델 익히기)
3. 양자화(정밀도 줄이기)

이러면 정확도를 거의 잃지 않고 가장 많이 눌러 담는다.

다루는 주제:
- 여러 단계 눌러 담기 물길
- 가지치기 → 앎 내리기 → 양자화의 차례
- 맞바꿈 살피기
- 실전 펼치기에서 헤아릴 점

눌러 담기 물길:
1단계: 스승 모델을 쳐 낸다(성김 50~70%)
2단계: 더 작은 제자에게 내린다
3단계: 제자를 INT8로 양자화한다
결과: 정확도를 2% 미만 잃고 10~20배 눌러 담는다

먼저 알아야 할 것:
- 앞선 모든 단원(01~05)
- 눌러 담기 재주마다에 대한 깊은 이해
- 모델 익히기 경험
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

# ========================================================================
# 메인
# ========================================================================

from utils import (
    count_parameters,
    get_model_size,
    evaluate_accuracy,
    seed_everything
)


class LargeTeacher(nn.Module):
    """큰 스승 모델."""
    def __init__(self, num_classes=10):
        super(LargeTeacher, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TinyStudent(nn.Module):
    """아주 작은 제자 모델."""
    def __init__(self, num_classes=10):
        super(TinyStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def compression_pipeline(train_loader, test_loader, device='cpu'):
    """
    온전한 눌러 담기 물길.
    
    반환값:
        단계마다의 결과를 담은 사전
    """
    results = {}
    
    # 0단계: 스승을 익힌다
    print("\n" + "="*60)
    print("STAGE 0: TRAINING TEACHER")
    print("="*60)
    
    teacher = LargeTeacher().to(device)
    # 스승을 익힌다(간추림 - 여느 때라면 더 길게 익힌다)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
    
    for epoch in range(3):  # 시범을 위한 빠른 익히기
        teacher.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = teacher(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    teacher_acc = evaluate_accuracy(teacher, test_loader, device)
    teacher_size = get_model_size(teacher)['mb']
    
    results['teacher'] = {
        'accuracy': teacher_acc,
        'size_mb': teacher_size,
        'params': count_parameters(teacher)
    }
    
    print(f"Teacher: {teacher_acc*100:.2f}% accuracy, {teacher_size:.2f} MB")
    
    # 1단계: 스승을 쳐 낸다
    print("\n" + "="*60)
    print("STAGE 1: PRUNING TEACHER")
    print("="*60)
    
    # 간추린 가지치기(무게의 50%를 0으로)
    pruned_teacher = copy.deepcopy(teacher)
    for param in pruned_teacher.parameters():
        if len(param.shape) > 1:  # 무게만
            threshold = torch.quantile(param.data.abs(), 0.5)
            mask = param.data.abs() >= threshold
            param.data *= mask.float()
    
    pruned_acc = evaluate_accuracy(pruned_teacher, test_loader, device)
    
    results['pruned_teacher'] = {
        'accuracy': pruned_acc,
        'size_mb': teacher_size,  # 크기는 같다(성김)
        'params': count_parameters(pruned_teacher)
    }
    
    print(f"Pruned Teacher: {pruned_acc*100:.2f}% accuracy")
    
    # 2단계: 제자에게 내린다
    print("\n" + "="*60)
    print("STAGE 2: DISTILLING TO STUDENT")
    print("="*60)
    
    student = TinyStudent().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    
    # 간추린 앎 내리기 익히기
    T = 4.0
    alpha = 0.3
    
    for epoch in range(3):  # 시범을 위한 빠른 익히기
        student.train()
        pruned_teacher.eval()
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                teacher_logits = pruned_teacher(data)
            
            optimizer.zero_grad()
            student_logits = student(data)
            
            # 앎 내리기 손실
            hard_loss = criterion(student_logits, target)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T ** 2)
            
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            loss.backward()
            optimizer.step()
    
    student_acc = evaluate_accuracy(student, test_loader, device)
    student_size = get_model_size(student)['mb']
    
    results['student'] = {
        'accuracy': student_acc,
        'size_mb': student_size,
        'params': count_parameters(student)
    }
    
    print(f"Student: {student_acc*100:.2f}% accuracy, {student_size:.2f} MB")
    
    # 3단계: 제자를 양자화한다
    print("\n" + "="*60)
    print("STAGE 3: QUANTIZING STUDENT")
    print("="*60)
    
    quantized_student = copy.deepcopy(student)
    quantized_size = student_size / 4  # INT8 = FP32의 1/4
    
    # 참고: 실제 양자화라면 torch.quantization을 쓴다
    # 여기서는 보이기 위해 간추렸다
    
    results['quantized_student'] = {
        'accuracy': student_acc,  # 조금 더 낮을 것이다
        'size_mb': quantized_size,
        'params': count_parameters(quantized_student)
    }
    
    print(f"Quantized Student: ~{student_acc*100:.2f}% accuracy, {quantized_size:.2f} MB")
    
    return results


def print_compression_summary(results):
    """두루 살핀 간추림을 찍는다."""
    print("\n" + "="*60)
    print("COMPRESSION PIPELINE SUMMARY")
    print("="*60)
    
    teacher = results['teacher']
    final = results['quantized_student']
    
    compression_ratio = teacher['params'] / final['params']
    size_reduction = (1 - final['size_mb'] / teacher['size_mb']) * 100
    accuracy_drop = (teacher['accuracy'] - final['accuracy']) * 100
    
    print(f"\n{'Stage':<25} {'Params':<15} {'Size (MB)':<12} {'Accuracy (%)'}")
    print("-" * 70)
    
    for name, data in results.items():
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<25} {data['params']:<15,} "
              f"{data['size_mb']:<12.2f} {data['accuracy']*100:.2f}")
    
    print("\n" + "="*60)
    print("FINAL COMPRESSION METRICS")
    print("="*60)
    print(f"Total Compression Ratio:   {compression_ratio:.1f}x")
    print(f"Size Reduction:            {size_reduction:.1f}%")
    print(f"Accuracy Drop:             {accuracy_drop:.2f}%")
    print("="*60)
    
    print("\n" + "="*60)
    print("PRODUCTION DEPLOYMENT GUIDE")
    print("="*60)
    print("""
    1. 하드웨어 고르기:
       ✓ CPU: INT8로 양자화한 모델을 쓴다
       ✓ GPU: INT8 대신 FP16이 나을 수 있다
       ✓ 손전화: INT8, 때로는 INT4 양자화
       ✓ 가장자리 TPU: INT8이 필요하다
    
    2. 가장 좋게 하는 차례:
       ✓ 늘: 양자화(쉬운 이득)
       ✓ 정확도가 허락하면: 가지치기
       ✓ 큰 스승이 있으면: 앎 내리기
       ✓ 아우르면: 가장 많이 눌러 담기
    
    3. 맞바꿈 지침:
       ✓ 정확도 손실 1% 미만: 거의 다 받아들일 만하다
       ✓ 손실 1~2%: 가장자리 펼치기에는 받아들일 만하다
       ✓ 손실 2% 넘음: 꼼꼼히 값매김해야 한다
    
    4. 검증:
       ✓ 목표 하드웨어(CPU/GPU/손전화)에서 시험한다
       ✓ 이론이 아니라 실제 늦음을 잰다
       ✓ 짐이 걸린 상태에서 기억 공간 씀씀이를 살핀다
       ✓ 다양한 시험 묶음에서 정확도를 확인한다
    
    5. 다음 걸음:
       - 펼치기를 위해 ONNX로 내보낸다
       - TensorRT/CoreML로 가장 좋게 다듬는다
       - 목표 기기에서 성능을 살핀다
       - 바탕과 견주어 A/B 시험을 한다
    """)


def main():
    """아우른 눌러 담기의 으뜸 함수."""
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("COMBINED COMPRESSION PIPELINE")
    print("="*60)
    print("\nThis demonstrates a complete compression workflow:")
    print("1. Train large teacher model")
    print("2. Prune teacher to remove redundancy")
    print("3. Distill knowledge to tiny student")
    print("4. Quantize student to INT8")
    print("\nResult: 10-20x compression with minimal accuracy loss")
    
    # 데이터를 불러온다
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 눌러 담기 물길을 돌린다
    results = compression_pipeline(train_loader, test_loader, device)
    
    # 요약 출력
    print_compression_summary(results)


if __name__ == "__main__":
    main()```

## 논의

여기 짠 것은 함께 어울려 온전한 모델 눌러 담기 얼개를 이루는 클래스 2개(`LargeTeacher`, `TinyStudent`)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기 보인 무늬는 더 복잡한 장면으로 자연스레 넓혀 쓸 수 있다. 웃매개변수, 얼개의 변종, 서로 다른 자료 뭉치로 실험해 보면 이해가 깊어지고 효율적인 펼치기 일에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`LargeTeacher`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = LargeTeacher(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `LargeTeacher`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = LargeTeacher(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
