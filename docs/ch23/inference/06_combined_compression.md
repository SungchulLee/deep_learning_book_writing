# 모델 압축 방법 결합하기

이 예제는 MNIST 분류기를 학습한 뒤 웨이트 마스킹, 작은 학생 모델로의 증류, 양자화 단계의 결과 정리를 연결한다. 앞의 세 과정에는 실제 연산이 있지만 마지막 양자화 단계는 복사와 크기 추정만 한다. 따라서 완성된 INT8 압축 파이프라인의 실측 결과로 읽어서는 안 된다.

## 1. 예제 코드

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
            
            # 앎 내리기 로스
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
       ✓ 정확도 로스 1% 미만: 거의 다 받아들일 만하다
       ✓ 로스 1~2%: 가장자리 펼치기에는 받아들일 만하다
       ✓ 로스 2% 넘음: 꼼꼼히 밸류해야 한다
    
    4. 검증:
       ✓ 목표 하드웨어(CPU/GPU/손전화)에서 시험한다
       ✓ 이론이 아니라 실제 늦음을 잰다
       ✓ 짐이 걸린 상태에서 기억 공간 씀씀이를 살핀다
       ✓ 다양한 시험 배치에서 정확도를 확인한다
    
    5. 다음 스텝:
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
    main()
```

## 2. 교사와 학생의 구조

두 모델의 입력은 B×1×28×28이고 출력은 B×10 로짓이다. 교사는 풀링 뒤 B×128×14×14, B×256×7×7을 만들고, 12544→512→10 분류기를 사용한다. 학생은 채널이 32와 64이며 3136→128→10 분류기를 사용한다. 교사만 분류기 중간에 드롭아웃이 있다.

바이어스를 포함한 파라미터 수는 교사 6,724,618개, 학생 421,642개로 약 15.95배 차이가 난다. 두 모델이 모두 FP32이면 웨이트 바이트 수도 같은 비율이다. 이는 구조를 작게 만든 효과이며 증류 로스 자체가 학생의 파라미터 수를 줄인 것은 아니다.

## 3. 웨이트 마스킹의 역할과 한계

교사를 3에포크 학습한 뒤 복사본의 각 행렬형 웨이트에서 절댓값 중앙값을 임계값으로 정한다. `>=`를 사용하므로 동점이 많으면 정확히 절반이 제거되지 않을 수 있다. 바이어스는 이 단계의 대상이 아니다.

마스크는 웨이트를 0으로 만들 뿐 텐서의 크기를 줄이거나 희소 형식으로 저장하지 않는다. 코드도 가지치기 후 크기를 원래 `teacher_size`로 기록한다. 제거 직후의 교사를 복구 파인튜닝하지 않고 학생의 교사로 쓰므로, 마스킹으로 손상된 예측도 증류 목표에 포함된다. 가지치기가 증류에 항상 도움이 된다고 가정할 수 없다.

## 4. 증류 로스와 학습 조건

학생은 정답의 교차 엔트로피와 교사 분포의 KL 발산을 결합해 3에포크 학습한다.

$$L=0.3\,\operatorname{CE}(z_s,y)+0.7\,T^2D_{KL}(p_t^{(T)}\|p_s^{(T)}),\qquad T=4.$$

`F.kl_div`에 학생의 로그 확률과 교사의 확률을 넣으므로 KL 방향은 교사에서 학생이다. 교사는 `eval()`과 `no_grad()`로 고정하고 학생만 갱신한다. 교사와 학생은 같은 클래스 순서를 사용해야 한다.

증류의 효과를 분리하려면 같은 작은 구조를 정답만으로 학습한 기준 학생이 필요하다. 이 예제에는 그 비교가 없어 학생과 교사의 정확도 차이를 증류 효과만으로 해석할 수 없다.

## 5. 마지막 단계는 실제 양자화가 아니다

`quantized_student=copy.deepcopy(student)`는 FP32 학생을 그대로 복사한다. 정수 변환이나 저비트 저장을 하지 않으며 `quantized_size=student_size/4`는 INT8을 가정한 계산이다. 정확도도 새로 평가하지 않고 학생의 값을 복사한다. 따라서 실제 파일 크기, 정수 추론 정확도와 속도는 측정되지 않았다.

`Total Compression Ratio`는 교사와 학생의 **파라미터 개수 비율**인 약 15.95배다. 반면 추정 바이트 비율에 가상의 4배를 곱하면 약 63.79배가 된다. 이 두 지표는 정의가 다르며 후자는 실제 저장 결과가 아니다. 추론 지연 시간을 측정하는 코드도 없다.

## 6. 정확도 단위와 공정한 평가

이 디렉터리의 `utils.evaluate_accuracy`는 0~100 백분율을 반환한다. 호출부가 다시 100을 곱해 출력하므로 표시 정확도가 잘못된다. 두 백분율의 차이도 이미 퍼센트포인트인데, 요약에서 다시 100을 곱한다. 본문의 품질 판단에는 이 단위 문제를 고려해야 한다.

`get_model_size()['mb']`는 파라미터와 버퍼 바이트를 1024²로 나눈 MiB이며 직렬화 파일이나 최대 실행 메모리가 아니다. 마스크·양자화 메타데이터와 실제 저장 형식을 포함한 비교도 별도로 필요하다.

출력되는 배포 지침과 “최소 로스로 10~20배 압축”은 실행에서 검증한 보편적 결론이 아니다. 압축 순서와 허용 품질 저하는 목적에 따라 정하고, 기준 학생·가지치기 복구 여부·실제 양자화를 분리한 비교로 각 단계의 기여를 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
배치 크기 4에서 교사와 학생의 두 풀링 출력 및 분류기 차원을 비교하라.

</div>

??? success "풀이"

    교사는 4×128×14×14 → 4×256×7×7 → 4×12544 → 4×512 → 4×10이다. 학생은 4×32×14×14 → 4×64×7×7 → 4×3136 → 4×128 → 4×10이다. 클래스 수는 같으므로 출력 분포를 비교할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 RGB 입력으로 바꾸려면 두 모델에서 어떤 차원을 바꾸어야 하는가?

</div>

??? success "풀이"

    첫 컨볼루션의 입력 채널을 둘 다 3으로 바꾼다. 두 풀링 뒤 공간 크기는 16×16이므로 교사의 첫 선형 입력은 256×16×16=65536, 학생은 64×16×16=16384가 된다. 중간 선형 출력과 클래스 수는 유지할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
가지치기 후 파라미터 개수가 그대로인 이유와 마지막 양자화 단계에서 실제로 바뀌는 항목을 설명하라.

</div>

??? success "풀이"

    가지치기는 기존 텐서 원소에 0을 대입하므로 원소 수와 dtype이 그대로다. 마지막 단계도 학생을 복사할 뿐 실제 dtype과 웨이트는 바꾸지 않는다. 바뀌는 것은 결과 사전의 추정 크기 값뿐이다. 실제 압축 효과를 주장하려면 저장 형식이나 구조를 변환하고 다시 측정해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
깊이가 달라지는 교사와 학생을 구성할 때 증류와 압축 평가에 필요한 조건을 설명하라. 아래 코드 조각은 무엇을 제공하는가?

</div>

??? success "풀이"

    아래 ModuleList는 하위 모듈 등록의 골격이며 YourBlock이 정의되지 않아 완성된 확장 코드가 아니다. 각 블록의 채널·공간 크기와 분류기 연결을 정해야 한다. 출력 클래스 순서와 수가 같으면 깊이가 달라도 출력 증류가 가능하다. 구조 변경의 효과와 증류 효과를 나누려면 같은 학생 구조의 비증류 기준과 실제 바이트·속도 측정이 필요하다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

압축 방법을 결합할 때는 각 단계에서 실제로 바뀐 텐서와 저장 형식을 확인한다. 이 예제는 교사 학습, 마스킹과 학생 증류를 포함하지만 최종 양자화 크기는 추정값이다. 정확도 단위 오류와 파라미터 비율·바이트 비율의 차이를 구분하고 완성된 배포 성능으로 확대 해석하지 않는다.
