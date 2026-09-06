# TensorBoard MNIST 주석본

TensorBoard는 학습 지표, 모델 구조, 평가 결과를 대화형으로 보여 준다. 이 스크립트는 MNIST 분류를 하면서 학습 손실, 정확도, 표본 이미지, 계산 그래프, 클래스별 정밀도-재현율 곡선을 남기는 법을 보인다. TensorBoard는 브라우저에서 `tensorboard --logdir=runs`으로 연다.

## 코드

```python
"""TensorBoard MNIST 주석본."""
# [Code Source](https://github.com/patrickloeber/pytorchTutorial)
# ============================================================================
# TensorBoard 시각화와 함께하는 MNIST 신경망 학습
# ============================================================================
# 이 스크립트는 다음을 보인다:
# - MNIST 데이터셋으로 신경망 학습하기
# - 시각화와 감시에 TensorBoard 쓰기
# - 학습 지표, 모델 그래프, 정밀도-재현율 곡선 기록하기
# ============================================================================

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ============================================================================
# TensorBoard 준비
# ============================================================================
# TensorBoard는 기계학습 실험을 위한 시각화 도구이다
# 지표, 모델 그래프를 비롯한 데이터를 기록하고 보여 준다
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# TensorBoard에 기록할 SummaryWriter 인스턴스 만들기
# 'runs/mnist1' 디렉터리에 TensorBoard 기록 파일이 모두 저장된다
# tensorboard --logdir=runs 으로 이 기록을 볼 수 있다
writer = SummaryWriter('runs/mnist1')
# ============================================================================

# ============================================================================
# 장치 설정
# ============================================================================
# CUDA(GPU)를 쓸 수 있는지 확인하고, 없으면 CPU를 쓴다
# 신경망은 GPU에서 학습하는 편이 훨씬 빠르다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================================================
# 초매개변수
# ============================================================================
# 이 값들이 모델의 구조와 학습 과정을 좌우한다
input_size = 784        # 28x28 화소를 1차원 벡터로 펼친 것
hidden_size = 500       # 은닉층의 뉴런 수
num_classes = 10        # MNIST에는 클래스가 10개 있다 (숫자 0~9)
num_epochs = 1          # 학습 데이터셋 전체를 훑는 횟수
batch_size = 64         # 모델을 갱신하기 전에 처리하는 표본의 수
learning_rate = 0.001   # 경사 하강 최적화의 걸음 크기

# ============================================================================
# 데이터셋 적재
# ============================================================================
# MNIST 데이터셋에는 학습 이미지 60,000장과 시험 이미지 10,000장이 있다
# 손글씨 숫자(0~9)이며 이미지마다 28x28 화소이다

# 학습 데이터셋 불러오기
train_dataset = torchvision.datasets.MNIST(
    root='./data',                          # 데이터를 저장하고 불러올 디렉터리
    train=True,                             # 학습 집합 불러오기
    transform=transforms.ToTensor(),        # PIL 이미지를 텐서로 바꾸기
    download=True                           # 없으면 내려받기
)

# 시험 데이터셋 불러오기
test_dataset = torchvision.datasets.MNIST(
    root='./data',                          # 일관성을 위해 같은 디렉터리
    train=False,                            # 시험 집합 불러오기
    transform=transforms.ToTensor()         # 같은 변환 적용
)

# ============================================================================
# 데이터 로더
# ============================================================================
# DataLoader가 배치 묶기, 섞기, 병렬 적재를 처리한다

# 일반화를 위해 섞는 학습 데이터 로더
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True                            # 에포크마다 데이터 섞기
)

# 섞지 않는 시험 데이터 로더 (평가에서는 순서가 중요하지 않다)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ============================================================================
# 표본 이미지 시각화
# ============================================================================
# 시각화를 위해 시험 데이터 배치 하나 가져오기
examples = iter(test_loader)
example_data, example_targets = next(examples)

# matplotlib으로 표본 이미지의 2x3 격자 만들기
for i in range(6):
    plt.subplot(2, 3, i+1)                  # 2x3 격자에 부분 그림 만들기
    plt.imshow(example_data[i][0], cmap='gray')  # 회색조 이미지 보이기
    plt.title(f'Label: {example_targets[i]}')    # 참 레이블 보이기
    plt.axis('off')                         # 깔끔하게 보이려고 축 감추기
# plt.show()  # Uncomment to display the plot

# ============================================================================
# TensorBoard: 표본 이미지 기록
# ============================================================================
# 이미지 격자를 만들어 TensorBoard에 기록
# 모델이 무엇을 다루는지 눈으로 보게 해 준다
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)

# 선택: 이미지만 보고 싶다면 여기서 writer를 닫고 끝낸다
# writer.close()
# sys.exit()
# ============================================================================

# ============================================================================
# 모델 정의
# ============================================================================
# 은닉층이 하나인 간단한 완전 연결 신경망 정의
class NeuralNet(nn.Module):
    """
    MNIST 분류를 위한 간단한 순방향 신경망.
    
    구조:
    - 입력층: 뉴런 784개 (28x28 이미지를 펼친 것)
    - 은닉층: ReLU 활성화를 쓰는 뉴런 500개
    - 출력층: 뉴런 10개 (숫자 클래스마다 하나)
    
    인수:
        input_size (int): 입력 특징의 수 (MNIST는 784)
        hidden_size (int): 은닉층의 뉴런 수
        num_classes (int): 출력 클래스의 수 (MNIST는 10)
    """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        
        # 첫 선형층: input_size -> hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)
        
        # ReLU 활성화 함수: max(0, x)
        # 복잡한 양상을 배우도록 비선형성을 넣는다
        self.relu = nn.ReLU()
        
        # 둘째 선형층: hidden_size -> num_classes
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        신경망을 통과하는 순전파.
        
        인수:
            x (torch.Tensor): 모양이 (batch_size, input_size)인 입력 텐서
        
        반환값:
            torch.Tensor: 모양이 (batch_size, num_classes)인 날 출력 점수(로짓)
        """
        # 첫 선형층 통과
        out = self.l1(x)
        
        # ReLU 활성화 적용
        out = self.relu(out)
        
        # 둘째 선형층 통과
        out = self.l2(out)
        
        # 참고: CrossEntropyLoss가 내부에서 적용하므로 여기서는 소프트맥스를 쓰지 않는다
        return out

# ============================================================================
# 모델 인스턴스 만들기
# ============================================================================
# 모델을 만들어 알맞은 장치(CPU 또는 GPU)로 옮긴다
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
print(f'\nModel Architecture:\n{model}')

# ============================================================================
# 손실 함수와 최적화기
# ============================================================================
# CrossEntropyLoss는 소프트맥스와 음의 로그가능도를 합친다
# 다중 클래스 분류 문제에 안성맞춤이다
criterion = nn.CrossEntropyLoss()

# Adam 최적화기: 적응형 학습률 최적화 알고리즘
# 초매개변수를 크게 조율하지 않아도 대체로 잘 통한다
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ============================================================================
# TensorBoard: 모델 그래프 기록
# ============================================================================
# 모델의 계산 그래프를 TensorBoard에 기록
# 신경망의 구조와 데이터의 흐름을 보여 준다
writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))

# 선택: 그래프만 보고 싶다면 여기서 writer를 닫고 끝낸다
# writer.close()
# sys.exit()
# ============================================================================

# ============================================================================
# 학습 루프
# ============================================================================
print('\n' + '='*60)
print('STARTING TRAINING')
print('='*60)

# TensorBoard 기록을 위해 진행 중인 지표를 담는 변수
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)

# 정해진 에포크 수만큼 데이터셋 전체를 훑는다
for epoch in range(num_epochs):
    # 학습 집합의 배치를 훑는다
    for i, (images, labels) in enumerate(train_loader):
        # ====================================================================
        # 데이터 준비
        # ====================================================================
        # 원래 모양: [batch_size, 1, 28, 28] (배치, 채널, 높이, 너비)
        # 바꾼 모양: [batch_size, 784] (이미지를 펼친다)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # ====================================================================
        # 순전파
        # ====================================================================
        # 모델의 예측 계산
        outputs = model(images)
        
        # 예측과 참 레이블 사이의 손실 계산
        loss = criterion(outputs, labels)
        
        # ====================================================================
        # 역전파와 최적화
        # ====================================================================
        # 앞 반복의 기울기 지우기
        # PyTorch는 기본으로 기울기를 누적한다
        optimizer.zero_grad()
        
        # 역전파로 기울기 계산
        loss.backward()
        
        # 계산한 기울기로 모델의 매개변수 갱신
        optimizer.step()
        
        # ====================================================================
        # 지표 누적
        # ====================================================================
        # 현재 손실을 누계에 더한다
        running_loss += loss.item()
        
        # 예측 클래스 얻기 (로짓이 최대인 인덱스)
        _, predicted = torch.max(outputs.data, 1)
        
        # 맞은 예측 세기
        running_correct += (predicted == labels).sum().item()
        
        # ====================================================================
        # TensorBoard에 기록 (100단계마다)
        # ====================================================================
        if (i+1) % 100 == 0:
            # 진행 상황을 콘솔에 출력
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{i+1}/{n_total_steps}], '
                  f'Loss: {loss.item():.4f}')
            
            # ----------------------------------------------------------------
            # 학습 손실 기록
            # ----------------------------------------------------------------
            # 최근 100 배치의 평균 손실
            writer.add_scalar('training loss', 
                            running_loss / 100, 
                            epoch * n_total_steps + i)
            
            # ----------------------------------------------------------------
            # 학습 정확도 기록
            # ----------------------------------------------------------------
            # 정확도 계산: 맞은 예측 / 전체 예측
            # predicted.size(0)이 배치 크기이다
            running_accuracy = running_correct / 100 / predicted.size(0)
            writer.add_scalar('accuracy', 
                            running_accuracy, 
                            epoch * n_total_steps + i)
            
            # 다음 100 배치를 위해 진행 지표 초기화
            running_correct = 0
            running_loss = 0.0

print('='*60)
print('TRAINING COMPLETED')
print('='*60 + '\n')

# ============================================================================
# 시험 집합에서의 모델 평가
# ============================================================================
print('='*60)
print('EVALUATING MODEL ON TEST SET')
print('='*60)

# 모든 배치의 예측과 레이블을 담을 목록
class_labels = []
class_preds = []

# 평가를 위해 기울기 계산 끄기 (메모리와 계산을 아낀다)
with torch.no_grad():
    n_correct = 0    # 맞은 예측의 총수
    n_samples = 0    # 표본의 총수
    
    # 시험 배치 훑기
    for images, labels in test_loader:
        # 데이터 준비 (학습과 같다)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # 순전파
        outputs = model(images)
        
        # 예측 클래스 얻기
        # torch.max은 값과 인덱스를 함께 돌려준다
        values, predicted = torch.max(outputs.data, 1)
        
        # 통계 누적
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        # ====================================================================
        # 정밀도-재현율 곡선을 위한 클래스 확률 계산
        # ====================================================================
        # 소프트맥스로 로짓을 확률로 바꾼다
        # 배치의 표본마다 이렇게 한다
        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]
        
        # 이 배치의 예측과 레이블 저장
        class_preds.append(class_probs_batch)
        class_labels.append(labels)
    
    # ========================================================================
    # 예측과 레이블 합치기
    # ========================================================================
    # stack: 새 차원을 만들어 텐서를 잇는다
    # cat: 이미 있는 차원을 따라 텐서를 잇는다
    
    # 목록의 목록을 텐서 하나로 바꾸기
    # 최종 모양: [10000, 10] (모든 표본, 모든 클래스의 확률)
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    
    # 모든 레이블 배치 잇기
    # 최종 모양: [10000] (모든 레이블)
    class_labels = torch.cat(class_labels)
    
    # ========================================================================
    # 정확도 계산하고 보이기
    # ========================================================================
    acc = 100.0 * n_correct / n_samples
    print(f'\nAccuracy of the network on the 10000 test images: {acc:.2f}%')
    
    # ========================================================================
    # TensorBoard: 정밀도-재현율 곡선 기록
    # ========================================================================
    # 정밀도-재현율 곡선은 정밀도와 재현율의 절충을 보여 준다
    # 문턱값을 달리할 때의 모델 성능을 이해하는 데 쓸모 있다
    # 숫자 클래스(0~9)마다 곡선을 따로 그린다
    
    classes = range(10)
    for i in classes:
        # 이진 레이블 만들기: 표본이 클래스 i이면 True, 아니면 False
        labels_i = class_labels == i
        
        # 클래스 i에 대한 예측 확률 얻기
        preds_i = class_preds[:, i]
        
        # 이 클래스의 정밀도-재현율 곡선을 TensorBoard에 추가
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    
    # TensorBoard writer 닫기
    writer.close()
    print('\nTensorBoard logs saved to: runs/mnist1')
    print('To view, run: tensorboard --logdir=runs')
    print('='*60)

# ============================================================================
# 스크립트 끝
# ============================================================================
# TensorBoard에서 결과를 보려면:
# 1. 터미널을 연다
# 2. 이 스크립트가 있는 디렉터리로 이동한다
# 3. tensorboard --logdir=runs 을 실행한다
# 4. 브라우저에서 http://localhost:6006 으로 접속한다
# ============================================================================


if __name__ == "__main__":
    pass
```

## 논의

TensorBoard 기록은 네 곳에 들어간다. 학습 전에는 데이터의 품질을 확인하려고 표본 이미지를 남기고, 구조를 보려고 계산 그래프를 남기며, 수렴을 살피려고 100 배치마다 학습 손실과 정확도를 남기고, 평가 뒤에는 클래스별 성능을 분석하려고 정밀도-재현율 곡선을 남긴다.

`SummaryWriter`은 TensorBoard가 읽는 `runs/mnist1` 디렉터리에 사건을 기록한다. `add_scalar`은 시계열 그래프를, `add_image`은 이미지 갤러리를, `add_graph`은 대화형 구조 도표를, `add_pr_curve`은 클래스별 정밀도-재현율 곡선을 만든다.

정밀도-재현율 곡선은 정확도 하나보다 훨씬 많은 것을 알려 준다. 숫자 클래스마다 분류 문턱값이 변할 때 정밀도와 재현율이 어떻게 달라지는지 보여 준다. 곡선이 오른쪽 위 모서리에 가까운 클래스는 잘 갈라지는 것이고, 곡선이 처지는 클래스는 다른 숫자와 헷갈린다는 뜻이다.

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

