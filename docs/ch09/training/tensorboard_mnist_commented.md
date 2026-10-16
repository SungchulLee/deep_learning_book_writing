# TensorBoard MNIST 주석본

MNIST를 완전 연결 뉴럴 넷트웍으로 학습하며 TensorBoard에 스칼라, 이미지, 모델 그래프와 클래스별 정밀도–재현율 곡선을 기록한다. 기록 항목의 의미와 평가 범위를 연결해서 읽는 예제이다.

## 1. 코드

```python
"""TensorBoard MNIST 주석본."""
# [Code Source](https://github.com/patrickloeber/pytorchTutorial)
# ============================================================================
# TensorBoard 시각화와 함께하는 MNIST 뉴럴 넷트웍 학습
# ============================================================================
# 이 스크립트는 다음을 보인다:
# - MNIST 데이터셋으로 뉴럴 넷트웍 학습하기
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
# 뉴럴 넷트웍은 GPU에서 학습하는 편이 훨씬 빠르다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================================================
# 초파라메터
# ============================================================================
# 이 값들이 모델의 구조와 학습 과정을 좌우한다
input_size = 784        # 28x28 화소를 1차원 벡터로 펼친 것
hidden_size = 500       # 은닉 레이어의 뉴런 수
num_classes = 10        # MNIST에는 클래스가 10개 있다 (숫자 0~9)
num_epochs = 1          # 학습 데이터셋 전체를 훑는 횟수
batch_size = 64         # 모델을 갱신하기 전에 처리하는 표본의 수
learning_rate = 0.001   # 경사 하강 최적화의 스텝 크기

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
# plt.show()  # 그림을 보려면 주석을 풀어라

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
# 은닉 레이어가 하나인 간단한 완전 연결 뉴럴 넷트웍 정의
class NeuralNet(nn.Module):
    """
    MNIST 분류를 위한 간단한 순방향 뉴럴 넷트웍.
    
    구조:
    - 입력 레이어: 뉴런 784개 (28x28 이미지를 펼친 것)
    - 은닉 레이어: ReLU 활성화를 쓰는 뉴런 500개
    - 출력 레이어: 뉴런 10개 (숫자 클래스마다 하나)
    
    인수:
        input_size (int): 입력 피처의 수 (MNIST는 784)
        hidden_size (int): 은닉 레이어의 뉴런 수
        num_classes (int): 출력 클래스의 수 (MNIST는 10)
    """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        
        # 첫 선형 레이어: input_size -> hidden_size
        self.l1 = nn.Linear(input_size, hidden_size)
        
        # ReLU 엑티베이션 함수: max(0, x)
        # 복잡한 양상을 배우도록 비선형성을 넣는다
        self.relu = nn.ReLU()
        
        # 둘째 선형 레이어: hidden_size -> num_classes
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        뉴럴 넷트웍을 통과하는 순전파.
        
        인수:
            x (torch.Tensor): 모양이 (batch_size, input_size)인 입력 텐서
        
        반환값:
            torch.Tensor: 모양이 (batch_size, num_classes)인 날 출력 점수(로짓)
        """
        # 첫 선형 레이어 통과
        out = self.l1(x)
        
        # ReLU 활성화 적용
        out = self.relu(out)
        
        # 둘째 선형 레이어 통과
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
# 로스 함수와 최적화기
# ============================================================================
# CrossEntropyLoss는 소프트맥스와 음의 로그가능도를 합친다
# 다중 클래스 분류 문제에 안성맞춤이다
criterion = nn.CrossEntropyLoss()

# Adam 최적화기: 적응형 러닝레이트 최적화 알고리즘
# 초파라메터를 크게 조율하지 않아도 대체로 잘 통한다
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ============================================================================
# TensorBoard: 모델 그래프 기록
# ============================================================================
# 모델의 계산 그래프를 TensorBoard에 기록
# 뉴럴 넷트웍의 구조와 데이터의 흐름을 보여 준다
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
        
        # 예측과 참 레이블 사이의 로스 계산
        loss = criterion(outputs, labels)
        
        # ====================================================================
        # 역전파와 최적화
        # ====================================================================
        # 앞 반복의 그래디언트 지우기
        # PyTorch는 기본으로 그래디언트를 누적한다
        optimizer.zero_grad()
        
        # 역전파로 그래디언트 계산
        loss.backward()
        
        # 계산한 그래디언트로 모델의 파라메터 갱신
        optimizer.step()
        
        # ====================================================================
        # 지표 누적
        # ====================================================================
        # 현재 로스를 누계에 더한다
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
            # 학습 로스 기록
            # ----------------------------------------------------------------
            # 최근 100 배치의 평균 로스
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

# 평가를 위해 그래디언트 계산 끄기 (메모리와 계산을 아낀다)
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
    # 임계값을 달리할 때의 모델 성능을 이해하는 데 쓸모 있다
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

## 2. 논의

NeuralNet은 784→500→10 구조로 로짓을 반환하며 교차 엔트로피에 소프트맥스를 별도로 적용하지 않는다. 평가에서만 각 표본의 10개 출력에 소프트맥스를 적용해 PR 곡선의 점수로 사용한다. 이 모델에는 드롭아웃이나 BatchNorm이 없어 eval() 누락이 해당 레이어의 동작 차이를 만들지는 않지만, 그 레이어를 추가한 모델에서는 평가 모드를 명시해야 한다.

SummaryWriter는 runs/mnist1에 이벤트 파일을 남긴다. add_scalar는 전역 단계에 따른 지표를, add_image는 이미지 텐서를, add_graph는 예시 입력으로 추적한 계산 그래프를 기록한다. 그래프 기록은 모든 가능한 입력과 분기에서 모델이 올바르다는 검증은 아니다. 100배치 단위 로스는 그 구간의 평균으로 전체 에포크 평균과 구분한다. 서로 다른 실행을 같은 태그와 디렉터리에 계속 기록하면 해석이 어려워질 수 있어 실험별 경로를 구분한다.

각 PR 곡선은 해당 숫자를 양성, 나머지를 음성으로 보는 one-vs-rest 평가이다. 임계값을 바꾸며 정밀도와 재현율을 계산하므로 10개 클래스 중 최댓값을 고르는 정확도와 다른 정보를 준다. 양성 비율도 곡선에 영향을 주며 클래스 간 순위를 곧바로 난이도 순위로 해석하지 않는다. 어느 숫자와 혼동하는지는 혼동 행렬을 함께 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
NeuralNet의 학습 파라메터 수를 계산하라.

</div>

??? success "연습문제 1 풀이"
    첫 선형 레이어는 784×500+500=392500개이고 출력 레이어는 500×10+10=5010개로 총 397510개이다. ReLU는 학습 파라메터가 없다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
숫자 3의 PR 곡선에서는 어떤 목표와 점수를 사용하는가?

</div>

??? success "연습문제 2 풀이"
    목표는 원래 레이블이 3인지 나타내는 이진 값이고 점수는 각 표본에 대한 클래스 3의 소프트맥스 확률이다. 이 점수에 대한 임계값을 바꾸며 one-vs-rest 정밀도와 재현율을 계산한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
실험을 여러 번 비교할 때 TensorBoard 기록에 어떤 메타데이터가 필요한가?

</div>

??? success "연습문제 3 풀이"
    모델 구조, 시드, 데이터 분할, 러닝레이트·배치 크기 등 설정과 소프트웨어 버전을 실행별로 남긴다. 전역 단계의 의미를 같게 하고 같은 테스트셋을 반복 모델 선택에 사용하지 않았는지도 구분한다.

## 정리하며

TensorBoard는 기록된 지표를 해석하기 위한 도구이다. 태그와 전역 단계, 집계 구간을 명확히 하고 PR 곡선의 이진 평가 의미를 정확히 읽어야 실험 비교에 도움이 된다.
