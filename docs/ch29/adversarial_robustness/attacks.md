# 적대적 공격의 기초

적대적 예제(adversarial example)는 모델의 잘못된 예측을 유도하도록 만든 입력이다. 이 페이지는 이미지 분류기의 입력에 제한된 섭동을 더하는 FGSM과 반복 공격을 설명한다. 작은 노름이 사람이 알아챌 수 없는 변화와 항상 같지는 않으며, 공격이 실패한 것과 모델의 강건성이 증명된 것도 다르다.

## 1. 위협 모델과 섭동 제약

원본 x, 정답 y에 대해 비표적 공격은 $f(x+\delta)\ne y$를, 표적 공격은 미리 정한 $t\ne y$에 대해 $f(x+\delta)=t$를 목표로 한다. 가능한 δ의 집합과 입력의 유효 범위를 함께 정한다.

- L∞ 제약은 각 스칼라 성분의 절댓값 변화를 ε 이하로 제한한다.
- L2 제약은 전체 유클리드 크기를 제한한다.
- 이른바 L0 값은 바뀐 성분 수를 센다. 엄밀한 노름은 아니며 RGB 성분 수와 픽셀 위치 수도 다르다.

화이트박스 공격은 모델과 기울기에 접근한다. 블랙박스 공격은 출력 질의나 대리 모델에서의 전이를 사용할 수 있다. 질의 예산, 반환 정보, 접근 가능한 모델과 전처리를 명시한다. 데이터 오염은 학습 단계의 개입이고 여기의 입력 공격은 추론 단계의 개입이므로 구분한다.

## 2. 섭동을 출력하는 생성기

```python
import torch
import torch.nn as nn

class AdversarialGenerator(nn.Module):
    """맞겨루기 생성기 같은 구조로 맞겨루기 흔들림을 만든다."""
    
    def __init__(self, input_channels=3, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1),
            nn.Tanh()  # [-1, 1]으로 내놓고 엡실론으로 잣수를 맞춘다
        )
    
    def forward(self, x):
        """들임 x의 맞겨루기 흔들림을 만든다."""
        features = self.encoder(x)
        perturbation = self.decoder(features)
        
        # 흔들림을 엡실론 공에 맞춰 잣수 조정한다
        perturbation = self.epsilon * perturbation
        
        # 맞겨루기 보기를 만든다
        x_adv = torch.clamp(x + perturbation, 0, 1)
        
        return x_adv, perturbation
```

이 네트워크는 입력 이미지를 받아 tanh로 제한한 섭동에 ε를 곱한다. 입력이 0~1이고 ε≥0이면 최종 clipping 후의 실제 섭동도 L∞≤ε를 만족한다. 반환하는 perturbation은 clipping **전** 값이므로 실제 x_adv−x와 다를 수 있다. 두 번의 stride 2 축소와 확대는 높이·너비가 4의 배수일 때 원래 크기를 복원한다.

코드에는 공격 목적함수나 생성기 학습, GAN 판별기 학습이 없다. 구조만으로 공격이 성공하거나 자연스러운 이미지를 만든다고 할 수 없다. 일반 GAN의 생성기 갱신과 입력 공격은 미분 가능한 모델을 통해 기울기를 얻는다는 공통점이 있지만, 최적화 변수와 목표 분포는 다르다.

## 3. 공격 평가의 분모

```python
def evaluate_attack(model, x_clean, x_adv, y_true, y_target=None):
    """
    맞겨루기 공격의 성공을 따진다.
    
    인수:
        model: 겨눌 분류기
        x_clean: 깨끗한 그림
        x_adv: 맞겨루기 그림
        y_true: 참 레이블
        y_target: 과녁 이름표(과녁 공격에 쓴다)
    
    반환값:
        공격 잣대 사전
    """
    model.eval()
    
    with torch.no_grad():
        # 깨끗한 자료의 맞힘률
        pred_clean = model(x_clean).argmax(dim=1)
        clean_acc = (pred_clean == y_true).float().mean().item()
        
        # 맞겨루기 헤아림
        pred_adv = model(x_adv).argmax(dim=1)
        adv_acc = (pred_adv == y_true).float().mean().item()
        
        # 공격 성공 비율(목표 없음)
        fooling_rate = (pred_adv != y_true).float().mean().item()
        
        metrics = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'fooling_rate': fooling_rate,
            'accuracy_drop': clean_acc - adv_acc,
        }
        
        # 목표 있는 공격의 성공
        if y_target is not None:
            target_success = (pred_adv == y_target).float().mean().item()
            metrics['target_success_rate'] = target_success
    
    return metrics
```

`fooling_rate`는 전체 표본에서의 공격 후 오분류율 1−adv_accuracy다. 원래 오분류된 표본도 포함하므로 '원래 맞혔던 표본을 공격으로 틀리게 만든 비율'과 다르다. 조건부 비표적 성공률은 다음처럼 같은 표본의 clean·adv 예측을 함께 사용해야 한다.

$$\mathrm{ASR}=\frac{\#\{i:f(x_i)=y_i,\ f(x_i^{adv})\ne y_i\}}
{\#\{i:f(x_i)=y_i\}}$$

분모가 0이면 별도 처리가 필요하다. 표적 성공률도 원래 정답·예측이 목표 클래스인 표본의 포함 여부를 밝혀야 한다. 평가 후 모델 모드를 원래대로 복원하는 기능은 이 함수에 없다.

```python
def perturbation_metrics(x_clean, x_adv):
    """
    흔들림 통계를 셈한다.
    
    인수:
        x_clean: 깨끗한 그림(배치)
        x_adv: 맞겨루기 그림(배치)
    
    반환값:
        흔들림 잣대 사전
    """
    delta = x_adv - x_clean
    
    # 표본마다 잣대
    l_inf = delta.abs().view(delta.size(0), -1).max(dim=1)[0]
    l_2 = delta.view(delta.size(0), -1).norm(p=2, dim=1)
    l_0 = (delta.abs() > 1e-6).view(delta.size(0), -1).sum(dim=1).float()
    
    return {
        'l_inf_mean': l_inf.mean().item(),
        'l_inf_max': l_inf.max().item(),
        'l_2_mean': l_2.mean().item(),
        'l_2_max': l_2.max().item(),
        'l_0_mean': l_0.mean().item(),
        'l_0_max': l_0.max().item(),
    }
```

섭동 통계는 실제 x_adv−x로 계산한다. 여기의 L0는 절댓값이 10⁻⁶보다 큰 스칼라 성분 수이므로 수학적인 정확한 비영 성분 수와 약간 다르다. 배치 평균뿐 아니라 최댓값을 확인해 제약 위반을 점검한다.

## 4. 시각화와 대상 분류기

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_adversarial_example(x_clean, x_adv, y_true, y_pred_clean, y_pred_adv,
                                   class_names=None, amplify=10):
    """
    깨끗한 그림, 흔들림, 맞겨루기 그림을 그려 본다.
    
    인수:
        x_clean: 깨끗한 그림 텐서 (C, H, W)
        x_adv: 맞겨루기 그림 텐서 (C, H, W)
        y_true: 참 이름표
        y_pred_clean: 깨끗한 그림의 헤아림
        y_pred_adv: 맞겨루기 그림의 헤아림
        class_names: 갈래 이름 목록(있으면)
        amplify: 눈에 띄게 흔들림을 키우는 갑절
    """
    # 흔들림을 셈한다
    delta = x_adv - x_clean
    
    # 보여 줄 수 있는 꼴로 바꾸기
    clean_img = x_clean.cpu().numpy().transpose(1, 2, 0)
    adv_img = x_adv.cpu().numpy().transpose(1, 2, 0)
    
    # 눈에 띄게 흔들림을 키운다
    pert_img = delta.cpu().numpy().transpose(1, 2, 0)
    pert_display = 0.5 + amplify * pert_img  # 0.5을 가운데로
    pert_display = np.clip(pert_display, 0, 1)
    
    # 회색을 다룬다
    if clean_img.shape[-1] == 1:
        clean_img = clean_img.squeeze(-1)
        adv_img = adv_img.squeeze(-1)
        pert_display = pert_display.squeeze(-1)
    
    # 갈래 이름 얻기
    if class_names is None:
        true_name = str(y_true)
        clean_name = str(y_pred_clean)
        adv_name = str(y_pred_adv)
    else:
        true_name = class_names[y_true]
        clean_name = class_names[y_pred_clean]
        adv_name = class_names[y_pred_adv]
    
    # 그림
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    cmap = 'gray' if clean_img.ndim == 2 else None
    
    axes[0].imshow(np.clip(clean_img, 0, 1), cmap=cmap)
    axes[0].set_title(f'Clean Image\nTrue: {true_name}\nPred: {clean_name}')
    axes[0].axis('off')
    
    axes[1].imshow(pert_display, cmap=cmap)
    axes[1].set_title(f'Perturbation (×{amplify})\n'
                      f'L∞: {delta.abs().max():.4f}\n'
                      f'L2: {delta.norm():.4f}')
    axes[1].axis('off')
    
    axes[2].imshow(np.clip(adv_img, 0, 1), cmap=cmap)
    axes[2].set_title(f'Adversarial Image\nPred: {adv_name}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
```

시각화는 한 이미지 C×H×W를 가정하며 CPU NumPy 변환 전에 기울기 추적에서 분리되어 있어야 한다. 섭동은 0.5 중심으로 확대하고 잘라 표시하므로 원래 크기의 입력 변화와 구분한다. 흑백 표시 눈금을 지정하지 않은 점도 비교 시 고려한다.

```python
class SimpleCNN(nn.Module):
    """공격 표적으로 쓰는 MNIST 가르기용 단순한 겹말기 신경망."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)           # (B, 32, 28, 28)
        x = self.relu(x)
        x = self.maxpool(x)         # (B, 32, 14, 14)
        x = self.conv2(x)           # (B, 64, 14, 14)
        x = self.relu(x)
        x = self.maxpool(x)         # (B, 64, 7, 7)
        x = self.flatten(x)         # (B, 64*7*7)
        x = self.fc1(x)             # (B, 128)
        x = self.relu(x)
        x = self.fc2(x)             # (B, 10)
        return x

def train_target_model(model, train_loader, epochs=5, lr=0.001, device='cpu'):
    """맞겨루기 공격 실험을 위해 표적 모델을 익힌다."""
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Loss: {total_loss/len(train_loader):.4f}, '
              f'Acc: {100*correct/total:.2f}%')
    
    return model
```

대상 CNN은 1×28×28→32×28×28→32×14×14→64×14×14→64×7×7→3136→128→10 로짓 순서다. 교차 엔트로피 손실에 로짓을 직접 전달한다. 이 보조 학습 함수는 일반 분류 학습이며 적대적 학습은 수행하지 않는다.

## 5. FGSM의 일차 근사

입력 손실의 기울기를 $g=\nabla_x\ell(f(x),y)$라 하자. 일차 근사 손실 증가 $g^T\delta$를 L∞≤ε에서 최대화하는 해는 $\delta=\epsilon\operatorname{sign}(g)$다. 따라서 비표적 FGSM은 다음과 같다.

$$x^{adv}=\operatorname{clip}_{[0,1]}\big(x+\epsilon\operatorname{sign}(g)\big)$$

이는 원래 비선형 손실의 전역 최적 공격이 아니라 **일차 근사 문제**의 해다. 0인 기울기 성분이나 clipping이 있으면 모든 좌표가 꼭 ±ε만큼 바뀌는 것도 아니다. [FGSM 논문](https://arxiv.org/abs/1412.6572)의 선형성 설명과 실제 네트워크의 비선형 동작을 구분한다.

```python
import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon, criterion=None):
    """
    그림 배치에 FGSM 공격을 한다.
    
    인수:
        model: 겨눌 분류기
        images: 들임 그림(requires_grad가 True여야 한다)
        labels: 참 이름표
        epsilon: 흔들림의 크기
        criterion: 잃음 함수(기본값: CrossEntropyLoss)
    
    반환값:
        adversarial_images: 흔들린 그림
        perturbation: 쓴 흔들림
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # 그림이 기울기를 요구하게 한다
    images = images.clone().detach().requires_grad_(True)
    
    # 순전파
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # 기울기를 얻으려 뒤먹임한다
    model.zero_grad()
    loss.backward()
    
    # 기울기 부호를 얻는다
    grad_sign = images.grad.data.sign()
    
    # 흔들림을 만든다
    perturbation = epsilon * grad_sign
    
    # 맞겨루기 그림을 만든다
    adversarial_images = images + perturbation
    
    # 올바른 범위 [0, 1]으로 가둔다
    adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
    return adversarial_images.detach(), perturbation.detach()

class FGSMAttack:
    """기능을 더한 FGSM 공격 감싸개 갈래."""
    
    def __init__(self, model, epsilon=0.1, criterion=None, device='cpu'):
        """
        인수:
            model: 공격할 표적 모델
            epsilon: 흔들림의 한계
            criterion: 손실 함수
            device: 공격을 돌릴 기기
        """
        self.model = model
        self.epsilon = epsilon
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
    
    def attack(self, images, labels):
        """
        맞겨루기 보기를 만든다.
        
        인수:
            images: 깨끗한 그림
            labels: 참 이름표
        
        반환값:
            맞겨루기 그림
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.model.eval()
        
        # 그림에 기울기 셈하기를 켠다
        images.requires_grad = True
        
        # 순전파
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        # 경사를 계산한다
        self.model.zero_grad()
        loss.backward()
        
        # FGSM 걸음
        grad_sign = images.grad.data.sign()
        perturbed_images = images + self.epsilon * grad_sign
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()
    
    def evaluate(self, dataloader, max_batches=None):
        """
        자료 묶음에서 공격의 성공을 따진다.
        
        인수:
            dataloader: 시험 자료 불러개
            max_batches: 따질 최대 배치 수
        
        반환값:
            공격 잣대를 담은 사전
        """
        self.model.eval()
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 깨끗한 자료의 헤아림
            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_pred = clean_outputs.argmax(dim=1)
                clean_correct += (clean_pred == labels).sum().item()
            
            # 맞겨루기 보기
            adv_images = self.attack(images, labels)
            
            # 맞겨루기 헤아림
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_pred = adv_outputs.argmax(dim=1)
                adv_correct += (adv_pred == labels).sum().item()
            
            total += labels.size(0)
        
        return {
            'clean_accuracy': clean_correct / total,
            'adversarial_accuracy': adv_correct / total,
            'attack_success_rate': (clean_correct - adv_correct) / clean_correct,
            'total_samples': total
        }
```

함수형 `fgsm_attack`은 입력 복사본에서 기울기를 구하지만 모델 모드는 바꾸지 않는다. 반환 perturbation은 clipping 전 값이다. 클래스형 공격은 eval을 사용하되 이전 모드를 복원하지 않고, 입력을 복제·분리하지도 않아 입력이 leaf 텐서인지와 기존 .grad 상태에 주의해야 한다. 두 방식 모두 backward로 모델 파라미터의 기울기도 계산하지만 optimizer.step을 호출하지 않아 파라미터를 업데이트하지는 않는다. 공격을 생성하는 구간 전체를 no_grad로 감싸면 필요한 입력 기울기를 구할 수 없다.

`FGSMAttack.evaluate`의 `(clean_correct-adv_correct)/clean_correct`는 정확도 감소를 나눈 값이다. 원래 오분류를 공격 후 맞힌 표본이 있으면 위에서 정의한 조건부 성공률과 달라진다. 빈 평가나 clean_correct=0도 처리하지 않는다.

```python
def untargeted_fgsm(model, images, labels, epsilon):
    """
    목표 없는 FGSM: 아무 잘못 가르기나 일으키려 손실을 가장 크게 한다.
    
    흔들림은 손실이 늘어나는 방향으로 움직여
    헤아림을 참 갈래에서 밀어낸다.
    """
    criterion = nn.CrossEntropyLoss()
    
    images = images.clone().detach().requires_grad_(True)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    # 손실이 늘어나는 방향으로 간다(양의 기울기 방향)
    perturbation = epsilon * images.grad.data.sign()
    
    adv_images = torch.clamp(images + perturbation, 0, 1)
    
    return adv_images.detach()
```

```python
def targeted_fgsm(model, images, target_labels, epsilon):
    """
    목표 있는 FGSM: 목표 갈래의 손실을 가장 작게 한다.
    
    흔들림은 손실이 줄어드는 방향으로 움직인다
    과녁 갈래에 대해 기울기를 구해 헤아림을 그쪽으로 끌어당긴다.
    """
    criterion = nn.CrossEntropyLoss()
    
    images = images.clone().detach().requires_grad_(True)
    
    outputs = model(images)
    loss = criterion(outputs, target_labels)
    
    model.zero_grad()
    loss.backward()
    
    # 손실이 줄어드는 방향으로 간다(음의 기울기 방향)
    # 이는 헤아림을 목표 갈래 쪽으로 민다
    perturbation = -epsilon * images.grad.data.sign()
    
    adv_images = torch.clamp(images + perturbation, 0, 1)
    
    return adv_images.detach()

def run_targeted_attack(model, dataloader, target_class, epsilon, device='cpu'):
    """
    모든 그림을 target_class으로 잘못 가르게 하는 목표 있는 FGSM 공격을 돌린다.
    
    인수:
        model: 겨눌 분류기
        dataloader: 시험 자료
        target_class: 그림을 잘못 가를 갈래
        epsilon: 흔들림의 크기
        device: 돌릴 장치
    
    반환값:
        공격 결과 사전
    """
    model.eval()
    model.to(device)
    
    success = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # 목표 이름표를 만든다
        target_labels = torch.full((batch_size,), target_class, device=device)
        
        # 이미 목표로 갈린 그림은 건너뛴다
        with torch.no_grad():
            pred = model(images).argmax(dim=1)
            
        # 맞겨루기 보기를 만든다
        adv_images = targeted_fgsm(model, images, target_labels, epsilon)
        
        # 이루었는지 살핀다
        with torch.no_grad():
            adv_pred = model(adv_images).argmax(dim=1)
            success += (adv_pred == target_class).sum().item()
        
        total += batch_size
    
    return {
        'target_class': target_class,
        'success_rate': success / total,
        'total_samples': total
    }
```

표적 FGSM은 목표 레이블에 대한 교차 엔트로피를 **줄이는** 방향이므로 부호 앞에 마이너스가 붙는다. `run_targeted_attack`은 원래 목표 클래스인 표본과 원래 오분류된 표본도 포함해 목표 도달 비율을 계산한다. 그 값을 새롭게 유도한 표적 오분류율이라고 그대로 해석하지 않는다.

## 6. 첫 번째 MNIST 예제의 전처리 불일치

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 단순한 겹말기 신경망을 뜻매김한다
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, epochs=5):
    """모델을 익힌다."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete")
    
    return model

def visualize_fgsm_attack(model, image, label, epsilons, class_names):
    """엡실론 값을 달리하여 FGSM 공격을 그려 본다."""
    
    fig, axes = plt.subplots(1, len(epsilons) + 1, figsize=(3 * (len(epsilons) + 1), 3))
    
    # 원래 이미지
    axes[0].imshow(image.squeeze().cpu(), cmap='gray')
    with torch.no_grad():
        pred = model(image.unsqueeze(0)).argmax().item()
    axes[0].set_title(f'Original\nPred: {class_names[pred]}')
    axes[0].axis('off')
    
    # 엡실론을 달리한 맞겨루기 그림
    for i, eps in enumerate(epsilons):
        adv_image, _ = fgsm_attack(
            model, 
            image.unsqueeze(0), 
            torch.tensor([label]), 
            eps
        )
        
        with torch.no_grad():
            adv_pred = model(adv_image).argmax().item()
        
        axes[i+1].imshow(adv_image.squeeze().cpu(), cmap='gray')
        axes[i+1].set_title(f'ε={eps}\nPred: {class_names[adv_pred]}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 준비
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_data = torchvision.datasets.MNIST('./data', train=True, 
                                            download=True, transform=transform)
    test_data = torchvision.datasets.MNIST('./data', train=False, 
                                           download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    
    # 모델을 학습시킨다
    model = SimpleCNN()
    model = train_model(model, train_loader, epochs=5)
    model.eval()
    
    # FGSM 공격개를 만든다
    attacker = FGSMAttack(model, epsilon=0.1)
    
    # 공격을 따진다
    results = attacker.evaluate(test_loader, max_batches=100)
    
    print("\nFGSM Attack Results:")
    print(f"Clean Accuracy: {results['clean_accuracy']:.2%}")
    print(f"Adversarial Accuracy: {results['adversarial_accuracy']:.2%}")
    print(f"Attack Success Rate: {results['attack_success_rate']:.2%}")
    
    # 시각화한다
    class_names = [str(i) for i in range(10)]
    image, label = next(iter(test_loader))
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.3]
    visualize_fgsm_attack(model, image.squeeze(0), label.item(), epsilons, class_names)

if __name__ == "__main__":
    main()
```

```
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz
Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 404: Not Found

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw

Epoch 1 complete
Epoch 2 complete
Epoch 3 complete
Epoch 4 complete
Epoch 5 complete

FGSM Attack Results:
Clean Accuracy: 99.00%
Adversarial Accuracy: 96.00%
Attack Success Rate: 3.03%
```

이 예제는 Normalize(0.5,0.5)로 이미지를 **−1~1**로 바꾸어 학습하지만 공격 함수는 결과를 **0~1로 자른다**. 따라서 원래 음수 픽셀이 clipping만으로 크게 바뀌어, 원본 기준 ε 제약을 위반할 수 있다. ε=0이어도 음수 값이 사라진다. 이 상태의 보존된 정확도·성공률 출력은 올바르게 일치시킨 FGSM 강건성 평가로 해석할 수 없다.

올바른 설명에서는 공격을 원래 0~1 이미지에 적용하고 정규화를 모델 앞의 미분 가능한 전처리로 포함하거나, 정규화 좌표에서 ε와 유효 범위를 함께 변환해야 한다. 이 페이지의 코드는 수정하지 않고 이러한 전제 불일치를 명시한다. 예제는 섞인 테스트 표본 중 최대 100개만 평가하며 전체 MNIST 성능과도 구분한다.

## 7. 표적 FGSM과 반복 공격

```python
import torch
import torch.nn as nn

def targeted_fgsm(model, images, target_labels, epsilon):
    """
    목표 있는 FGSM 공격.
    
    잃음을 가장 크게 하는 대신(참 갈래에서 밀어내는 대신)
    잃음을 가장 작게 한다(과녁 갈래로 끌어당긴다).
    
    인수:
        model: 겨눌 분류기
        images: 들임 그림
        target_labels: 바라는 목표 갈래
        epsilon: 흔들림의 크기
    
    반환값:
        맞겨루기 그림
    """
    criterion = nn.CrossEntropyLoss()
    
    # 베끼고 기울기를 켠다
    images = images.clone().detach().requires_grad_(True)
    
    # 순전파
    outputs = model(images)
    loss = criterion(outputs, target_labels)
    
    # 역전파
    model.zero_grad()
    loss.backward()
    
    # 음의 기울기 방향으로 간다(목표 쪽으로 손실을 가장 작게 한다)
    perturbation = -epsilon * images.grad.data.sign()
    
    # 맞겨루기 보기를 만든다
    adv_images = torch.clamp(images + perturbation, 0, 1)
    
    return adv_images.detach()
```

같은 표적 FGSM 함수를 다시 제시한 블록이다. 표적 레이블과 입력·모델의 장치가 같고 입력 범위가 0~1이라는 전제가 필요하다.

```python
def targeted_bim(model, images, target_labels, epsilon, alpha, num_iter):
    """
    과녁을 둔 기본 되풀이 방법(되풀이 FGSM).
    
    목표 갈래 쪽으로 작은 걸음을 여러 번 내딛는다.
    
    인수:
        model: 겨눌 분류기
        images: 들임 그림
        target_labels: 바라는 목표 갈래
        epsilon: 가장 큰 온 흔들림(L_inf 한계)
        alpha: 되풀이마다 걸음 크기
        num_iter: 되풀이 횟수
    
    반환값:
        맞겨루기 그림
    """
    criterion = nn.CrossEntropyLoss()
    
    # 본디 그림에서 시작한다
    adv_images = images.clone().detach()
    
    for i in range(num_iter):
        adv_images.requires_grad_(True)
        
        outputs = model(adv_images)
        loss = criterion(outputs, target_labels)
        
        model.zero_grad()
        loss.backward()
        
        # 목표 쪽으로 작은 걸음
        adv_images = adv_images - alpha * adv_images.grad.sign()
        
        # 본디 둘레의 엡실론 공으로 도로 쏜다
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, 0, 1).detach()
    
    return adv_images
```

표적 BIM은 목표 손실을 줄이는 부호 방향으로 α만큼 이동한 뒤 원본의 ε 상자와 유효 입력 범위로 되돌린다. 반복 중 원본을 고정하며 입력이 유효 범위 안에 있어야 두 clipping이 의도한 제약을 준다.

```python
def targeted_pgd(model, images, target_labels, epsilon, alpha, num_iter, 
                 random_start=True):
    """
    아무 첫자리매김을 갖춘 목표 있는 PGD 공격.
    
    인수:
        model: 겨눌 분류기
        images: 들임 그림
        target_labels: 바라는 목표 갈래
        epsilon: 가장 큰 흔들림(L_inf)
        alpha: 걸음 크기
        num_iter: 되풀이 횟수
        random_start: 엡실론 공 안의 아무 점에서 시작할지 여부
    
    반환값:
        맞겨루기 그림
    """
    criterion = nn.CrossEntropyLoss()
    
    # 엡실론 공 안의 아무 첫자리매김
    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
    else:
        adv_images = images.clone().detach()
    
    for i in range(num_iter):
        adv_images.requires_grad_(True)
        
        outputs = model(adv_images)
        loss = criterion(outputs, target_labels)
        
        model.zero_grad()
        loss.backward()
        
        # 기울기 내려가기(목표 쪽으로 손실을 가장 작게 한다)
        adv_images = adv_images - alpha * adv_images.grad.sign()
        
        # 엡실론 공으로 쏜다
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, 0, 1).detach()
    
    return adv_images

class TargetedPGDAttack:
    """목표 있는 PGD 공격 감싸개."""
    
    def __init__(self, model, epsilon=0.3, alpha=0.01, num_iter=40, 
                 random_start=True, device='cpu'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.random_start = random_start
        self.device = device
    
    def attack(self, images, target_labels):
        """특정 갈래를 겨눈 맞겨루기 보기를 만든다."""
        images = images.to(self.device)
        target_labels = target_labels.to(self.device)
        
        self.model.eval()
        
        return targeted_pgd(
            self.model, images, target_labels,
            self.epsilon, self.alpha, self.num_iter, self.random_start
        )
    
    def evaluate(self, dataloader, target_class, max_samples=1000):
        """
        목표 있는 공격의 성공을 따진다.
        
        인수:
            dataloader: 시험 자료
            target_class: 겨눌 갈래
            max_samples: 따질 최대 표본 수
        
        반환값:
            공격 잣대
        """
        self.model.eval()
        
        total = 0
        successful_attacks = 0
        already_target = 0
        
        for images, labels in dataloader:
            if total >= max_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)
            
            # 목표 이름표를 만든다
            targets = torch.full((batch_size,), target_class, 
                                device=self.device, dtype=torch.long)
            
            # 본디 헤아림을 살핀다
            with torch.no_grad():
                original_pred = self.model(images).argmax(dim=1)
            
            # 이미 목표로 갈린 그림을 센다
            already_target += (original_pred == target_class).sum().item()
            
            # 맞겨루기 보기를 만든다
            adv_images = self.attack(images, targets)
            
            # 맞겨루기 헤아림을 살핀다
            with torch.no_grad():
                adv_pred = self.model(adv_images).argmax(dim=1)
            
            successful_attacks += (adv_pred == target_class).sum().item()
            total += batch_size
        
        return {
            'target_class': target_class,
            'success_rate': successful_attacks / total,
            'already_target_rate': already_target / total,
            'total_samples': total
        }
```

표적 PGD는 무작위 시작을 추가할 수 있다. 유효한 0~1 원본에서는 초기 ε 상자와 유효 범위의 교집합 안에 들어간다. 반복 수·α·재시작 횟수와 성공 판정이 공격 결과를 바꾸며, 반복 공격이 모든 입력에서 FGSM보다 반드시 성공적이라는 보장은 없다. [강건 최적화 연구](https://arxiv.org/abs/1706.06083)는 이런 내부 공격과 외부 모델 학습을 구분해 다룬다.

`TargetedPGDAttack.evaluate`도 이미 목표로 분류된 표본을 포함한 전체 목표 도달 비율을 반환한다. max_samples는 배치 시작에서만 검사하므로 마지막 배치만큼 한도를 넘길 수 있다. 클래스의 device 인수는 모델 자체를 이동시키지 않는다.

## 8. C&W 형태의 L2 공격

```python
def cw_targeted_attack(model, images, target_labels, c=1.0, kappa=0, 
                        num_iter=1000, lr=0.01, device='cpu'):
    """
    카를리니-와그너 L2 목표 있는 공격.
    
    Minimizes: ||δ||_2 + c * max(Z(x+δ)_t - max_{j≠t} Z(x+δ)_j, -κ)
    
    여기서 Z은 로짓, t은 목표 갈래, κ은 자신도 여유이다.
    
    인수:
        model: 겨눌 분류기
        images: 들임 그림
        target_labels: 목표 갈래
        c: 자신도 매개변수
        kappa: 자신도 여유
        num_iter: 가장 좋게 하기 되풀이 횟수
        lr: 학습률
        device: 기기
    
    반환값:
        맞겨루기 그림
    """
    images = images.to(device)
    target_labels = target_labels.to(device)
    
    # 흔들림을 tanh으로 잣수 맞춘 변수로 첫자리매김한다
    # tanh을 쓰면 내놓기가 가둬진다
    w = torch.zeros_like(images, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([w], lr=lr)
    
    best_adv = images.clone()
    best_l2 = float('inf') * torch.ones(images.size(0), device=device)
    
    for step in range(num_iter):
        # tanh으로 w을 올바른 그림 범위로 바꾼다
        adv_images = 0.5 * (torch.tanh(w) + 1)  # [0, 1]으로 옮긴다
        
        # L2 거리
        l2_dist = ((adv_images - images) ** 2).view(images.size(0), -1).sum(dim=1)
        
        # 로짓을 얻는다
        logits = model(adv_images)
        
        # C&W 손실: 목표 갈래의 로짓을 가장 크게 한다
        # f(x') = max(max{Z(x')_j : j ≠ t} - Z(x')_t, -κ)
        target_logits = logits.gather(1, target_labels.view(-1, 1)).squeeze()
        
        # 목표를 뺀 최대 로짓
        other_logits = logits.clone()
        other_logits.scatter_(1, target_labels.view(-1, 1), -float('inf'))
        max_other_logits = other_logits.max(dim=1)[0]
        
        # f 함수
        f_loss = torch.clamp(max_other_logits - target_logits + kappa, min=0)
        
        # 전체 손실
        loss = l2_dist.sum() + c * f_loss.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 가장 좋은 맞겨루기 보기를 좇는다(L2이 가장 작으면서 이룬 것)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            is_successful = pred == target_labels
            is_better = l2_dist < best_l2
            should_update = is_successful & is_better
            
            best_l2 = torch.where(should_update, l2_dist, best_l2)
            for i in range(images.size(0)):
                if should_update[i]:
                    best_adv[i] = adv_images[i]
    
    return best_adv.detach()
```

코드는 $x^{adv}=(\tanh w+1)/2$로 범위를 제한하고 제곱 L2 거리와 목표 로짓 마진 벌점의 합을 최소화한다. [C&W 논문](https://arxiv.org/abs/1608.04644)의 아이디어를 단순화한 구현이며 전역 최소 섭동을 보장하지 않는다. c에 대한 탐색이 없고 w=0, 즉 회색 이미지에서 시작한다. 명시적인 L∞≤ε 제약도 없다.

κ는 마진 목표에 영향을 주지만 저장 시에는 argmax가 목표인지 여부만 확인한다. 따라서 저장된 성공 샘플이 κ 마진까지 만족한다고 보장할 수 없다. 성공하지 못한 표본은 원본으로 남으며 성공 마스크는 따로 반환하지 않는다. 모델을 eval로 전환하거나 모델 파라미터의 누적 기울기를 초기화하는 기능도 이 함수에는 없다.

## 9. 목표 클래스 선택

```python
def random_target(true_labels, num_classes):
    """참 갈래와 다른 아무 목표를 고른다."""
    targets = torch.randint(0, num_classes, true_labels.shape)
    # 목표가 참 이름표와 다르게 한다
    same_mask = targets == true_labels
    targets[same_mask] = (targets[same_mask] + 1) % num_classes
    return targets
```

`random_target`은 CPU에서 후보를 만들고 정답과 같으면 다음 클래스로 바꾼다. 클래스가 3개 이상이면 그 다음 클래스가 더 자주 선택되어 **정답을 제외한 균등 선택이 아니다**. CUDA 레이블과의 장치 불일치도 주의한다. num_classes≥2가 필요하다.

```python
def least_likely_target(model, images):
    """모델이 가장 자신 없어 하는 갈래를 고른다."""
    with torch.no_grad():
        logits = model(images)
        return logits.argmin(dim=1)
```

최소 로짓 목표는 가장 낮게 예측한 클래스를 택하지만 원래 정답과 반드시 다르지는 않다.

```python
def most_confusing_target(model, images, true_labels):
    """모델이 둘째로 자신 있어 하는 갈래를 고른다."""
    with torch.no_grad():
        logits = model(images)
        # 참 갈래 로짓을 -inf로 둔다
        logits.scatter_(1, true_labels.view(-1, 1), -float('inf'))
        return logits.argmax(dim=1)
```

`most_confusing_target`은 정답 클래스를 제외한 최대 로짓을 고른다. 최소 로짓 목표와는 공격 난이도와 의미가 다르며 평가 결과에서 선택 규칙을 밝힌다.

## 10. 두 번째 예제의 범위

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 겹말기 신경망 뜻매김(앞과 같다)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def visualize_targeted_attack(model, image, true_label, target_label, 
                               attack_fn, attack_params, class_names):
    """목표 있는 공격을 그려 본다."""
    
    # 본디 헤아림
    model.eval()
    with torch.no_grad():
        original_pred = model(image.unsqueeze(0)).argmax().item()
    
    # 맞겨루기 보기를 만든다
    target_tensor = torch.tensor([target_label])
    adv_image = attack_fn(model, image.unsqueeze(0), target_tensor, **attack_params)
    
    # 맞겨루기 헤아림
    with torch.no_grad():
        adv_pred = model(adv_image).argmax().item()
    
    # 흔들림을 셈한다
    perturbation = adv_image.squeeze(0) - image
    
    # 시각화한다
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image.squeeze().cpu(), cmap='gray')
    axes[0].set_title(f'Original\nTrue: {class_names[true_label]}\n'
                      f'Pred: {class_names[original_pred]}')
    axes[0].axis('off')
    
    # 눈에 띄게 흔들림을 키운다
    pert_display = 0.5 + 10 * perturbation.squeeze().cpu()
    axes[1].imshow(pert_display.clamp(0, 1), cmap='gray')
    axes[1].set_title(f'Perturbation (×10)\n'
                      f'L∞: {perturbation.abs().max():.4f}\n'
                      f'L2: {perturbation.norm():.4f}')
    axes[1].axis('off')
    
    axes[2].imshow(adv_image.squeeze().cpu(), cmap='gray')
    success = "✓" if adv_pred == target_label else "✗"
    axes[2].set_title(f'Adversarial {success}\n'
                      f'Target: {class_names[target_label]}\n'
                      f'Pred: {class_names[adv_pred]}')
    axes[2].axis('off')
    
    plt.suptitle('Targeted Adversarial Attack', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return adv_pred == target_label

def main():
    # 자료와 모델을 불러온다
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_data = torchvision.datasets.MNIST('./data', train=False, 
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    
    # 미리 익힌 모델을 불러온다(앞 마디의 익히기 코드를 쓴다고 본다)
    model = SimpleCNN()
    # model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()
    
    class_names = [str(i) for i in range(10)]
    
    # 공격할 표본을 찾는다
    for image, label in test_loader:
        if label.item() != 3:  # 3이 아닌 그림을 찾는다
            break
    
    # 공격: "3"으로 갈리게 해 본다
    target_label = 3
    
    print("Testing Targeted Attacks:")
    print("="*50)
    
    # 목표 있는 FGSM
    print("\n1. Targeted FGSM (ε=0.3):")
    success = visualize_targeted_attack(
        model, image.squeeze(0), label.item(), target_label,
        targeted_fgsm, {'epsilon': 0.3}, class_names
    )
    print(f"   Success: {success}")
    
    # 목표 있는 PGD
    print("\n2. Targeted PGD (ε=0.3, 40 iterations):")
    success = visualize_targeted_attack(
        model, image.squeeze(0), label.item(), target_label,
        lambda m, x, t, **kw: targeted_pgd(m, x, t, epsilon=0.3, alpha=0.01, num_iter=40),
        {}, class_names
    )
    print(f"   Success: {success}")

if __name__ == "__main__":
    main()
```

```
Testing Targeted Attacks:
==================================================

1. Targeted FGSM (ε=0.3):
   Success: False

2. Targeted PGD (ε=0.3, 40 iterations):
   Success: True
```

두 번째 main은 ToTensor의 0~1 이미지를 쓰므로 공격의 clipping 범위와 맞는다. 그러나 **새로 초기화한 CNN을 학습하거나 체크포인트에서 불러오지 않는다**. 따라서 보존된 한 이미지의 FGSM 실패·PGD 성공 출력은 학습된 MNIST 분류기의 강건성 비교가 아니다. 표본을 하나 골라 시각화한 결과만으로 성공률이나 방법 간 일반적인 순위를 주장할 수도 없다.

## 11. 강건성을 평가하는 기준

ε는 원래 픽셀 단위와 전처리에 맞춰 해석한다. 8/255나 0.3 같은 값도 데이터와 위협 모델에 따른 실험 조건이며, 크면 거의 100% 성공하거나 작으면 눈에 보이지 않는다는 보편적 보장은 없다. 표적 공격은 가능한 실패 중 특정 클래스 도달을 요구하지만 실제 위험의 크기는 응용과 실패 비용에 달려 있다.

공격이 찾은 오분류는 취약성의 증거지만 공격 실패는 허용 집합 안에 오분류가 없다는 증명이 아니다. 기울기 가림과 최적화 실패를 검토하고 필요한 경우 재시작·다른 공격·방어를 고려한 평가를 사용한다. 적대적 학습의 계산비용과 깨끗한 데이터 정확도에 대한 영향도 구현·데이터·목적함수에 따라 평가해야 하며 일정한 배수나 필연적 성능 저하로 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
L∞ 제약에서 일차 근사 손실을 최대화하는 섭동을 유도하라.

</div>

??? success "연습문제 1 풀이"

    기울기 g를 고정하면 목적함수는 Σgᵢδᵢ다. 각 좌표에서 |δᵢ|≤ε 아래 gᵢ와 같은 부호의 끝점을 택하면 최대가 되므로 δᵢ=ε sign(gᵢ)다. gᵢ=0이면 그 좌표의 모든 허용값이 같은 목적값을 준다. 원래 비선형 손실의 전역 최대해라는 뜻은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
Normalize(0.5,0.5)를 적용한 이미지에 0~1 clipping을 사용하면 어떤 문제가 생기는가?

</div>

??? success "연습문제 2 풀이"

    정규화 이미지의 유효 범위는 −1~1이다. 음수 값을 0으로 올리면 원본에서의 변화가 ε를 넘을 수 있고 ε=0이어도 입력이 변한다. 공격 좌표, ε와 clipping 범위를 모두 같은 표현에 맞춰야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
조건부 비표적 성공률과 전체 공격 후 오분류율을 구분하라.

</div>

??? success "연습문제 3 풀이"

    조건부 성공률은 원래 맞힌 표본 중 공격 후 틀린 비율이다. 전체 오분류율에는 원래부터 틀렸던 표본도 포함한다. clean 정확도와 adv 정확도의 단순 차이는 공격 후 새로 맞힌 표본을 상쇄하므로 조건부 성공률과 다를 수 있다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 보존된 중요도 샘플링 코드는 이 페이지의 어떤 공격을 구현하는가?

</div>

??? success "연습문제 4 풀이"

    어떤 공격도 구현하지 않는다. 표준 정규분포의 E[X²]=1을 t분포 제안으로 추정하는 별도 통계 예제다. 원래 연습문제와 코드가 공격 주제에 연결되지 않으므로 공격 구현의 증거로 해석하지 않는다. 양의 자유도의 t 제안과 이 정규분포·다항 함수 조합에서는 제곱 가중 적분이 유한하지만, 꼬리가 무거운 제안이라는 사실만으로 임의의 목표·함수에서 분산이 유한하다고 일반화할 수는 없다.

    ```python
    import numpy as np
    from scipy import stats

    def importance_sampling_x_squared(n_samples=10000, df=5):
        target = stats.norm(0, 1)
        proposal = stats.t(df=df)
        x = proposal.rvs(n_samples)
        weights = target.pdf(x) / proposal.pdf(x)
        f_x = x ** 2
        estimate = np.mean(weights * f_x)
        return estimate  # 1.0에 가까워야 함

    print(f"Estimate: {importance_sampling_x_squared():.4f}")
    print(f"True value: 1.0000")
    ```


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
GAN 생성기 갱신과 입력 공격에서 최적화 변수는 각각 무엇인가?

</div>

??? success "연습문제 5 풀이"

    입력 공격은 대상 모델을 고정하고 입력 또는 섭동을 바꾼다. GAN은 판별기를 통과한 기울기로 생성기 파라미터를 바꾸며 잠재 분포에서 나온 출력 전체를 학습한다. 미분 가능한 상대 모델을 이용한다는 공통점만으로 두 문제의 균형·수렴 성질이 같다고 할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
아래 FGSM 한 줄을 사용하기 전에 필요한 전제와 한계를 설명하라.

</div>

??? success "연습문제 6 풀이"

    x.grad가 올바른 레이블 손실에 대해 계산되어 있고 x가 0~1 범위이며 ε≥0이어야 한다. clipping 후의 실제 섭동은 x_adv−x로 확인한다. 노름 제약이 작아도 사람이 변화를 못 느낀다는 보장은 없으며 공격 한 번의 실패가 강건성 인증은 아니다.

    ```python
    x_adv = (x + eps * x.grad.sign()).clamp(0, 1)
    ```


---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
표적 공격 성공률을 보고할 때 분모를 어떻게 명시해야 하는가?

</div>

??? success "연습문제 7 풀이"

    전체 데이터인지, 정답이 목표와 다른 표본인지, 원래 올바르게 분류되고 예측도 목표가 아닌 표본인지 명시한다. 목표 클래스 선택 방식, 노름·ε, 반복 수·스텝·재시작, 접근 권한, 깨끗한 정확도와 표본 수도 함께 보고한다. 이 페이지의 표적 evaluate는 이미 목표로 분류된 표본도 포함한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
고차원에서 작은 좌표별 변화가 큰 손실 변화를 줄 수 있는 이유를 설명하라.

</div>

??? success "연습문제 8 풀이"

    선형화한 손실에서 FGSM의 증가량은 εΣ|gᵢ|=ε‖g‖₁이다. 여러 좌표의 영향이 같은 방향으로 합쳐질 수 있다. 이는 유용한 설명이지만 모든 적대적 예제의 유일한 원인이라는 정리는 아니다. 데이터 특징, 결정 경계와 전처리도 영향을 주며 강건성과 깨끗한 정확도의 관계도 조건에 따라 다르다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
섭동 생성 네트워크와 잠재 공간 공격을 어떻게 구분하는가?

</div>

??? success "연습문제 9 풀이"

    섭동 네트워크는 주어진 원본 근처의 제약된 변화를 출력하도록 학습할 수 있다. 잠재 공간 공격은 생성기의 z를 바꾸어 대상 모델의 예측을 바꾸려 하며 원본과의 픽셀 노름 제약을 자동으로 만족하지 않는다. 생성기 출력이라는 이유만으로 자연스러움이나 원래 의미가 보존된다고 보장할 수도 없다. 이 페이지에는 첫 네트워크의 학습 루프가 없다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
PGD 적대적 학습의 계산비용과 강건성을 고정된 배수·순위로 단정할 수 없는 이유는 무엇인가?

</div>

??? success "연습문제 10 풀이"

    내부 공격의 여러 입력 기울기 계산과 외부 파라미터 갱신이 필요해 일반 학습보다 비용이 커질 수 있다. 실제 시간은 네트워크, 배치, 기울기 계산·재사용과 공격 설정에 달려 있다. 공격 횟수 증가가 곧 같은 비율의 실행시간이나 모든 위협에 대한 방어를 보장하지 않는다. 학습에 사용한 공격과 별도의 평가 조건에서 확인해야 한다.



## 정리하며

적대적 공격은 입력 표현, 노름·허용 범위, 접근 권한과 목표를 먼저 정해야 한다. FGSM은 일차 근사의 한 단계 공격이고 반복 공격도 성공을 보장하지 않는다. 전처리와 성공률의 분모를 확인하며, 보존된 첫 예제의 범위 불일치와 두 번째 예제의 미학습 모델을 실제 강건성 결과와 구분한다.
