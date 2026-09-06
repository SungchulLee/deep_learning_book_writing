# 맞겨루기 공격의 바탕
맞겨루기 공격은 들임 자료에 꼼꼼히 빚은 흔들림을 넣어 기계 배움 모델을 주무른다. 사람은 알아채지 못할 때가 많은 이 흔들림이 모델로 하여금 아주 자신 있게 틀린 헤아림을 하게 만든다.

## 맞겨루기 공격 들어가기

### 맞겨루기 보기란 무엇인가?

맞겨루기 보기는 모델이 잘못하도록 일부러 빚은 기계 배움 모델의 들임이다. 그림 가르기에서는 다음과 같다.

$$x_{adv} = x + \delta$$

여기서 각 기호는 다음과 같다.

- $x$은 본디 들임이다
- $\delta$은 작은 흔들림이다
- $x_{adv}$은 맞겨루기 보기이다

흔들림 $\delta$은 $x_{adv}$이 보기에 $x$과 비슷하도록 작게 묶는다(흔히 $L_p$ 잣대로 잰다).

### 맞겨루기 보기는 왜 있는가?

맞겨루기 보기가 있는 까닭을 설명하는 가설이 여럿 있다.

1. **선형 가설(Goodfellow 외, 2015)**: 깊은 신경망은 차원이 높은 공간에서 사실상 선형이다. 작은 흔들림이 차원을 가로질러 쌓인다.

2. **튼튼하지 않은 특징**: 모델이 헤아리는 데는 쓸모 있으나 흔들림에 튼튼하지 않은 특징에 기댈 수 있다.

3. **결정 가장자리의 기하**: 복잡하고 차원 높은 결정 가장자리에는 들임이 조금만 바뀌어도 가장자리를 넘는 자리가 있을 수 있다.

## 위협 모델

### 공격 상황

| 상황 | 공격자가 아는 것 | 보기 |
|----------|-------------------|---------|
| **흰 상자** | 모델 전체 접근(얼개, 무게, 기울기) | 기울기에 바탕한 공격 |
| **검은 상자** | 묻기만 가능(들임 → 내놓기) | 옮기기 공격, 묻기 공격 |
| **회색 상자** | 일부만 앎(얼개는 알되 무게는 모름) | 얼개에 맞춘 공격 |

### 공격 목표

1. **목표 없는 공격**: 어떤 틀린 갈래로든 잘못 가르게 한다

$$\text{Find } \delta: f(x + \delta) \neq y_{true}$$

2. **목표 있는 공격**: 정해진 목표 갈래로 잘못 가르게 한다

$$\text{Find } \delta: f(x + \delta) = y_{target}$$

## 흔들림의 묶음

흔들림은 흔히 $L_p$ 잣대로 묶는다.

### L무한 잣대(최대 흔들림)

$$\|\delta\|_\infty = \max_i |\delta_i| \leq \epsilon$$

- 화소 하나의 최대 바뀜을 가둔다
- 흔한 묶음: $[0, 1]$ 안의 그림에서 $\epsilon = 8/255$

### L2 잣대(유클리드 거리)

$$\|\delta\|_2 = \sqrt{\sum_i \delta_i^2} \leq \epsilon$$

- 흔들림의 온 크기를 가둔다
- 다른 화소가 작으면 낱낱의 화소는 더 크게 바뀔 수 있다

### L0 잣대(성긴 흔들림)

$$\|\delta\|_0 = |\{i : \delta_i \neq 0\}| \leq k$$

- 고친 화소의 수를 가둔다
- 성긴 맞겨루기 헝겊에 쓴다

## 맞겨루기 만들개 맥락에서의 공격 갈래

### 맞겨루기 만들개에 대한 공격

맞겨루기 만들개는 저만의 맞겨루기 약점을 지닌다.

1. **독 넣기 공격**: 익히기 자료에 나쁜 표본을 넣는다
2. **빠져나가기 공격**: 가름개를 속이는 들임을 빚는다
3. **추론 공격**: 익히기 자료에 대한 앎을 뽑아낸다
4. **모델 빼내기**: 만들개가 배운 분포를 훔친다

### 맞겨루기 만들개가 만든 맞겨루기 보기

맞겨루기 만들개로 맞겨루기 보기를 만들 수 있다.

```python
import torch
import torch.nn as nn

class AdversarialGenerator(nn.Module):
    """맞겨루기 만들개 같은 얼개로 맞겨루기 흔들림을 만든다."""
    
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

## 공격 성공 잣대

### 가르기 잣대

```python
def evaluate_attack(model, x_clean, x_adv, y_true, y_target=None):
    """
    맞겨루기 공격의 성공을 따진다.
    
    인수:
        model: 겨눌 가름개
        x_clean: 깨끗한 그림
        x_adv: 맞겨루기 그림
        y_true: 참 레이블
        y_target: Target labels (for targeted attacks)
    
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

### 흔들림 잣대

```python
def perturbation_metrics(x_clean, x_adv):
    """
    흔들림 통계를 셈한다.
    
    인수:
        x_clean: Clean images (batch)
        x_adv: Adversarial images (batch)
    
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

## 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_adversarial_example(x_clean, x_adv, y_true, y_pred_clean, y_pred_adv,
                                   class_names=None, amplify=10):
    """
    깨끗한 그림, 흔들림, 맞겨루기 그림을 그려 본다.
    
    인수:
        x_clean: Clean image tensor (C, H, W)
        x_adv: Adversarial image tensor (C, H, W)
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

## 시험에 쓰는 흔한 신경망

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

## 요약

| 개념 | 설명 |
|---------|-------------|
| **맞겨루기 보기** | 작은 흔들림으로 잘못 가르게 하는 들임 |
| **흰 상자 공격** | 모델 전체에 접근하며 기울기를 셈할 수 있다 |
| **검은 상자 공격** | 묻기만 가능하다 |
| **목표 없는 공격** | 아무 잘못 가르기나 일으킨다 |
| **목표 있는 공격** | 정해진 잘못 가르기를 일으킨다 |
| **$L_\infty$ 묶음** | 최대 화소 바뀜을 가둔다 |
| **$L_2$ 묶음** | 온 흔들림 크기를 가둔다 |
| **속임 비율** | 이룬 공격의 몫 |

이 바탕을 아는 것은 신경망을 공격하고 지키는 데 모두 꼭 필요하며, 가름개가 맞겨루기 보기의 표적이 될 수 있는 맞겨루기 만들개에서도 그렇다.

---

# 빠른 기울기 부호 방법(FGSM)

Goodfellow 외가 2015년에 내놓은 빠른 기울기 부호 방법(FGSM)은 들임에 대한 손실의 기울기를 쓰는 단순하면서도 잘 듣는 흰 상자 맞겨루기 공격이다.

## 수학적 바탕

### FGSM 공격

FGSM은 기울기 방향으로 한 걸음을 내디뎌 맞겨루기 보기를 만든다.

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))$$

여기서 각 기호는 다음과 같다.

- $x$은 본디 들임이다
- $\epsilon$은 흔들림의 크기(공격 세기)이다
- $\nabla_x \mathcal{L}$은 들임에 대한 손실의 기울기이다
- $\text{sign}(\cdot)$은 기울기 성분마다 부호를 취한다

### 왜 부호인가?

부호 함수를 쓰면 흔들림이 $L_\infty$ 묶음을 채우게 된다.

$$\|\delta\|_\infty = \epsilon$$

화소마다 꼭 $\pm\epsilon$만큼 흔들려 묶음 안에서 흔들림을 가장 크게 한다.

### 선형 어림 풀이

FGSM은 손실의 일차 테일러 펼침에 바탕한다.

$$\mathcal{L}(x + \delta) \approx \mathcal{L}(x) + \delta^T \nabla_x \mathcal{L}(x)$$

$\|\delta\|_\infty \leq \epsilon$ 아래서 손실 늘어남을 가장 크게 하려면:

$$\delta^* = \arg\max_{\|\delta\|_\infty \leq \epsilon} \delta^T \nabla_x \mathcal{L}(x) = \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x))$$

## 구현

### 기본 FGSM 공격

```python
import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon, criterion=None):
    """
    그림 묶음에 FGSM 공격을 한다.
    
    인수:
        model: 겨눌 가름개
        images: Input images (requires_grad should be True)
        labels: 참 이름표
        epsilon: 흔들림의 크기
        criterion: Loss function (default: CrossEntropyLoss)
    
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
            max_batches: 따질 최대 묶음 수
        
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

### 목표 없는 FGSM

여느 FGSM 공격은 목표가 없다. 곧 그저 손실을 가장 크게 한다.

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

### 목표 있는 FGSM

목표 있는 FGSM은 정해진 목표 갈래로 잘못 가르게 하려 한다.

```python
def targeted_fgsm(model, images, target_labels, epsilon):
    """
    목표 있는 FGSM: 목표 갈래의 손실을 가장 작게 한다.
    
    흔들림은 손실이 줄어드는 방향으로 움직인다
    with respect to the target class, pulling predictions toward it.
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
        model: 겨눌 가름개
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

## 온전한 보기

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

## 엡실론 고르기

$\epsilon$을 어떻게 고르느냐가 공격 성공과 흔들림이 눈에 띄는 정도 사이의 맞바꿈을 다스린다.

| 엡실론 | 효과 |
|---------|--------|
| 작음(0.01-0.05) | 미묘한 바뀜, 낮은 성공 비율 |
| 보통(0.1-0.2) | 눈에 띄는 바뀜, 높은 성공 비율 |
| 큼(0.3 이상) | 아주 눈에 띔, 거의 100% 성공 |

### 흔한 엡실론 값

- **MNIST**: $\epsilon = 0.3$([0, 1] 안의 그림)
- **CIFAR-10**: $\epsilon = 8/255 \approx 0.031$
- **ImageNet**: $\epsilon = 4/255$ 또는 \$8/255$

## FGSM의 한계

1. **한 걸음**: FGSM은 기울기 걸음을 한 번만 쓴다. 되풀이 방법(PGD)이 더 세다
2. **선형 어림**: 손실이 거의 선형이라고 보는데 그렇지 않을 수 있다
3. **기울기 가리기**: FGSM에 맞서 익힌 모델이 기울기를 숨기는 법을 배울 수 있다
4. **옮겨감**: FGSM 보기가 다른 모델로 잘 옮겨 가지 않을 수 있다

## 요약

| 갈래 | FGSM |
|--------|------|
| **갈래** | 흰 상자, 기울기 바탕 |
| **묶음** | $L_\infty$ |
| **걸음** | 한 번 |
| **복잡도** | 뒤먹임 O(1)번 |
| **센 곳** | 빠르고 단순하다 |
| **여린 곳** | 흔들림이 가장 좋지는 않다 |

FGSM은 바탕 공격으로서, 그리고 맞겨루기 튼튼함을 이해하는 데 여전히 중요하다. 단순해서 빠른 따지기와 맞겨루기 익히기에 쓸모 있다.

---

# 목표 있는 맞겨루기 공격

목표 있는 맞겨루기 공격은 아무 틀린 갈래가 아니라 정해진 목표 갈래로 잘못 가르게 하려 한다. 이런 공격은 더 어렵지만 실제 상황에서는 더 위험하다.

## 목표 있는 공격과 목표 없는 공격

### 목표 없는 공격

$$\text{Find } \delta: f(x + \delta) \neq y_{true}$$

목표: 아무 잘못 가르기나 일으킨다.

### 목표 있는 공격

$$\text{Find } \delta: f(x + \delta) = y_{target}$$

목표: 정해진 목표 갈래로 잘못 가르게 한다.

### 실제의 차이

| 갈래 | 목표 없음 | 목표 있음 |
|--------|------------|----------|
| 어려움 | 더 쉽다 | 더 어렵다 |
| 다스림 | 낮다 | 높다 |
| 실제 세상의 위험 | 보통 | 높다 |
| 보기 | 그림을 알아보지 못함 | 얼굴을 허락된 사용자로 알아봄 |

## 목표 있는 FGSM

가장 단순한 목표 있는 공격은 목표 쪽으로 손실을 가장 작게 하도록 FGSM을 고친 것이다.

```python
import torch
import torch.nn as nn

def targeted_fgsm(model, images, target_labels, epsilon):
    """
    목표 있는 FGSM 공격.
    
    Instead of maximizing loss (pushing away from true class),
    we minimize loss (pulling toward target class).
    
    인수:
        model: 겨눌 가름개
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

## 되풀이 목표 있는 공격

### 기본 되풀이 방법(BIM) - 목표 있음

```python
def targeted_bim(model, images, target_labels, epsilon, alpha, num_iter):
    """
    Targeted Basic Iterative Method (Iterative FGSM).
    
    목표 갈래 쪽으로 작은 걸음을 여러 번 내딛는다.
    
    인수:
        model: 겨눌 가름개
        images: 들임 그림
        target_labels: 바라는 목표 갈래
        epsilon: Maximum total perturbation (L_inf bound)
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

### 쏜 기울기 내려가기(PGD) - 목표 있음

```python
def targeted_pgd(model, images, target_labels, epsilon, alpha, num_iter, 
                 random_start=True):
    """
    아무 첫자리매김을 갖춘 목표 있는 PGD 공격.
    
    인수:
        model: 겨눌 가름개
        images: 들임 그림
        target_labels: 바라는 목표 갈래
        epsilon: Maximum perturbation (L_inf)
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

## 카를리니-와그너(C&W) 목표 있는 공격

C&W 공격은 가장 센 목표 있는 공격 가운데 하나로, 가장 좋게 하기로 가장 작은 흔들림을 찾는다.

```python
def cw_targeted_attack(model, images, target_labels, c=1.0, kappa=0, 
                        num_iter=1000, lr=0.01, device='cpu'):
    """
    카를리니-와그너 L2 목표 있는 공격.
    
    Minimizes: ||δ||_2 + c * max(Z(x+δ)_t - max_{j≠t} Z(x+δ)_j, -κ)
    
    여기서 Z은 로짓, t은 목표 갈래, κ은 자신도 여유이다.
    
    인수:
        model: 겨눌 가름개
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

## 목표 고르기 방책

### 아무 목표

```python
def random_target(true_labels, num_classes):
    """참 갈래와 다른 아무 목표를 고른다."""
    targets = torch.randint(0, num_classes, true_labels.shape)
    # 목표가 참 이름표와 다르게 한다
    same_mask = targets == true_labels
    targets[same_mask] = (targets[same_mask] + 1) % num_classes
    return targets
```

### 가장 그럴듯하지 않은 목표

```python
def least_likely_target(model, images):
    """모델이 가장 자신 없어 하는 갈래를 고른다."""
    with torch.no_grad():
        logits = model(images)
        return logits.argmin(dim=1)
```

### 가장 헷갈리는 목표

```python
def most_confusing_target(model, images, true_labels):
    """모델이 둘째로 자신 있어 하는 갈래를 고른다."""
    with torch.no_grad():
        logits = model(images)
        # 참 갈래 로짓을 -inf으로 둔다
        logits.scatter_(1, true_labels.view(-1, 1), -float('inf'))
        return logits.argmax(dim=1)
```

## 온전한 보기

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

## 요약

| 공격 | 걸음 | 가장 좋게 하기 | 알맞은 곳 |
|--------|-------|--------------|----------|
| 목표 있는 FGSM | 1 | 없음 | 빠른 공격 |
| 목표 있는 BIM | 여러 번 | 없음 | 되풀이 다듬기 |
| 목표 있는 PGD | 여러 번 | 쏜 기울기 내려가기 | 센 공격 |
| C&W | 많이 | Adam | 가장 작은 흔들림 |

목표 있는 공격은 더 어렵지만 모델의 잘못된 움직임을 꼭 집어 다스릴 수 있게 한다. 얼굴 알아보기나 스스로 움직이는 얼개처럼 보안이 중요한 쓰임새에서 특히 위험하다.

## 연습문제

**연습문제 1.**
과녁 적분이 끝이 있는데도 중요도 표집의 흩어짐이 왜 끝없을 수 있는지 설명하여라.

??? success "연습문제 1 풀이"
    중요도 표집 어림자의 흩어짐은 $\text{Var}_q[w(x) f(x)]$에 비례하며, 여기서 $w(x) = p(x)/q(x)$은 중요도 무게이다. $q(x)$의 꼬리가 $p(x) f(x)$보다 가벼우면, $q$은 확률을 거의 주지 않는데 $p$은 주는 구역에서 비 $p(x)/q(x)$이 한없이 커질 수 있다. 그러면 이따금 어림값을 좌우하는 몹시 큰 무게가 생겨, 적분 $\mathbb{E}_p[f(X)]$이 끝이 있는데도 흩어짐이 끝없어진다(또는 사실상 끝없어진다).

---

**연습문제 2.**
중요도 무게 $w_1, \ldots, w_N$으로 나타낸 실효 표본 크기(ESS)의 공식을 이끌어 내어라.

??? success "연습문제 2 풀이"
    ESS은 무게 준 표본이 과녁 분포의 독립 표본 몇 개에 맞먹는지를 잰다:

    $$\text{ESS} = \frac{\left(\sum_{i=1}^N w_i\right)^2}{\sum_{i=1}^N w_i^2}$$

    무게가 모두 같으면($w_i = c$) ESS $= N$이다. 무게 하나가 좌우하면 ESS $\approx 1$이다. 이는 스스로 고르게 하는 중요도 표집 어림자의 흩어짐을 과녁에서 뽑은 독립 동일 분포 표본의 흩어짐에 견주어 뜯어보면 나온다.

---

**연습문제 3.**
중요도 표집으로 $\mathbb{E}_p[f(X)]$을 어림할 때 가장 좋은 제안 분포가 $q^*(x) \propto |f(x)| p(x)$임을 보여라.

??? success "연습문제 3 풀이"
    중요도 표집 어림자의 흩어짐은 $\text{Var}_q\left[\frac{f(X)p(X)}{q(X)}\right] / N$이다. 제약 $\int q(x) dx = 1$ 아래 라그랑주 곱수로 이를 $q$에 대해 가장 작게 하면 $q^*(x) = |f(x)| p(x) / \int |f(x')| p(x') dx'$이 나온다. $f \geq 0$일 때 이것이 흩어짐 0인 제안이다(어림자가 표본 하나로 정확한 답을 되돌린다). 실전에서 $q^*$은 우리가 셈하려는 바로 그 적분을 필요로 하므로 쓸 수 없다.

---

**연습문제 4.**
$X \sim \mathcal{N}(0,1)$일 때 $t$분포를 제안으로 써서 $\mathbb{E}[X^2]$의 단순한 중요도 표집 어림자를 구현하여라.

??? success "연습문제 4 풀이"
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
    $t$분포는 가우스보다 꼬리가 무거워 중요도 무게의 흩어짐이 끝이 있음을 보장한다.
