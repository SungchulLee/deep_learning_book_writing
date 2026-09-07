# 앎 옮기기
## 두루 보기

앎 옮기기는 크고 많이 담는 "스승" 모형의 앎을 작고 잘 드는 "제자" 모형으로 옮긴다. 제자는 굳은 이름표만이 아니라 스승의 부드러운 낌새 분포에서 배우며, 갈래끼리의 사이와 판단의 금에 대한 더 넉넉한 소식을 담는다. 이로써 밑천이 넉넉지 않은 자리에도 됨됨이 좋은 모형을 내놓을 수 있다.

## 왜 하는가

큰 모형은 맞음이 뛰어나지만 내놓는 값이 비싸다.

| 어려움 | 미침 |
|-----------|--------|
| 늦음이 큼 | 쓰는 이가 답답하고 제때 요건을 어긴다 |
| 기억 자리가 큼 | 끝단 장치에 들어가지 않고 DRAM 값이 비싸다 |
| 셈 값이 큼 | 크게 늘리면 미루어 봄이 비싸다 |
| 힘을 많이 씀 | 손전화의 배터리가 닳고 자료 집의 값이 든다 |

앎 옮기기는 작은 제자 모형이 큰 스승 모형을 흉내 내도록 익혀 이를 푼다. 제자를 처음부터 익히는 것보다 맞음이 낫다.

```
┌─────────────────────────────────────────────────────────────────┐
│                        앎 옮기기                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐         ┌───────────────┐                   │
│  │     스승      │         │     제자      │                   │
│  │  (큰 CNN)     │         │  (작은 CNN)   │                   │
│  └───────┬───────┘         └───────┬───────┘                   │
│          │                         │                            │
│          ▼                         ▼                            │
│    ┌───────────┐             ┌───────────┐                     │
│    │  부드러운 │             │  부드러운 │                     │
│    │  과녁     │────────────▶│  과녁     │  ← 앎 옮기기       │
│    │ (로짓)    │             │ (로짓)    │    잃음             │
│    └───────────┘             └───────────┘                     │
│                                    │                            │
│                                    │                            │
│                              ┌───────────┐                     │
│                              │   굳은    │                     │
│                              │   과녁    │  ← 가름             │
│                              │ (이름표)  │    잃음             │
│                              └───────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 이론 밑바탕

### 부드러운 과녁과 어둠 속의 앎

앎 옮기기의 고갱이 깨침은 스승의 날임 낌새가 굳은 이름표만 있을 때보다 더 많은 소식을 담는다는 것이다.

**굳은 이름표:**

$$y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] \quad \text{(cat)}$$

**부드러운 이름표(스승의 날임):**

$$p = [0.01, 0.02, 0.85, 0.05, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005]$$

부드러운 이름표는 스승이 이 그림을 자동차나 비행기보다 개(갈래 3)와 범(갈래 4)에 더 가깝다고 본다는 것을 알려 준다. 굳은 이름표에는 아예 없는 소식이다.

힌턴 등은 이 덧붙은 소식을 "어둠 속의 앎"이라 불렀다. 틀린 갈래들의 견준 낌새에 담긴 앎이다.

**굳은 이름표의 문제:**

굳은 이름표(원핫)은 종요로운 소식을 잃는다.

- 자신함 99%의 "고양이" → [0, 0, 1, 0, 0]
- 자신함 51%의 "고양이" → [0, 0, 1, 0, 0]

자신함이 크게 다른데도 익힘 신호는 똑같다.

### 온도 잣대 잡기

부드러운 이름표에서 소식을 더 뽑아내려고 소프트맥스의 "온도"를 올린다.

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

여기서 $z_i$은 로짓이고 $T$은 온도다.

**온도가 미치는 것:**

- $T = 1$: 여느 소프트맥스(뾰족한 분포)
- $T > 1$: 더 부드러운 분포, 갈래끼리의 사이에 대한 소식이 더 많다
- $T \to \infty$: 고른 분포

온도가 높을수록 갈래끼리의 낌새 차이가 뚜렷해져 부드러운 과녁이 제자에게 더 많은 것을 알려 준다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


def softmax_with_temperature(logits: torch.Tensor, 
                             temperature: float = 1.0) -> torch.Tensor:
    """
    온도 잣대를 곁들인 소프트맥스.
    
    온도가 높으면(T > 1):
    - 낌새 분포가 더 부드럽다
    - 갈래끼리의 사이에 대한 소식이 더 많다
    
    온도가 낮으면(T < 1):
    - 분포가 더 날카롭다
    - 미루어 봄을 더 자신한다
    
    T = 1: 여느 소프트맥스
    """
    return F.softmax(logits / temperature, dim=-1)


# 보기: 온도가 미치는 것
logits = torch.tensor([5.0, 2.0, 0.5, 0.1, 0.1])

print("온도가 소프트맥스에 미치는 것:")
print("-" * 50)
for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
    probs = softmax_with_temperature(logits, T)
    entropy = -(probs * probs.log()).sum()
    print(f"T={T:>4}: {probs.numpy().round(3)}  (엔트로피: {entropy:.3f})")
```

날임:
```
T= 0.5: [0.991 0.009 0.    0.    0.   ]  (엔트로피: 0.063)
T= 1.0: [0.936 0.047 0.01  0.007 0.007]  (엔트로피: 0.317)
T= 2.0: [0.757 0.17  0.039 0.017 0.017]  (엔트로피: 0.742)
T= 5.0: [0.44  0.26  0.13  0.085 0.085]  (엔트로피: 1.386)
T=10.0: [0.32  0.25  0.17  0.13  0.13 ]  (엔트로피: 1.531)
```

온도가 높을수록 엔트로피가 커지며 갈래끼리의 사이가 드러난다.

### 앎 옮기기 잃음

온 앎 옮기기 잃음은 두 몫을 아우른다.

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{hard}} + (1 - \alpha) \cdot \mathcal{L}_{\text{soft}}$$

**굳은 잃음(참 이름표와의 여느 엇갈린 엔트로피):**

$$\mathcal{L}_{\text{hard}} = -\sum_i y_i \log(p_i^{\text{student}})$$

**부드러운 잃음(스승과의 KL 갈림):**

$$\mathcal{L}_{\text{soft}} = T^2 \cdot D_{\text{KL}}\left(p^{\text{teacher}}(T) \| p^{\text{student}}(T)\right)$$

$$= T^2 \cdot \sum_i p_i^{\text{teacher}}(T) \log\frac{p_i^{\text{teacher}}(T)}{p_i^{\text{student}}(T)}$$

$T^2$ 값은 온도에 따라 기울기 크기가 바뀌는 것을 메운다.

### 기울기 살피기

제자의 로짓 $z_i^s$에 대한 부드러운 잃음의 기울기는

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i^s} = \frac{1}{T}\left(p_i^s(T) - p_i^t(T)\right)$$

온도가 높으면 이 기울기는 제자가 가장 큰 값만이 아니라 스승의 온 날임 분포를 맞추도록 이끈다. 잃음의 $T^2$ 잣대는 온도를 어떻게 고르든 기울기의 크기가 알맞게 남도록 한다.

## PyTorch로 짜기

### 밑바탕 앎 옮기기

```python
class DistillationLoss(nn.Module):
    """
    굳은 과녁과 부드러운 과녁을 아우른 앎 옮기기 잃음.
    
    L_total = α * L_hard + (1-α) * L_soft
    
    여기서:
    - L_hard = CrossEntropy(제자 날임, 참 이름표)
    - L_soft = T² * KL_div(제자 부드러움, 스승 부드러움)
    """
    
    def __init__(self,
                 temperature: float = 4.0,
                 alpha: float = 0.5):
        """
        Args:
            temperature: 소프트맥스 온도(높을수록 분포가 부드럽다)
            alpha: 굳은 잃음의 짐(부드러운 잃음에는 1-alpha)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        앎 옮기기 잃음을 셈한다.
        
        Args:
            student_logits: 제자의 날것 날임 (B, num_classes)
            teacher_logits: 스승의 날것 날임 (B, num_classes)
            labels: 참 이름표 (B,)
            
        Returns:
            온 잃음과 잃음 몫을 담은 사전
        """
        # 굳은 잃음(제자 대 참 이름표)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 부드러운 잃음(제자 대 스승)
        # 붙임말: 제자에는 F.log_softmax, 스승에는 F.softmax
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        soft_loss = self.kl_loss(student_soft, teacher_soft)
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # 아우른 잃음
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'total_loss': total_loss.item()
        }


def train_student_with_distillation(student: nn.Module,
                                    teacher: nn.Module,
                                    train_loader: torch.utils.data.DataLoader,
                                    test_loader: torch.utils.data.DataLoader,
                                    epochs: int = 20,
                                    temperature: float = 4.0,
                                    alpha: float = 0.5,
                                    lr: float = 1e-3,
                                    device: str = 'cpu') -> nn.Module:
    """
    앎 옮기기로 제자 모형을 익힌다.
    
    Args:
        student: 익힐 제자 모형
        teacher: 미리 익힌 스승 모형(얼려 둔다)
        train_loader: 익힘 자료
        test_loader: 시험 자료
        epochs: 익힘 판 수
        temperature: 앎 옮기기 온도
        alpha: 굳은 잃음의 짐
        lr: 배움 비율
        device: 익힐 장치
        
    Returns:
        익힌 제자 모형
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()  # 스승을 얼린다
    
    # 스승의 매개변수에 기울기가 붙지 않게 한다
    for param in teacher.parameters():
        param.requires_grad = False
    
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        student.train()
        epoch_losses = {'hard': 0, 'soft': 0, 'total': 0}
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # 스승의 미루어 봄을 얻는다(기울기 없이)
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            # 제자의 앞으로 걸음
            optimizer.zero_grad()
            student_logits = student(data)
            
            # 앎 옮기기 잃음
            loss, loss_dict = criterion(student_logits, teacher_logits, target)
            
            # 되돌아가고 가장 좋게 한다
            loss.backward()
            optimizer.step()
            
            # 잃음을 쌓는다
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[f'{key}_loss']
        
        scheduler.step()
        
        # 따진다
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = student(data)
                _, pred = output.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), 'best_student.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"{epoch+1}/{epochs}판 - "
                  f"굳은: {epoch_losses['hard']/len(train_loader):.4f}, "
                  f"부드러운: {epoch_losses['soft']/len(train_loader):.4f}, "
                  f"맞음: {100*acc:.2f}%")
    
    # 가장 좋은 모형을 얹는다
    student.load_state_dict(torch.load('best_student.pth'))
    return student
```

### 보기: CNN 앎 옮기기

```python
class TeacherCNN(nn.Module):
    """큰 스승 모형."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class StudentCNN(nn.Module):
    """작은 제자 모형(매개변수가 6배 적다)."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 매개변수 수를 견준다
def compare_models():
    teacher = TeacherCNN()
    student = StudentCNN()
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"스승 매개변수: {teacher_params:,}")
    print(f"제자 매개변수: {student_params:,}")
    print(f"눌러 담은 견줌: {teacher_params/student_params:.1f}배")
```

## 한발 더 나간 앎 옮기기 방법

### 결에 기댄 앎 옮기기(FitNets)

날임 낌새를 넘어 가운데 결까지 옮길 수 있으며, 이는 더 센 이끔 신호가 된다.

```python
class FeatureDistillationLoss(nn.Module):
    """
    결에 기댄 앎 옮기기(FitNets / 눈길 옮기기).
    
    스승과 제자의 가운데 결을 맞추어
    날임만 옮길 때보다 더 센 이끔 신호를 준다.
    """
    
    def __init__(self,
                 teacher_channels: int,
                 student_channels: int,
                 temperature: float = 4.0,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 spatial_matching: bool = True):
        """
        Args:
            teacher_channels: 스승 결 그림의 갈래 수
            student_channels: 제자 결 그림의 갈래 수
            temperature: 날임 앎 옮기기 온도
            alpha: 굳은 잃음의 짐
            beta: 결 잃음의 짐
            spatial_matching: 자리 눈길 그림을 맞출지
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.spatial_matching = spatial_matching
        
        # 차수를 맞추는 비추기 켜
        if teacher_channels != student_channels:
            self.projector = nn.Conv2d(student_channels, teacher_channels, 1)
        else:
            self.projector = nn.Identity()
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                student_features: torch.Tensor,
                teacher_features: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        결 앎 옮기기 잃음을 셈한다.
        
        Args:
            student_logits: 제자의 날임 로짓 (B, num_classes)
            teacher_logits: 스승의 날임 로짓 (B, num_classes)
            student_features: 제자의 가운데 결 (B, C_s, H, W)
            teacher_features: 스승의 가운데 결 (B, C_t, H, W)
            labels: 참 이름표 (B,)
            
        Returns:
            온 잃음과 잃음 몫을 담은 사전
        """
        # 굳은 잃음
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 부드러운 잃음
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(soft_student, soft_teacher)
        
        # 결 맞추기 잃음
        student_proj = self.projector(student_features)
        
        if self.spatial_matching:
            # 눈길 옮기기: 자리 눈길 그림을 맞춘다
            student_attn = self._spatial_attention(student_proj)
            teacher_attn = self._spatial_attention(teacher_features)
            feature_loss = self.mse_loss(student_attn, teacher_attn)
        else:
            # 곧바로 결 맞추기(잣대를 맞춘 뒤)
            student_norm = F.normalize(student_proj.flatten(2), dim=2)
            teacher_norm = F.normalize(teacher_features.detach().flatten(2), dim=2)
            feature_loss = self.mse_loss(student_norm, teacher_norm)
        
        # 잃음을 아우른다
        total_loss = (
            self.alpha * hard_loss +
            (1 - self.alpha - self.beta) * self.temperature ** 2 * soft_loss +
            self.beta * feature_loss
        )
        
        return total_loss, {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'feature_loss': feature_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _spatial_attention(self, features: torch.Tensor) -> torch.Tensor:
        """
        자리 눈길 그림을 셈한다: 갈래에 걸친 살림 제곱의 합.
        
        Args:
            features: 결 그림 (B, C, H, W)
            
        Returns:
            잣대 맞춘 눈길 그림 (B, H, W)
        """
        attention = (features ** 2).sum(dim=1)  # (B, H, W)
        attention = attention / (attention.sum(dim=(1, 2), keepdim=True) + 1e-8)
        return attention


def attention_transfer_loss(student_attention: torch.Tensor,
                           teacher_attention: torch.Tensor) -> torch.Tensor:
    """
    홀로 쓰는 눈길 옮기기 잃음.
    
    자리 눈길 그림(모형이 "보는" 자리)을 맞춘다.
    """
    # 눈길 그림의 잣대를 맞춘다
    student_norm = F.normalize(
        student_attention.pow(2).mean(1).view(student_attention.size(0), -1), 
        dim=1
    )
    teacher_norm = F.normalize(
        teacher_attention.pow(2).mean(1).view(teacher_attention.size(0), -1), 
        dim=1
    )
    
    return (student_norm - teacher_norm).pow(2).mean()
```

### 여러 켜 앎 옮기기

가운데 켜 여럿에서 옮기면 이끔이 더 넉넉해진다.

```python
class MultiLayerDistillation(nn.Module):
    """
    가운데 켜 여럿에서 앎을 옮기기.
    
    여러 깊이의 결을 맞추어 이끔을 더 넉넉하게 한다.
    """
    
    def __init__(self,
                 layer_configs: List[Dict],  # [{'t': name, 's': name, 'w': weight}, ...]
                 temperature: float = 4.0,
                 alpha: float = 0.5,
                 beta: float = 0.1):
        """
        Args:
            layer_configs: 스승/제자 켜 이름과 짐을 담은 사전들의 목록
            temperature: 날임 앎 옮기기 온도
            alpha: 굳은 잃음의 짐
            beta: 온 결 잃음의 짐
        """
        super().__init__()
        self.layer_configs = layer_configs
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor],
                labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        여러 켜 앎 옮기기 잃음을 셈한다.
        """
        loss_dict = {}
        
        # 굳은 잃음
        hard_loss = self.ce_loss(student_logits, labels)
        loss_dict['hard_loss'] = hard_loss.item()
        
        # 부드러운 잃음
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        loss_dict['soft_loss'] = soft_loss.item()
        
        # 켜 여럿에서의 결 옮기기
        feature_loss = 0.0
        for config in self.layer_configs:
            t_name, s_name = config['t'], config['s']
            weight = config.get('w', 1.0)
            
            if t_name in teacher_features and s_name in student_features:
                t_feat = teacher_features[t_name].detach()
                s_feat = student_features[s_name]
                
                # 잣대를 맞추고 MSE을 셈한다
                fl = F.mse_loss(
                    F.normalize(s_feat.flatten(1), dim=1),
                    F.normalize(t_feat.flatten(1), dim=1)
                )
                feature_loss += weight * fl
                loss_dict[f'feature_{s_name}'] = fl.item()
        
        loss_dict['feature_total'] = feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss
        
        # 아우른다
        output_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        total_loss = output_loss + self.beta * feature_loss
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
```

### 스스로 앎 옮기기

제자와 스승이 같은 얼개다. 모형이 제 스스로에게서 앎을 옮긴다.

```python
class SelfDistillation(nn.Module):
    """
    스스로 앎 옮기기: 모형이 제 앞선 미루어 봄에서 배운다.
    
    갈래:
    1. 다시 난 그물: 익힌 뒤 같은 얼개로 옮긴다
    2. 깊은 서로 배움: 그물 둘이 서로 가르친다
    3. 앎 옮기기로서의 이름표 매끄럽게 하기: 부드러운 이름표가 넌지시 스승 노릇을 한다
    """
    
    def __init__(self,
                 model: nn.Module,
                 temperature: float = 4.0,
                 alpha: float = 0.5):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        
        # 앞 판의 부드러운 과녁을 담아 둔다
        self.soft_targets = {}
    
    def compute_soft_targets(self,
                            data_loader: torch.utils.data.DataLoader,
                            device: str = 'cpu'):
        """
        익힘 보기 모두의 부드러운 과녁을 셈해 담는다.
        """
        self.model.eval()
        self.soft_targets = {}
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(device)
                logits = self.model(data)
                soft = F.softmax(logits / self.temperature, dim=1)
                
                for i, s in enumerate(soft):
                    idx = batch_idx * data_loader.batch_size + i
                    self.soft_targets[idx] = s.cpu()
    
    def train_step(self,
                   data: torch.Tensor,
                   labels: torch.Tensor,
                   indices: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        스스로 앎 옮기기의 익힘 한 걸음.
        """
        self.model.train()
        
        optimizer.zero_grad()
        logits = self.model(data)
        
        # 굳은 잃음
        hard_loss = F.cross_entropy(logits, labels)
        
        # 부드러운 잃음(담아 둔 과녁에서)
        if self.soft_targets:
            soft_targets = torch.stack([self.soft_targets[i.item()] for i in indices])
            soft_targets = soft_targets.to(data.device)
            
            student_soft = F.log_softmax(logits / self.temperature, dim=1)
            soft_loss = F.kl_div(student_soft, soft_targets, reduction='batchmean')
            soft_loss = soft_loss * (self.temperature ** 2)
        else:
            soft_loss = torch.tensor(0.0)
        
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        total_loss.backward()
        optimizer.step()
        
        return {
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item() if isinstance(soft_loss, torch.Tensor) else 0,
            'total_loss': total_loss.item()
        }


def self_distillation_training(model: nn.Module,
                               train_loader: torch.utils.data.DataLoader,
                               epochs: int = 100,
                               distill_epochs_start: int = 50,
                               temperature: float = 4.0,
                               device: str = 'cpu') -> nn.Module:
    """
    스스로 앎 옮기기: 모형이 제 지난 미루어 봄에서 배운다.
    
    처음 익힘이 끝나면 모형 제 부드러운 미루어 봄을
    덧붙은 이끔으로 쓴다.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    past_predictions = {}  # 앞 판의 미루어 봄을 담아 둔다
    
    for epoch in range(epochs):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # 여느 잃음
            loss = criterion(output, target)
            
            # 몸풀기 뒤 스스로 앎 옮기기를 더한다
            if epoch >= distill_epochs_start and batch_idx in past_predictions:
                soft_loss = F.kl_div(
                    F.log_softmax(output / temperature, dim=-1),
                    F.softmax(past_predictions[batch_idx] / temperature, dim=-1),
                    reduction='batchmean'
                )
                loss = 0.5 * loss + 0.5 * (temperature ** 2) * soft_loss
            
            loss.backward()
            optimizer.step()
            
            # 다음 판을 위해 미루어 봄을 담는다
            past_predictions[batch_idx] = output.detach()
    
    return model
```

### 여러 스승 앎 옮기기

앎을 더 넉넉히 하려고 스승 여럿을 모둠으로 쓴다.

$$p^{\text{ensemble}} = \frac{1}{K}\sum_{k=1}^{K} p^{\text{teacher}_k}$$

```python
class MultiTeacherDistillation(nn.Module):
    """
    스승 모형 여럿에서 앎을 옮기기.
    
    스승이 여럿이면 보는 눈이 여러 가지라 스승 하나일 때보다
    제자의 두루 미침이 나아지는 일이 잦다.
    """
    
    def __init__(self,
                 teachers: List[nn.Module],
                 aggregation: str = 'mean',
                 temperature: float = 4.0):
        """
        Args:
            teachers: 스승 모형의 목록
            aggregation: 'mean', 'weighted', 'max' 가운데 하나
            temperature: 앎 옮기기 온도
        """
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.aggregation = aggregation
        self.temperature = temperature
        
        if aggregation == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(teachers)) / len(teachers))
    
    def get_teacher_soft_targets(self,
                                 data: torch.Tensor) -> torch.Tensor:
        """
        스승 모두의 부드러운 과녁을 모은다.
        """
        soft_targets = []
        
        for teacher in self.teachers:
            teacher.eval()
            with torch.no_grad():
                logits = teacher(data)
                soft = F.softmax(logits / self.temperature, dim=1)
                soft_targets.append(soft)
        
        soft_targets = torch.stack(soft_targets, dim=0)  # (K, B, C)
        
        if self.aggregation == 'mean':
            return soft_targets.mean(dim=0)
        elif self.aggregation == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            return (soft_targets * weights.view(-1, 1, 1)).sum(dim=0)
        elif self.aggregation == 'max':
            return soft_targets.max(dim=0)[0]
        else:
            raise ValueError(f"모르는 모으기: {self.aggregation}")
```

### 살아 있는 앎 옮기기(깊은 서로 배움)

스승과 제자를 한꺼번에 익힌다.

```python
class OnlineDistillation(nn.Module):
    """
    깊은 서로 배움: 익히는 동안 그물끼리 서로 가르친다.
    
    미리 익힌 스승이 있어야 하지 않다. 그물 여럿이 함께 배우며,
    흔히 한쪽으로만 옮기는 여느 길보다 낫다.
    """
    
    def __init__(self,
                 models: List[nn.Module],
                 temperature: float = 4.0):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.temperature = temperature
    
    def forward(self,
                data: torch.Tensor,
                labels: torch.Tensor) -> Tuple[List[torch.Tensor], Dict[str, float]]:
        """
        서로 배우며 모형 모두의 앞으로 걸음.
        """
        logits_list = [model(data) for model in self.models]
        
        losses = []
        loss_dict = {}
        
        for i, logits in enumerate(logits_list):
            # 굳은 잃음
            hard_loss = F.cross_entropy(logits, labels)
            
            # 부드러운 잃음: 다른 모형 모두에게서 배운다
            soft_loss = 0.0
            student_soft = F.log_softmax(logits / self.temperature, dim=1)
            
            for j, other_logits in enumerate(logits_list):
                if i != j:
                    with torch.no_grad():
                        teacher_soft = F.softmax(other_logits / self.temperature, dim=1)
                    soft_loss += F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            
            soft_loss = soft_loss / (len(self.models) - 1) * (self.temperature ** 2)
            total_loss = hard_loss + soft_loss
            losses.append(total_loss)
            loss_dict[f'model_{i}_loss'] = total_loss.item()
        
        return losses, loss_dict
```

## 차근차근 앎 옮기기

아주 크게 눌러 담으려면 가운데 크기의 모형을 거쳐 차근차근 옮긴다.

```
큰 스승 → 가운데 모형 → 작은 제자
```

걸음마다가 가장 작은 모형으로 곧바로 옮기는 것보다 쉽다.

```python
def progressive_distillation(teachers: List[nn.Module],
                            train_loader: torch.utils.data.DataLoader,
                            test_loader: torch.utils.data.DataLoader,
                            epochs_per_stage: int = 20,
                            temperature: float = 4.0,
                            device: str = 'cpu') -> nn.Module:
    """
    모형 사슬을 거치는 차근차근 앎 옮기기.
    
    Args:
        teachers: 큰 것에서 작은 것 차례의 모형 목록
                  [큰 스승, 가운데 모형, 작은 제자]
        train_loader: 익힘 자료
        test_loader: 시험 자료
        epochs_per_stage: 옮기기 도막마다의 익힘 판 수
        temperature: 앎 옮기기 온도
        device: 익힐 장치
        
    Returns:
        끝으로 익힌 작은 제자
    """
    for i in range(len(teachers) - 1):
        teacher = teachers[i]
        student = teachers[i + 1]
        
        print(f"\n{'='*60}")
        print(f"{i+1}도막: 모형 {i} → 모형 {i+1}으로 옮기는 중")
        print(f"{'='*60}")
        
        student = train_student_with_distillation(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs_per_stage,
            temperature=temperature,
            device=device
        )
    
    return teachers[-1]
```

## 하이퍼파라미터 고르기

### 온도 고르기

| 온도 | 미침 | 잘 맞는 자리 |
|-------------|--------|----------|
| 1~2 | 날카로운 분포 | 이미 자신하는 스승 |
| 3~5 | 알맞은 부드러움 | 두루 쓰기(기본값: 4) |
| 10~20 | 아주 부드러움 | 갈래끼리 닮음이 걸릴 때, 얼개 차이가 클 때 |
| 20 넘음 | 거의 고름 | 흔히 너무 부드럽다 |

### 알파 고르기

| 알파 | 풀이 | 쓸 때 |
|-------|---------------|-------------|
| 0.0 | 부드러운 잃음만 | 아주 자신하고 맞는 스승 |
| 0.3~0.5 | 고르게 | 두루 쓰기(기본값) |
| 0.7~0.9 | 거의 굳은 잃음 | 못 미더운 스승이나 곱게 맞추기 |
| 1.0 | 굳은 잃음만 | 앎 옮기기 없음(밑금) |

### 어느 방법을 언제 쓸까

| 형편 | 즐겨 쓸 길 |
|----------|---------------|
| 스승과 제자의 맞음 차이가 큼 | 온도를 높인다(8~20) |
| 얼개가 비슷함 | 결 옮기기를 쓴다 |
| 제자가 아주 작음 | 차근차근 옮기기를 쓴다 |
| 익힘 자료가 적음 | 앎 옮기기가 크게 도움이 된다 |
| 모둠 스승 | 스승 여럿을 아우른다 |
| 미리 익힌 스승이 없음 | 깊은 서로 배움을 쓴다 |

### 제자 얼개 꾸미기

제자는 내놓을 만큼 작되 스승의 앎을 담을 만큼 커야 한다.

- **너무 작으면**: 얽힌 판단의 금을 배우지 못한다
- **너무 크면**: 보람이 줄고 눌러 담는 뜻이 사라진다

어림 규칙: 스승 매개변수의 10~30%에서 비롯해 맞음 요건에 따라 손본다.

### 가장 좋은 온도 찾기

```python
def find_optimal_temperature(teacher: nn.Module,
                            student_class: type,
                            train_loader: torch.utils.data.DataLoader,
                            val_loader: torch.utils.data.DataLoader,
                            temperatures: List[float] = [1, 2, 4, 8, 16, 20],
                            quick_epochs: int = 10,
                            device: str = 'cpu') -> float:
    """
    격자 뒤지기로 가장 좋은 앎 옮기기 온도를 찾는다.
    """
    results = []
    
    for T in temperatures:
        # 해 볼 때마다 새 제자
        student = student_class()
        
        # 빠른 앎 옮기기 익힘
        trained = train_student_with_distillation(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=quick_epochs,
            temperature=T,
            device=device
        )
        
        # 따진다
        accuracy = evaluate_accuracy(trained, val_loader, device)
        results.append((T, accuracy))
        print(f"온도 {T}: {accuracy*100:.2f}%")
    
    best_T = max(results, key=lambda x: x[1])[0]
    print(f"\n가장 좋은 온도: {best_T}")
    return best_T
```

## 따지는 자

```python
def evaluate_distillation(teacher: nn.Module,
                          student: nn.Module,
                          student_baseline: nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          device: str = 'cpu') -> Dict[str, float]:
    """
    앎 옮기기가 얼마나 잘 듣는지 두루 따진다.
    """
    def get_accuracy(model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return correct / total
    
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    teacher_acc = get_accuracy(teacher)
    student_acc = get_accuracy(student)
    baseline_acc = get_accuracy(student_baseline)
    
    results = {
        'teacher_accuracy': teacher_acc,
        'student_distilled_accuracy': student_acc,
        'student_baseline_accuracy': baseline_acc,
        'distillation_improvement': student_acc - baseline_acc,
        'gap_to_teacher': teacher_acc - student_acc,
        'compression_ratio': count_params(teacher) / count_params(student),
        'parameter_reduction': 1 - count_params(student) / count_params(teacher)
    }
    
    return results


def evaluate_distillation_agreement(teacher: nn.Module,
                                   student: nn.Module,
                                   test_loader: torch.utils.data.DataLoader,
                                   device: str = 'cpu') -> Dict[str, float]:
    """
    제자가 스승의 결을 얼마나 잘 흉내 내는지 따진다.
    """
    teacher.eval()
    student.eval()
    
    teacher_correct = 0
    student_correct = 0
    agreement = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            teacher_out = teacher(data)
            student_out = student(data)
            
            teacher_pred = teacher_out.argmax(dim=1)
            student_pred = student_out.argmax(dim=1)
            
            teacher_correct += (teacher_pred == target).sum().item()
            student_correct += (student_pred == target).sum().item()
            agreement += (teacher_pred == student_pred).sum().item()
            total += target.size(0)
    
    return {
        'teacher_accuracy': teacher_correct / total,
        'student_accuracy': student_correct / total,
        'teacher_student_agreement': agreement / total,
        'knowledge_transfer_efficiency': student_correct / teacher_correct
    }
```

## 간추림

앎 옮기기는 잘 드는 모형을 내놓을 수 있게 한다.

1. **고갱이 깨침**: 작은 제자가 큰 스승을 흉내 내도록 익힌다
2. **부드러운 과녁**: 온도로 낌새의 사이를 지킨다
3. **잃음 함수**: 굳은 이름표와 부드러운 과녁을 아우른다
4. **한발 더 나간 방법**: 결 맞추기, 눈길 옮기기, 스스로 옮기기
5. **온도**: 얼개 차이가 클수록 높인다

고갱이로 즐겨 쓸 길:

- 온도 4~8, 알파 0.5에서 비롯한다
- 얼개가 비슷하면 결 옮기기를 쓴다
- 제자가 스승의 결을 맞추는지 따진다
- 제자가 아주 작으면 차근차근 옮기기를 헤아린다
- 미리 익힌 스승이 없으면 깊은 서로 배움을 쓴다

## 살펴볼 거리

1. Hinton, G., Vinyals, O., & Dean, J. "Distilling the Knowledge in a Neural Network." arXiv 2015.
2. Romero, A., et al. "FitNets: Hints for Thin Deep Nets." ICLR 2015.
3. Zagoruyko, S. & Komodakis, N. "Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer." ICLR 2017.
4. Zhang, Y., et al. "Deep Mutual Learning." CVPR 2018.
5. Furlanello, T., et al. "Born-Again Neural Networks." ICML 2018.
6. Gou, J., et al. "Knowledge Distillation: A Survey." IJCV 2021.

## 익힘 문제

**익힘 1.**
이 마디에서 다룬 다듬기 재주들을 맞음 잃음, 미루어 봄 빨라짐, 짜기의 번거로움으로 견주어 맞바꿈을 밝혀라.

??? success "익힘 1 풀이"
    재주마다 맞바꿈의 결이 다르다. 수 줄이기(INT8)은 흔히 2~4배 빨라지면서 맞음 잃음이 1% 미만이고, 틀이 받쳐 주므로 짜는 품이 가운데쯤이다. 쳐내기는 성김의 결에 따라 빨라짐이 들쭉날쭉하며(짜임새 있는 쳐내기가 쇠 붙임새에 더 맞다) 맞음 잃음은 1~3%이다. 앎 옮기기는 얼개 자체의 미루어 봄 값은 그대로 두되 더 작은 제자를 써서 2~10배로 눌러 담고 맞음 잃음은 1~5%이다. 신경 얼개 찾기는 가장 좋은 얼개를 찾아 주지만 찾는 데 엄청난 셈이 든다(GPU 수천 시간). 금융 쓰임에서는 받아들일 수 있는 맞음 잃음이 어긋남의 값에 매인다. $\square$

---

**익힘 2.**
단순한 앞먹임 그물에 익힘 뒤 수 줄이기(INT8)을 짜 넣고, 잣대 자료 꾸러미에서 맞음이 얼마나 떨어지고 미루어 봄이 얼마나 빨라지는지 재어라.

??? success "익힘 2 풀이"
    PyTorch의 수 줄이기 API을 쓴다. (1) float32 모형을 밑금 맞음까지 익힌다. (2) 움직이는 수 줄이기에는 `torch.quantization.quantize_dynamic`을 쓰고, 붙박인 수 줄이기에는 본보기 자료로 눈금을 맞춘다. (3) 미루어 보는 때(묶음 1000개의 평균)와 시험 꾸러미의 맞음을 잰다. 흔한 결과: CPU에서 1.5~3배 빨라지고, 움직이는 수 줄이기는 맞음이 0.5% 미만, 눈금 맞춘 붙박인 수 줄이기는 0.2% 미만 떨어진다. 모형 크기는 약 4배 줄어든다(FP32에서 INT8으로). 고갱이: 붙박인 수 줄이기에는 내놓을 자리의 자료를 잘 드러내는 눈금 맞추기 꾸러미가 있어야 한다. $\square$

---

**익힘 3.**
내놓은 모형의 자료 옮겨감, 뜻 옮겨감, 됨됨이 떨어짐을 짚어내는 서비스 지켜보기 얼개를 꾸며라. 자와 알림 문턱을 밝혀라.

??? success "익힘 3 풀이"
    세 켜를 지켜본다. (1) 자료 옮겨감: KS 시험이나 PSI(무리 든든함 지수)으로 들임 결의 분포를 좇는다. 어떤 결이든 PSI > 0.2이면 알린다. (2) 뜻 옮겨감: 미루어 봄 분포의 옮겨감과 (얻을 수 있으면) 참 이름표 분포를 좇는다. 미루어 봄의 평균이 밑금 동안에서 잣대 어긋남 2배 넘게 옮겨가면 알린다. (3) 모형 떨어짐: 굴러가는 창으로 살아 있는 맞음과 잃음을 좇는다. 맞음이 밑금보다 3% 넘게 떨어지거나 늦음이 약속을 넘으면(p99 > 50ms 따위) 알린다. Grafana으로 판을 만들고, Prometheus에 자를 담고, PagerDuty으로 알림을 보낸다. $\square$

---

**익힘 4.**
금융 거래 얼개의 늦음 요건이 웹 서비스와 밑바탕부터 다른 까닭을 밝혀라. 이것이 내놓기 다듬기 꾀에 어떻게 걸리는가?

??? success "익힘 4 풀이"
    웹 서비스는 100~500ms의 늦음과 이따금의 치솟음을 받아 준다. 거래 얼개는 붙박이로 1밀리초 아래(고빈도 거래에서는 흔히 100마이크로초 미만)여야 한다. 그래서 다듬는 꾀가 달라진다. (1) 쓰레기 치우기의 멈춤을 없앤다(파이썬 대신 C++ 미루어 봄). (2) 기억을 미리 다 잡아 둔다(그때그때 잡지 않는다). (3) 실을 알맹이에 붙박는다(자리 바꿈을 없앤다). (4) 늦음이 가장 걸리는 길목에는 FPGA이나 ASIC을 쓴다. (5) 수 줄이기는 있어야 하되 붙박이지 않은 반올림을 들여서는 안 된다. 묶음 미루어 봄은 쓸 수 없다(판단 하나하나가 늦음에 걸린다). 내놓기 더미는 나름보다 가장 나쁜 자리의 늦음(p99.9)을 앞세운다. $\square$
