# 지식 증류

---

## 1. 개관

지식 증류는 크고 용량이 큰 "교사" 모델의 앎을 작고 효율적인 "학생" 모델로 옮긴다. 학생은 하드 레이블만이 아니라 교사가 내는 소프트 확률 분포에서도 배운다. 이 분포는 클래스 사이의 유사도와 결정 경계에 대해 하드 레이블보다 많은 정보를 담는다. 이로써 계산 자원이 넉넉지 않은 환경에도 성능 좋은 모델을 배포할 수 있다.

---

## 2. 왜 하는가

큰 모델은 정확도가 뛰어나지만 배포 비용이 크다.

| 어려움 | 영향 |
|-----------|--------|
| 지연이 큼 | 사용자 경험이 나빠지고 실시간 요건을 어긴다 |
| 메모리가 큼 | 엣지 장치에 들어가지 않고 DRAM 비용이 크다 |
| 계산 비용이 큼 | 서비스 규모를 키우면 추론 비용이 커진다 |
| 전력을 많이 씀 | 휴대폰의 배터리가 닳고 데이터 센터의 전력 비용이 든다 |

지식 증류는 작은 학생 모델이 큰 교사 모델을 흉내 내도록 트레인시켜 이를 푼다. 같은 학생을 처음부터 트레인하는 것보다 대개 정확도가 높다.

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
│    │  타깃     │────────────▶│  타깃     │  ← 앎 옮기기       │
│    │ (로짓)    │             │ (로짓)    │    잃음             │
│    └───────────┘             └───────────┘                     │
│                                    │                            │
│                                    │                            │
│                              ┌───────────┐                     │
│                              │   굳은    │                     │
│                              │   타깃    │  ← 가름             │
│                              │ (이름표)  │    잃음             │
│                              └───────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 이론 기초

### 소프트 타깃과 암흑 지식(dark knowledge)

지식 증류의 핵심 통찰은 교사의 출력 확률이 하드 레이블만 있을 때보다 더 많은 정보를 담는다는 것이다.

**하드 레이블:**

$$y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] \quad \text{(cat)}$$

**소프트 레이블(교사의 출력):**

$$p = [0.01, 0.02, 0.85, 0.05, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005]$$

소프트 레이블은 교사가 이 그림을 자동차나 비행기보다 개(클래스 3)와 호랑이(클래스 4)에 더 가깝다고 본다는 것을 알려 준다. 하드 레이블에는 아예 없는 정보다.

힌턴 등은 이 덧붙은 정보를 "암흑 지식(dark knowledge)"이라 불렀다. 틀린 클래스들의 상대 확률에 담긴 앎이다.

**하드 레이블의 문제:**

하드 레이블(원핫)은 중요한 정보를 잃는다.

- 확신도 99%의 "고양이" → [0, 0, 1, 0, 0]
- 확신도 51%의 "고양이" → [0, 0, 1, 0, 0]

확신도가 크게 다른데도 트레인 신호는 똑같다.

### 온도 스케일링

소프트 레이블에서 정보를 더 뽑아내려고 소프트맥스의 "온도"를 올린다.

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

여기서 $z_i$는 로짓이고 $T$는 온도다.

**온도의 영향:**

- $T = 1$: 보통의 소프트맥스(뾰족한 분포)
- $T > 1$: 더 부드러운 분포, 클래스끼리의 사이에 대한 정보가 더 많다
- $T \to \infty$: 균등 분포

온도가 높을수록 틀린 클래스들의 작은 확률이 커져, 그들 사이의 상대적 차이가 드러나고 소프트 타깃이 학생에게 더 많은 것을 알려 준다.

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
    - 확률 분포가 더 부드럽다
    - 클래스끼리의 사이에 대한 소식이 더 많다
    
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

**출력:**

```
온도가 소프트맥스에 미치는 것:
--------------------------------------------------
T= 0.5: [0.997 0.002 0.    0.    0.   ]  (엔트로피: 0.020)
T= 1.0: [0.93  0.046 0.01  0.007 0.007]  (엔트로피: 0.326)
T= 2.0: [0.666 0.149 0.07  0.057 0.057]  (엔트로피: 1.069)
T= 5.0: [0.37  0.203 0.15  0.139 0.139]  (엔트로피: 1.524)
T=10.0: [0.277 0.206 0.177 0.17  0.17 ]  (엔트로피: 1.590)
```

온도가 높을수록 엔트로피가 커지며 클래스끼리의 사이가 드러난다.

### 지식 증류 로스

전체 지식 증류 로스는 두 몫을 아우른다.

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{hard}} + (1 - \alpha) \cdot \mathcal{L}_{\text{soft}}$$

**하드 로스(참 레이블과의 보통 교차 엔트로피):**

$$\mathcal{L}_{\text{hard}} = -\sum_i y_i \log(p_i^{\text{student}})$$

**소프트 로스(교사와의 KL 발산):**

$$\mathcal{L}_{\text{soft}} = T^2 \cdot D_{\text{KL}}\left(p^{\text{teacher}}(T) \| p^{\text{student}}(T)\right)$$

$$= T^2 \cdot \sum_i p_i^{\text{teacher}}(T) \log\frac{p_i^{\text{teacher}}(T)}{p_i^{\text{student}}(T)}$$

$T^2$를 곱하는 것은 온도에 따라 그래디언트 크기가 바뀌는 것을 상쇄하려는 것이다.

### 그래디언트 살피기

학생의 로짓 $z_i^s$에 대한 소프트 로스의 기울기는

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i^s} = \frac{1}{T}\left(p_i^s(T) - p_i^t(T)\right)$$

온도가 높으면 이 그래디언트는 학생이 가장 큰 값만이 아니라 교사의 출력 분포 전체를 맞추도록 이끈다. 로스의 $T^2$ 스케일은 온도를 어떻게 고르든 그래디언트의 크기가 알맞게 남도록 한다.

---

## 4. PyTorch로 구현하기

### 기초 지식 증류

```python
class DistillationLoss(nn.Module):
    """
    굳은 타깃과 부드러운 타깃을 아우른 앎 옮기기 잃음.
    
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
        lr: 러닝 비율
        device: 익힐 장치
        
    Returns:
        익힌 제자 모형
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()  # 스승을 얼린다
    
    # 스승의 파라메터에 그래디언트가 붙지 않게 한다
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
            
            # 스승의 미루어 봄을 얻는다(그래디언트 없이)
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            # 제자의 앞으로 스텝
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

### 보기: CNN 지식 증류

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
    """작은 제자 모형(파라메터가 6배 적다)."""
    
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

# 파라메터 수를 견준다
def compare_models():
    teacher = TeacherCNN()
    student = StudentCNN()
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"스승 파라메터: {teacher_params:,}")
    print(f"제자 파라메터: {student_params:,}")
    print(f"눌러 담은 견줌: {teacher_params/student_params:.1f}배")
```

---

## 5. 심화 지식 증류 방법

### 특성 기반 지식 증류(FitNets)

출력 확률을 넘어 중간 표현까지 옮길 수 있으며, 이는 더 센 지도 신호가 된다.

```python
class FeatureDistillationLoss(nn.Module):
    """
    결에 기댄 앎 옮기기(FitNets / 어텐션 옮기기).
    
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
            teacher_channels: 스승 결 그림의 클래스 수
            student_channels: 제자 결 그림의 클래스 수
            temperature: 날임 앎 옮기기 온도
            alpha: 굳은 잃음의 짐
            beta: 결 잃음의 짐
            spatial_matching: 자리 어텐션 그림을 맞출지
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
            # 어텐션 옮기기: 자리 어텐션 그림을 맞춘다
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
        자리 어텐션 그림을 셈한다: 클래스에 걸친 살림 제곱의 합.
        
        Args:
            features: 결 그림 (B, C, H, W)
            
        Returns:
            잣대 맞춘 어텐션 그림 (B, H, W)
        """
        attention = (features ** 2).sum(dim=1)  # (B, H, W)
        attention = attention / (attention.sum(dim=(1, 2), keepdim=True) + 1e-8)
        return attention

def attention_transfer_loss(student_attention: torch.Tensor,
                           teacher_attention: torch.Tensor) -> torch.Tensor:
    """
    홀로 쓰는 어텐션 옮기기 잃음.
    
    자리 어텐션 그림(모형이 "보는" 자리)을 맞춘다.
    """
    # 어텐션 그림의 잣대를 맞춘다
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

### 여러 레이어 지식 증류

중간 레이어 여럿에서 옮기면 지도 신호가 더 풍부해진다.

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

### 자기 증류

학생과 교사가 같은 구조다. 모델이 자기 자신에게서 지식을 옮겨 받는다.

```python
class SelfDistillation(nn.Module):
    """
    스스로 앎 옮기기: 모형이 제 앞선 미루어 봄에서 배운다.
    
    클래스:
    1. 다시 난 네트워크: 익힌 뒤 같은 구조로 옮긴다
    2. 깊은 서로 러닝: 네트워크 둘이 서로 가르친다
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
        
        # 앞 판의 부드러운 타깃을 담아 둔다
        self.soft_targets = {}
    
    def compute_soft_targets(self,
                            data_loader: torch.utils.data.DataLoader,
                            device: str = 'cpu'):
        """
        익힘 보기 모두의 부드러운 타깃을 셈해 담는다.
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
        스스로 앎 옮기기의 익힘 한 스텝.
        """
        self.model.train()
        
        optimizer.zero_grad()
        logits = self.model(data)
        
        # 굳은 잃음
        hard_loss = F.cross_entropy(logits, labels)
        
        # 부드러운 잃음(담아 둔 타깃에서)
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

### 다중 교사 지식 증류

더 풍부한 지식을 얻으려고 교사 여럿을 앙상블로 쓴다.

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
        스승 모두의 부드러운 타깃을 모은다.
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

### 온라인 지식 증류(딥 상호 트레인)

교사와 학생을 한꺼번에 트레인한다.

```python
class OnlineDistillation(nn.Module):
    """
    깊은 서로 러닝: 익히는 동안 네트워크끼리 서로 가르친다.
    
    미리 익힌 스승이 있어야 하지 않다. 네트워크 여럿이 함께 배우며,
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
        서로 배우며 모형 모두의 앞으로 스텝.
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

---

## 6. 점진적 지식 증류

아주 크게 압축하려면 중간 크기의 모델을 거쳐 차근차근 옮긴다.

```
큰 스승 → 가운데 모형 → 작은 제자
```

각 단계는 가장 작은 모델로 곧바로 옮기는 것보다 쉽다.

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

---

## 7. 하이퍼파라미터 고르기

### 온도 고르기

| 온도 | 영향 | 잘 맞는 경우 |
|-------------|--------|----------|
| 1~2 | 날카로운 분포 | 이미 확신이 강한 교사 |
| 3~5 | 알맞은 부드러움 | 일반적인 경우(기본값: 4) |
| 10~20 | 아주 부드러움 | 클래스 간 유사도가 중요할 때, 구조 차이가 클 때 |
| 20 초과 | 거의 균등 | 대개 너무 부드럽다 |

### 알파 고르기

| 알파 | 의미 | 쓸 때 |
|-------|---------------|-------------|
| 0.0 | 소프트 로스만 | 확신이 강하고 정확한 교사 |
| 0.3~0.5 | 균형 | 일반적인 경우(기본값) |
| 0.7~0.9 | 거의 하드 로스 | 믿기 어려운 교사, 또는 미세 조정 |
| 1.0 | 하드 로스만 | 지식 증류 없음(베이스라인) |

### 어느 방법을 언제 쓸까

| 상황 | 권장 방법 |
|----------|---------------|
| 교사와 학생의 정확도 차이가 큼 | 온도를 높인다(8~20) |
| 구조가 비슷함 | 특성 증류를 쓴다 |
| 학생이 아주 작음 | 점진적 증류를 쓴다 |
| 트레인 데이터가 적음 | 지식 증류가 크게 도움이 된다 |
| 앙상블 교사 | 다중 교사 증류를 쓴다 |
| 사전 트레인된 교사가 없음 | 딥 상호 트레인을 쓴다 |

### 학생 구조 설계

학생은 배포할 수 있을 만큼 작되 교사의 지식을 담을 만큼 커야 한다.

- **너무 작으면**: 복잡한 결정 경계를 배우지 못한다
- **너무 크면**: 얻는 이득이 줄고 압축하는 의미가 사라진다

경험칙: 교사 파라메터의 10~30%에서 시작해 정확도 요건에 따라 조정한다.

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

---

## 8. 평가 지표

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

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
45장이 다루는 최적화 기법(양자화·가지치기·지식 증류·아키텍처 탐색)을 정확도 손실, 추론 속도 향상, 구현의 번거로움으로 비교하여 트레이드오프를 설명하여라.

</div>

??? success "풀이"
    기법마다 트레이드오프의 양상이 다르다. 양자화(INT8)는 흔히 2~4배 빨라지면서 정확도 손실이 1% 미만이고, 프레임워크가 지원하므로 구현 부담이 중간쯤이다. 가지치기는 희소성의 형태에 따라 속도 향상이 들쭉날쭉하며(구조적 가지치기가 하드웨어에 더 맞다) 정확도 손실은 1~3%이다. 지식 증류는 추론 방식 자체는 그대로 두되 더 작은 학생을 써서 2~10배로 압축하고 정확도 손실은 1~5%이다. 뉴럴 네트워크 아키텍처 탐색은 가장 좋은 구조를 찾아 주지만 탐색에 엄청난 계산이 든다(GPU 수천 시간). 금융 응용에서는 받아들일 수 있는 정확도 손실이 오류의 비용에 달려 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
단순한 순방향 네트워크에 트레인 후 양자화(INT8)를 구현하고, 표준 데이터셋에서 정확도가 얼마나 떨어지고 추론이 얼마나 빨라지는지 재어라.

</div>

??? success "풀이"
    PyTorch의 양자화 API를 쓴다. (1) float32 모델을 베이스라인 정확도까지 트레인한다. (2) 동적 양자화에는 `torch.quantization.quantize_dynamic`을 쓰고, 정적 양자화에는 대표 데이터로 캘리브레이션한다. (3) 추론 지연(배치 1000개의 평균)과 테스트 데이터의 정확도를 잰다. 흔한 결과: CPU에서 1.5~3배 빨라지고, 동적 양자화는 정확도가 0.5% 미만, 캘리브레이션한 정적 양자화는 0.2% 미만 떨어진다. 모델 크기는 약 4배 줄어든다(FP32에서 INT8로). 핵심: 정적 양자화에는 배포 환경의 데이터를 잘 대표하는 캘리브레이션 데이터가 있어야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
배포된 모델의 데이터 드리프트, 개념 드리프트, 성능 저하를 탐지하는 프로덕션 모니터링 시스템을 설계하여라. 지표와 알림 임계를 밝혀라.

</div>

??? success "풀이"
    세 층위를 모니터링한다. (1) 데이터 드리프트: KS 검정이나 PSI(모집단 안정성 지수)로 입력 피처의 분포를 추적한다. 어느 피처든 PSI > 0.2이면 알린다. (2) 개념 드리프트: 예측 분포의 이동과 (얻을 수 있으면) 실제 레이블 분포를 추적한다. 예측의 평균이 베이스라인 기간 대비 표준편차 2배 넘게 이동하면 알린다. (3) 모델 성능 저하: 슬라이딩 윈도로 실시간 정확도와 로스를 추적한다. 정확도가 베이스라인보다 3% 넘게 떨어지거나 지연이 SLA를 넘으면(p99 > 50ms 등) 알린다. Grafana로 대시보드를 만들고, Prometheus에 지표를 저장하고, PagerDuty로 알림을 보낸다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
금융 트레이딩 시스템의 지연 요건이 웹 서비스와 근본적으로 다른 까닭을 설명하여라. 이것이 배포 최적화 전략에 어떤 영향을 주는가?

</div>

??? success "풀이"
    웹 서비스는 100~500ms의 지연과 이따금의 지연 급증을 허용한다. 트레이딩 시스템은 결정적으로(deterministic) 1밀리초 아래(고빈도 거래에서는 흔히 100마이크로초 미만)여야 한다. 그래서 최적화 전략이 달라진다. (1) 가비지 컬렉션 멈춤을 없앤다(파이썬 대신 C++ 추론). (2) 메모리를 미리 다 할당해 둔다(동적으로 할당하지 않는다). (3) 스레드를 코어에 고정한다(컨텍스트 스위칭을 없앤다). (4) 지연이 가장 중요한 경로에는 FPGA나 ASIC을 쓴다. (5) 양자화는 필요하되 비결정적인 반올림을 들여서는 안 된다. 배치 추론은 쓸 수 없다(판단 하나하나가 지연에 민감하다). 배포 스택은 처리량보다 최악의 경우 지연(p99.9)을 우선한다. $\square$

## 정리하며

지식 증류는 효율적인 모델을 배포할 수 있게 한다.

1. **핵심 통찰**: 작은 학생이 큰 교사를 흉내 내도록 트레인한다
2. **소프트 타깃**: 온도로 클래스 사이의 상대 확률을 보존한다
3. **로스 함수**: 하드 레이블과 소프트 타깃을 아우른다
4. **심화 방법**: 특성 맞추기, 어텐션 전이, 자기 증류
5. **온도**: 구조 차이가 클수록 높인다

핵심 권장 사항:

- 온도 4~8, 알파 0.5에서 시작한다
- 구조가 비슷하면 특성 증류를 쓴다
- 학생이 교사의 특성을 맞추는지 평가한다
- 학생이 아주 작으면 점진적 증류를 고려한다
- 사전 트레인된 교사가 없으면 딥 상호 트레인을 쓴다

**살펴볼 거리**

1. Hinton, G., Vinyals, O., & Dean, J. "Distilling the Knowledge in a Neural Network." arXiv 2015.
2. Romero, A., et al. "FitNets: Hints for Thin Deep Nets." ICLR 2015.
3. Zagoruyko, S. & Komodakis, N. "Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer." ICLR 2017.
4. Zhang, Y., et al. "Deep Mutual Learning." CVPR 2018.
5. Furlanello, T., et al. "Born-Again Neural Networks." ICML 2018.
6. Gou, J., et al. "Knowledge Distillation: A Survey." IJCV 2021.
