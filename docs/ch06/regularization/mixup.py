"""mixup — mixup 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch06/regularization/mixup.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

import torch
import torch.nn as nn
import numpy as np

def mixup_data(x: torch.Tensor, y: torch.Tensor, 
               alpha: float = 0.2) -> tuple:
    """
    데이터 배치에 믹스업을 적용한다.
    
    인수:
        x: 입력 배치, 모양 (batch_size, ...)
        y: 레이블(클래스 인덱스), 모양 (batch_size,)
        alpha: 베타분포의 매개변수. 클수록 더 많이 섞인다.
        
    반환값:
        mixed_x: 섞인 입력
        y_a: 원래 레이블
        y_b: 순열을 적용한 레이블
        lam: 혼합 계수
    """
    # Beta(alpha, alpha)에서 섞는 비율을 뽑는다. 대칭이라 평균은 늘 0.5지만
    # 모양이 alpha에 달렸다. alpha가 작으면(0.2 따위) 0이나 1 가까이 몰려
    # "거의 원본"인 표본이 많아지고, alpha가 1이면 균등분포가 되어
    # 절반씩 섞인 표본이 많아진다
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0   # 믹스업을 끈 것과 같다

    batch_size = x.size(0)
    # 새 배치를 따로 불러오지 않고 같은 배치를 뒤섞어 제 자신과 짝짓는다.
    # 데이터 로더를 건드리지 않아도 되므로 어디에나 끼워 넣기 쉽다
    index = torch.randperm(batch_size, device=x.device)

    # 입력은 여기서 바로 섞는다
    mixed_x = lam * x + (1 - lam) * x[index]
    # 레이블은 섞지 않고 두 벌을 그대로 돌려준다. 분류에서는 y가
    # 클래스 번호(정수)라 lam*y_a + (1-lam)*y_b 가 뜻이 없기 때문이다.
    # 대신 아래 mixup_criterion에서 손실을 그 비율로 섞는다
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """
    믹스업 손실을 두 표준 손실의 가중 결합으로 계산한다.
    
    인수:
        criterion: 바탕이 되는 손실 함수 (예: CrossEntropyLoss)
        pred: 모델의 예측
        y_a: 첫째 레이블 집합
        y_b: 둘째 레이블 집합
        lam: 혼합 계수
    """
    # 손실을 섞는 것이 레이블을 섞는 것과 같아지는 까닭.
    # 교차 엔트로피는 레이블에 대해 선형이므로
    #   CE(pred, lam*y_a + (1-lam)*y_b) = lam*CE(pred, y_a) + (1-lam)*CE(pred, y_b)
    # 가 성립한다. 그래서 원-핫을 만들어 섞지 않고도 같은 결과를 얻는다.
    #
    # 주의: 이 등식은 교차 엔트로피처럼 레이블에 선형인 손실에서만
    # 성립한다. 초점 손실이나 다른 비선형 손실에는 그대로 쓸 수 없다
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_mixup(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 0.2,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    믹스업 증강으로 모델을 학습시킨다.
    
    인수:
        model: 신경망
        train_loader: 학습 데이터
        val_loader: 검증 데이터
        alpha: 믹스업 보간의 세기
        epochs: 학습 에포크 수
        lr: 학습률
        
    반환값:
        학습 이력
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 믹스업을 쓰는 학습
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 믹스업 적용.
            # 컷믹스의 학습 루프와 달리 확률로 켜고 끄지 않고 모든
            # 배치에 건다. 다만 베타분포에서 뽑은 lam이 0이나 1에
            # 가까운 배치는 사실상 원본이므로, 깨끗한 이미지도
            # 자연히 섞여 들어간다
            mixed_x, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha)
            
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            
            loss.backward()
            optimizer.step()
            # 이 손실은 섞인 이름표에 대한 값이라 아래 검증 손실과
            # 같은 자로 잰 값이 아니다. 두 곡선을 겹쳐 그리면 안 된다
            train_loss += loss.item()
        
        # 검증 (믹스업 없음).
        # 믹스업은 학습에만 건다. 시험에서 만날 것은 섞이지 않은
        # 이미지이므로, 평가는 그 조건에서 해야 뜻이 있다
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_correct/val_total:.4f}")
    
    return history

class ManifoldMixupModel(nn.Module):
    """
    무작위 은닉층에서 다양체 믹스업을 지원하는 모델.
    
    참고: Verma 등, "Manifold Mixup: Better Representations by
               은닉 상태 사이 메우기"(ICML 2019)
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        # 인덱싱을 위해 층을 목록으로 만든다
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ))
            prev_dim = hidden_dim
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x, mixup_layer=None, lam=None, index=None):
        """
        다양체 믹스업을 선택적으로 쓰는 순전파.
        
        인수:
            x: 입력 텐서
            mixup_layer: 믹스업을 적용할 층의 인덱스 (None이면 믹스업 없음)
            lam: 혼합 계수
            index: 배치에 대한 순열 인덱스
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 고른 층에서 믹스업 적용
            if mixup_layer is not None and i == mixup_layer:
                x = lam * x + (1 - lam) * x[index]
        
        return self.output(x)

def train_step_manifold_mixup(model, X_batch, y_batch, criterion, 
                               optimizer, alpha=0.2):
    """다양체 믹스업을 쓰는 학습 단계 하나."""
    optimizer.zero_grad()
    
    # 섞을 층을 무작위로 고르기
    n_layers = len(model.layers)
    mixup_layer = np.random.randint(0, n_layers + 1)  # +1은 입력 공간을 포함한다
    
    # 혼합 계수 뽑기
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    # 순열 만들기
    batch_size = X_batch.size(0)
    index = torch.randperm(batch_size, device=X_batch.device)
    
    if mixup_layer == 0:
        # 입력 공간에서의 믹스업
        mixed_x = lam * X_batch + (1 - lam) * X_batch[index]
        outputs = model(mixed_x)
    else:
        # 은닉층에서의 믹스업
        outputs = model(X_batch, mixup_layer=mixup_layer - 1, 
                       lam=lam, index=index)
    
    # 섞인 레이블
    loss = lam * criterion(outputs, y_batch) + (1 - lam) * criterion(outputs, y_batch[index])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

class BatchMixup:
    """
    여러 짝짓기 전략을 갖춘 유연한 믹스업.
    """
    
    def __init__(self, alpha=0.2, strategy='random'):
        """
        인수:
            alpha: 베타분포의 매개변수
            strategy: 짝짓기 전략 — 'random', 'cross_class', 'same_class'
        """
        self.alpha = alpha
        self.strategy = strategy
    
    def __call__(self, x, y):
        if self.alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        
        if self.strategy == 'random':
            index = torch.randperm(batch_size, device=x.device)
        
        elif self.strategy == 'cross_class':
            # 각 표본을 다른 클래스의 표본과 짝짓기
            index = self._cross_class_permutation(y)
        
        elif self.strategy == 'same_class':
            # 각 표본을 같은 클래스의 표본과 짝짓기.
            # 이 경우 y_a와 y_b가 같아 이름표는 섞이지 않는다. 즉
            # 정칙화가 아니라 같은 클래스 안에서 새 표본을 지어내는
            # 쪽에 가깝고, 결정 경계를 매끄럽게 하는 믹스업 본래의
            # 효과는 사라진다
            index = self._same_class_permutation(y)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # 컷믹스와 갈리는 지점이다. 컷믹스는 조각을 오려 붙여 어느
        # 화소든 두 이미지 가운데 하나에서 오지만, 믹스업은 화소마다
        # 두 이미지를 겹쳐 반투명하게 만든다. 그래서 lam은 넓이의
        # 비율이 아니라 밝기의 비율이며, 다시 계산할 일도 없다
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam
    
    def _cross_class_permutation(self, y):
        """서로 다른 클래스를 짝짓는 순열을 만든다."""
        batch_size = y.size(0)
        index = torch.randperm(batch_size, device=y.device)
        
        # 되도록 클래스를 가로질러 짝지으려 시도한다.
        # 같은 클래스끼리 섞으면 이름표가 그대로라 배울 것이 없으므로
        # 그런 짝을 풀어 준다. 배치가 한 클래스로만 채워졌거나 클래스가
        # 둘뿐이면 풀 상대가 없어 그대로 남는다. 보장이 아니라 발견법이다
        for i in range(batch_size):
            if y[i] == y[index[i]]:
                # 클래스가 다른 교환 상대 찾기.
                # 앞의 컷믹스판과 달리 i+1부터 훑는다. 이미 손본 앞쪽을
                # 다시 건드리지 않으므로 고쳐 놓은 짝이 도로 망가지지 않는다
                for j in range(i + 1, batch_size):
                    if y[i] != y[index[j]] and y[j] != y[index[i]]:
                        index[i], index[j] = index[j].clone(), index[i].clone()
                        break
        return index
    
    def _same_class_permutation(self, y):
        """같은 클래스를 짝짓는 순열을 만든다."""
        batch_size = y.size(0)
        index = torch.arange(batch_size, device=y.device)
        
        # 클래스 안에서 섞기
        for c in y.unique():
            class_mask = (y == c).nonzero(as_tuple=True)[0]
            if len(class_mask) > 1:
                perm = class_mask[torch.randperm(len(class_mask))]
                index[class_mask] = perm
        
        return index

def mixup_regression(x, y, alpha=0.2):
    """
    회귀 과제를 위한 믹스업.
    
    회귀의 목푯값은 이미 연속이므로 레이블 섞기는
    단순한 보간이 된다.
    """
    # 섞는 비율 lam을 Beta(alpha, alpha)에서 뽑는다.
    # alpha가 작으면(0.2 따위) 뽑히는 값이 0이나 1 가까이 몰려
    # "거의 원본"인 표본이 많아진다. alpha가 1이면 균등분포가 되어
    # 절반씩 섞인 표본이 많아진다. 곧 alpha가 섞기의 세기를 정한다
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0   # alpha=0이면 섞지 않는다(믹스업을 끈 것과 같다)

    # 배치를 뒤섞은 인덱스. 새 배치를 따로 만들지 않고 같은 배치를
    # 제 자신과 짝지어 섞는 것이 믹스업의 요령이다
    index = torch.randperm(x.size(0), device=x.device)

    # 입력과 목푯값을 같은 lam으로 섞는다. 두 곳에 같은 비율을 써야
    # "입력을 섞으면 답도 그만큼 섞인다"는 선형성 가정이 지켜진다
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    # 분류에서는 레이블이 원-핫이라 손실 쪽에서 lam으로 나누어 셈해야 하지만,
    # 회귀는 목푯값이 이미 연속이므로 이렇게 곧바로 섞으면 끝난다
    return mixed_x, mixed_y

def mixup_with_label_smoothing(model, x, y, alpha=0.2, epsilon=0.05):
    """
    믹스업을 가벼운 레이블 평활화와 결합한다.
    
    믹스업이 이미 레이블을 부드럽게 하므로 epsilon을 줄여 쓴다.
    """
    num_classes = 10  # 필요에 따라 조정하라
    
    # 믹스업
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    
    # 믹스업에서 온 부드러운 목표
    y_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    y_onehot_perm = torch.zeros(x.size(0), num_classes, device=x.device)
    y_onehot_perm.scatter_(1, y[index].unsqueeze(1), 1.0)
    
    soft_targets = lam * y_onehot + (1 - lam) * y_onehot_perm
    
    # 추가 레이블 평활화 적용
    soft_targets = (1 - epsilon) * soft_targets + epsilon / num_classes
    
    # 손실을 계산한다
    logits = model(mixed_x)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1).mean()
    
    return loss

def mixup_or_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, 
                     cutmix_prob=0.5):
    """배치마다 믹스업이나 컷믹스 중 하나를 무작위로 적용한다."""
    if np.random.random() < cutmix_prob:
        # 컷믹스 적용 (구현은 cutmix.md 참고)
        return cutmix_data(x, y, alpha=cutmix_alpha)
    else:
        return mixup_data(x, y, alpha=mixup_alpha)
