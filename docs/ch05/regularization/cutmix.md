# 컷믹스

컷믹스는 한 이미지에서 직사각형 조각을 잘라 다른 이미지에 붙여 학습 표본을 만들고, 조각의 넓이에 비례하여 레이블을 섞는 데이터 증강 겸 정칙화 기법이다. 컷아웃의 공간적 가림 효과와 믹스업의 레이블 섞기 효과를 결합하여, 섞인 목표로부터 배우면서도 모델이 물체의 전체 범위에 주의를 기울이도록 이끈다.

---

## 1. 수학적 정식화

### 핵심 연산

$x \in \mathbb{R}^{C \times H \times W}$인 두 학습 예 $(x_A, y_A)$과 $(x_B, y_B)$이 주어지면 컷믹스는 새 표본을 만든다.

$$
\tilde{x} = \mathbf{M} \odot x_A + (\mathbf{1} - \mathbf{M}) \odot x_B
$$

$$
\tilde{y} = \lambda \, y_A + (1 - \lambda) \, y_B
$$

여기서 $\mathbf{M} \in \{0, 1\}^{H \times W}$은 잘라 낼 영역을 나타내는 이진 마스크이고, $\odot$은 (채널 방향으로 방송되는) 원소별 곱이며, $\lambda$은 가려지지 않은 넓이의 비율로 정해진다.

$$
\lambda = 1 - \frac{(x_2 - x_1)(y_2 - y_1)}{W \cdot H}
$$

여기서 $(x_1, y_1, x_2, y_2)$은 직사각형 절단 영역의 좌표이다.

### 경계 상자 뽑기

절단 영역은 균등하게 뽑는다. 혼합 비율 $\lambda_0 \sim \text{Beta}(\alpha, \alpha)$이 주어지면 다음과 같이 한다.

1. 절단 비율을 계산한다. $r = \sqrt{1 - \lambda_0}$
2. 절단 크기를 정한다. $r_w = r \cdot W$, $r_h = r \cdot H$
3. 중심을 뽑는다. $c_x \sim \text{Uniform}(0, W)$, $c_y \sim \text{Uniform}(0, H)$
4. (이미지 경계로 잘라 낸) 경계 상자를 계산한다.

$$
x_1 = \max(0, \lfloor c_x - r_w/2 \rfloor), \quad x_2 = \min(W, \lfloor c_x + r_w/2 \rfloor)
$$

$$
y_1 = \max(0, \lfloor c_y - r_h/2 \rfloor), \quad y_2 = \min(H, \lfloor c_y + r_h/2 \rfloor)
$$

5. (잘라 낸 뒤의) 실제 상자 넓이로부터 $\lambda$을 다시 계산한다.

$$
\lambda = 1 - \frac{(x_2 - x_1)(y_2 - y_1)}{W \cdot H}
$$

경계에서의 잘라 냄이 실제 넓이 비율을 바꾸므로 다시 계산하는 것이 중요하다.

### 관련 방법들과의 비교

| 방법 | 입력의 변형 | 레이블의 변형 |
|--------|-------------------|-------------------|
| 컷아웃 | 영역을 0으로 가림 | 바뀌지 않음 |
| 믹스업 | 전역적인 선형 혼합 | 선형 보간 |
| 컷믹스 | 다른 이미지의 조각을 붙임 | 넓이에 비례 |

컷믹스는 앞선 두 기법의 한계를 다룬다. 컷아웃은 화소를 정보 없는 0으로 바꿔 낭비한다. 믹스업은 자연스러운 시각 입력과 거리가 먼, 전역적으로 뒤섞인 이미지를 만든다. 컷믹스는 모든 화소가 정보를 지니게 하면서도 국소적으로 그럴듯한 이미지를 만든다.

---

## 2. 컷믹스가 통하는 이유

### 학습 화소의 온전한 활용

조각을 0(또는 상수 채움값)으로 바꾸는 컷아웃과 달리 컷믹스는 조각을 다른 학습 이미지의 내용으로 바꾼다. 즉 증강된 이미지의 모든 화소가 의미 정보를 지니게 되어 학습 데이터를 더 효율적으로 쓴다.

### 위치 파악의 촉진

레이블이 각 원본 이미지의 넓이에 비례하므로 모델은 부분적인 모습만 보고도 물체를 알아보도록 배워야 한다. 이는 가장 두드러진 조각 하나에 기대는 대신 여러 변별적 영역에 주의를 기울이게 한다.

### 부분 정보를 통한 정칙화

원래 내용의 일부만 보이는 이미지를 보여 줌으로써 컷믹스는 모델이 어느 한 공간 영역에 의존하지 못하게 하고 더 전체적인 특징 학습을 이끈다.

---

## 3. PyTorch 구현

### 기본적인 컷믹스

```python
import torch
import torch.nn as nn
import numpy as np

def rand_bbox(size, lam):
    """
    컷믹스를 위한 무작위 경계 상자를 만든다.
    
    인수:
        size: (batch_size, 채널, 높이, 너비)
        lam: 베타분포에서 뽑은 혼합 비율
        
    반환값:
        경계 상자의 좌표 (x1, y1, x2, y2)
    """
    # size에서 H, W만 꺼내 쓴다. 즉 상자 하나를 뽑아 배치의 모든 이미지에
    # 똑같은 자리로 적용한다. 표본마다 다른 자리를 오리려면 아래
    # saliency_cutmix처럼 반복문을 돌아야 한다
    H, W = size[2], size[3]
    # 제곱근이 붙는 까닭은 넓이와 길이의 관계 때문이다. 오려 낼 넓이의
    # 비율이 1-lam이어야 하는데 상자는 가로세로를 함께 줄이므로,
    # 각 변을 r배 하면 넓이는 r^2배가 된다. 따라서 r = sqrt(1-lam)이다
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 균등 무작위 중심
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 이미지 경계로 자르기.
    # 중심이 가장자리에 가까우면 상자가 밖으로 삐져나가 잘린다. 그래서
    # 실제 오려진 넓이가 뜻한 것보다 작아질 수 있고, 아래에서 lam을
    # 다시 계산하는 까닭이 바로 이것이다
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2

def cutmix_data(x: torch.Tensor, y: torch.Tensor,
                alpha: float = 1.0) -> tuple:
    """
    이미지 배치에 컷믹스를 적용한다.
    
    인수:
        x: 입력 배치, 모양 (batch_size, C, H, W)
        y: 레이블(클래스 인덱스), 모양 (batch_size,)
        alpha: 베타분포의 매개변수
        
    반환값:
        x_cutmix: 증강된 이미지
        y_a: 원래 레이블
        y_b: 순열을 적용한 레이블
        lam: 조정된 혼합 계수 (잘라 낸 뒤)
    """
    # alpha=1이면 베타분포가 [0, 1] 균등분포가 되어 조각 크기가 고르게
    # 뽑힌다. alpha를 키우면 0.5 근처에 몰려 반반씩 섞이고, 0에 가깝게
    # 하면 양 끝에 몰려 거의 원본이거나 거의 통째로 바뀐다
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    # 섞을 짝을 같은 배치 안에서 찾는다. 데이터를 더 읽지 않아도 되니
    # 공짜에 가깝다. 다만 배치가 작으면 짝의 다양성도 줄어든다.
    # 순열이라 자기 자신과 짝지어질 수도 있는데, 그때는 원본을 원본에
    # 붙이는 셈이라 아무 일도 일어나지 않는다
    index = torch.randperm(batch_size, device=x.device)

    # 경계 상자 생성
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)

    # 자르고 붙이기.
    # clone이 없으면 x를 제자리에서 고치게 되는데, 오른쪽에서 조각을
    # 읽어 오는 대상도 같은 x다. 덮어쓰기와 읽기가 섞여 결과가 망가진다
    x_cutmix = x.clone()
    x_cutmix[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # 실제 상자 넓이에 맞추어 lambda 다시 계산.
    # 이 줄이 컷믹스의 정확성을 지킨다. 가장자리에서 상자가 잘렸으면
    # 뽑아 둔 lam과 실제 넓이가 어긋나므로, 이름표를 섞을 비율은
    # 뽑은 값이 아니라 정말로 오려진 넓이에서 나와야 한다
    _, _, H, W = x.shape
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)

    # 이름표는 섞지 않고 둘 다 그대로 돌려준다. 섞는 일은 아래
    # cutmix_criterion이 손실 쪽에서 맡는다
    y_a, y_b = y, y[index]
    return x_cutmix, y_a, y_b, lam

def cutmix_criterion(criterion: nn.Module, pred: torch.Tensor,
                     y_a: torch.Tensor, y_b: torch.Tensor,
                     lam: float) -> torch.Tensor:
    """컷믹스 손실을 가중 결합으로 계산한다."""
    # 섞인 이름표를 만들어 넣는 대신, 딱딱한 이름표 둘로 손실을 두 번
    # 재어 가중 평균한다. 교차 엔트로피가 목표 분포에 대해 선형이라
    # 두 방식의 값이 같기 때문이다. 이렇게 하면 원-핫 벡터를 만들지 않고
    # 클래스 인덱스를 그대로 쓸 수 있어 편하다.
    # 주의: lam은 스칼라여야 한다. 표본마다 다른 lam을 (B,) 텐서로 넘기면
    # 결과가 스칼라가 아니게 되어 backward()에서 오류가 난다
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

### 완전한 학습 루프

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_cutmix(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    alpha: float = 1.0,
    cutmix_prob: float = 0.5,
    epochs: int = 100,
    lr: float = 0.001
) -> dict:
    """
    컷믹스 증강으로 모델을 학습시킨다.
    
    인수:
        model: CNN 모델
        train_loader: 학습 데이터
        val_loader: 검증 데이터
        alpha: 컷믹스를 위한 베타분포의 매개변수
        cutmix_prob: 배치마다 컷믹스를 적용할 확률
        epochs: 학습 에포크 수
        lr: 학습률
        
    반환값:
        학습 이력
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 확률 cutmix_prob으로 컷믹스 적용.
            # 배치 단위로 켜고 끈다. 한 배치는 전부 섞이거나 전부
            # 안 섞이거나 둘 중 하나다. 섞이지 않은 배치를 남겨 두는
            # 까닭은, 모델이 실제 시험에서 만날 깨끗한 이미지도
            # 계속 보아야 하기 때문이다
            if np.random.random() < cutmix_prob:
                X_mixed, y_a, y_b, lam = cutmix_data(X_batch, y_batch, alpha)
                outputs = model(X_mixed)
                loss = cutmix_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 스케줄러는 에포크마다 한 번 부른다. 배치마다 부르면 학습률이
        # 예정보다 훨씬 빨리 0으로 떨어진다
        scheduler.step()
        
        # 검증 (컷믹스 없음).
        # 증강은 학습에만 건다. 그래서 train_loss와 val_loss는 성격이
        # 달라, 섞인 이미지의 손실과 깨끗한 이미지의 손실을 견주는 셈이다.
        # 컷믹스가 잘 들으면 학습 손실이 검증 손실보다 높게 나올 수도
        # 있는데, 이는 이상이 아니다
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
```

---

## 4. 컷믹스의 변형

### 클래스를 고려한 컷믹스

합성 결과의 다양성을 최대로 하려고 서로 다른 클래스의 이미지에서 조각을 고른다.

```python
def class_aware_cutmix(x, y, alpha=1.0):
    """
    서로 다른 클래스의 표본을 우선하여 짝짓는 컷믹스.
    
    이렇게 하면 모델이 정말로 다른 범주에 속하는 영역들을
    구별해야만 한다.
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)
    
    # 클래스를 가로지르는 순열 만들기
    index = torch.randperm(batch_size, device=x.device)
    
    # 되도록 다른 클래스가 되도록 시도한다.
    # 같은 클래스끼리 짝지어지면 섞어도 배울 것이 없다. 고양이에 고양이를
    # 붙여 놓고 "고양이 반 고양이 반"이라 가르치는 꼴이기 때문이다.
    # 이중 반복문이라 배치 크기의 제곱에 비례해 느리고, 어디까지나
    # 최선을 다하는 발견법이라 모든 짝이 갈라진다는 보장은 없다
    for i in range(batch_size):
        if y[i] == y[index[i]]:
            for j in range(batch_size):
                if y[i] != y[index[j]] and y[j] != y[index[i]]:
                    # clone이 필요한 까닭은 오른쪽이 먼저 다 계산된 뒤
                    # 대입되는 것이 아니라, 같은 텐서의 원소를 서로
                    # 가리키고 있어 한쪽을 덮어쓰면 다른 쪽도 바뀌기
                    # 때문이다
                    index[i], index[j] = index[j].clone(), index[i].clone()
                    break
    
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    
    x_cutmix = x.clone()
    x_cutmix[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    _, _, H, W = x.shape
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)
    
    return x_cutmix, y, y[index], lam
```

### 여러 이미지를 쓰는 컷믹스

두 개보다 많은 이미지의 조각을 섞는다.

```python
def multi_cutmix(x, y, n_patches=4, alpha=1.0):
    """
    격자 배치로 여러 이미지의 조각을 쓰는 컷믹스.
    
    인수:
        x: 입력 배치 (B, C, H, W)
        y: 레이블 (B,)
        n_patches: 격자 칸의 수 (완전제곱수여야 한다)
        alpha: 혼합 매개변수
        
    반환값:
        증강된 배치와 부드러운 레이블 벡터
    """
    B, C, H, W = x.shape
    # 배치에 나타난 가장 큰 레이블로 갈래 수를 짐작한다. 배치에 없는
    # 갈래가 있으면 이 값이 실제보다 작아지므로, 실무에서는 인자로 받는 편이 낫다
    num_classes = y.max().item() + 1
    grid_size = int(np.sqrt(n_patches))   # 4개면 2x2 격자

    patch_h = H // grid_size
    patch_w = W // grid_size

    x_mixed = x.clone()   # 원본을 건드리지 않도록 사본에 덮어쓴다
    soft_labels = torch.zeros(B, num_classes, device=x.device)

    total_area = H * W

    for gi in range(grid_size):
        for gj in range(grid_size):
            # 칸마다 새로 뒤섞는다. 칸이 저마다 다른 이미지에서 오므로
            # 한 장에 여러 갈래가 섞이고, 그래서 "다중" 컷믹스다
            index = torch.randperm(B, device=x.device)

            h_start = gi * patch_h
            # 마지막 칸은 H까지 늘린다. H가 grid_size로 나누어떨어지지
            # 않을 때 남는 줄을 버리지 않기 위해서다. 그래서 마지막 칸이
            # 조금 더 클 수 있고, 아래에서 넓이를 그때그때 다시 잰다
            h_end = (gi + 1) * patch_h if gi < grid_size - 1 else H
            w_start = gj * patch_w
            w_end = (gj + 1) * patch_w if gj < grid_size - 1 else W

            # 그 칸만 다른 이미지의 같은 자리 값으로 갈아 끼운다
            x_mixed[:, :, h_start:h_end, w_start:w_end] = \
                x[index, :, h_start:h_end, w_start:w_end]

            # 레이블은 넓이 몫만큼 섞는다. 이것이 컷믹스의 핵심 규약으로,
            # "화면의 몇 할을 차지하는가"가 곧 "정답에 몇 할을 기여하는가"다
            patch_area = (h_end - h_start) * (w_end - w_start)
            weight = patch_area / total_area

            # 부드러운 레이블 누적.
            # scatter_(1, 자리, 1.0)이 원-핫을 만든다. y[index]를
            # unsqueeze(1)로 (B,1) 꼴로 만들어야 열 방향 자리로 읽힌다
            patch_labels = torch.zeros(B, num_classes, device=x.device)
            patch_labels.scatter_(1, y[index].unsqueeze(1), 1.0)
            soft_labels += weight * patch_labels

    # 넓이 몫을 모두 더하면 1이므로 soft_labels의 각 행도 합이 1이 된다.
    # 곧 확률분포이며, 손실은 이 부드러운 목표에 대한 교차 엔트로피로 셈한다
    return x_mixed, soft_labels
```

### 현저도로 이끄는 컷믹스

기울기 기반 현저도를 써서 가장 중요한 곳에 절단 영역을 놓는다.

```python
def saliency_cutmix(model, x, y, alpha=1.0):
    """
    조각이 정보가 있는 영역과 겹치도록 현저도 지도로
    이끄는 컷믹스.
    
    참고: Uddin 등, "SaliencyMix" (2020)
    """
    model.eval()
    # 가중치가 아니라 입력에 대해 미분한다. 그래서 requires_grad를
    # 파라미터가 아닌 입력 텐서에 켠다
    x_sal = x.clone().requires_grad_(True)
    
    # 현저도 계산.
    # 손실을 입력으로 미분한 크기가 곧 "이 화소를 조금 바꾸면 판단이
    # 얼마나 흔들리는가"이다. 값이 큰 자리가 모델이 보고 있는 자리다
    outputs = model(x_sal)
    loss = nn.CrossEntropyLoss()(outputs, y)
    # 주의: 이 backward는 모델 파라미터에도 기울기를 쌓는다. 학습 루프
    # 안에서 optimizer.zero_grad() 뒤에 이 함수를 부르면 그 기울기가
    # 남아 진짜 학습 기울기에 더해진다. 이 함수를 부른 뒤에
    # zero_grad()를 다시 불러 주어야 안전하다
    loss.backward()
    
    # 채널 방향으로 평균 내어 색을 지운다. 어느 색 채널이 반응했는지가
    # 아니라 어느 위치가 중요한지만 남기려는 것이다
    saliency = x_sal.grad.abs().mean(dim=1)  # (B, H, W)
    
    model.train()
    
    # lambda 뽑기
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    index = torch.randperm(B, device=x.device)
    
    # 출처 이미지에서 가장 두드러진 위치에 상자를 놓는다
    x1_list, y1_list, x2_list, y2_list = [], [], [], []
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = max(1, int(W * cut_rat))
    cut_h = max(1, int(H * cut_rat))
    
    # 앞의 rand_bbox와 달리 표본마다 상자를 따로 잡는다. 이미지마다
    # 두드러진 자리가 다르기 때문이다
    for b in range(B):
        # 붙일 쪽이 아니라 가져올 쪽(index[b])의 현저도를 본다.
        # 알맹이가 든 조각을 오려 와야 이름표를 섞는 뜻이 산다.
        # 무작위 상자는 배경만 오려 와 이름표만 흐려 놓을 때가 있다
        sal = saliency[index[b]]
        # 가장 두드러진 중심점 찾기.
        # argmax는 평탄화된 위치를 주므로 W로 나누고 남겨 (행, 열)로
        # 되돌린다
        flat_idx = sal.argmax().item()
        cy, cx = flat_idx // W, flat_idx % W
        
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        x1_list.append(x1); y1_list.append(y1)
        x2_list.append(x2); y2_list.append(y2)
    
    # 표본마다 컷믹스 적용
    x_cutmix = x.clone()
    # 상자가 표본마다 다르니 lam도 표본마다 다르다.
    # 주의: 그래서 이 함수가 돌려주는 lam은 스칼라가 아니라 (B,) 텐서이며,
    # 위의 cutmix_criterion에 그대로 넘길 수 없다. 표본별 가중치를 쓰려면
    # reduction='none'으로 손실을 표본마다 받아 lam을 곱한 뒤
    # 평균 내는 손실 함수가 따로 필요하다
    lam_actual = torch.zeros(B, device=x.device)
    
    for b in range(B):
        x_cutmix[b, :, y1_list[b]:y2_list[b], x1_list[b]:x2_list[b]] = \
            x[index[b], :, y1_list[b]:y2_list[b], x1_list[b]:x2_list[b]]
        area = (x2_list[b] - x1_list[b]) * (y2_list[b] - y1_list[b])
        lam_actual[b] = 1 - area / (H * W)
    
    return x_cutmix, y, y[index], lam_actual
```

---

## 5. 다른 증강과 결합하기

### 컷믹스 + 표준 증강 파이프라인

```python
import torchvision.transforms as T

def get_cutmix_training_pipeline(image_size=32):
    """
    표준 증강 파이프라인. 컷믹스는 변환 파이프라인이 아니라
    배치 수준(학습 루프 안)에서 적용한다.
    """
    # 훈련용: 이미지 하나하나에 무작위 변형을 건다.
    # 컷믹스가 여기 없는 까닭은, 컷믹스가 이미지 두 장을 섞는 연산이라
    # 이미지 하나만 받는 transform 자리에 들어갈 수 없기 때문이다.
    # 배치가 만들어진 뒤 학습 루프 안에서 걸어야 한다
    train_transform = T.Compose([
        T.RandomCrop(image_size, padding=4),   # 위치를 조금씩 옮긴다
        T.RandomHorizontalFlip(),              # 좌우를 뒤집는다
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # 검증용에는 무작위 변형을 걸지 않는다. 평가는 되풀이해도 같은 값이
    # 나와야 하고, 증강은 훈련 자료를 늘리려는 장치이지 평가의 일부가 아니다.
    # 정규화만 훈련과 똑같이 걸어 두 분포를 맞춘다
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    return train_transform, val_transform
```

### 컷믹스 또는 믹스업 (무작위 선택)

배치마다 컷믹스와 믹스업 중 하나를 무작위로 고르는 것이 흔한 관행이다.

```python
def cutmix_or_mixup(x, y, cutmix_alpha=1.0, mixup_alpha=0.2, 
                     cutmix_prob=0.5):
    """
    배치마다 컷믹스나 믹스업을 무작위로 적용한다.
    
    요즘의 여러 학습 요령(예: DeiT, EfficientNetV2)에서
    쓰는 전략이다.
    """
    if np.random.random() < cutmix_prob:
        return cutmix_data(x, y, alpha=cutmix_alpha)
    else:
        from mixup import mixup_data  # mixup.md 참고
        return mixup_data(x, y, alpha=mixup_alpha)
```

---

## 6. 실무 지침

### 초매개변수 선택

| 매개변수 | 권장값 | 비고 |
|-----------|-------------|-------|
| $\alpha$ | 1.0 | 컷믹스의 표준값. $\text{Beta}(1,1)$은 Uniform[0,1] |
| `cutmix_prob` | 0.5 – 1.0 | 믹스업과 함께 쓸 때는 0.5 |
| 이미지 크기 | 아무거나 | 여러 해상도에서 잘 통한다 |

### 컷믹스를 쓸 때

1. **이미지 분류**: CNN과 비전 트랜스포머를 위한 강력한 정칙화 장치이다
2. **물체 검출**: 공간 구조가 보존되므로 믹스업보다 낫다
3. **학습 데이터가 적을 때**: 데이터가 부족할 때 크게 나아진다
4. **세밀한 인식**: 여러 변별적 부위에 주의를 기울이게 한다

### 컷믹스를 피할 때

1. **이미지가 아닌 데이터**: 표 형식, 텍스트, 1차원 신호에는 공간적 절단이 뜻을 갖지 않는다
2. **화소 단위 과제**: 정확한 공간 레이블이 필요한 경우(분할에서는 조심해서 쓰라)
3. **아주 작은 이미지**: 이미지가 이미 작으면 절단 영역이 너무 작아 효과가 없을 수 있다

### 평가

컷믹스는 **평가 시점에 결코 적용하지 않는다**. 검증과 시험에는 손대지 않은 깨끗한 이미지를 쓰라.

---

## 연습문제

**연습문제 1.**
컷믹스 알고리즘과 혼합 비율 $\lambda$이 잘라 낼 넓이를 정하는 방식을 설명하라.

??? success "연습문제 1 풀이"
    컷믹스는 이미지 B에서 직사각형 영역을 잘라 이미지 A에 붙인다. $\lambda \sim \text{Beta}(\alpha, \alpha)$일 때 넓이 비율은 $1 - \lambda$이다. 상자의 좌표는 $r_x, r_y \sim \text{Uniform}(0, W), (0, H)$이고 너비는 $r_w = W\sqrt{1-\lambda}$, 높이는 $r_h = H\sqrt{1-\lambda}$이다.

---

**연습문제 2.**
컷믹스를 PyTorch로 구현하라.

??? success "연습문제 2 풀이"
    ```python
    def cutmix(x, y, alpha=1.0):
        lam = torch.distributions.Beta(alpha, alpha).sample()
        idx = torch.randperm(x.size(0))
        W, H = x.size(2), x.size(3)
        cut_w = int(W * (1 - lam)**0.5)
        cut_h = int(H * (1 - lam)**0.5)
        cx, cy = torch.randint(W, (1,)), torch.randint(H, (1,))
        x1 = torch.clamp(cx - cut_w//2, 0, W)
        x2 = torch.clamp(cx + cut_w//2, 0, W)
        y1 = torch.clamp(cy - cut_h//2, 0, H)
        y2 = torch.clamp(cy + cut_h//2, 0, H)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam_adj = 1 - (x2-x1)*(y2-y1)/(W*H)
        return x, y, y[idx], lam_adj
    ```

---

**연습문제 3.**
학습에서 대체로 컷아웃보다 컷믹스를 선호하는 이유는 무엇인가?

??? success "연습문제 3 풀이"
    컷아웃은 영역을 지우고 0으로 채워 화소를 낭비하고 정보량을 줄인다. 컷믹스는 그 영역을 다른 이미지의 내용으로 바꾸므로 모든 화소가 정보를 지닌다. 또한 컷믹스는 각 이미지에서 보이는 넓이에 비례하는 부드러운 레이블을 제공한다.

---

**연습문제 4.**
컷믹스가 분류 신경망의 위치 파악 능력을 높이는 방식을 설명하라.

??? success "연습문제 4 풀이"
    일부가 다른 클래스로 바뀐 이미지를 분류하게 함으로써 컷믹스는 신경망이 (가장 변별적인 부위만이 아니라) 물체의 모든 부위에 주의를 기울이도록 학습시킨다. 이는 물체의 위치 파악을 개선하고 가림에 대한 견고성을 높인다.

## 정리하며

이 마당은 수학적 정식화、컷믹스가 통하는 이유、PyTorch 구현、컷믹스의 변형을 차례로 짚었다.

**참고 문헌**

1. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*.
2. Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
3. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of CNNs with Cutout. *arXiv*.
4. Uddin, A. F. M. S., et al. (2020). SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization. *ICLR Workshop*.
