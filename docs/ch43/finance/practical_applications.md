# 모형 아리송함을 손에 잡히게 쓰기

아리송함 재기는 미루어 봄을 점 어림에서 낌새 분포로 바꾸어 여러 손에 잡히는 쓰임을 이룬다. 이 마디는 서비스 얼개에서 아리송함이 가장 크게 힘을 쓰는 자리를 다룬다.

---

## 1. 아리송함을 쓰는 살아 있는 배움

### 문제

이름표를 다는 일은 비싸다. 살아 있는 배움은 아리송함을 써서 가장 알려 주는 바가 큰 보기를 골라 이름표를 달게 하므로 다는 값을 50~70% 줄인다.

### 얻기 함수

**가장 큰 엔트로피**:

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} H(y|\mathbf{x}, \mathcal{D}) = -\sum_{k} p(y=k|\mathbf{x}) \log p(y=k|\mathbf{x})$$

**가장 큰 흩어짐**:

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} \text{Var}[p(y|\mathbf{x}, \mathcal{D})]$$

**BALD**(어긋남으로 하는 베이즈 살아 있는 배움):

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} I(y; \mathbf{w}|\mathbf{x}, \mathcal{D}) = H(y|\mathbf{x}) - \mathbb{E}_{\mathbf{w}}[H(y|\mathbf{x}, \mathbf{w})]$$

### PyTorch로 짜기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Tuple, Optional

class ActiveLearner:
    """
    아리송함으로 보기를 고르는 살아 있는 배움 얼개.
    
    흐름:
    1. 이름표가 달린 작은 꾸러미로 모형을 익힌다
    2. 이름표 없는 못에서 아리송함을 곁들여 미루어 본다
    3. 가장 아리송한 보기를 골라 이름표를 달게 한다
    4. 익힘 꾸러미에 더하고 다시 익힌다
    5. 예산이 다할 때까지 되풀이한다
    """
    
    def __init__(self, model: nn.Module, acquisition: str = 'entropy'):
        """
        Args:
            model: 아리송함을 어림할 수 있는 모형
            acquisition: 'entropy', 'variance', 'bald', 'random' 가운데 하나
        """
        self.model = model
        self.acquisition = acquisition
        self.labeled_indices = []
        self.query_history = []
    
    def compute_acquisition_scores(self, probs: torch.Tensor, 
                                   all_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        보기를 고르는 얻기 점수를 셈한다.
        
        Args:
            probs: 평균 미루어 본 낌새 (N, K)
            all_probs: BALD에 쓸 모둠/MC 미루어 봄 모두 (M, N, K)
        
        Returns:
            보기마다의 점수(클수록 알려 주는 바가 크다)
        """
        epsilon = 1e-10
        
        if self.acquisition == 'entropy':
            # 미루어 보는 분포의 엔트로피
            scores = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        
        elif self.acquisition == 'variance':
            # 흩어짐(가장 큰 낌새의 아리송함)
            scores = 1 - probs.max(dim=-1)[0]
        
        elif self.acquisition == 'bald':
            if all_probs is None:
                raise ValueError("BALD에는 모둠 미루어 봄이 있어야 한다")
            
            # BALD = H(y|x,D) - E_w[H(y|x,w)]
            # 온 엔트로피
            mean_probs = all_probs.mean(dim=0)
            total_entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=-1)
            
            # 바라는 엔트로피
            individual_entropy = -torch.sum(all_probs * torch.log(all_probs + epsilon), dim=-1)
            expected_entropy = individual_entropy.mean(dim=0)
            
            scores = total_entropy - expected_entropy
        
        elif self.acquisition == 'random':
            scores = torch.rand(probs.shape[0])
        
        else:
            raise ValueError(f"모르는 얻기 함수: {self.acquisition}")
        
        return scores
    
    def select_samples(self, pool_loader: DataLoader, 
                       n_samples: int, 
                       n_mc_samples: int = 50) -> np.ndarray:
        """
        못에서 가장 알려 주는 바가 큰 보기를 고른다.
        
        Args:
            pool_loader: 이름표 없는 못의 DataLoader
            n_samples: 고를 보기의 수
            n_mc_samples: 아리송함을 어림할 MC 표본 수
        
        Returns:
            고른 보기의 손가락질
        """
        self.model.eval()
        
        all_scores = []
        all_probs_list = []
        
        with torch.no_grad():
            for x, _ in pool_loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                
                # 모형이 받쳐 주면 MC 미루어 봄을 얻는다
                if hasattr(self.model, 'mc_predict'):
                    mean_probs, _, mc_probs = self.model.mc_predict(
                        x, n_samples=n_mc_samples, return_samples=True
                    )
                    all_probs_list.append(mc_probs)
                else:
                    logits = self.model(x)
                    mean_probs = F.softmax(logits, dim=-1)
                    mc_probs = None
                
                scores = self.compute_acquisition_scores(mean_probs, mc_probs)
                all_scores.append(scores)
        
        all_scores = torch.cat(all_scores)
        
        # 위에서 k개를 고른다
        _, top_indices = torch.topk(all_scores, k=min(n_samples, len(all_scores)))
        selected = top_indices.cpu().numpy()
        
        self.query_history.append(selected.tolist())
        
        return selected

def run_active_learning_experiment(model_class, 
                                    train_dataset,
                                    test_dataset,
                                    n_initial: int = 100,
                                    n_queries: int = 5,
                                    query_size: int = 50,
                                    acquisition: str = 'entropy') -> dict:
    """
    살아 있는 배움 해 봄을 온전히 돌린다.
    
    이름표 수에 따른 맞음의 배움 굽이를 돌려준다.
    """
    # 처음에 아무렇게나 고른 이름표 꾸러미
    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)
    
    labeled_indices = all_indices[:n_initial].tolist()
    pool_indices = all_indices[n_initial:n_initial + 1000].tolist()  # 못의 크기를 마디 짓는다
    
    results = {
        'n_labeled': [],
        'accuracy': [],
        'acquisition': acquisition
    }
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    for query_round in range(n_queries + 1):
        print(f"\n{query_round}바퀴: 이름표 달린 보기 {len(labeled_indices)}개")
        
        # 모형을 만들고 익힌다
        model = model_class()
        labeled_subset = Subset(train_dataset, labeled_indices)
        train_loader = DataLoader(labeled_subset, batch_size=64, shuffle=True)
        
        # 익힘
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(5):
            for x, y in train_loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                
                loss = criterion(model(x), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 따지기
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += len(y)
        
        accuracy = correct / total
        results['n_labeled'].append(len(labeled_indices))
        results['accuracy'].append(accuracy)
        
        print(f"  시험 맞음: {accuracy:.4f}")
        
        # 새 보기를 물어 온다(마지막 바퀴는 뺀다)
        if query_round < n_queries and pool_indices:
            # 못 실개를 만든다
            pool_subset = Subset(train_dataset, pool_indices)
            pool_loader = DataLoader(pool_subset, batch_size=256, shuffle=False)
            
            # 보기를 고른다
            learner = ActiveLearner(model, acquisition=acquisition)
            selected_pool_indices = learner.select_samples(pool_loader, query_size)
            
            # 본디 손가락질로 되돌리고 고친다
            selected_original = [pool_indices[i] for i in selected_pool_indices]
            labeled_indices.extend(selected_original)
            pool_indices = [i for i in pool_indices if i not in selected_original]
    
    return results

def compare_acquisition_functions():
    """
    여러 얻기 함수를 견준다.
    """
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 10)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    results = {}
    
    for acquisition in ['random', 'entropy', 'variance']:
        print(f"\n{'='*50}")
        print(f"얻기 함수: {acquisition}")
        print('='*50)
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        results[acquisition] = run_active_learning_experiment(
            SimpleModel, train_dataset, test_dataset,
            n_initial=50, n_queries=5, query_size=50,
            acquisition=acquisition
        )
    
    return results
```

---

## 2. 골라 미루어 보기(물릴 수 있는 길)

### 깨침

들임 모두에 대해 미루어 보는 대신, 아리송함이 크면 미루어 봄을 물린다.

- **받음**: 아리송함 < 문턱 → 미루어 본다
- **물림**: 아리송함 ≥ 문턱 → 삼간다(사람이 살피게 한다)

### 무릅씀과 덮음의 맞바꿈

$$\text{Coverage} = \frac{\text{# predictions made}}{\text{# total samples}}$$

$$\text{Selective Risk} = \frac{\text{# wrong predictions}}{\text{# predictions made}}$$

**목표**: 받아들일 만한 무릅씀을 지키면서 덮음을 가장 크게 한다.

### 짜기

```python
class SelectivePredictor:
    """
    아리송함 문턱을 쓰는 골라 미루어 보기.
    
    맞바꿈: 문턱이 높을수록 맞음은 오르나 덮음은 준다
    """
    
    def __init__(self, model: nn.Module, threshold: float = 0.5):
        """
        Args:
            model: 아리송함을 어림하는 모형
            threshold: 물리는 아리송함 문턱
        """
        self.model = model
        self.threshold = threshold
    
    def predict(self, x: torch.Tensor, 
                n_mc_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        물릴 수 있는 길을 갖추어 미루어 본다.
        
        Returns:
            predictions: 미루어 본 이름표
            accept_mask: 참거짓 가리개(True = 미루어 봄을 받음)
            uncertainty: 아리송함 점수
        """
        if hasattr(self.model, 'mc_predict'):
            mean_probs, uncertainty = self.model.mc_predict(x, n_samples=n_mc_samples)
        else:
            with torch.no_grad():
                logits = self.model(x)
                mean_probs = F.softmax(logits, dim=-1)
                uncertainty = 1 - mean_probs.max(dim=-1)[0]
        
        predictions = mean_probs.argmax(dim=-1)
        accept_mask = uncertainty < self.threshold
        
        return predictions, accept_mask, uncertainty
    
    def evaluate(self, test_loader: DataLoader, 
                 n_mc_samples: int = 50) -> dict:
        """
        골라 미루어 보기의 됨됨이를 따진다.
        """
        all_preds = []
        all_labels = []
        all_accept = []
        all_uncertainty = []
        
        for x, y in test_loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            preds, accept, unc = self.predict(x, n_mc_samples)
            
            all_preds.append(preds)
            all_labels.append(y)
            all_accept.append(accept)
            all_uncertainty.append(unc)
        
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        accept = torch.cat(all_accept)
        uncertainty = torch.cat(all_uncertainty)
        
        # 자
        coverage = accept.float().mean().item()
        
        if accept.sum() > 0:
            selective_accuracy = (preds[accept] == labels[accept]).float().mean().item()
        else:
            selective_accuracy = 0.0
        
        overall_accuracy = (preds == labels).float().mean().item()
        
        # 물린 보기의 맞음(더 낮아야 한다)
        if (~accept).sum() > 0:
            rejected_accuracy = (preds[~accept] == labels[~accept]).float().mean().item()
        else:
            rejected_accuracy = 1.0
        
        return {
            'coverage': coverage,
            'selective_accuracy': selective_accuracy,
            'overall_accuracy': overall_accuracy,
            'rejected_accuracy': rejected_accuracy,
            'n_accepted': accept.sum().item(),
            'n_rejected': (~accept).sum().item()
        }

def find_optimal_threshold(model: nn.Module,
                            val_loader: DataLoader,
                            target_accuracy: float = 0.95,
                            n_mc_samples: int = 50) -> Tuple[float, dict]:
    """
    과녁 맞음을 이루면서 덮음이 가장 큰 문턱을 찾는다.
    """
    # 미루어 봄과 아리송함을 모두 모은다
    all_preds = []
    all_labels = []
    all_uncertainty = []
    
    model.eval()
    
    with torch.no_grad():
        for x, y in val_loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            if hasattr(model, 'mc_predict'):
                mean_probs, unc = model.mc_predict(x, n_samples=n_mc_samples)
            else:
                logits = model(x)
                mean_probs = F.softmax(logits, dim=-1)
                unc = 1 - mean_probs.max(dim=-1)[0]
            
            all_preds.append(mean_probs.argmax(dim=-1))
            all_labels.append(y)
            all_uncertainty.append(unc)
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    uncertainty = torch.cat(all_uncertainty)
    
    correct = (preds == labels)
    
    # 문턱을 두 쪽으로 갈라 찾는다
    thresholds = torch.linspace(uncertainty.min(), uncertainty.max(), 100)
    
    best_threshold = 0.0
    best_coverage = 0.0
    
    for thresh in thresholds:
        accept = uncertainty < thresh
        
        if accept.sum() > 0:
            sel_acc = correct[accept].float().mean().item()
            coverage = accept.float().mean().item()
            
            if sel_acc >= target_accuracy and coverage > best_coverage:
                best_coverage = coverage
                best_threshold = thresh.item()
    
    return best_threshold, {
        'threshold': best_threshold,
        'coverage': best_coverage,
        'target_accuracy': target_accuracy
    }

def plot_risk_coverage_curve(model: nn.Module,
                              test_loader: DataLoader,
                              n_mc_samples: int = 50):
    """
    무릅씀-덮음 맞바꿈 굽이를 그린다.
    """
    import matplotlib.pyplot as plt
    
    # 자료를 모은다
    all_preds = []
    all_labels = []
    all_uncertainty = []
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            if hasattr(model, 'mc_predict'):
                mean_probs, unc = model.mc_predict(x, n_samples=n_mc_samples)
            else:
                logits = model(x)
                mean_probs = F.softmax(logits, dim=-1)
                unc = 1 - mean_probs.max(dim=-1)[0]
            
            all_preds.append(mean_probs.argmax(dim=-1))
            all_labels.append(y)
            all_uncertainty.append(unc)
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    uncertainty = torch.cat(all_uncertainty)
    correct = (preds == labels)
    
    # 굽이를 셈한다
    thresholds = torch.linspace(uncertainty.min(), uncertainty.max(), 100)
    coverages = []
    risks = []
    accuracies = []
    
    for thresh in thresholds:
        accept = uncertainty < thresh
        coverage = accept.float().mean().item()
        
        if accept.sum() > 0:
            risk = 1 - correct[accept].float().mean().item()  # 어긋남 비율
            accuracy = correct[accept].float().mean().item()
        else:
            risk = 0.0
            accuracy = 1.0
        
        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(accuracy)
    
    # 그림
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(coverages, risks, 'b-', linewidth=2)
    axes[0].set_xlabel('덮음')
    axes[0].set_ylabel('무릅씀(어긋남 비율)')
    axes[0].set_title('무릅씀-덮음 굽이')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(coverages, accuracies, 'g-', linewidth=2)
    axes[1].axhline(y=correct.float().mean().item(), color='r', linestyle='--', 
                   label=f'온 맞음: {correct.float().mean():.4f}')
    axes[1].set_xlabel('덮음')
    axes[1].set_ylabel('골라 미루어 본 맞음')
    axes[1].set_title('맞음-덮음 굽이')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

---

## 3. 밖 분포 알아내기

익힘 분포와 다른 들임을 짚어내는 데 아리송함을 쓴다.

```python
def evaluate_ood_detection(in_dist_uncertainty: np.ndarray,
                           ood_uncertainty: np.ndarray) -> dict:
    """
    아리송함 점수로 밖 분포 알아내기를 따진다.
    
    Args:
        in_dist_uncertainty: 분포 안 자료의 아리송함
        ood_uncertainty: 밖 분포 자료의 아리송함
    
    Returns:
        자: AUROC, FPR@95TPR, 알아내기 맞음
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # 모으고 이름표를 만든다(0 = 분포 안, 1 = 밖 분포)
    all_uncertainty = np.concatenate([in_dist_uncertainty, ood_uncertainty])
    labels = np.concatenate([
        np.zeros(len(in_dist_uncertainty)),
        np.ones(len(ood_uncertainty))
    ])
    
    # AUROC
    auroc = roc_auc_score(labels, all_uncertainty)
    
    # TPR 95%에서의 FPR
    fpr, tpr, thresholds = roc_curve(labels, all_uncertainty)
    idx_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95_tpr = fpr[idx_95]
    
    # 가장 좋은 문턱에서의 알아내기 맞음
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    predictions = (all_uncertainty > optimal_threshold).astype(int)
    accuracy = (predictions == labels).mean()
    
    return {
        'auroc': auroc,
        'fpr_at_95_tpr': fpr_at_95_tpr,
        'detection_accuracy': accuracy,
        'optimal_threshold': optimal_threshold,
        'in_dist_mean': in_dist_uncertainty.mean(),
        'ood_mean': ood_uncertainty.mean()
    }

def ood_detection_example():
    """
    밖 분포 알아내기 보기: MNIST으로 익히고 Fashion-MNIST을 짚어낸다.
    """
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 분포 안: MNIST
    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # 밖 분포: Fashion-MNIST(그림 크기는 같고 밭이 다름)
    fmnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    mnist_loader = DataLoader(mnist_test, batch_size=256, shuffle=False)
    fmnist_loader = DataLoader(fmnist_test, batch_size=256, shuffle=False)
    
    # MNIST으로 모형을 익힌다(줄여 적음)
    model = MCDropoutModel(784, [256, 128], 10, dropout_rate=0.3)
    # ... 익힘 코드 ...
    
    # 아리송함을 셈한다
    model.eval()
    
    mnist_uncertainty = []
    fmnist_uncertainty = []
    
    with torch.no_grad():
        for x, _ in mnist_loader:
            x = x.view(x.size(0), -1)
            _, unc = model.mc_predict(x, n_samples=50)
            mnist_uncertainty.append(unc.cpu().numpy())
        
        for x, _ in fmnist_loader:
            x = x.view(x.size(0), -1)
            _, unc = model.mc_predict(x, n_samples=50)
            fmnist_uncertainty.append(unc.cpu().numpy())
    
    mnist_unc = np.concatenate(mnist_uncertainty)
    fmnist_unc = np.concatenate(fmnist_uncertainty)
    
    # 따진다
    metrics = evaluate_ood_detection(mnist_unc, fmnist_unc)
    
    print("밖 분포 알아내기 결과")
    print("=" * 40)
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"TPR 95%에서의 FPR: {metrics['fpr_at_95_tpr']:.4f}")
    print(f"알아내기 맞음: {metrics['detection_accuracy']:.4f}")
    
    return metrics
```

---

## 4. 고갱이로 챙길 것

!!! success "간추림"

    1. **살아 있는 배움**: 아리송함으로 가장 알려 주는 바가 큰 보기를 고른다(이름표 값 50~70% 아낌)
    2. **골라 미루어 보기**: 아리송한 미루어 봄은 물려 사람이 살피게 한다
    3. **밖 분포 알아내기**: 아리송함이 크면 밖 분포 들임임을 뜻한다
    4. **맞바꿈**: 덮음 대 맞음, 셈 값 대 아리송함 됨됨이

---

## 5. 좋은 버릇

| 쓰임 | 즐겨 쓸 길 |
|-------------|----------------|
| 살아 있는 배움 | 됨됨이는 BALD, 단순함은 엔트로피 |
| 골라 미루어 보기 | 따짐 꾸러미로 문턱을 맞추고 덮음을 지켜본다 |
| 밖 분포 알아내기 | 아리송함을 다른 신호(되살림 어긋남 따위)와 함께 쓴다 |
| 서비스 | 아리송함 분포를 적어 두고 옮겨감이 보이면 알린다 |

---

## 연습문제

1. **살아 있는 배움 견주기**: CIFAR-10에서 아무렇게나, 엔트로피, BALD 얻기 함수를 견주어라. 배움 굽이를 그려라.

2. **골라 미루어 보기**: MNIST에서 맞음 99%을 이루는 문턱을 찾아라. 덮음은 얼마인가?

3. **밖 분포 알아내기**: CIFAR-10으로 익히고 SVHN으로 시험하여라. 아리송함이 둘을 얼마나 잘 갈라내는가?

## 정리하며

이 마당은 아리송함을 쓰는 살아 있는 배움、골라 미루어 보기(물릴 수 있는 길)、밖 분포 알아내기、고갱이로 챙길 것을 차례로 짚었다.

**살펴볼 거리**

- Gal, Y., et al. (2017). "Deep Bayesian Active Learning with Image Data"
- Geifman, Y., & El-Yaniv, R. (2017). "Selective Classification for Deep Neural Networks"
- Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples"
