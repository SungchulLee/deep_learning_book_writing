# 평가

소수 예시 학습을 위한 평가 도구. 소수 예시 학습 모델을 평가하는 함수들로, 지표를 비롯하여

모자란 데이터나 서로 이어진 데이터에서 효율적으로 배우는 것은 오늘날 깊은 학습의 한가운데 놓인 어려움이다. 이 모듈은 모델이 앞선 앎을 살려 새 과제에 재빨리 맞추어 가게 하는 소수 예시 학습 기법을 보여 준다.

## 1. 코드

```python
"""
소수 예시 학습을 위한 평가 도구

소수 예시 학습 모델을 평가하는 함수들로, 지표와
믿음 구간, 표준 평가 규약을 담는다.
"""

import torch
import numpy as np
from scipy import stats

# ========================================================================
# 메인
# ========================================================================


def compute_accuracy(predictions, labels):
    """
    가려내기 정확도를 셈한다.
    
    인수:
        predictions: (n,) - 맞힌 부류 이름표
        labels: (n,) - 참 부류 이름표
    
    반환값:
        accuracy: [0, 1] 안의 실수
    """
    correct = (predictions == labels).float()
    return correct.mean().item()


def compute_confidence_interval(accuracies, confidence=0.95):
    """
    정확도 측정값의 믿음 구간을 셈한다.
    
    인수:
        accuracies: 정확도 값의 목록이나 배열
        confidence: 믿음 수준(기본값 0.95는 95% 믿음 구간)
    
    반환값:
        mean: 평균 정확도
        ci: 믿음 구간(반너비)
    """
    accuracies = np.array(accuracies)
    mean = np.mean(accuracies)
    std_error = stats.sem(accuracies)
    
    # 믿음 구간을 셈한다
    ci = std_error * stats.t.ppf((1 + confidence) / 2, len(accuracies) - 1)
    
    return mean, ci


def evaluate_few_shot_model(model, dataloader, n_episodes=600):
    """
    여러 에피소드에서 소수 예시 학습 모델을 평가한다.
    
    인수:
        model: forward(support, support_labels, query)를 갖춘 소수 예시 학습 모델
        dataloader: (support, support_labels, query, query_labels)를 내주는 DataLoader
        n_episodes: 평가할 에피소드 개수
    
    반환값:
        mean_accuracy: 에피소드에 걸친 평균 정확도
        ci: 95% 믿음 구간
        accuracies: 에피소드별 정확도 목록
    """
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for episode_idx, (support, support_labels, query, query_labels) in enumerate(dataloader):
            if episode_idx >= n_episodes:
                break
            
            # 배치 차원이 있으면 없앤다
            if support.dim() == 5:
                support = support.squeeze(0)
                support_labels = support_labels.squeeze(0)
                query = query.squeeze(0)
                query_labels = query_labels.squeeze(0)
            
            # 순전파
            logits = model(support, support_labels, query)
            
            # 예측을 셈한다
            predictions = torch.argmax(logits, dim=1)
            
            # 이 에피소드의 정확도를 셈한다
            accuracy = compute_accuracy(predictions, query_labels)
            accuracies.append(accuracy)
    
    # 통계를 셈한다
    mean_acc, ci = compute_confidence_interval(accuracies)
    
    return mean_acc, ci, accuracies


def evaluate_with_multiple_runs(model, data, labels, n_way, k_shot, n_query, n_episodes=600):
    """
    에피소드를 그때그때 만들어 모델을 평가한다.
    
    인수:
        model: 소수 예시 모델
        data: 쓸 수 있는 모든 데이터
        labels: 쓸 수 있는 모든 이름표
        n_way: 에피소드마다의 부류 개수
        k_shot: 부류마다의 받침 보기
        n_query: 부류마다의 물음 보기
        n_episodes: 평가할 에피소드 개수
    
    반환값:
        mean_accuracy, confidence_interval, all_accuracies
    """
    from data_loader import create_episode
    
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for _ in range(n_episodes):
            # 에피소드를 만든다
            support, support_labels, query, query_labels = create_episode(
                data, labels, n_way, k_shot, n_query
            )
            
            # 순전파
            logits = model(support, support_labels, query)
            predictions = torch.argmax(logits, dim=1)
            
            # 정확도를 계산한다
            accuracy = compute_accuracy(predictions, query_labels)
            accuracies.append(accuracy)
    
    mean_acc, ci = compute_confidence_interval(accuracies)
    return mean_acc, ci, accuracies


def evaluate_cross_domain(model, source_data, source_labels, target_data, target_labels,
                          n_way, k_shot, n_query, n_episodes=600):
    """
    영역을 넘나드는 소수 예시 학습을 평가한다.
    
    바탕 영역에서 익히고 대상 영역에서 시험하여 일반화를 잰다.
    """
    model.eval()
    accuracies = []
    
    from data_loader import create_episode
    
    with torch.no_grad():
        for _ in range(n_episodes):
            # 대상 영역에서 에피소드를 만든다
            support, support_labels, query, query_labels = create_episode(
                target_data, target_labels, n_way, k_shot, n_query
            )
            
            # 평가한다
            logits = model(support, support_labels, query)
            predictions = torch.argmax(logits, dim=1)
            accuracy = compute_accuracy(predictions, query_labels)
            accuracies.append(accuracy)
    
    mean_acc, ci = compute_confidence_interval(accuracies)
    return mean_acc, ci, accuracies


def compute_confusion_matrix(predictions, labels, n_classes):
    """
    소수 예시 예측의 혼동 행렬을 셈한다.
    
    인수:
        predictions: (n,) - 맞힌 이름표
        labels: (n,) - 참 이름표
        n_classes: 부류 개수
    
    반환값:
        confusion_matrix: (n_classes, n_classes) 행렬
    """
    confusion = torch.zeros(n_classes, n_classes)
    
    for pred, true in zip(predictions, labels):
        confusion[true, pred] += 1
    
    return confusion


def evaluate_per_class_accuracy(predictions, labels, n_classes):
    """
    부류별 정확도를 셈한다.
    
    반환값:
        per_class_acc: (n_classes,) - 부류마다의 정확도
    """
    per_class_acc = []
    
    for c in range(n_classes):
        class_mask = (labels == c)
        if class_mask.sum() > 0:
            class_predictions = predictions[class_mask]
            class_labels = labels[class_mask]
            accuracy = compute_accuracy(class_predictions, class_labels)
            per_class_acc.append(accuracy)
        else:
            per_class_acc.append(0.0)
    
    return torch.tensor(per_class_acc)


class FewShotEvaluator:
    """
    소수 예시 학습 모델을 두루 평가하는 평가기.
    """
    def __init__(self, model):
        self.model = model
        self.results = {
            'accuracies': [],
            'losses': [],
            'per_class_accuracies': []
        }
    
    def evaluate_episode(self, support, support_labels, query, query_labels):
        """
        에피소드 하나에서 평가한다.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(support, support_labels, query)
            predictions = torch.argmax(logits, dim=1)
            
            accuracy = compute_accuracy(predictions, query_labels)
            loss = torch.nn.functional.cross_entropy(logits, query_labels).item()
            
            n_classes = len(torch.unique(support_labels))
            per_class_acc = evaluate_per_class_accuracy(predictions, query_labels, n_classes)
            
            self.results['accuracies'].append(accuracy)
            self.results['losses'].append(loss)
            self.results['per_class_accuracies'].append(per_class_acc)
        
        return accuracy, loss
    
    def get_summary(self):
        """
        평가의 요약 통계를 얻는다.
        """
        mean_acc, ci_acc = compute_confidence_interval(self.results['accuracies'])
        mean_loss = np.mean(self.results['losses'])
        
        # 부류별 정확도의 평균
        if self.results['per_class_accuracies']:
            avg_per_class = torch.stack(self.results['per_class_accuracies']).mean(dim=0)
        else:
            avg_per_class = None
        
        summary = {
            'mean_accuracy': mean_acc,
            'accuracy_95_ci': ci_acc,
            'mean_loss': mean_loss,
            'per_class_accuracy': avg_per_class,
            'n_episodes': len(self.results['accuracies'])
        }
        
        return summary
    
    def reset(self):
        """
        평가 결과를 되돌린다.
        """
        self.results = {
            'accuracies': [],
            'losses': [],
            'per_class_accuracies': []
        }


def print_evaluation_results(mean_acc, ci, n_episodes):
    """
    평가 결과를 보기 좋게 찍는다.
    """
    print("=" * 50)
    print("Few-Shot Learning Evaluation Results")
    print("=" * 50)
    print(f"Number of episodes: {n_episodes}")
    print(f"Mean accuracy: {mean_acc*100:.2f}%")
    print(f"95% Confidence interval: ±{ci*100:.2f}%")
    print(f"Accuracy range: [{(mean_acc-ci)*100:.2f}%, {(mean_acc+ci)*100:.2f}%]")
    print("=" * 50)


# 사용 예
if __name__ == "__main__":
    from prototypical_networks import PrototypicalNetwork, ConvEncoder
    from data_loader import create_episode
    
    # 임시 데이터 만들기
    n_samples = 500
    n_classes = 20
    data = torch.randn(n_samples, 1, 28, 28)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # 모델 생성
    encoder = ConvEncoder(input_channels=1, hidden_dim=64, output_dim=64)
    model = PrototypicalNetwork(encoder)
    
    # 평가한다
    print("Evaluating 5-way 1-shot...")
    mean_acc, ci, accuracies = evaluate_with_multiple_runs(
        model, data, labels,
        n_way=5, k_shot=1, n_query=15,
        n_episodes=100
    )
    print_evaluation_results(mean_acc, ci, 100)
    
    # 5-갈래 5-예시를 평가한다
    print("\nEvaluating 5-way 5-shot...")
    mean_acc, ci, accuracies = evaluate_with_multiple_runs(
        model, data, labels,
        n_way=5, k_shot=5, n_query=15,
        n_episodes=100
    )
    print_evaluation_results(mean_acc, ci, 100)
    
    # FewShotEvaluator를 쓴다
    evaluator = FewShotEvaluator(model)
    
    for i in range(10):
        support, support_labels, query, query_labels = create_episode(
            data, labels, n_way=5, k_shot=1, n_query=15
        )
        evaluator.evaluate_episode(support, support_labels, query, query_labels)
    
    summary = evaluator.get_summary()
    print("\nEvaluator Summary:")
    print(f"Mean accuracy: {summary['mean_accuracy']*100:.2f}%")
    print(f"95% CI: ±{summary['accuracy_95_ci']*100:.2f}%")
    print(f"Mean loss: {summary['mean_loss']:.4f}")```

## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 파이토치 코드로 소수 예시 학습의 핵심 개념을 보여 준다. 모듈 방식의 짜임 덕분에 낱낱의 부품을 살펴보고 다른 과제나 데이터셋에 맞추어 고치기 쉽다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 메타 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 설계 결정을 짚어라. 구체적인 구현 선택 세 가지를 들고 각각이 소수 예시 학습에 왜 알맞은지 설명하라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
평가 구현을 검증하는 두루 갖춘 시험 함수를 작성하라. 빈 입력, 원소가 하나인 입력, 아주 큰 입력, 극단적인 값(0, 아주 큰 수)을 가진 입력 같은 모서리 경우를 시험하라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_fewshotevaluator():
        model = FewShotEvaluator(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — 평가

이 구현은 깔끔하고 읽기 쉬운 파이토치 코드로 소수 예시 학습의 핵심 개념을 보여 준다.

핵심 클래스는 `FewShotEvaluator`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
