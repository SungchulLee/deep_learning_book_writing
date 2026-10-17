# 평가

퓨샷 모델은 여러 에피소드에서 평가하고 평균 정확도와 에피소드 간 변동을 함께 보고한다. 이 페이지는 정확도, t 기반 신뢰구간, 혼동행렬과 클래스별 지표를 계산하는 도구를 설명한다.

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
    
    # 5-클래스 5-예시를 평가한다
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
    print(f"Mean loss: {summary['mean_loss']:.4f}")
```

## 2. 논의

에피소드 정확도 $a_i$가 n개이면 평균과 t 기반 신뢰구간의 반폭은 $\bar a\pm t_{n-1,1-\alpha/2}s/\sqrt n$으로 계산한다. 코드의 ci는 양 끝점이 아니라 반폭이다. n은 2 이상이어야 하며, 에피소드의 독립성 및 표본 수에 따른 근사 타당성을 고려한다. 고정된 데이터셋에서 반복 추출한 구간은 그 데이터와 에피소드 생성 규칙에 대한 불확실성을 나타내며 새로운 데이터셋까지 포괄하지 않는다.

모델 호출은 (support, support_labels, query)이며 반환값을 (쿼리 수, 클래스 수)로 가정한다. FewShotEvaluator는 교차 엔트로피도 계산하므로 로짓이 필요하다. 확률을 반환하는 매칭 네트워크에 그대로 쓰면 로스의 의미가 달라진다. 전체 평가가 no_grad 안에 있으므로 서포트에서 경사 기반 적응을 하는 MAML에는 그대로 적용할 수 없다. 배치 차원 제거는 5차원 이미지와 에피소드 배치 크기 1만 가정한다.

클래스별 정확도에서 쿼리가 없는 클래스는 코드상 0이지만 실제 정확도가 0이라고 관측한 것은 아니다. 또 에피소드 로컬 라벨 0이 매번 다른 원본 클래스를 뜻할 수 있으므로 같은 위치끼리 평균 낸 값을 특정 클래스의 성능으로 부르면 안 된다. evaluate_cross_domain은 source_data와 source_labels를 사용하지 않고 타깃 에피소드만 평가한다. 소스 학습이나 도메인 적응을 수행하는 함수는 아니다. 주 예제는 무작위 데이터이며, create_episode가 요청한 표본 수를 확보하는지도 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
에피소드 평균 정확도와 전체 쿼리 정확도는 언제 같은가?

</div>

??? success "풀이"
    모든 에피소드의 쿼리 수가 같으면 같다. 크기가 다르면 에피소드 평균은 각 과제에 같은 웨이트를 주지만 전체 쿼리 정확도는 표본 수에 비례한 웨이트를 준다. 어떤 평균인지 평가 결과와 함께 기록한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
compute_accuracy와 신뢰구간 함수에 필요한 입력 조건을 설명하라.

</div>

??? success "풀이"
    예측과 정답은 같은 길이의 1차원 클래스 인덱스여야 하며 빈 입력을 제외한다. (B,1)과 (B,)를 비교하면 브로드캐스팅으로 잘못된 행렬이 생길 수 있다. 신뢰구간에는 최소 두 개의 유한한 정확도와 0과 1 사이의 confidence가 필요하다. 장치 이동은 함수가 자동 수행하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
서로 다른 에피소드의 클래스별 지표를 어떻게 집계해야 하는가?

</div>

??? success "풀이"
    원본 클래스 ID로 로컬 라벨을 되돌려 정답 수와 표본 수를 누적한다. 관측이 없는 클래스는 미관측으로 표시한다. 모든 에피소드에 같은 N개 로컬 라벨이 있어도 클래스의 의미가 같지는 않다. N 자체가 다르면 현재 torch.stack도 사용할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 로스나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

평균 정확도에는 에피소드 수, 클래스·샷·쿼리 구성과 신뢰구간의 의미를 함께 적는다. 경사 기반 적응, 확률 출력과 클래스별 집계에는 모델에 맞는 평가 경로가 필요하다.
