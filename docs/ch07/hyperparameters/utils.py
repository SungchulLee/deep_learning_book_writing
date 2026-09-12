"""7.9 초매개변수 조율 절이 함께 쓰는 도움 함수.

이 절의 예제들(격자 찾기, 마구잡이 찾기, 베이즈 최적화, 이어지는 반 줄이기)은
모두 같은 자료와 같은 보고 형식을 쓴다. 그 공통부를 여기에 모아 둔다.
"""

import numpy as np
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__all__ = ["load_sample_dataset", "print_results", "plot_search_results",
           "create_param_grid_summary", "plot_parameter_importance"]


def load_sample_dataset(name="iris", test_size=0.3, random_state=42, scale=True):
    """예제가 함께 쓰는 자료를 불러와 학습/시험으로 나눈다.

    인수:
        name: 'iris', 'wine', 'synthetic' 가운데 하나
        test_size: 시험 집합의 비율
        random_state: 씨앗. 예제끼리 견주려면 같은 값을 써야 한다
        scale: 특징을 평균 0, 표준편차 1로 맞출지 여부.
               SVM처럼 크기에 민감한 모델에는 반드시 필요하다

    반환값:
        X_train, X_test, y_train, y_test
    """
    if name == "iris":
        data = load_iris()
        X, y = data.data, data.target
    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
    elif name == "synthetic":
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=3, random_state=random_state,
        )
    else:
        raise ValueError(
            f"모르는 자료 이름: {name!r}. 'iris', 'wine', 'synthetic' 가운데 하나를 쓴다."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        # 고르게 맞추는 값은 반드시 학습 집합에서만 구한다.
        # 전체에서 구하면 시험 자료의 정보가 학습에 새어 든다.
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def print_results(method_name, best_params, best_score, search_time, test_score=None):
    """찾기 결과를 절 전체에서 같은 모양으로 알린다.

    방법마다 형식이 다르면 나란히 견주기 어려우므로 여기서 한 번에 정한다.
    """
    print("\n" + "=" * 60)
    print(f"{method_name}")
    print("=" * 60)
    print(f"  가장 좋은 조절 값:")
    for key, value in sorted(best_params.items()):
        print(f"    {key:<24} {value}")
    print(f"  맞대 보기 점수 : {best_score:.4f}")
    if test_score is not None:
        print(f"  시험 점수      : {test_score:.4f}")
    print(f"  걸린 값        : {search_time:.2f}초")


def plot_search_results(results, title="Hyperparameter Search", ax=None):
    """찾기가 나아가는 모습을 그린다.

    인수:
        results: 시도마다의 점수를 담은 차례열, 또는 sklearn의 cv_results_ 사전
        title: 그림 제목
        ax: 그릴 Axes. 없으면 새로 만든다
    """
    import matplotlib.pyplot as plt

    if isinstance(results, dict) and "mean_test_score" in results:
        scores = np.asarray(results["mean_test_score"])
    else:
        scores = np.asarray(list(results))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    trials = np.arange(1, len(scores) + 1)
    ax.plot(trials, scores, "o-", markersize=4, alpha=0.6, label="per-trial score")
    # 지금까지의 가장 좋은 점수. 찾기가 실제로 나아가는지는 이 선이 말해 준다
    ax.plot(trials, np.maximum.accumulate(scores), "r-", linewidth=2,
            label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_sample_dataset("iris")
    print(f"iris: X_train {X_train.shape}, X_test {X_test.shape}")
    print_results(
        method_name="보기",
        best_params={"n_estimators": 100, "max_depth": 5},
        best_score=0.95,
        search_time=1.23,
        test_score=0.93,
    )


def create_param_grid_summary(param_grid, verbose=True):
    """격자에 몇 가지 짝이 들어 있는지 세어 알린다.

    격자 찾기의 값은 짝의 수에 그대로 비례한다. 돌리기 전에 이 수를 보아
    두면 몇 분이 걸릴 일인지 몇 시간이 걸릴 일인지 가늠할 수 있다.

    인수:
        param_grid: {이름: [값들]} 사전, 또는 그런 사전들의 목록

    반환값:
        맞대 보기 한 겹에서 시험할 짝의 총 수
    """
    grids = param_grid if isinstance(param_grid, list) else [param_grid]

    total = 0
    for grid in grids:
        combos = 1
        for values in grid.values():
            combos *= len(values)
        total += combos

    if verbose:
        print("\n격자 살펴보기")
        for grid in grids:
            for name, values in grid.items():
                print(f"  {name:<22} {len(values):>3}가지  {list(values)}")
        print(f"  합쳐서 {total}가지 짝")
    return total


def plot_parameter_importance(cv_results, param_name, ax=None):
    """조절 값 하나를 바꿀 때 점수가 어떻게 달라지는지 그린다.

    인수:
        cv_results: sklearn 찾기 객체의 cv_results_
        param_name: 볼 조절 값의 이름(앞의 'param_'은 빼고 준다)
        ax: 그릴 Axes. 없으면 새로 만든다

    같은 값끼리 묶어 평균과 흩어짐을 함께 보이므로, 그 조절 값이 실제로
    점수를 움직이는지 아니면 상관없는지가 드러난다.
    """
    import matplotlib.pyplot as plt

    key = param_name if param_name.startswith("param_") else f"param_{param_name}"
    if key not in cv_results:
        raise KeyError(
            f"{key}가 cv_results_에 없다. 있는 것: "
            f"{[k for k in cv_results if k.startswith('param_')]}"
        )

    values = [v for v in cv_results[key]]
    scores = np.asarray(cv_results["mean_test_score"])

    # 값마다 점수를 모은다. 숫자가 아닐 수도 있으므로 문자열로 묶는다.
    groups = {}
    for v, s in zip(values, scores):
        groups.setdefault(str(v), []).append(s)

    labels = list(groups)
    means = [float(np.mean(groups[k])) for k in labels]
    stds = [float(np.std(groups[k])) for k in labels]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(len(labels)), means, yerr=stds, fmt="o-", capsize=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(param_name)
    ax.set_ylabel("mean CV score")
    ax.set_title(f"Effect of {param_name}")
    ax.grid(alpha=0.3)
    return ax
