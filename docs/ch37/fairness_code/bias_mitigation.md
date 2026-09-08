# 치우침 눅이기

깊은 배움의 치우침 눅이기 재주. 기계 배움 모형의 치우침을 줄이는 여러 길.

기계 배움 얼개에서 고름을 지키는 일은 윤리로도 마땅하고 참으로도 걸린 문제다. 이 꾸러미는 깊은 배움 모형의 치우침을 알아내고, 재고, 눅이는 재주를 보이며, 이론의 고름 잣대를 손에 잡히는 코드로 이어 준다.

## 1. 코드

```python
"""
깊은 배움의 치우침 눅이기 재주
기계 배움 모형의 치우침을 줄이는 여러 길.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler

# ========================================================================
# 메인
# ========================================================================


class ReweighingMitigation:
    """
    미리 다듬어 눅이기: 익힘 표본에 짐을 다시 매긴다.

    지켜야 할 됨됨이와 이름표에 따라 익힘 표본마다 다른 짐을 매겨
    고름을 이룬다.
    """

    def __init__(self):
        self.weights = None

    def compute_weights(
        self,
        y: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> np.ndarray:
        """
        짐 다시 매기기에 쓸 표본 짐을 셈한다.

        Args:
            y: 이름표
            sensitive_attr: 예민한 됨됨이

        Returns:
            표본 짐
        """
        weights = np.ones(len(y))

        # 서로 다른 값을 얻는다
        attr_values = np.unique(sensitive_attr)
        label_values = np.unique(y)

        # 바라는 낌새와 본 낌새를 셈한다
        n = len(y)

        for attr_val in attr_values:
            for label_val in label_values:
                # 본 낌새
                mask = (sensitive_attr == attr_val) & (y == label_val)
                p_observed = np.sum(mask) / n

                # 바라는 낌새(서로 남남이라고 여긴다)
                p_attr = np.sum(sensitive_attr == attr_val) / n
                p_label = np.sum(y == label_val) / n
                p_expected = p_attr * p_label

                # 짐을 매긴다
                if p_observed > 0:
                    weight = p_expected / p_observed
                    weights[mask] = weight

        self.weights = weights
        return weights


class AdversarialDebiasing(nn.Module):
    """
    익히며 눅이기: 맞겨루며 치우침 걷어내기.

    맞겨루는 익힘으로 치우침을 걷어낸다. 가름개는 겨눈 것을
    미루어 보도록 배우고, 맞수는 가름개의 나타냄에서 예민한
    됨됨이를 알아내려 한다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1
    ):
        super(AdversarialDebiasing, self).__init__()

        # 결 부호기
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 가름개(겨눈 이름표를 미루어 본다)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

        # 맞수(예민한 됨됨이를 미루어 본다)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        앞으로 걸음.

        Args:
            x: 들임 결

        Returns:
            (가름개 미루어 봄, 맞수 미루어 봄) 짝
        """
        features = self.encoder(x)
        y_pred = self.classifier(features)
        a_pred = self.adversary(features)
        return y_pred, a_pred


def train_adversarial_debiasing(
    model: AdversarialDebiasing,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.001,
    adversary_weight: float = 0.5
) -> AdversarialDebiasing:
    """
    맞겨루며 치우침 걷어내는 모형을 익힌다.

    Args:
        model: AdversarialDebiasing 모형
        X_train: 익힘 결
        y_train: 익힘 이름표
        sensitive_train: 예민한 됨됨이
        epochs: 익힘 시대의 수
        learning_rate: 배움 빠르기
        adversary_weight: 맞수 잃음의 짐

    Returns:
        익힌 모형
    """
    # 텐서로 바꾼다
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    s_tensor = torch.FloatTensor(sensitive_train).unsqueeze(1)

    # 가장 좋게 하는 개
    optimizer_clf = optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=learning_rate
    )
    optimizer_adv = optim.Adam(model.adversary.parameters(), lr=learning_rate)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        # 맞수를 익힌다
        model.adversary.train()
        model.encoder.eval()

        optimizer_adv.zero_grad()
        _, a_pred = model(X_tensor)
        adv_loss = criterion(a_pred, s_tensor)
        adv_loss.backward()
        optimizer_adv.step()

        # 가름개를 익힌다(맞수 잃음은 크게, 가름개 잃음은 작게)
        model.classifier.train()
        model.encoder.train()
        model.adversary.eval()

        optimizer_clf.zero_grad()
        y_pred, a_pred = model(X_tensor)

        clf_loss = criterion(y_pred, y_tensor)
        adv_loss_for_clf = -criterion(a_pred, s_tensor)  # 크게 하려고 음수로

        total_loss = clf_loss + adversary_weight * adv_loss_for_clf
        total_loss.backward()
        optimizer_clf.step()

        if (epoch + 1) % 20 == 0:
            print(f"시대 {epoch+1}/{epochs}, 가름개 잃음: {clf_loss.item():.4f}, "
                  f"맞수 잃음: {adv_loss.item():.4f}")

    return model


class FairRepresentationLearning(nn.Module):
    """
    예민한 소식을 걷어내어 고른 나타냄을 배운다.

    변이 자동 부호기 결의 길로 예민한 됨됨이에 흔들리지 않는
    나타냄을 배운다.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64
    ):
        super(FairRepresentationLearning, self).__init__()

        # 부호기
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 푸는 개
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 예민한 됨됨이 미루개(정칙화에 쓴다)
        self.sensitive_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        앞으로 걸음.

        Args:
            x: 들임 결

        Returns:
            (숨은 나타냄, 되세움, 예민한 됨됨이 미루어 봄) 짝
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        s_pred = self.sensitive_predictor(z)
        return z, x_recon, s_pred


class ThresholdOptimization:
    """
    뒤에 다듬어 눅이기: 판단 문턱을 가장 좋게 한다.

    무리마다 다른 가름 문턱을 써서 고름 매임을 채운다.
    """

    def __init__(self, fairness_constraint: str = 'demographic_parity'):
        """
        문턱 다듬개의 첫자리를 잡는다.

        Args:
            fairness_constraint: 고름 매임의 갈래
                ('demographic_parity', 'equal_opportunity', 'equalized_odds')
        """
        self.fairness_constraint = fairness_constraint
        self.thresholds = {}

    def optimize_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_attr: np.ndarray,
        num_thresholds: int = 100
    ) -> Dict[int, float]:
        """
        무리마다 가장 좋은 문턱을 찾는다.

        Args:
            y_true: 참 이름표
            y_pred_proba: 미루어 본 낌새
            sensitive_attr: 예민한 됨됨이
            num_thresholds: 해 볼 문턱의 수

        Returns:
            무리를 가장 좋은 문턱에 이어 주는 사전
        """
        groups = np.unique(sensitive_attr)
        thresholds_to_try = np.linspace(0, 1, num_thresholds)

        best_thresholds = {}
        best_fairness = float('inf')

        # 온 문턱 어우름을 격자로 훑는다
        for t0 in thresholds_to_try:
            for t1 in thresholds_to_try:
                thresholds = {groups[0]: t0, groups[1]: t1}

                # 문턱을 건다
                y_pred = np.zeros_like(y_pred_proba)
                for group, threshold in thresholds.items():
                    mask = sensitive_attr == group
                    y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)

                # 고름 자를 셈한다
                fairness_score = self._calculate_fairness(
                    y_true, y_pred, sensitive_attr
                )

                if fairness_score < best_fairness:
                    best_fairness = fairness_score
                    best_thresholds = thresholds.copy()

        self.thresholds = best_thresholds
        return best_thresholds

    def _calculate_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> float:
        """매임에 따라 고름 자를 셈한다."""
        groups = np.unique(sensitive_attr)

        if self.fairness_constraint == 'demographic_parity':
            rates = []
            for group in groups:
                mask = sensitive_attr == group
                rates.append(np.mean(y_pred[mask]))
            return abs(rates[0] - rates[1])

        elif self.fairness_constraint == 'equal_opportunity':
            tpr_list = []
            for group in groups:
                mask = (sensitive_attr == group) & (y_true == 1)
                if np.sum(mask) > 0:
                    tpr = np.sum((y_pred == 1) & mask) / np.sum(mask)
                    tpr_list.append(tpr)
                else:
                    tpr_list.append(0)
            return abs(tpr_list[0] - tpr_list[1])

        return 0.0

    def predict(
        self,
        y_pred_proba: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> np.ndarray:
        """
        다듬은 문턱으로 미루어 본다.

        Args:
            y_pred_proba: 미루어 본 낌새
            sensitive_attr: 예민한 됨됨이

        Returns:
            두 값 미루어 봄
        """
        y_pred = np.zeros_like(y_pred_proba)

        for group, threshold in self.thresholds.items():
            mask = sensitive_attr == group
            y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)

        return y_pred


def example_usage():
    """치우침 눅이기 재주를 쓰는 보기."""
    np.random.seed(42)

    # 지어낸 자료를 만든다
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    sensitive_attr = np.random.randint(0, 2, n_samples)

    # 치우친 이름표를 만든다
    y = np.random.randint(0, 2, n_samples)
    y[sensitive_attr == 0] = np.random.choice([0, 1], np.sum(sensitive_attr == 0), p=[0.3, 0.7])
    y[sensitive_attr == 1] = np.random.choice([0, 1], np.sum(sensitive_attr == 1), p=[0.7, 0.3])

    print("=" * 60)
    print("치우침 눅이기 재주 보이기")
    print("=" * 60)

    # 1. 짐 다시 매기기
    print("\n1. 짐 다시 매기기")
    print("-" * 60)
    reweigh = ReweighingMitigation()
    weights = reweigh.compute_weights(y, sensitive_attr)
    print(f"표본 짐을 셈했다. 고른 짐: {np.mean(weights):.4f}")
    print(f"짐 너비: [{np.min(weights):.4f}, {np.max(weights):.4f}]")

    # 2. 문턱 다듬기
    print("\n2. 문턱 다듬기")
    print("-" * 60)
    y_pred_proba = np.random.rand(n_samples)
    threshold_opt = ThresholdOptimization(fairness_constraint='demographic_parity')
    optimal_thresholds = threshold_opt.optimize_thresholds(
        y, y_pred_proba, sensitive_attr, num_thresholds=20
    )
    print(f"가장 좋은 문턱: {optimal_thresholds}")

    print("\n눅이기 재주의 첫자리를 잘 잡았다!")


if __name__ == "__main__":
    example_usage()
```

## 2. 논의

이 짜보기는 함께 어우러져 온전한 깊은 배움 얼개를 이루는 클래스 4개(`ReweighingMitigation`, `AdversarialDebiasing`, `FairRepresentationLearning`, `ThresholdOptimization`)를 세운다. 클래스마다 따로 떨어진 조각을 감싸므로 코드가 조각으로 나뉘고 넓히기 쉽다. `forward` 방법은 PyTorch가 저절로 미분하는 데 쓰는 셈 그래프를 세운다.

익힘 돌기는 여느 PyTorch 결을 따른다. 앞으로 걸음으로 미루어 봄을 셈하고, 잃음을 셈하고, 되짚기로 기울기를 셈하고, 가장 좋게 하는 개로 매개변수를 고친다. 시대마다 자를 좇으면 모여 가는 결이 드러나고 덜 맞추기나 지나치게 맞추기 같은 탈을 짚어내기 좋다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 기계 배움 일감에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
맡긴 첫자리 잡기로 만든 `ReweighingMitigation`에서 배울 수 있는 매개변수의 온 수를 셈하여라. 짐과 치우침을 아울러 켜마다 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)`마다 짐 매개변수가 `in_features * out_features`개, 치우침 매개변수가 `out_features`개다(`bias=False`이 아니라면). `nn.Conv2d(in_c, out_c, k)`마다 짐 매개변수가 `in_c * out_c * k * k`개, 치우침 매개변수가 `out_c`개다. `nn.Embedding(num, dim)`이면 매개변수가 `num * dim`개다. 온 켜에 걸쳐 더한다. `sum(p.numel() for p in model.parameters())`으로 살펴볼 수 있다.

---

**연습문제 2.**
가장 좋게 하는 개를 Adam(`torch.optim.Adam`에 `lr=0.001`)으로 갈음하고 본디 것과 익힘이 모여 가는 결을 견주어라. 둘의 잃음 곡선을 한 그림에 그려라.

??? success "연습문제 2 풀이"
    가장 좋게 하는 개를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 갈음한다. Adam은 매개변수마다 맞춰 가는 배움 빠르기와 밀림 어림을 지니므로 앞선 시대에 흔히 더 빨리 모인다. Adam의 잃음 곡선은 첫 몇 시대에 더 가파르게 떨어지지만 가장 좋은 자리 언저리에서 밀림을 곁들인 SGD보다 조금 더 흔들릴 수 있다. 고르게 견주려면 아무렇게나 하는 씨앗과 시대 수를 같게 하고 둘 다 돌려라.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
`ReweighingMitigation`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    못 박아 둔 켜를 이렇게 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서는 `for layer in self.layers: x = layer(x)`으로 돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 매개변수를 가장 좋게 하기에 올린다. 시험은 이렇게 한다. `for n in [2, 4, 8]: model = ReweighingMitigation(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 치우침 눅이기

이 짜보기는 함께 어우러져 온전한 깊은 배움 얼개를 이루는 클래스 4개(`ReweighingMitigation`, `AdversarialDebiasing`, `FairRepresentationLearning`, `ThresholdOptimization`)를 세운다.

고갱이 갈래는 `ReweighingMitigation`, `AdversarialDebiasing`, `FairRepresentationLearning`, `ThresholdOptimization`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
