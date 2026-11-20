# 바이어스 완화

딥러닝의 바이어스 완화 기법. 머신러닝 모델의 바이어스를 줄이는 여러 길.

머신러닝 시스템에서 공정성을 지키는 일은 윤리적으로도 마땅하고 실무적으로도 중요한 문제다. 이 꾸러미는 딥러닝 모델의 바이어스를 탐지하고, 측정하고, 완화하는 기법을 보이며, 이론의 공정성 지표를 실제 코드로 연결해 준다.

## 1. 코드

??? note "코드 (420줄)"

    ```python
    """
    딥러닝의 치우침 눅이기 재주
    기계 러닝 모형의 치우침을 줄이는 여러 길.
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
        공정성을 이룬다.
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

            # 바라는 확률과 본 확률을 셈한다
            n = len(y)

            for attr_val in attr_values:
                for label_val in label_values:
                    # 본 확률
                    mask = (sensitive_attr == attr_val) & (y == label_val)
                    p_observed = np.sum(mask) / n

                    # 바라는 확률(서로 남남이라고 여긴다)
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

        맞겨루는 익힘으로 치우침을 걷어낸다. 분류기는 겨눈 것을
        미루어 보도록 배우고, 맞수는 분류기의 나타냄에서 예민한
        됨됨이를 알아내려 한다.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            output_dim: int = 1
        ):
            super(AdversarialDebiasing, self).__init__()

            # 결 인코더
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            # 분류기(겨눈 이름표를 미루어 본다)
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
            앞으로 스텝.

            Args:
                x: 들임 결

            Returns:
                (분류기 미루어 봄, 맞수 미루어 봄) 짝
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
            learning_rate: 러닝 빠르기
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

            # 분류기를 익힌다(맞수 잃음은 크게, 분류기 잃음은 작게)
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
                print(f"시대 {epoch+1}/{epochs}, 분류기 잃음: {clf_loss.item():.4f}, "
                      f"맞수 잃음: {adv_loss.item():.4f}")

        return model


    class FairRepresentationLearning(nn.Module):
        """
        예민한 소식을 걷어내어 고른 나타냄을 배운다.

        변이 자동 인코더 결의 길로 예민한 됨됨이에 흔들리지 않는
        나타냄을 배운다.
        """

        def __init__(
            self,
            input_dim: int,
            latent_dim: int = 32,
            hidden_dim: int = 64
        ):
            super(FairRepresentationLearning, self).__init__()

            # 인코더
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
            앞으로 스텝.

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
        뒤에 다듬어 눅이기: 판단 임계를 가장 좋게 한다.

        무리마다 다른 가름 임계를 써서 공정성 제약을 채운다.
        """

        def __init__(self, fairness_constraint: str = 'demographic_parity'):
            """
            임계 다듬개의 첫자리를 잡는다.

            Args:
                fairness_constraint: 공정성 제약의 클래스
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
            무리마다 가장 좋은 임계를 찾는다.

            Args:
                y_true: 참 이름표
                y_pred_proba: 미루어 본 확률
                sensitive_attr: 예민한 됨됨이
                num_thresholds: 해 볼 임계의 수

            Returns:
                무리를 가장 좋은 임계에 이어 주는 사전
            """
            groups = np.unique(sensitive_attr)
            thresholds_to_try = np.linspace(0, 1, num_thresholds)

            best_thresholds = {}
            best_fairness = float('inf')

            # 온 임계 어우름을 격자로 훑는다
            for t0 in thresholds_to_try:
                for t1 in thresholds_to_try:
                    thresholds = {groups[0]: t0, groups[1]: t1}

                    # 임계를 건다
                    y_pred = np.zeros_like(y_pred_proba)
                    for group, threshold in thresholds.items():
                        mask = sensitive_attr == group
                        y_pred[mask] = (y_pred_proba[mask] >= threshold).astype(int)

                    # 공정성 지표를 셈한다
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
            """매임에 따라 공정성 지표를 셈한다."""
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
            다듬은 임계로 미루어 본다.

            Args:
                y_pred_proba: 미루어 본 확률
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

        # 2. 임계 다듬기
        print("\n2. 임계 다듬기")
        print("-" * 60)
        y_pred_proba = np.random.rand(n_samples)
        threshold_opt = ThresholdOptimization(fairness_constraint='demographic_parity')
        optimal_thresholds = threshold_opt.optimize_thresholds(
            y, y_pred_proba, sensitive_attr, num_thresholds=20
        )
        print(f"가장 좋은 임계: {optimal_thresholds}")

        print("\n눅이기 재주의 첫자리를 잘 잡았다!")


    if __name__ == "__main__":
        example_usage()
    ```


**출력:**

```
============================================================
치우침 눅이기 재주 보이기
============================================================

1. 짐 다시 매기기
------------------------------------------------------------
표본 짐을 셈했다. 고른 짐: 1.0000
짐 너비: [0.7141, 1.6633]

2. 임계 다듬기
------------------------------------------------------------
가장 좋은 임계: {0: 0.0, 1: 0.0}

눅이기 재주의 첫자리를 잘 잡았다!
```

## 2. 논의

이 구현은 함께 어우러져 온전한 딥러닝 구조를 이루는 클래스 4개(`ReweighingMitigation`, `AdversarialDebiasing`, `FairRepresentationLearning`, `ThresholdOptimization`)를 세운다. 클래스마다 따로 떨어진 조각을 감싸므로 코드가 조각으로 나뉘고 넓히기 쉽다. `forward` 방법은 PyTorch가 저절로 미분하는 데 쓰는 계산 그래프를 세운다.

학습 루프는 일반적인 PyTorch 패턴을 따른다. 순전파로 예측을 계산하고, 로스를 계산하고, 역전파로 그래디언트를 계산하고, 옵티마이저로 파라미터를 갱신한다. 에폭마다 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 탐지하기 좋다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 머신러닝 과제에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 초기화로 만든 `ReweighingMitigation`에서 학습 가능한 파라미터의 총수를 계산하여라. 웨이트와 바이어스를 아울러 레이어마다 나누어 세어라.

</div>

??? success "풀이"
    `nn.Linear(in_features, out_features)`마다 웨이트 파라미터가 `in_features * out_features`개, 바이어스 파라미터가 `out_features`개다(`bias=False`이 아니라면). `nn.Conv2d(in_c, out_c, k)`마다 웨이트 파라미터가 `in_c * out_c * k * k`개, 바이어스 파라미터가 `out_c`개다. `nn.Embedding(num, dim)`이면 파라미터가 `num * dim`개다. 모든 레이어에 걸쳐 더한다. `sum(p.numel() for p in model.parameters())`으로 살펴볼 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam(`torch.optim.Adam`에 `lr=0.001`)으로 바꾸고 원래 것과 학습 수렴 양상을 비교하여라. 둘의 로스 곡선을 한 그림에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 파라미터마다 맞춰 가는 러닝레이트와 모멘텀 추정을 지니므로 초기 에폭에 흔히 더 빨리 수렴한다. Adam의 로스 곡선은 첫 몇 시대에 더 가파르게 떨어지지만 가장 좋은 자리 언저리에서 모멘텀을 곁들인 SGD보다 조금 더 흔들릴 수 있다. 고르게 견주려면 아무렇게나 하는 시드와 시대 수를 같게 하고 둘 다 돌려라.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    흔한 실패 유형은 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 모니터링해 탐지한다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 기록). 그래디언트 클리핑, 더 나은 초기화(Xavier/Kaiming), 구조 수정(잔차 연결, 정규화 레이어)으로 고친다. (2) **과적합** -- 학습 로스는 줄어드는데 검증 로스가 오르면 탐지한다. 정칙화(드롭아웃, 웨이트 감쇠, 데이터 증강)나 모델 축소로 고친다. 학습 지표와 검증 지표를 늘 함께 모니터링해 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`ReweighingMitigation`을 레이어나 블록의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라미터를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하라. 레이어 2개, 4개, 8개로 시험하여라.

</div>

??? success "풀이"
    고정해 둔 레이어를 이렇게 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서는 `for layer in self.layers: x = layer(x)`으로 돈다. 일반 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라미터를 최적화에 올린다. 시험은 이렇게 한다. `for n in [2, 4, 8]: model = ReweighingMitigation(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 바이어스 완화

이 구현은 함께 어우러져 온전한 딥러닝 구조를 이루는 클래스 4개(`ReweighingMitigation`, `AdversarialDebiasing`, `FairRepresentationLearning`, `ThresholdOptimization`)를 세운다.

핵심 클래스는 `ReweighingMitigation`, `AdversarialDebiasing`, `FairRepresentationLearning`, `ThresholdOptimization`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
