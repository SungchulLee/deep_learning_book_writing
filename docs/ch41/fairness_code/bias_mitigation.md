# 바이어스 완화

딥러닝의 바이어스 완화 기법. 머신러닝 모델의 바이어스를 줄이는 여러 방법을 다룬다.

머신러닝 시스템에서 공정성을 지키는 일은 윤리적으로도 마땅하고 실무적으로도 중요한 문제다. 이 꾸러미는 바이어스를 완화하는 기법 네 가지를 보인다. 학습 전에 데이터를 고치는 전처리(재가중), 학습 중에 공정성 목표를 넣는 학습 중 처리(적대적 디바이어싱, 공정한 표현 학습), 학습된 모델의 출력을 조정하는 후처리(임계값 최적화)다.

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

이 구현은 바이어스 완화의 세 단계에 해당하는 클래스 4개(`ReweighingMitigation`, `AdversarialDebiasing`, `FairRepresentationLearning`, `ThresholdOptimization`)를 정의한다. 전처리인 재가중은 보호 속성과 레이블이 독립일 때의 기대 확률을 관측 확률로 나눈 값을 표본 가중치로 매긴다. 학습 중 처리인 두 뉴럴 네트워크는 인코더가 만든 표현에서 보호 속성을 맞히려는 적대자(또는 예측기)를 두어, 표현에서 집단 정보를 걷어내도록 학습한다. 후처리인 임계값 최적화는 집단마다 분류 임계값을 따로 골라 공정성 지표를 맞춘다.

예제가 실제로 돌리는 것은 재가중과 임계값 최적화 둘뿐이고, 두 뉴럴 네트워크와 적대적 학습 루프는 정의만 되어 있다. 재가중 결과는 식대로 나왔다. 가중치의 평균이 정확히 1.0000인 것은 이 가중치가 전체 표본 수를 보존하도록 정의되었기 때문이고, 범위 [0.7141, 1.6633]은 많이 나타난 조합(집단 0의 양성, 집단 1의 음성)의 가중치를 줄이고 적게 나타난 조합의 가중치를 키운 결과다.

임계값 최적화의 결과 {0: 0.0, 1: 0.0}은 쓸모없는 해다. 두 집단 모두 임계값이 0이면 모든 표본을 양성으로 예측하므로 인구통계적 동등성 차이가 0이 되어 목표를 '완벽히' 만족한다. 정확도를 목표에 넣지 않고 공정성 지표만 최소화하면 이런 자명한 해에 빠진다. 실제로 쓰려면 공정성 제약을 만족하는 임계값 가운데 정확도가 가장 높은 것을 골라야 한다. 게다가 예측 확률이 레이블과 무관한 난수라서, 이 예제는 알고리즘이 돌아가는 모습만 보일 뿐 완화 효과를 보이지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
`ReweighingMitigation`에는 학습 가능한 파라미터가 있는가? 예제와 같이 피처가 20개일 때 `AdversarialDebiasing(input_dim=20)`을 기본값으로 만들면 학습 가능한 파라미터는 모두 몇 개인지, 인코더·분류기·적대자로 나누어 세어라.

</div>

??? success "풀이"
    `ReweighingMitigation`은 데이터의 빈도로 가중치를 계산할 뿐이라 학습 가능한 파라미터가 없다. `nn.Linear(m, n)`은 웨이트 $mn$개와 바이어스 $n$개를 가진다. 인코더는 $20 \times 64 + 64 = 1344$와 $64 \times 64 + 64 = 4160$을 더해 5504개다. 분류기는 $64 \times 32 + 32 = 2080$과 $32 \times 1 + 1 = 33$을 더해 2113개이고, 적대자도 구조가 같아 2113개다. 모두 $5504 + 2113 + 2113 = 9730$개다. 분류기를 학습하는 옵티마이저는 인코더와 분류기의 7617개를, 적대자의 옵티마이저는 나머지 2113개를 맡는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
예제는 집단 0의 양성 비율을 0.7, 집단 1의 양성 비율을 0.3으로 두고 레이블을 만든다. 두 집단의 크기가 같다고 보고, 재가중이 네 칸(집단 × 레이블)에 매길 가중치를 손으로 계산하여라. 출력의 범위 [0.7141, 1.6633]과 견주고, 가중치의 평균이 왜 정확히 1인지 보여라.

</div>

??? success "풀이"
    $P(A = a) = 0.5$이고 $P(Y = 1) = 0.5 \times 0.7 + 0.5 \times 0.3 = 0.5$이므로, 독립일 때의 기대 확률은 네 칸 모두 $0.25$다. 관측 확률은 (집단 0, 양성)과 (집단 1, 음성)이 $0.5 \times 0.7 = 0.35$, 나머지 두 칸이 $0.15$다. 가중치는 $0.25/0.35 \approx 0.714$와 $0.25/0.15 \approx 1.667$이고, 출력의 0.7141과 1.6633은 표본 1,000개에서 잰 빈도라 이 값에서 조금 벗어났다. 평균은 칸마다 (표본 수 × 가중치)를 더해 $n$으로 나눈 것이다. 칸 $c$의 표본 수가 $n P_{\text{obs}}(c)$이므로 그 합은 $n \sum_c P_{\text{obs}}(c) \cdot P_{\text{exp}}(c)/P_{\text{obs}}(c) = n \sum_c P_{\text{exp}}(c) = n$이 되어, 평균은 표본에서 잰 확률로도 정확히 1이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    (1) **임계값 최적화가 자명한 해에 빠진다.** 출력의 {0: 0.0, 1: 0.0}이 그것이다. 공정성 지표만 최소화하므로 모두 양성 또는 모두 음성으로 예측하는 해가 언제나 최적이다. 고른 임계값에서 정확도와 양성 예측 비율을 함께 출력하면 바로 탐지된다. 공정성 차이가 허용치 $\varepsilon$ 이하인 임계값 쌍 가운데 정확도가 가장 높은 것을 고르도록 목표를 바꾸어 고친다. (2) **적대자를 속이는 방향이 틀릴 수 있다.** 분류기 쪽 로스는 적대자의 BCE에 음수를 붙여 최대화한다. 그러면 인코더는 적대자가 「모르게」 만드는 대신 「거꾸로 확신하게」 만들어도 로스를 줄일 수 있고, 그 표현에는 보호 속성이 그대로 남는다(뒤집어 읽으면 그만이다). 적대자의 정확도가 50%보다 뚜렷이 낮아지면 이 실패다. 적대자의 목표를 0.5로 두는 로스나 그래디언트 반전 레이어를 쓰고, 학습이 끝난 표현 위에 새 분류기를 따로 학습해 보호 속성이 정말 걷혔는지 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
`FairRepresentationLearning`의 인코더를 은닉 레이어의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 파라미터를 더하고 `nn.ModuleList`로 깊이를 바꿀 수 있는 구조를 구현하라. 레이어 2개, 4개, 8개로 시험하여라. (`ReweighingMitigation`은 레이어가 없는 계산이라 이 문제의 대상이 아니다.)

</div>

??? success "풀이"
    고정해 둔 레이어를 이렇게 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서는 `for layer in self.layers: x = layer(x)`으로 돈다. 일반 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch가 모든 파라미터를 최적화에 올린다. 시험은 이렇게 한다. `for n in [2, 4, 8]: model = FairRepresentationLearning(input_dim=20, num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`. 은닉 레이어 하나($64 \times 64$)가 늘 때마다 파라미터가 4160개씩 는다.

## 정리하며

**다룬 것** — 바이어스 완화

이 구현은 바이어스 완화의 전처리·학습 중 처리·후처리에 해당하는 클래스 4개(`ReweighingMitigation`, `AdversarialDebiasing`, `FairRepresentationLearning`, `ThresholdOptimization`)를 정의하고, 그 가운데 재가중과 임계값 최적화를 지어낸 데이터에 돌려 본다.

앞의 연습문제 4개로 네 클래스의 크기, 재가중의 계산, 그리고 두 가지 실패 유형을 스스로 따져 볼 수 있다.
