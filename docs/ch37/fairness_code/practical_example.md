# 손에 잡히는 보기

손에 잡히는 보기: 치우침 알아내기와 눅이기. 지어낸 대출 받아들임 자료에서 치우침을 알아내고 눅이는 것을 보인다.

기계 배움 얼개에서 고름을 지키는 일은 윤리로도 마땅하고 참으로도 걸린 문제다. 이 꾸러미는 깊은 배움 모형의 치우침을 알아내고, 재고, 눅이는 재주를 보이며, 이론의 고름 잣대를 손에 잡히는 코드로 이어 준다.

## 1. 코드

```python
"""
손에 잡히는 보기: 치우침 알아내기와 눅이기
지어낸 대출 받아들임 자료에서 치우침을 알아내고 눅이는 것을 보인다.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple

# ========================================================================
# 메인
# ========================================================================


class LoanApprovalDataset:
    """치우침을 심은 지어낸 대출 받아들임 자료를 만든다."""

    @staticmethod
    def generate_data(
        n_samples: int = 2000,
        bias_strength: float = 0.3,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        치우친 지어낸 대출 받아들임 자료를 만든다.

        Args:
            n_samples: 표본의 수
            bias_strength: 치우침의 세기(0~1)
            random_state: 아무렇게나 하는 씨앗

        Returns:
            결과 겨눔이 든 DataFrame
        """
        np.random.seed(random_state)

        # 결을 만든다
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.randint(20000, 200000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'employment_years': np.random.randint(0, 40, n_samples),
            'loan_amount': np.random.randint(5000, 500000, n_samples),
            'num_credit_cards': np.random.randint(0, 10, n_samples),
            'debt_to_income': np.random.uniform(0, 1, n_samples),
        }

        # 지켜야 할 됨됨이
        data['gender'] = np.random.choice(['Male', 'Female'], n_samples)
        data['race'] = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples)

        df = pd.DataFrame(data)

        # 치우침을 넣어 겨눔을 만든다
        # 받아들일 낌새를 마땅한 인자로 잡는다
        base_score = (
            (df['credit_score'] - 300) / 550 * 0.4 +
            (df['income'] - 20000) / 180000 * 0.3 +
            (1 - df['debt_to_income']) * 0.2 +
            (df['employment_years'] / 40) * 0.1
        )

        # 지켜야 할 됨됨이로 치우침을 더한다
        bias_factor = np.where(
            df['gender'] == 'Male',
            1 + bias_strength,
            1 - bias_strength
        )

        biased_score = base_score * bias_factor
        biased_score = np.clip(biased_score, 0, 1)

        # 두 값 받아들임으로 바꾼다
        df['approved'] = (biased_score > 0.5).astype(int)

        return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    모형에 쓸 자료를 채비한다.

    Args:
        df: 들임 데이터프레임

    Returns:
        (X, y, 성별 두 값, 인종 두 값) 짝
    """
    # 결을 고른다
    feature_cols = ['age', 'income', 'credit_score', 'employment_years',
                    'loan_amount', 'num_credit_cards', 'debt_to_income']

    X = df[feature_cols].values
    y = df['approved'].values

    # 살피기 좋게 지켜야 할 됨됨이를 두 값으로 바꾼다
    gender_binary = (df['gender'] == 'Male').astype(int).values
    race_binary = (df['race'] == 'White').astype(int).values

    return X, y, gender_binary, race_binary


def detect_bias_in_data(df: pd.DataFrame):
    """
    모형에 앞서 자료의 치우침을 알아낸다.

    Args:
        df: 들임 데이터프레임
    """
    print("\n" + "=" * 80)
    print("자료 켜의 치우침 살피기")
    print("=" * 80)

    # 성별 받아들임 비율
    print("\n성별 받아들임 비율:")
    print("-" * 40)
    gender_stats = df.groupby('gender')['approved'].agg(['mean', 'count'])
    print(gender_stats)

    male_rate = df[df['gender'] == 'Male']['approved'].mean()
    female_rate = df[df['gender'] == 'Female']['approved'].mean()
    print(f"\n성별 받아들임 틈: {abs(male_rate - female_rate):.4f}")
    print(f"달리 미침 비: {min(male_rate/female_rate, female_rate/male_rate):.4f}")

    # 인종 받아들임 비율
    print("\n\n인종 받아들임 비율:")
    print("-" * 40)
    race_stats = df.groupby('race')['approved'].agg(['mean', 'count'])
    print(race_stats)

    # 지켜야 할 무리마다의 결 셈속
    print("\n\n성별 결 셈속:")
    print("-" * 40)
    feature_cols = ['income', 'credit_score', 'debt_to_income']
    for col in feature_cols:
        male_mean = df[df['gender'] == 'Male'][col].mean()
        female_mean = df[df['gender'] == 'Female'][col].mean()
        print(f"{col}: 남={male_mean:.2f}, 여={female_mean:.2f}, "
              f"차이={abs(male_mean - female_mean):.2f}")


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> RandomForestClassifier:
    """
    치우침을 눅이지 않은 밑금 모형을 익힌다.

    Args:
        X_train: 익힘 결
        y_train: 익힘 이름표
        X_test: 시험 결
        y_test: 시험 이름표

    Returns:
        익힌 모형
    """
    print("\n" + "=" * 80)
    print("밑금 모형(치우침을 눅이지 않음)")
    print("=" * 80)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n맞음률: {accuracy:.4f}")
    print("\n가름 알림:")
    print(classification_report(y_test, y_pred))

    return model


def analyze_model_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gender: np.ndarray,
    race: np.ndarray
):
    """
    모형의 미루어 봄이 고른지 살핀다.

    Args:
        y_true: 참 이름표
        y_pred: 미루어 본 이름표
        gender: 성별 됨됨이(두 값)
        race: 인종 됨됨이(두 값)
    """
    print("\n" + "=" * 80)
    print("모형 고름 살피기")
    print("=" * 80)

    # 성별 고름
    print("\n성별 고름 자:")
    print("-" * 40)

    # 양수 미루어 봄 비율
    male_pos_rate = np.mean(y_pred[gender == 1])
    female_pos_rate = np.mean(y_pred[gender == 0])
    print(f"양수 미루어 봄 비율(남): {male_pos_rate:.4f}")
    print(f"양수 미루어 봄 비율(여): {female_pos_rate:.4f}")
    print(f"통계 고름 차이: {abs(male_pos_rate - female_pos_rate):.4f}")

    # 참 양수 비율
    male_mask = (gender == 1) & (y_true == 1)
    female_mask = (gender == 0) & (y_true == 1)

    if np.sum(male_mask) > 0:
        male_tpr = np.sum((y_pred == 1) & male_mask) / np.sum(male_mask)
    else:
        male_tpr = 0

    if np.sum(female_mask) > 0:
        female_tpr = np.sum((y_pred == 1) & female_mask) / np.sum(female_mask)
    else:
        female_tpr = 0

    print(f"참 양수 비율(남): {male_tpr:.4f}")
    print(f"참 양수 비율(여): {female_tpr:.4f}")
    print(f"고른 틈 차이: {abs(male_tpr - female_tpr):.4f}")

    # 무리마다의 맞음률
    male_acc = accuracy_score(y_true[gender == 1], y_pred[gender == 1])
    female_acc = accuracy_score(y_true[gender == 0], y_pred[gender == 0])
    print(f"\n맞음률(남): {male_acc:.4f}")
    print(f"맞음률(여): {female_acc:.4f}")
    print(f"맞음률 차이: {abs(male_acc - female_acc):.4f}")


def train_with_reweighing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gender_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> RandomForestClassifier:
    """
    짐 다시 매기기로 치우침을 눅이며 모형을 익힌다.

    Args:
        X_train: 익힘 결
        y_train: 익힘 이름표
        gender_train: 익힘에 쓸 성별 됨됨이
        X_test: 시험 결
        y_test: 시험 이름표

    Returns:
        익힌 모형
    """
    print("\n" + "=" * 80)
    print("짐 다시 매기기를 쓴 모형")
    print("=" * 80)

    # 표본 짐을 셈한다
    weights = np.ones(len(y_train))

    n = len(y_train)
    for gender_val in [0, 1]:
        for label_val in [0, 1]:
            mask = (gender_train == gender_val) & (y_train == label_val)
            p_observed = np.sum(mask) / n

            p_gender = np.sum(gender_train == gender_val) / n
            p_label = np.sum(y_train == label_val) / n
            p_expected = p_gender * p_label

            if p_observed > 0:
                weights[mask] = p_expected / p_observed

    print(f"표본 짐을 셈했다.")
    print(f"짐 셈속: 가장 작음={np.min(weights):.4f}, "
          f"가장 큼={np.max(weights):.4f}, 고름={np.mean(weights):.4f}")

    # 짐을 실어 익힌다
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n맞음률: {accuracy:.4f}")
    print("\n가름 알림:")
    print(classification_report(y_test, y_pred))

    return model


def main():
    """으뜸 돌리기 함수."""
    print("=" * 80)
    print("깊은 배움에서의 치우침과 고름 - 손에 잡히는 보기")
    print("대출 받아들임 자료")
    print("=" * 80)

    # 자료를 만든다
    print("\n지어낸 대출 받아들임 자료를 만드는 중...")
    df = LoanApprovalDataset.generate_data(n_samples=2000, bias_strength=0.3)
    print(f"자료를 만들었다: 표본 {len(df)}개")
    print(f"받아들임 비율: {df['approved'].mean():.4f}")

    # 자료의 치우침을 알아낸다
    detect_bias_in_data(df)

    # 자료를 채비한다
    X, y, gender, race = prepare_data(df)
    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        X, y, gender, test_size=0.3, random_state=42
    )

    # 밑금 모형을 익힌다
    baseline_model = train_baseline_model(X_train, y_train, X_test, y_test)
    y_pred_baseline = baseline_model.predict(X_test)

    # 밑금의 고름을 살핀다
    analyze_model_fairness(y_test, y_pred_baseline, gender_test, gender_test)

    # 치우침을 눅이며 익힌다
    fair_model = train_with_reweighing(
        X_train, y_train, gender_train, X_test, y_test
    )
    y_pred_fair = fair_model.predict(X_test)

    # 고른 모형을 살핀다
    analyze_model_fairness(y_test, y_pred_fair, gender_test, gender_test)

    # 모형을 견준다
    print("\n" + "=" * 80)
    print("모형 견주기")
    print("=" * 80)

    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    fair_acc = accuracy_score(y_test, y_pred_fair)

    baseline_spd = abs(
        np.mean(y_pred_baseline[gender_test == 1]) -
        np.mean(y_pred_baseline[gender_test == 0])
    )
    fair_spd = abs(
        np.mean(y_pred_fair[gender_test == 1]) -
        np.mean(y_pred_fair[gender_test == 0])
    )

    print(f"\n밑금 모형:")
    print(f"  맞음률: {baseline_acc:.4f}")
    print(f"  통계 고름 차이: {baseline_spd:.4f}")

    print(f"\n고른 모형(짐 다시 매기기):")
    print(f"  맞음률: {fair_acc:.4f}")
    print(f"  통계 고름 차이: {fair_spd:.4f}")

    print(f"\n나아진 만큼:")
    print(f"  맞음률 바뀜: {fair_acc - baseline_acc:.4f}")
    print(f"  고름 나아짐: {baseline_spd - fair_spd:.4f}")

    print("\n" + "=" * 80)
    print("살피기 마침")
    print("=" * 80)


if __name__ == "__main__":
    main()```

## 2. 논의

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 깊은 배움의 고갱이가 되는 생각을 보여 준다. 조각으로 나눈 얼개 덕에 부분마다 따로 살피고 다른 일이나 자료에 맞추어 고치기 쉽다.

여기서 보인 결은 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 얼개의 갈래, 여러 자료를 바꿔 가며 해 보면 이해가 깊어지고 기계 배움 일감에 대한 감이 몸에 붙는다.

## 연습문제

**연습문제 1.**
코드를 읽고 고갱이가 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 깊은 배움에 알맞은지 밝혀라.

??? success "연습문제 1 풀이"
    설계 판단은 짜보기마다 다르나 흔히 이런 것이 있다. (1) 살림 함수 고르기 -- ReLU 갈래는 기울기가 잦아들지 않아 익히기가 빠르다. (2) 고르게 하는 꾀 -- 묶음 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 익힘을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 그물에서 기울기가 흐르게 한다. 고른 것마다 나타내는 힘, 셈 값, 익힘의 든든함 사이의 맞바꿈을 드러낸다.

---

**연습문제 2.**
들임의 꼴과 자료 갈래가 바라는 대로인지 살피는 들임 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 들임에는 알아듣기 쉬운 어긋남 알림을 띄워라.

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 첫머리에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 종요로운 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 벌레잡기를 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

**연습문제 3.**
이 짜보기가 무너질 만한 결 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

??? success "연습문제 3 풀이"
    흔히 무너지는 결은 이렇다. (1) **기울기가 사라지거나 터짐** -- 기울기 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 켜마다 `param.grad.norm()` 적기). 기울기 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 얼개 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 익힘 잃음은 줄어드는데 살핌 잃음이 오르면 짚어낸다. 정칙화(드롭아웃, 짐 줄이기, 자료 늘리기)나 모형 크기 줄이기로 고친다. 익힘과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

**연습문제 4.**
손에 잡히는 보기 짜보기를 살피는 두루 갖춘 시험 함수를 써라. 빈 들임, 원소 하나짜리 들임, 아주 큰 들임, 그리고 끝자락 값(0, 아주 큰 수)이 든 들임 같은 가장자리 자리를 시험하여라.

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_loanapprovaldataset():
        model = LoanApprovalDataset(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 묶음
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 기울기 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    얼개가 끝에서 끝까지 익히기를 받치는지 알려면 기울기 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 손에 잡히는 보기

이 짜보기는 깔끔하고 읽기 쉬운 PyTorch 코드로 깊은 배움의 고갱이가 되는 생각을 보여 준다.

고갱이 갈래는 `LoanApprovalDataset`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
