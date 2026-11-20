# 실전 예제

실전 예제: 바이어스 탐지와 완화. 합성 대출 승인 데이터에서 바이어스를 탐지하고 완화하는 것을 보인다.

머신러닝 시스템에서 공정성을 지키는 일은 윤리적으로도 마땅하고 실무적으로도 중요한 문제다. 이 꾸러미는 딥러닝 모델의 바이어스를 탐지하고, 측정하고, 완화하는 기법을 보이며, 이론의 공정성 지표를 실제 코드로 연결해 준다.

## 1. 코드

```python
"""
손에 잡히는 보기: 치우침 탐지와 눅이기
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
        # 받아들일 확률을 마땅한 인자로 잡는다
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
    치우침을 눅이지 않은 베이스라인 모형을 익힌다.

    Args:
        X_train: 익힘 결
        y_train: 익힘 이름표
        X_test: 시험 결
        y_test: 시험 이름표

    Returns:
        익힌 모형
    """
    print("\n" + "=" * 80)
    print("베이스라인 모형(치우침을 눅이지 않음)")
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
    print("모형 공정성 살피기")
    print("=" * 80)

    # 성별 공정성
    print("\n성별 공정성 지표:")
    print("-" * 40)

    # 양수 미루어 봄 비율
    male_pos_rate = np.mean(y_pred[gender == 1])
    female_pos_rate = np.mean(y_pred[gender == 0])
    print(f"양수 미루어 봄 비율(남): {male_pos_rate:.4f}")
    print(f"양수 미루어 봄 비율(여): {female_pos_rate:.4f}")
    print(f"통계적 패리티 차이: {abs(male_pos_rate - female_pos_rate):.4f}")

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
          f"가장 큼={np.max(weights):.4f}, 평균={np.mean(weights):.4f}")

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
    print("딥러닝에서의 치우침과 공정성 - 손에 잡히는 보기")
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

    # 베이스라인 모형을 익힌다
    baseline_model = train_baseline_model(X_train, y_train, X_test, y_test)
    y_pred_baseline = baseline_model.predict(X_test)

    # 베이스라인의 공정성을 살핀다
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

    print(f"\n베이스라인 모형:")
    print(f"  맞음률: {baseline_acc:.4f}")
    print(f"  통계적 패리티 차이: {baseline_spd:.4f}")

    print(f"\n고른 모형(짐 다시 매기기):")
    print(f"  맞음률: {fair_acc:.4f}")
    print(f"  통계적 패리티 차이: {fair_spd:.4f}")

    print(f"\n나아진 만큼:")
    print(f"  맞음률 바뀜: {fair_acc - baseline_acc:.4f}")
    print(f"  공정성 나아짐: {baseline_spd - fair_spd:.4f}")

    print("\n" + "=" * 80)
    print("살피기 마침")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

??? note "전체 출력 (129줄)"

    ```
    ================================================================================
    딥러닝에서의 치우침과 공정성 - 손에 잡히는 보기
    대출 받아들임 자료
    ================================================================================

    지어낸 대출 받아들임 자료를 만드는 중...
    자료를 만들었다: 표본 2000개
    받아들임 비율: 0.4290

    ================================================================================
    자료 켜의 치우침 살피기
    ================================================================================

    성별 받아들임 비율:
    ----------------------------------------
                mean  count
    gender                 
    Female  0.098802   1002
    Male    0.760521    998

    성별 받아들임 틈: 0.6617
    달리 미침 비: 0.1299


    인종 받아들임 비율:
    ----------------------------------------
                  mean  count
    race                     
    Asian     0.440083    484
    Black     0.458586    495
    Hispanic  0.387033    509
    White     0.431641    512


    성별 결 셈속:
    ----------------------------------------
    income: 남=111531.49, 여=110872.04, 차이=659.45
    credit_score: 남=583.28, 여=574.10, 차이=9.18
    debt_to_income: 남=0.50, 여=0.50, 차이=0.00

    ================================================================================
    베이스라인 모형(치우침을 눅이지 않음)
    ================================================================================

    맞음률: 0.6550

    가름 알림:
                  precision    recall  f1-score   support

               0       0.67      0.76      0.71       338
               1       0.63      0.52      0.57       262

        accuracy                           0.66       600
       macro avg       0.65      0.64      0.64       600
    weighted avg       0.65      0.66      0.65       600


    ================================================================================
    모형 공정성 살피기
    ================================================================================

    성별 공정성 지표:
    ----------------------------------------
    양수 미루어 봄 비율(남): 0.3545
    양수 미루어 봄 비율(여): 0.3688
    통계적 패리티 차이: 0.0143
    참 양수 비율(남): 0.4589
    참 양수 비율(여): 0.9677
    고른 틈 차이: 0.5089

    맞음률(남): 0.5819
    맞음률(여): 0.7276
    맞음률 차이: 0.1456

    ================================================================================
    짐 다시 매기기를 쓴 모형
    ================================================================================
    표본 짐을 셈했다.
    짐 셈속: 가장 작음=0.5636, 가장 큼=4.3886, 평균=1.0000

    맞음률: 0.6617

    가름 알림:
                  precision    recall  f1-score   support

               0       0.68      0.76      0.72       338
               1       0.63      0.54      0.58       262

        accuracy                           0.66       600
       macro avg       0.66      0.65      0.65       600
    weighted avg       0.66      0.66      0.66       600


    ================================================================================
    모형 공정성 살피기
    ================================================================================

    성별 공정성 지표:
    ----------------------------------------
    양수 미루어 봄 비율(남): 0.3679
    양수 미루어 봄 비율(여): 0.3754
    통계적 패리티 차이: 0.0075
    참 양수 비율(남): 0.4762
    참 양수 비율(여): 1.0000
    고른 틈 차이: 0.5238

    맞음률(남): 0.5953
    맞음률(여): 0.7276
    맞음률 차이: 0.1323

    ================================================================================
    모형 견주기
    ================================================================================

    베이스라인 모형:
      맞음률: 0.6550
      통계적 패리티 차이: 0.0143

    고른 모형(짐 다시 매기기):
      맞음률: 0.6617
      통계적 패리티 차이: 0.0075

    나아진 만큼:
      맞음률 바뀜: 0.0067
      공정성 나아짐: 0.0067

    ================================================================================
    살피기 마침
    ================================================================================
    ```


## 2. 논의

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 딥러닝의 핵심 아이디어를 보여 준다. 모듈로 나눈 구조 덕에 부분마다 따로 이해하고 다른 과제나 데이터에 맞추어 고치기 쉽다.

여기서 보인 기법은 더 까다로운 상황으로도 자연스럽게 확장된다. 하이퍼파라미터, 구조의 종류, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 머신러닝 과제에 대한 감이 생긴다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 딥러닝에 알맞은지 설명하라.

</div>

??? success "풀이"
    설계 선택은 구현마다 다르나 흔히 이런 것이 있다. (1) 활성화 함수 선택 -- ReLU 계열은 그래디언트가 소실되지 않아 학습이 빠르다. (2) 정규화 전략 -- 배치 정규화가 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 -- 있으면 스킵 경로를 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 형태와 데이터 타입이 기대한 대로인지 확인하는 입력 검증을 주요 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아보기 쉬운 오류 메시지를 띄워라.

</div>

??? success "풀이"
    `forward` 메서드(또는 해당 함수)의 앞부분에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 검사를 더한다. 형태를 검사하려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아보기 쉬운 오류 메시지는 디버깅을 크게 앞당기고 코드를 재사용하기 안전하게 한다.

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
실전 예제 구현을 검사하는 포괄적인 테스트 함수를 작성하라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 극단값(0, 아주 큰 수)이 든 입력 같은 경계 사례를 테스트하여라.

</div>

??? success "풀이"
    경계 조건을 두루 다루는 테스트 함수를 작성한다.
    ```python
    def test_loanapprovaldataset():
        model = LoanApprovalDataset(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 엔드투엔드 학습을 지원하는지 알려면 그래디언트 흐름을 테스트하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 손에 잡히는 보기

이 구현은 깔끔하고 읽기 쉬운 PyTorch 코드로 딥러닝의 핵심 아이디어를 보여 준다.

핵심 클래스는 `LoanApprovalDataset`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
