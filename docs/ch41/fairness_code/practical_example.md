# 실전 예제

실전 예제: 바이어스 탐지와 완화. 합성 대출 승인 데이터에서 바이어스를 탐지하고 완화하는 것을 보인다.

머신러닝 시스템에서 공정성을 지키는 일은 윤리적으로도 마땅하고 실무적으로도 중요한 문제다. 이 예제는 레이블에 성별 바이어스를 심은 대출 데이터를 만들고, 데이터의 바이어스를 잰 뒤, 랜덤 포레스트 베이스라인과 재가중을 쓴 모델의 정확도와 공정성 지표를 견준다. 뉴럴 네트워크가 아니라 scikit-learn 모델을 쓴다.

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

데이터 단계의 바이어스는 아주 크다. 승인 비율이 남성 0.761, 여성 0.099로 차이가 0.6617, 이질적 영향 비가 0.1299다. 그런데 소득·신용 점수·부채 비율의 평균은 성별 사이에 거의 같다. 피처는 성별과 무관하게 뽑았고, 바이어스는 점수에 곱한 계수 $1 \pm 0.3$으로만 레이블에 들어갔기 때문이다.

베이스라인 모델은 성별을 입력으로 받지 않으며, 성별을 짐작하게 하는 대리 피처도 없다. 그래서 양성 예측 비율이 남성 0.3545, 여성 0.3688로 거의 같고 통계적 패리티 차이는 0.0143에 그친다. 대신 레이블이 성별에 크게 기대므로 정확도는 0.6550에 머물고, 기회 균등 차이가 0.5089로 크다. 승인된 여성은 거의 다 승인으로 예측하고(TPR 0.9677) 승인된 남성은 절반 남짓만 맞힌다(TPR 0.4589). 곧 이 모델은 레이블에 심은 바이어스를 따라 배우지 못한 것이고, 어느 지표로 재느냐에 따라 「공정하다」와 「불공정하다」가 갈린다. 실제 데이터에는 우편번호처럼 보호 속성을 대신 드러내는 피처가 흔하므로, 보호 속성을 빼는 것만으로 이런 결과가 나오리라 기대하면 안 된다.

재가중을 쓴 모델은 정확도 0.6617, 패리티 차이 0.0075, 기회 균등 차이 0.5238이다. 출력의 「공정성 나아짐 0.0067」은 이 패리티 차이의 변화다. 그러나 시험 표본이 600개(집단마다 약 300개)이면 양성 예측 비율 차이의 표준오차만 해도 대략 0.04이므로, 0.0067의 변화는 분할 하나에서 생긴 잡음과 가를 수 없고 기회 균등 차이는 오히려 조금 커졌다. 모델이 이미 성별을 보지 못하므로 성별과 레이블의 상관을 걷어내는 재가중이 바꿀 것이 거의 없었던 셈이다. 효과를 말하려면 분할과 시드를 바꾸어 여러 번 재고 그 스프레드를 보고해야 한다.

코드에 대해 하나 덧붙인다. `analyze_model_fairness`는 인종 배열도 받도록 되어 있으나 `main`이 그 자리에 `gender_test`를 넘기고 함수 안에서도 쓰지 않으므로, 인종에 대한 모델 공정성은 이 예제에서 재지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심 설계 선택을 짚어라. 구현에서 고른 것 셋을 들고, 저마다 왜 그렇게 했는지와 그 한계를 설명하라.

</div>

??? success "풀이"
    (1) 바이어스를 점수에 곱하는 계수 $1 \pm 0.3$으로 레이블에만 심는다 -- 데이터 바이어스와 모델 바이어스를 가르기 쉽지만, 피처가 성별과 무관해 대리 변수 문제가 생기지 않으므로 현실보다 쉬운 상황이 된다. (2) 모델 입력에서 보호 속성을 뺀다 -- 「보호 속성을 모르면 공정하다」는 생각을 시험하는 셈이다. 이 데이터에서는 패리티 차이가 작게 나오지만 그것은 대리 피처가 없어서이고, 기회 균등 차이는 0.5를 넘는다. (3) 재가중을 성별 × 레이블 네 칸의 빈도로 계산한다 -- 구현이 간단하고 평균 가중치가 1로 보존되지만, 모델이 이미 성별을 보지 못하면 바꿀 것이 거의 없다. 덧붙여, 학습·시험 분할을 한 번만 하므로 두 모델의 차이가 잡음인지 가릴 수 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 예제는 모델에 성별을 넣지 않았다. 피처 가운데 하나를 성별과 상관되게 바꾸면(예: 남성의 소득 평균을 높인다) 베이스라인의 통계적 패리티 차이는 어떻게 바뀔 것이라 예상하는가? 재가중은 그때 어떤 역할을 하는가?

</div>

??? success "풀이"
    소득이 성별의 대리 변수가 되므로, 모델은 소득을 통해 레이블에 심은 성별 바이어스를 일부 배울 수 있다. 그러면 남성의 양성 예측 비율이 올라가고 패리티 차이가 지금의 0.0143보다 커질 것이다. 상관이 강할수록 차이는 데이터 단계의 0.66 쪽으로 다가간다. 이때 재가중은 성별과 레이블이 독립인 것처럼 표본 가중치를 매기므로, 「소득이 높으면 승인」이라는 관계 가운데 성별에서 온 부분의 무게를 줄인다. 지금 예제에서는 재가중이 거의 아무것도 바꾸지 못했지만, 대리 피처가 있으면 효과가 드러날 여지가 생긴다. 이 예상은 실제로 데이터를 바꾸어 여러 시드로 돌려 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 실패할 만한 유형 둘을 들고, 저마다 어떻게 탐지하고 고칠지 설명하라.

</div>

??? success "풀이"
    (1) **한 번의 분할로 효과를 판정한다.** 패리티 차이 0.0143 → 0.0075 같은 변화는 시험 표본 600개의 표준오차(약 0.04)보다 훨씬 작다. 분할의 `random_state`와 모델 시드를 바꾸어 여러 번 돌리고 차이의 평균과 스프레드를 보고하면 탐지된다. 스프레드보다 작은 차이는 효과로 적지 않는다. (2) **분석 함수가 엉뚱한 배열을 받는다.** `main`이 `analyze_model_fairness`의 인종 자리에 `gender_test`를 넘긴다. 함수가 그 인자를 쓰지 않아 오류도 경고도 없다. 인자를 쓰지 않으면 알려 주는 린터나, 인종별 지표를 실제로 출력해 두 결과가 성별과 똑같이 나오는지 보는 것으로 탐지한다. 인종 배열도 `train_test_split`에 함께 넣어 `race_test`를 만들고 그것을 넘기도록 고친다.

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
    위 틀은 뉴럴 네트워크용이라 `LoanApprovalDataset`에는 그대로 맞지 않는다. 이 예제에서 시험할 것은 이렇다. `generate_data`가 요청한 표본 수와 열을 돌려주는지, 같은 `random_state`이면 같은 데이터를 내는지, `bias_strength=0`이면 성별 승인 비율의 차이가 표본 오차 수준으로 줄어드는지, `bias_strength`를 키우면 차이가 단조롭게 커지는지 본다. 경계 사례로는 표본이 하나뿐이거나 한 성별의 승인 비율이 0인 경우를 넣는다. 후자에서 `detect_bias_in_data`의 이질적 영향 비는 0으로 나누기 경고와 함께 `inf`나 `nan`을 낼 수 있으므로 그 처리를 확인한다.

## 정리하며

**다룬 것** — 실전 예제

이 예제는 레이블에 성별 바이어스를 심은 합성 대출 데이터에서 데이터 단계의 큰 바이어스(승인 비율 차이 0.66)를 재고, 성별을 보지 않는 랜덤 포레스트가 패리티 차이는 작지만 기회 균등 차이는 큰 모델이 됨을 보인다. 재가중의 효과는 이 설정에서 잡음과 가를 수 없을 만큼 작았다.

핵심 클래스는 `LoanApprovalDataset`이며, 앞의 연습문제 4개로 그 설계와 약점을 스스로 따져 볼 수 있다.
