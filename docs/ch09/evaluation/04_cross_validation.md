# 교차 검증

교차 검증은 표본을 학습·평가 역할로 나누는 규칙을 반복하는 방법이다. 무작위·레이어화·그룹·시간 분할 중 실제 일반화 대상을 반영하는 방식을 선택한다.

## 1. 코드

??? note "코드 (419줄)"

    ```python
    """
    교차 검증 기법
    ============================

    모델 평가를 위한 교차 검증 전략을 두루 다룬다.

    다루는 기법:
    - K겹 교차 검증
    - 레이어화 K겹
    - 하나 남기기 (LOO)
    - 시계열 분할
    - 배치 K겹
    - 되풀이 K겹
    """

    import numpy as np
    from sklearn.model_selection import (

    # ========================================================================
    # 메인
    # ========================================================================
        KFold, StratifiedKFold, LeaveOneOut, LeavePOut,
        TimeSeriesSplit, GroupKFold, RepeatedKFold,
        cross_val_score, cross_validate
    )
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.datasets import make_classification, make_regression


    class CrossValidationDemo:
        """
        여러 교차 검증 기법을 보인다
        """

        @staticmethod
        def kfold_cv(X, y, model, n_splits=5, scoring='accuracy'):
            """
            표준 K겹 교차 검증

            과정:
            1. 데이터를 크기가 같은 K개 겹으로 나눈다
            2. 겹마다:
               - K-1개 겹으로 학습한다
               - 남은 겹으로 시험한다
            3. 모든 겹의 성능을 평균한다

            장점:
            - 간단하고 널리 쓰인다
            - 데이터를 잘 쓴다

            단점:
            - (분류에서) 클래스 분포를 지키지 못할 수 있다
            - 작은 데이터셋에서는 분산이 크다

            인수:
                X: 피처
                y: 목푯값
                model: scikit-learn 모델
                n_splits: 겹의 수 (기본값 5)
                scoring: 평가할 지표
            """
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

            print(f"\nK-Fold Cross-Validation (K={n_splits})")
            print(f"Scores for each fold: {scores}")
            print(f"Mean score: {scores.mean():.4f}")
            print(f"Std deviation: {scores.std():.4f}")
            print(f"95% Confidence Interval: [{scores.mean() - 1.96*scores.std():.4f}, "
                  f"{scores.mean() + 1.96*scores.std():.4f}]")

            return scores

        @staticmethod
        def stratified_kfold_cv(X, y, model, n_splits=5, scoring='accuracy'):
            """
            레이어화 K겹 교차 검증

            과정:
            - K겹과 비슷하지만 겹마다 클래스 분포를 지킨다
            - 겹마다 클래스별 표본의 비율이 대체로 같도록 한다

            쓸 때:
            - 분류 과제
            - 불균형한 데이터셋
            - 겹에 걸쳐 클래스가 고르게 들어가기를 바랄 때

            장점:
            - 불균형한 데이터셋에 더 낫다
            - 분류에서 더 믿을 만한 추정을 준다

            인수:
                X: 피처
                y: 목푯값 (분류 레이블이어야 한다)
                model: scikit-learn 분류기
                n_splits: 겹의 수
                scoring: 평가할 지표
            """
            skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=skfold, scoring=scoring)

            print(f"\nStratified K-Fold Cross-Validation (K={n_splits})")
            print(f"Scores for each fold: {scores}")
            print(f"Mean score: {scores.mean():.4f}")
            print(f"Std deviation: {scores.std():.4f}")

            # 클래스 분포 확인
            print("\nClass distribution in original data:")
            unique, counts = np.unique(y, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"  Class {label}: {count} ({count/len(y)*100:.1f}%)")

            return scores

        @staticmethod
        def leave_one_out_cv(X, y, model, scoring='accuracy'):
            """
            하나 남기기 교차 검증 (LOO)

            과정:
            - K = n(표본의 수)인 K겹의 특수한 경우
            - 각 표본이 시험 집합으로 한 번씩 쓰인다
            - n-1개로 학습하고 1개로 시험하기를 n번 되풀이한다

            장점:
            - 학습 데이터를 최대로 쓴다
            - 결정적이다 (무작위성이 없다)
            - 바이어스가 작다

            단점:
            - 계산 비용이 크다 (반복이 n번이다!)
            - 분산이 크다
            - 큰 데이터셋에는 알맞지 않다

            쓸 때:
            - 아주 작은 데이터셋 (n < 100)
            - 계산 시간을 감당할 수 있을 때

            인수:
                X: 피처
                y: 목푯값
                model: scikit-learn 모델
                scoring: 평가할 지표
            """
            if len(X) > 100:
                print("\nWarning: LOO is computationally expensive for large datasets!")
                print(f"This will perform {len(X)} iterations.")

            loo = LeaveOneOut()
            scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)

            print(f"\nLeave-One-Out Cross-Validation")
            print(f"Number of iterations: {len(scores)}")
            print(f"Mean score: {scores.mean():.4f}")
            print(f"Std deviation: {scores.std():.4f}")

            return scores

        @staticmethod
        def time_series_split_cv(X, y, model, n_splits=5, scoring='neg_mean_squared_error'):
            """
            시계열 분할 교차 검증

            과정:
            - 시간 순서를 지킨다 (섞지 않는다!)
            - 학습 집합은 커지고 시험 집합은 굴러간다
            - 겹 1: [0:n]으로 학습하고 [n:2n]으로 시험
            - 겹 2: [0:2n]으로 학습하고 [2n:3n]으로 시험
            - etc.

            쓸 때:
            - 시계열 데이터
            - 시간적 의존이 중요할 때
            - 미래가 과거에 영향을 주면 안 될 때

            매우 중요: 데이터가 시간 순으로 정렬되어 있어야 한다!

            인수:
                X: 피처 (시간 순 정렬)
                y: 목푯값 (시간 순 정렬)
                model: scikit-learn 모델
                n_splits: 분할의 수
                scoring: 평가할 지표
            """
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)

            print(f"\nTime Series Split Cross-Validation (n_splits={n_splits})")
            print(f"Scores for each split: {scores}")
            print(f"Mean score: {scores.mean():.4f}")
            print(f"Std deviation: {scores.std():.4f}")

            print("\nSplit details:")
            for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
                print(f"  Fold {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}")

            return scores

        @staticmethod
        def group_kfold_cv(X, y, groups, model, n_splits=5, scoring='accuracy'):
            """
            배치 K겹 교차 검증

            과정:
            - 같은 배치의 표본이 학습과 시험에 함께 나오지 않게 한다
            - 개별 표본이 아니라 배치를 기준으로 나눈다

            쓸 때:
            - 데이터에 자연스러운 배치가 있을 때 (예: 환자, 회사, 실험)
            - 배치 안의 표본이 서로 독립이 아닐 때
            - 새 배치로의 일반화를 시험하고 싶을 때

            예: 여러 병원에서 모은 의료 데이터
            - 같은 환자가 학습과 시험에 함께 들어가면 안 된다
            - 새 병원으로의 일반화를 시험하고 싶다

            인수:
                X: 피처
                y: 목푯값
                groups: 표본마다의 배치 레이블 배열
                model: scikit-learn 모델
                n_splits: 분할의 수
                scoring: 평가할 지표
            """
            gkfold = GroupKFold(n_splits=n_splits)
            scores = cross_val_score(model, X, y, groups=groups, cv=gkfold, scoring=scoring)

            print(f"\nGroup K-Fold Cross-Validation (K={n_splits})")
            print(f"Number of unique groups: {len(np.unique(groups))}")
            print(f"Scores for each fold: {scores}")
            print(f"Mean score: {scores.mean():.4f}")
            print(f"Std deviation: {scores.std():.4f}")

            return scores

        @staticmethod
        def repeated_kfold_cv(X, y, model, n_splits=5, n_repeats=3, scoring='accuracy'):
            """
            되풀이 K겹 교차 검증

            과정:
            - 무작위 분할을 달리하며 K겹 교차 검증을 여러 번 한다
            - 성능 추정의 분산을 줄인다

            장점:
            - 더 튼튼한 추정
            - 특정 분할의 영향을 줄인다

            단점:
            - 계산 비용이 더 크다 (반복이 K × n_repeats번)

            쓸 때:
            - 더 믿을 만한 추정이 필요할 때
            - 데이터셋의 크기가 감당할 때

            인수:
                X: 피처
                y: 목푯값
                model: scikit-learn 모델
                n_splits: 되풀이마다의 겹 수
                n_repeats: K겹을 되풀이할 횟수
                scoring: 평가할 지표
            """
            rkfold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
            scores = cross_val_score(model, X, y, cv=rkfold, scoring=scoring)

            print(f"\nRepeated K-Fold Cross-Validation")
            print(f"K={n_splits}, Repeats={n_repeats}, Total iterations={n_splits*n_repeats}")
            print(f"Mean score: {scores.mean():.4f}")
            print(f"Std deviation: {scores.std():.4f}")
            print(f"Min score: {scores.min():.4f}")
            print(f"Max score: {scores.max():.4f}")

            return scores

        @staticmethod
        def cross_validate_detailed(X, y, model, cv=5, scoring=None):
            """
            여러 지표와 시간 측정을 곁들인 자세한 교차 검증

            학습 점수, 시험 점수, 적합과 채점에 걸린 시간을 돌려준다

            인수:
                X: 피처
                y: 목푯값
                model: scikit-learn 모델
                cv: 교차 검증 전략 또는 겹의 수
                scoring: 지표의 사전 또는 지표 하나
            """
            results = cross_validate(
                model, X, y, cv=cv, scoring=scoring,
                return_train_score=True,
                return_estimator=False
            )

            print("\nDetailed Cross-Validation Results")
            print("-" * 40)

            for key, values in results.items():
                if key.startswith('test_') or key.startswith('train_'):
                    metric_name = key.replace('test_', '').replace('train_', '')
                    print(f"{key}:")
                    print(f"  Mean: {values.mean():.4f}")
                    print(f"  Std: {values.std():.4f}")
                elif 'time' in key:
                    print(f"{key}: {values.mean():.4f}s (avg)")

            return results


    def cv_strategy_selection_guide():
        """
        알맞은 교차 검증 전략을 고르는 안내
        """
        guide = """
        교차 검증 전략 선택 안내
        =========================================

        분류 (균형 잡힘):
            → K겹 (K=5 또는 10)

        분류 (불균형):
            → 레이어화 K겹 (K=5 또는 10)
            → 믿을 만한 추정에 반드시 필요하다!

        회귀:
            → K겹 (K=5 또는 10)
            → 목푯값을 구간으로 나눈 레이어화 K겹 (심화)

        작은 데이터셋 (n < 100):
            → 하나 남기기 (계산이 감당된다면)
            → K=10, 또는 K=n인 K겹

        큰 데이터셋 (n > 10,000):
            → K=3이나 5인 K겹 (더 빠르다)
            → 한 번의 학습-시험 분할로 충분할 수 있다

        시계열:
            → 시계열 분할
            → 보통의 K겹을 절대 쓰지 마라 (시간 순서를 어긴다!)

        묶인 데이터 (예: 환자마다 표본이 여럿):
            → 배치 K겹
            → 학습과 시험 사이의 데이터 유출을 막는다

        튼튼한 추정이 필요할 때:
            → 되풀이 K겹 (K=5, 되풀이 3~10회)
            → 분류라면 레이어화를 쓴다

        일반적인 좋은 관행:
        =======================

        기본 선택:
            → 분류에는 레이어화 K겹 (K=5)
            → 회귀에는 K겹 (K=5)

        K=5와 K=10 중 무엇을 쓸까:
            → K=5: 더 빠르고 대부분의 경우에 알맞다
            → K=10: 추정이 더 안정적이다. 시간이 되면 쓴다

        언제나:
            → 재현성을 위해 random_state를 지정한다
            → shuffle=True를 쓴다 (시계열은 예외!)
            → 점수의 평균과 표준편차를 함께 보고한다

        절대 하지 말 것:
            → 불균형 분류에 보통의 K겹을 쓰기
            → 시계열 데이터를 섞기
            → 시험 집합을 어느 겹에든 넣기
        """
        print(guide)


    # 사용 예
    if __name__ == "__main__":
        print("=" * 60)
        print("CROSS-VALIDATION TECHNIQUES DEMONSTRATION")
        print("=" * 60)

        # 예시 분류 데이터 생성
        X_clf, y_clf = make_classification(
            n_samples=500, n_features=20, n_informative=15,
            n_redundant=5, n_classes=2, random_state=42
        )

        # 예시 회귀 데이터 생성
        X_reg, y_reg = make_regression(
            n_samples=500, n_features=10, random_state=42
        )

        # 모델들
        clf_model = LogisticRegression(max_iter=1000, random_state=42)
        reg_model = LinearRegression()

        demo = CrossValidationDemo()

        # 1. K겹
        print("\n" + "=" * 60)
        print("1. K-FOLD CROSS-VALIDATION (Classification)")
        print("=" * 60)
        demo.kfold_cv(X_clf, y_clf, clf_model, n_splits=5)

        # 2. 레이어화 K겹
        print("\n" + "=" * 60)
        print("2. STRATIFIED K-FOLD CROSS-VALIDATION")
        print("=" * 60)
        demo.stratified_kfold_cv(X_clf, y_clf, clf_model, n_splits=5)

        # 3. 시계열 분할
        print("\n" + "=" * 60)
        print("3. TIME SERIES SPLIT (Regression)")
        print("=" * 60)
        demo.time_series_split_cv(X_reg, y_reg, reg_model, n_splits=5)

        # 4. 전략 선택 안내
        print("\n" + "=" * 60)
        print("4. STRATEGY SELECTION GUIDE")
        print("=" * 60)
        cv_strategy_selection_guide()
    ```


??? note "전체 출력 (103줄)"

    ```
    ============================================================
    CROSS-VALIDATION TECHNIQUES DEMONSTRATION
    ============================================================

    ============================================================
    1. K-FOLD CROSS-VALIDATION (Classification)
    ============================================================

    K-Fold Cross-Validation (K=5)
    Scores for each fold: [0.89 0.89 0.81 0.87 0.86]
    Mean score: 0.8640
    Std deviation: 0.0294
    95% Confidence Interval: [0.8064, 0.9216]

    ============================================================
    2. STRATIFIED K-FOLD CROSS-VALIDATION
    ============================================================

    Stratified K-Fold Cross-Validation (K=5)
    Scores for each fold: [0.88 0.87 0.92 0.95 0.75]
    Mean score: 0.8740
    Std deviation: 0.0683

    Class distribution in original data:
      Class 0: 246 (49.2%)
      Class 1: 254 (50.8%)

    ============================================================
    3. TIME SERIES SPLIT (Regression)
    ============================================================

    Time Series Split Cross-Validation (n_splits=5)
    Scores for each split: [-8.19282661e-27 -2.33407784e-26 -2.03585725e-25 -1.02190087e-26
     -1.37782802e-26]
    Mean score: -0.0000
    Std deviation: 0.0000

    Split details:
      Fold 1: Train size=85, Test size=83
      Fold 2: Train size=168, Test size=83
      Fold 3: Train size=251, Test size=83
      Fold 4: Train size=334, Test size=83
      Fold 5: Train size=417, Test size=83

    ============================================================
    4. STRATEGY SELECTION GUIDE
    ============================================================

        교차 검증 전략 선택 안내
        =========================================
        
        분류 (균형 잡힘):
            → K겹 (K=5 또는 10)
        
        분류 (불균형):
            → 레이어화 K겹 (K=5 또는 10)
            → 믿을 만한 추정에 반드시 필요하다!
        
        회귀:
            → K겹 (K=5 또는 10)
            → 목푯값을 구간으로 나눈 레이어화 K겹 (심화)
        
        작은 데이터셋 (n < 100):
            → 하나 남기기 (계산이 감당된다면)
            → K=10, 또는 K=n인 K겹
        
        큰 데이터셋 (n > 10,000):
            → K=3이나 5인 K겹 (더 빠르다)
            → 한 번의 학습-시험 분할로 충분할 수 있다
        
        시계열:
            → 시계열 분할
            → 보통의 K겹을 절대 쓰지 마라 (시간 순서를 어긴다!)
        
        묶인 데이터 (예: 환자마다 표본이 여럿):
            → 배치 K겹
            → 학습과 시험 사이의 데이터 유출을 막는다
        
        튼튼한 추정이 필요할 때:
            → 되풀이 K겹 (K=5, 되풀이 3~10회)
            → 분류라면 레이어화를 쓴다
        
        일반적인 좋은 관행:
        =======================
        
        기본 선택:
            → 분류에는 레이어화 K겹 (K=5)
            → 회귀에는 K겹 (K=5)
        
        K=5와 K=10 중 무엇을 쓸까:
            → K=5: 더 빠르고 대부분의 경우에 알맞다
            → K=10: 추정이 더 안정적이다. 시간이 되면 쓴다
        
        언제나:
            → 재현성을 위해 random_state를 지정한다
            → shuffle=True를 쓴다 (시계열은 예외!)
            → 점수의 평균과 표준편차를 함께 보고한다
        
        절대 하지 말 것:
            → 불균형 분류에 보통의 K겹을 쓰기
            → 시계열 데이터를 섞기
            → 시험 집합을 어느 겹에든 넣기
        
    ```


## 2. 논의

KFold는 표본을 거의 같은 크기의 폴드로 나누며, StratifiedKFold는 클래스 비율을 가능한 범위에서 유지한다. 소수 클래스의 표본이 너무 적으면 모든 폴드에 그 클래스를 넣는 것은 불가능하다.

GroupKFold는 같은 사람·장치 등 같은 그룹이 양쪽 분할에 들어가지 않게 한다. 여기서 group은 학습 미니배치와 다르다. TimeSeriesSplit은 앞선 관측으로 학습하고 뒤 구간으로 평가하며, gap이나 최대 학습 크기 같은 설정도 있다.

각 폴드 안에서 전처리를 학습해야 한다. 시간 순서를 지키는 것만으로 목표 기간 겹침이나 미래 정보 피처의 누출을 해결할 수는 없다. 폴드 점수도 학습 데이터를 공유하므로 독립 반복 관측은 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
환자당 이미지가 여러 장이면 무작위 이미지 분할에 어떤 문제가 생기는가?

</div>

??? success "풀이"
    같은 환자의 유사 이미지가 양쪽에 들어가 새 환자에 대한 성능을 과대평가할 수 있다. 환자 단위 그룹 분할로 일반화 대상을 맞춘다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
scaler는 교차 검증의 어느 단계에서 fit해야 하는가?

</div>

??? success "풀이"
    각 폴드의 학습 부분에서만 fit하고 해당 검증 부분에는 transform한다. 전체 데이터를 먼저 표준화하면 검증 통계량이 학습에 들어간다. Pipeline으로 함께 묶을 수 있다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
레이어화가 표본 독립성이나 미래 예측의 타당성을 보장하는가?

</div>

??? success "풀이"
    아니다. 레이어화는 클래스 비율을 조정할 뿐 같은 개체의 상관이나 시간 순서를 처리하지 않는다. 과제에 맞는 그룹·시간 구조를 우선 반영한다.

## 정리하며

분할 방식은 새 표본·새 그룹·미래 시점 중 무엇을 예측할지에 맞춰 정한다. 전처리를 폴드 안에 넣고 교차 검증 점수의 의존성과 모델 선택 효과를 고려한다.
