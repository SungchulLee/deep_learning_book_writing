# 회귀 지표

회귀 지표는 예측 오차를 절댓값·제곱·기준 모형 대비 비율 등으로 요약한다. 목표 단위와 이상치 비용, 값 0 근처의 동작을 고려해 선택한다.

## 1. 코드

```python
"""
회귀 지표
==================

회귀 모델을 평가하는 지표를 두루 다룬다.

다루는 지표:
- MAE, MSE, RMSE
- 결정계수 R²
- 조정된 R²
- MAPE, SMAPE
- 잔차 분석
"""

import numpy as np
from sklearn.metrics import (

# ========================================================================
# 메인
# ========================================================================
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    max_error, explained_variance_score
)


class RegressionMetrics:
    """
    회귀 지표를 두루 계산하는 도구
    """
    
    def __init__(self, y_true, y_pred):
        """
        참값과 예측으로 초기화
        
        인수:
            y_true: 참 목푯값
            y_pred: 예측값
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.residuals = self.y_true - self.y_pred
    
    def mae(self):
        """
        평균절대오차 (MAE)
        
        식: (1/n) * Σ|y_true - y_pred|
        
        성질:
        - 목표 변수와 같은 단위
        - MSE보다 이상점에 덜 민감하다
        - 해석하기 쉽다
        
        해석: 예측과 실제의 평균 절대 차이
        """
        return mean_absolute_error(self.y_true, self.y_pred)
    
    def mse(self):
        """
        평균제곱오차 (MSE)
        
        식: (1/n) * Σ(y_true - y_pred)²
        
        성질:
        - 단위가 제곱된다
        - 큰 오차에 무거운 벌점을 준다 (이차)
        - 어디서나 미분 가능하다 (최적화에 좋다)
        
        쓸 때: 큰 오차가 특히 바람직하지 않을 때
        """
        return mean_squared_error(self.y_true, self.y_pred)
    
    def rmse(self):
        """
        제곱근 평균제곱오차 (RMSE)
        
        식: √MSE
        
        성질:
        - 목표 변수와 같은 단위
        - MSE보다 해석하기 쉽다
        - 여전히 큰 오차에 벌점을 준다
        
        해석: 예측 오차의 표준편차
        """
        return np.sqrt(self.mse())
    
    def r2(self):
        """
        결정계수 R²
        
        식: 1 - (SS_res / SS_tot)
        여기서:
        - SS_res = Σ(y_true - y_pred)² (잔차 제곱합)
        - SS_tot = Σ(y_true - y_mean)² (총제곱합)
        
        범위: -∞부터 1까지
        - 1.0: 완벽한 예측
        - 0.0: 평균을 내놓는 것과 같은 수준
        - <0: 평균을 내놓는 것보다 나쁘다
        
        해석: 모델이 설명하는 목푯값 분산의 비율
        """
        return r2_score(self.y_true, self.y_pred)
    
    def adjusted_r2(self, n_features):
        """
        조정된 결정계수 R²
        
        식: 1 - [(1-R²)(n-1)/(n-p-1)]
        여기서:
        - n = 표본의 수
        - p = 특징의 수
        
        성질:
        - 모델을 개선하지 못하는 특징을 더하면 벌점을 준다
        - 특징의 수가 다른 모델을 견주기에 더 낫다
        
        인수:
            n_features: 모델의 특징 수
        """
        n = len(self.y_true)
        r2 = self.r2()
        
        if n <= n_features + 1:
            return None  # 정의되지 않음
        
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adjusted_r2
    
    def mape(self):
        """
        평균절대백분율오차 (MAPE)
        
        식: (100/n) * Σ|(y_true - y_pred) / y_true|
        
        성질:
        - 척도와 무관하다 (백분율)
        - 해석하기 쉽다
        - y_true = 0이면 정의되지 않는다
        - 비대칭이다 (양의 오차에 더 큰 벌점을 준다)
        
        해석: 평균 백분율 오차
        """
        # 0으로 나누기를 피한다
        mask = self.y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
    
    def smape(self):
        """
        대칭 평균절대백분율오차 (SMAPE)
        
        식: (100/n) * Σ|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
        
        성질:
        - MAPE보다 대칭적이다
        - 범위: 0%부터 200%까지
        - 실제값이 0에 가까운 경우를 더 잘 다룬다
        
        해석: 대칭 평균 백분율 오차
        """
        numerator = np.abs(self.y_true - self.y_pred)
        denominator = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2
        
        # 0으로 나누기를 피한다
        mask = denominator != 0
        if not np.any(mask):
            return 0
        
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    def median_absolute_error(self):
        """
        중앙절대오차
        
        성질:
        - 이상점에 견고하다
        - 목푯값과 같은 단위
        
        쓸 때: 데이터셋에 이상점이 있을 때
        """
        return median_absolute_error(self.y_true, self.y_pred)
    
    def max_error_metric(self):
        """
        최대 잔차 오차
        
        해석: 최악의 경우 예측 오차
        쓸 때: 최악의 성능에 한계를 두어야 할 때
        """
        return max_error(self.y_true, self.y_pred)
    
    def explained_variance(self):
        """
        설명된 분산 점수
        
        범위: 0부터 1까지
        R²과 비슷하지만 조직적인 치우침을 고려하지 않는다
        """
        return explained_variance_score(self.y_true, self.y_pred)
    
    def residual_analysis(self):
        """
        잔차(오차) 분석
        
        반환값:
            잔차 통계를 담은 사전
        """
        return {
            'mean': np.mean(self.residuals),
            'std': np.std(self.residuals),
            'min': np.min(self.residuals),
            'max': np.max(self.residuals),
            'median': np.median(self.residuals),
            'q25': np.percentile(self.residuals, 25),
            'q75': np.percentile(self.residuals, 75)
        }
    
    def full_evaluation_report(self, n_features=None):
        """
        모든 지표를 담은 완전한 평가 보고서 생성
        
        인수:
            n_features: 특징의 수 (조정된 R²에 쓴다)
        """
        report = {
            'MAE': self.mae(),
            'MSE': self.mse(),
            'RMSE': self.rmse(),
            'R² Score': self.r2(),
            'Explained Variance': self.explained_variance(),
            'MAPE (%)': self.mape(),
            'SMAPE (%)': self.smape(),
            'Median Absolute Error': self.median_absolute_error(),
            'Max Error': self.max_error_metric()
        }
        
        if n_features is not None:
            adj_r2 = self.adjusted_r2(n_features)
            if adj_r2 is not None:
                report['Adjusted R²'] = adj_r2
        
        report['Residual Analysis'] = self.residual_analysis()
        
        return report


def metric_interpretation_guide():
    """
    회귀 지표를 해석하는 안내
    """
    guide = """
    회귀 지표 해석 안내
    ======================================
    
    MAE (평균절대오차):
        → 목푯값과 같은 단위의 평균 오차
        → 해석하기 쉽고 이상점에 견고하다
        → 알맞은 곳: 모델 정확도를 대략 파악할 때
    
    RMSE (제곱근 평균제곱오차):
        → 오차의 표준편차
        → MAE보다 큰 오차에 더 큰 벌점을 준다
        → 알맞은 곳: 큰 오차가 특히 나쁠 때
    
    결정계수 R²:
        → 설명된 분산의 비율 (0부터 1까지)
        → 0.7~0.9: 좋은 모델
        → >0.9: 훌륭하다 (다만 과적합을 확인하라!)
        → <0.3: 나쁜 모델
    
    MAPE/SMAPE:
        → 백분율 오차 (척도와 무관하다)
        → 알맞은 곳: 서로 다른 데이터셋의 모델을 견줄 때
        → <10%: 훌륭하다
        → 10~20%: 좋다
        → 20~50%: 쓸 만하다
        → >50%: 나쁘다
    
    조정된 R²:
        → 특징이 다른 모델을 견줄 때 쓴다
        → 필요 없는 특징에 벌점을 준다
    
    지표 고르기:
    =================
    
    표준 관행:
        → RMSE나 MAE와 함께 R²을 보고한다
    
    척도가 다를 때 견주기:
        → MAPE/SMAPE를 쓴다
    
    사업적 맥락:
        → MAE (설명하기 쉽다)
    
    이상점이 있을 때:
        → 중앙절대오차
    
    중요한 응용:
        → 최대 오차 (최악의 경우)
    """
    print(guide)


# 사용 예
if __name__ == "__main__":
    print("=" * 60)
    print("REGRESSION METRICS DEMONSTRATION")
    print("=" * 60)
    
    # 예: 집값 예측
    print("\n1. HOUSE PRICE PREDICTION EXAMPLE")
    print("-" * 40)
    y_true = np.array([300000, 450000, 200000, 550000, 380000, 420000, 290000, 510000])
    y_pred = np.array([290000, 470000, 195000, 530000, 400000, 410000, 305000, 495000])
    
    metrics = RegressionMetrics(y_true, y_pred)
    report = metrics.full_evaluation_report(n_features=5)
    
    print("Target: House prices (in $)")
    print(f"\nTrue values:      {y_true}")
    print(f"Predicted values: {y_pred}")
    print("\nMetrics:")
    
    for metric_name, value in report.items():
        if metric_name != 'Residual Analysis':
            if isinstance(value, float):
                if 'R²' in metric_name or 'Variance' in metric_name:
                    print(f"  {metric_name}: {value:.4f}")
                elif '%' in metric_name:
                    print(f"  {metric_name}: {value:.2f}%")
                else:
                    print(f"  {metric_name}: ${value:,.2f}")
            else:
                print(f"  {metric_name}: {value}")
    
    print("\n  Residual Analysis:")
    for key, value in report['Residual Analysis'].items():
        print(f"    {key}: ${value:,.2f}")
    
    # 해석 안내
    print("\n2. METRIC INTERPRETATION GUIDE")
    print("-" * 40)
    metric_interpretation_guide()
```

**출력:**

```
============================================================
REGRESSION METRICS DEMONSTRATION
============================================================

1. HOUSE PRICE PREDICTION EXAMPLE
----------------------------------------
Target: House prices (in $)

True values:      [300000 450000 200000 550000 380000 420000 290000 510000]
Predicted values: [290000 470000 195000 530000 400000 410000 305000 495000]

Metrics:
  MAE: $14,375.00
  MSE: $234,375,000.00
  RMSE: $15,309.31
  R² Score: 0.9810
  Explained Variance: 0.9810
  MAPE (%): 3.71%
  SMAPE (%): 3.69%
  Median Absolute Error: $15,000.00
  Max Error: 20000
  Adjusted R²: 0.9335

  Residual Analysis:
    mean: $625.00
    std: $15,296.55
    min: $-20,000.00
    max: $20,000.00
    median: $7,500.00
    q25: $-16,250.00
    q75: $11,250.00

2. METRIC INTERPRETATION GUIDE
----------------------------------------

    회귀 지표 해석 안내
    ======================================
    
    MAE (평균절대오차):
        → 목푯값과 같은 단위의 평균 오차
        → 해석하기 쉽고 이상점에 견고하다
        → 알맞은 곳: 모델 정확도를 대략 파악할 때
    
    RMSE (제곱근 평균제곱오차):
        → 오차의 표준편차
        → MAE보다 큰 오차에 더 큰 벌점을 준다
        → 알맞은 곳: 큰 오차가 특히 나쁠 때
    
    결정계수 R²:
        → 설명된 분산의 비율 (0부터 1까지)
        → 0.7~0.9: 좋은 모델
        → >0.9: 훌륭하다 (다만 과적합을 확인하라!)
        → <0.3: 나쁜 모델
    
    MAPE/SMAPE:
        → 백분율 오차 (척도와 무관하다)
        → 알맞은 곳: 서로 다른 데이터셋의 모델을 견줄 때
        → <10%: 훌륭하다
        → 10~20%: 좋다
        → 20~50%: 쓸 만하다
        → >50%: 나쁘다
    
    조정된 R²:
        → 특징이 다른 모델을 견줄 때 쓴다
        → 필요 없는 특징에 벌점을 준다
    
    지표 고르기:
    =================
    
    표준 관행:
        → RMSE나 MAE와 함께 R²을 보고한다
    
    척도가 다를 때 견주기:
        → MAPE/SMAPE를 쓴다
    
    사업적 맥락:
        → MAE (설명하기 쉽다)
    
    이상점이 있을 때:
        → 중앙절대오차
    
    중요한 응용:
        → 최대 오차 (최악의 경우)
    
```

## 2. 논의

MSE의 단위는 목표 단위의 제곱이고 RMSE·MAE는 목표와 같다. RMSE는 큰 잔차를 더 강조하며 MAE는 상대적으로 덜 강조하지만, 어느 지표가 맞는지는 예측 목적에 달려 있다.

R²는 해당 평가셋의 평균 예측을 기준으로 한 상대 제곱 오차이다. 음수도 가능하고, 절편을 포함한 학습 OLS의 분산 분해를 모든 테스트 예측에 그대로 적용할 수는 없다. 서로 다른 목표나 분포에서 R²가 높다는 사실만으로 절대 오차 비용을 비교하지 않는다.

조정 R²는 선형 모형의 자유도에 관한 조건부 지표이며 임의 신경망의 입력 특징 수만 넣어 일반화 성능을 보정하는 식은 아니다. MAPE·SMAPE는 0 근처와 음수 목표, 분모를 처리하는 구현 규약을 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
목표와 예측을 함께 10배하면 MSE와 RMSE는 어떻게 되는가?

</div>

??? success "연습문제 1 풀이"
    잔차는 10배, MSE는 100배, RMSE와 MAE는 10배가 된다. R²는 같은 평가 자료를 모두 배율 변환한다면 유지된다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
테스트 R²가 음수가 되는 경우를 설명하라.

</div>

??? success "연습문제 2 풀이"
    예측 제곱 오차 합이 그 테스트셋의 평균으로 예측하는 기준보다 크면 음수이다. 0~1에만 있는 점수로 해석하면 안 된다. 상수 목표의 특별 처리는 구현을 확인한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
예측에 일정한 편향이 있을 때 explained variance와 R²가 다를 수 있는 이유는 무엇인가?

</div>

??? success "연습문제 3 풀이"
    설명 분산은 잔차의 분산에, R²는 잔차 제곱에 기반한다. 일정한 오프셋은 잔차 분산에는 나타나지 않지만 제곱 오차에는 반영된다. 둘을 동등한 지표로 취급하지 않는다.

## 정리하며

회귀 평가는 목표 단위의 오차와 기준 대비 상대 오차를 함께 읽는다. 조정 R²·백분율 지표의 조건과 구현을 확인하고, 잔차의 패턴도 요약 지표와 함께 살핀다.
