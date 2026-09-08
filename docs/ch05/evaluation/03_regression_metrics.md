# 회귀 지표

회귀 지표는 예측한 연속값과 실제값의 어긋남을 잰다. 흔히 쓰는 지표로 MAE, MSE, RMSE, 결정계수 $R^2$이 있으며 이상점에 대한 민감도와 해석하기 쉬움이 저마다 다르다. 어떤 지표를 고를지는 큰 오차에 불균형하게 벌점을 주어야 하는지, 척도와 무관한 비교가 필요한지에 달렸다.

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

## 2. 논의

MAE와 RMSE는 모두 목표 변수와 같은 단위로 재지만, RMSE는 제곱하므로 큰 오차에 더 무거운 벌점을 준다. 이상점이 있을 때에는 MAE와 중앙절대오차가 더 튼튼한 추정을 준다. 설명된 분산의 비율을 나타내는 $R^2$은 목표의 척도가 달라도 비교할 수 있게 해 준다.

조정된 $R^2$은 예측력을 높이지 못하는 특징을 더한 모델에 벌점을 주므로, 특징의 수가 다른 모델을 견줄 때 $R^2$보다 믿을 만하다. MAPE와 SMAPE는 척도와 무관한 백분율 오차를 주지만 참값이 0에 가까우면 정의되지 않거나 믿을 수 없다.

잔차 분석은 조직적인 양상을 드러내어 요약 지표를 보완한다. 잔차가 예측값에 대해 어떤 추세를 보인다면 모델이 설명하지 못한 구조가 있다는 뜻이다. 잔차가 이분산적이라면(예측의 크기에 따라 분산이 달라진다면) 로그 변환이나 가중 회귀가 도움이 될 수 있다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

## 정리하며

**다룬 것** — 회귀 지표

MAE와 RMSE는 모두 목표 변수와 같은 단위로 재지만, RMSE는 제곱하므로 큰 오차에 더 무거운 벌점을 준다.

핵심 클래스는 `RegressionMetrics`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
