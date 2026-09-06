# 1장: 사이킷런

사이킷런(scikit-learn)은 고전적 기계학습을 위한 표준적인 파이썬 인터페이스를 제공한다. 이 장에서는 환경 설정, API 설계 철학, 전처리 도구, 모델 계열, 평가 방법론, 그리고 PyTorch와의 통합 패턴을 다룬다. 사이킷런을 먼저 이해하면 딥러닝 작업 흐름에 그대로 이어지는 기본 규율과 파이프라인 사고방식을 갖출 수 있다.

## 환경 설정

파이썬 기반 기계학습과 딥러닝을 위한 개발 환경을 구성한다.

- 개발 환경 구축 -- 시스템 도구, Miniforge, VS Code, 격리된 파이썬 환경의 설치와 설정
- 기본 설정 -- 프로젝트 디렉터리 구조, 필수 라이브러리, 주피터 사용자 설정, Git 설정
- 패키지 관리 -- conda와 pip 비교, 채널, 의존성 충돌, 재현 가능한 환경 명세
- 가상 환경 -- conda와 venv를 이용한 환경 격리, 환경 내보내기와 재현
- IDE와 주피터 -- Jupyter Notebook, JupyterLab, Spyder, PyCharm, VS Code, Google Colab

## 기초

사이킷런 전체를 관통하는 API 관례, 추정기 인터페이스, 파이프라인 설계.

- API 개요 -- 통일된 `fit`/`predict`/`transform` 인터페이스와 매개변수 관례
- 추정기 인터페이스 -- `BaseEstimator`, `TransformerMixin`, `ClassifierMixin`, 사용자 정의 추정기 작성
- 파이프라인 설계 -- `Pipeline`, `ColumnTransformer`, `FeatureUnion`, 캐싱, 데이터 누출 방지

## 전처리

원시 특징을 모델이 쓸 수 있는 표현으로 변환한다.

- 스케일러 -- `StandardScaler`, `MinMaxScaler`, `RobustScaler`, 멱변환
- 인코더 -- `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`, 목표 인코딩, 해싱
- 결측값 대치 -- `SimpleImputer`, `KNNImputer`, `IterativeImputer`, 결측 지시자
- 특징 선택 -- 차원 축소를 위한 필터, 래퍼, 임베디드 방법
- 변환기 -- 다항식 확장, 이산화, PCA, t-SNE, 텍스트 특징 공학

## 모델

선형 모델부터 앙상블까지의 지도 학습 알고리즘.

- [선형 모델](models/linear.md) -- `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`
- [트리 모델](models/trees.md) -- `DecisionTreeClassifier`/`Regressor`, 분할 기준, 가지치기, 시각화
- 앙상블 방법 -- `RandomForest`, `GradientBoosting`, `AdaBoost`, 스태킹, 투표
- [SVM](models/svm.md) -- `SVC`, `SVR`, 커널 기법, 정칙화, 스케일 조정 요구사항
- [최근접 이웃](models/neighbors.md) -- `KNeighborsClassifier`/`Regressor`, 거리 척도, `BallTree`, `KDTree`
- 나이브 베이즈 -- `GaussianNB`, `MultinomialNB`, `BernoulliNB`, 조건부 독립

## 모델 선택

분할, 검증, 하이퍼파라미터 탐색에 대한 원칙 있는 접근.

- 교차 검증 -- K-Fold, 층화, LOOCV, `TimeSeriesSplit`, `GroupKFold`, 중첩 CV
- 격자 탐색 -- `GridSearchCV`, 매개변수 격자, 다중 지표 평가
- 무작위 탐색 -- `RandomizedSearchCV`, 분포 지정, 격자 대비 효율
- 베이즈 최적화 -- 대리 모델, 획득 함수, `scikit-optimize`, `Optuna`

## 평가 지표

분류, 회귀, 군집화의 모델 성능을 정량화한다.

- 분류 지표 -- 정확도, 정밀도, 재현율, F1, ROC-AUC, PR-AUC, 혼동 행렬
- 회귀 지표 -- MSE, RMSE, MAE, 결정계수, MAPE
- [군집화 지표](metrics/clustering.md) -- 실루엣, 칼린스키-하라바스, 데이비스-볼딘, 조정 랜드 지수
- 사용자 정의 스코어러 -- `make_scorer`, 업무별 손실 함수, 비대칭 비용

## PyTorch 통합

사이킷런 작업 흐름과 딥러닝을 잇는다.

- Skorch -- `NeuralNetClassifier`/`NeuralNetRegressor`로 PyTorch 모듈을 sklearn 추정기로 감싸기
- 사용자 정의 추정기 -- PyTorch 모델에 대해 `fit`/`predict`/`score`를 직접 구현하기
- 혼합 파이프라인 -- sklearn 전처리, PyTorch 모델, sklearn 평가의 결합

## 금융 응용

계량 금융을 위한 영역별 패턴.

- 팩터 모델 -- 횡단면 회귀, Fama-MacBeth, 팩터 적재로서의 특징 중요도
- 신용 평가 -- 불균형 분류, 스코어카드 개발, 규제 제약
- 시계열 교차 검증 -- 전진 검증, 퍼징, 엠바고, 조합적 퍼지 CV

## 설치 안내

플랫폼별 파이썬 개발 환경 구축 안내.

- 설치 개요 -- 모든 플랫폼에 대한 개요와 빠른 시작 링크
- macOS 설치 안내 -- macOS에서의 Homebrew, Miniforge, VS Code 설정
- Windows 설치 안내 -- Windows에서의 Chocolatey, Miniconda, VS Code 설정
- Linux 설치 안내 -- Ubuntu, Fedora, Arch에서의 Miniforge와 VS Code 설정
- macOS 빠른 참조 -- macOS용 한 줄 설치 명령
- Windows 빠른 참조 -- Windows용 한 줄 설치 명령
- Linux 빠른 참조 -- Linux용 한 줄 설치 명령
