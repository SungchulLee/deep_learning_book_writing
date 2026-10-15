# 25장: 차원 축소

이 장에서는 PCA의 선형 부분공간 근사부터 커널과 매니폴드 방법까지 차원 축소의 목표와 구현을 살펴본다. 분산·재구성·거리·이웃 관계는 서로 다른 보존 기준이다. 같은 데이터라도 목적과 전처리에 따라 적절한 방법이 달라진다.

## 1. PCA의 수학과 구현

- [PCA의 기초](pca/pca_fundamentals.md): 중심화, 로딩과 점수, 설명 분산과 재구성 오차
- [PCA 유도](pca/pca_derivation.md): 분산 최대화와 오차 최소화의 동등성, SVD와 해의 유일성
- [커널 PCA](pca/kernel_pca.md): 특징 공간의 중심화, 고윳값 정규화와 프리이미지 문제
- [NumPy 2차원 예제](pca/01_pca_2d_numpy.md): SVD로 방향을 구하고 투영·재구성하기
- [scikit-learn 2차원 예제](pca/02_pca_2d_sklearn.md): 적합·변환과 분산 속성, whitening과 평가 누출
- [PCA를 위한 PyTorch 기초](pca/03_pytorch_basics.md): 텐서·장치·자동 미분과 파라미터 갱신
- [PyTorch 2차원 예제](pca/04_pca_2d_pytorch.md): 배치 SVD와 고정·데이터 의존 기저의 미분
- [MNIST PCA](pca/05_pca_mnist_pytorch.md): 784차원 이미지의 저차원 표현과 같은 데이터 재구성

## 2. 거리와 이웃 기반 방법

- [MDS](manifold/mds.md): 고전 MDS의 그람 행렬 근사, 계량·비계량 stress와 PCA의 관계
- [매니폴드 방법 비교](manifold/01_manifold_comparison.md): MDS·Isomap·LLE·t-SNE의 목표와 전처리·평가 기준
- [t-SNE 자세히 살펴보기](manifold/02_tsne_deep_dive.md): 퍼플렉서티, 최적화와 초기화, 시각화의 해석 한계
- [MDS 구현 자세히 살펴보기](manifold/03_mds_deep_dive.md): 고유분해와 직접 좌표 최적화, 코드의 계산 범위

## 3. 금융 데이터 응용

- [수익률 곡선 PCA](finance/yield_curve.md): 수준·기울기·곡률의 경험적 해석과 요인 위험, 데이터·추정 조건

## 정리하며

PCA의 설명 분산은 모든 예측 정보의 보존율이 아니며, 시각화의 군집과 거리는 원래 공간의 관계로 바로 해석할 수 없다. 학습·평가 데이터를 구분하고 평균·스케일·분모를 일관되게 사용한다. 차원 수 감소와 실제 저장 공간·실행 속도도 별도로 평가해야 한다.
