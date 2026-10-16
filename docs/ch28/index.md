# 29장: 자기회귀 모델

자기회귀 모델은 사슬 규칙으로 결합 분포를 조건부 분포의 곱으로 표현한다. 이 장은 그 확률적 기초를 시계열 점 예측기, 문자 언어 모델, PixelCNN과 WaveNet 계열의 인과적 구조, 자기회귀 Transformer와 연결한다. 모든 예제가 확률적 생성 모델 전체를 구현하는 것은 아니므로 각 모듈의 출력·로스·생성 계약을 구분한다.

## 1. 사슬 규칙과 학습

- [자기회귀 인수분해](factorization/factorization.md): 첫 관측의 분포, 조건부 우도와 샘플링 폴리시
- [Teacher forcing](training/teacher_forcing.md): 관측 접두사에서의 우도 학습과 자유 생성의 차이
- [Scheduled sampling](training/scheduled_sampling.md): 혼합 입력 일정과 최대우도·통계적 일관성의 차이

정규화된 조건부를 계산할 수 있으면 주어진 모형의 로그확률·밀도를 계산할 수 있다. 그러나 유한 모델이 참 분포를 정확히 학습했다는 뜻은 아니다. 온도·top-k·top-p는 생성 분포를 바꾸며 낮은 검증 NLL이 모든 생성 품질을 보장하지도 않는다.

## 2. 시계열·문자 기초 예제

- [선형·비선형 AR 예측기](foundations/ar_model.md): 시차 순서, 재귀 점 예측과 안정성
- [시계열 데이터 구성](foundations/data.md): 인공 과정, 슬라이딩 창과 시간 분할
- [시계열 예측기 학습](foundations/train.md): 전체 배치 MSE와 한 스텝·다중 스텝 평가
- [문자 데이터 구성](foundations/data_2.md): 사전, 문맥당 타깃과 미등록 문자
- [문자 LSTM과 Transformer](foundations/model.md): 다음 문자 logits와 생성 상태 관리
- [문자 모델 학습](foundations/train_2.md): 교차엔트로피·perplexity와 자유 생성

기초 시계열 모듈은 조건부 잡음 분포가 없는 점 예측기다. 문자 예제는 고정 문맥에서 다음 문자 하나를 학습한다. 의존 모듈·타깃 정렬·상태 캐시 문제를 각 절에서 설명하며, 표시된 코드만으로 완성된 실행 경로가 보장되는 것은 아니다.

## 3. 이미지·컨볼루션 자기회귀 구조

- [이진 PixelCNN](pixelcnn/pixelcnn.md): A·B 마스크와 베르누이 픽셀 분포
- [PixelCNN 학습](pixelcnn/train.md): 이진 MNIST의 픽셀당 NLL과 순차 생성
- [인과적 확장 컨볼루션](wavenet/01_wavenet_basic.md): dilation과 리셉티브 필드·반복 타깃의 의미
- [게이트 WaveNet](wavenet/02_wavenet_gated.md): 게이트·잔차·스킵 경로와 회귀 타깃

픽셀 마스크는 미래 정보를 차단하면서 학습 시 모든 위치를 병렬 계산하게 한다. 생성은 픽셀별 순차 결정이다. WaveNet 예제는 오디오 분포 전체가 아닌 인공 시계열 회귀이므로 출력 채널·지평·리셉티브 필드를 정확히 읽어야 한다.

## 4. 자기회귀 Transformer

- [Transformer의 구조와 생성](transformers/transformers.md): 인과 마스크, 위치 표현, GPT 로스와 KV cache

이 절은 기본 GPT와 MQA·GQA·SwiGLU·RMSNorm·체크포인팅 등의 확장 골격을 구분한다. 패딩 키 마스크와 패딩 타깃 제외는 다르며, 캐시에는 레이어별 상태와 위치 오프셋·문맥 한계의 일관된 계약이 필요하다.

## 정리하며

자기회귀 모델에서는 관측 순서·입력과 타깃의 이동·확률 분포·로스 단위를 먼저 정한다. 한 스텝 평가와 자유 롤아웃, 점 예측과 확률적 생성, 학습 병렬성과 생성 의존성을 구분해야 구조의 장점과 구현 한계를 정확히 비교할 수 있다.
