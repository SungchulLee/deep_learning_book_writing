# 22장: 컴퓨터 비전

컴퓨터 비전의 이미지 분류·분할·객체 탐지·비디오 이해를 다룬다. 각 과제가 예측하는 대상과 입력·출력의 차이를 먼저 이해하고, 구조의 아이디어를 코드의 실제 호출 경로와 연결한다. 전체 모델, 일부 구성 요소를 보여 주는 축약 예제, 안내 문구만 있는 시작용 코드는 구현 범위가 다르다. 코드 이름이나 출력 메시지만으로 학습·성능·배포가 검증되었다고 해석하지 않는다.

## 1. 이미지 분류

이미지 전체의 클래스를 예측하는 구조를 살펴본다. 컨볼루션·잔차 연결·피처 재사용·그룹 및 깊이별 컨볼루션·attention과 모델 확장의 관계를 비교한다. 원 논문 모델과 축약 코드의 깊이·채널·head가 같은지 확인한다.

이 절은 **읽는 차례가 아니라 카탈로그**다(27쪽). 그래서 발표 순서가 아니라 **설계 아이디어**로 묶었다.

**깊이와 잔차**

- [VGGNet](image_classification/vgg.md) — 작은 컨볼루션을 반복하는 일관된 설계
- [GoogLeNet 인셉션](image_classification/03_google_net_inception.md) · [인셉션 V3](image_classification/14_inception_v3.md) — 여러 커널을 병렬로, 그리고 분해해서
- [Xception](image_classification/15_xception.md) — 인셉션을 depthwise separable 로 밀고 간다
- [ResNet](image_classification/04_resnet.md) — 잔차 경로 · [ResNeXt](image_classification/22_res_next.md) — cardinality 를 깊이·폭과 **별도 축**으로
- [DenseNet](image_classification/05_densenet.md) · [심화 DenseNet](image_classification/06_densenet_advanced.md) — 더하지 않고 **채널로 잇는다**
- [PyramidNet](image_classification/29_pyramidnet.md) — 단계 경계가 아니라 블록마다 조금씩 넓힌다

**값싸게 만들기**

- [SqueezeNet](image_classification/12_squeeze_net.md) — Fire 모듈로 웨이트를 줄인다
- [심화 MobileNet](image_classification/05_mobilenet_advanced.md) · [MobileNet V2](image_classification/11_mobilenet_v2.md) — depthwise separable, 그리고 역병목
- [ShuffleNet](image_classification/13_shuffle_net.md) — 그룹 컨볼루션 + 채널 셔플. 여기서 group 은 **채널** 그룹이지 미니배치가 아니다
- [GhostNet](image_classification/26_ghost_net.md) — 값싼 변환으로 피처를 불린다 · [MixNet](image_classification/27_mixnet.md) — 그룹마다 커널 크기를 달리

**크기를 설계하기**

- [EfficientNet](image_classification/07_efficientnet.md) · [심화 EfficientNet](image_classification/04_efficientnet_advanced.md) — 깊이·폭·해상도를 **함께** 조절하는 복합 스케일링
- [EfficientNet V2](image_classification/17_efficient_net_v2.md) — 학습 속도까지 넣고 다시 탐색한다
- [NASNet](image_classification/16_nas_net.md) — 작은 과제에서 찾은 셀을 옮긴다
- [RegNet](image_classification/21_regnet.md) — 모델 하나가 아니라 **설계 공간의 규칙**을 찾는다

**채널에 어텐션 걸기**

- [SENet](image_classification/23_se_net.md) — 채널별 통계로 게이트를 만든다
- [CBAM](image_classification/24_cbam.md) — 채널 어텐션 뒤에 공간 어텐션
- [ResNeSt](image_classification/25_res_nest.md) — 분기들을 채널 어텐션으로 결합한다(radix 는 분기 수다)

**해상도와 위치를 지키기**

- [HRNet](image_classification/28_hrnet.md) — 높은 해상도 경로를 **끝까지** 들고 간다
- [CoordConv](image_classification/30_coordconv.md) — 좌표 채널을 붙여 절대 위치를 쉽게 쓰게 한다

**CNN 을 현대화하기**

- [ConvNeXt](image_classification/06_convnext.md) — 큰 depthwise 커널과 LayerNorm 으로 ResNet 을 다시 짠다
- [ConvNeXt V2](image_classification/20_convnext_v2.md) — 마스크드 오토인코더 사전 학습과 GRN 을 함께 설계한다

여러 구조의 파라메터·MAC·활성 메모리와 실제 지연시간은 같은 척도가 아니다. 출력 클래스 수와 입력 해상도·장치·학습 조건을 맞춰 비교한다. 이 장의 개별 구조 페이지는 설계 원리를 설명하며 모두 동일한 성능 실험을 제공하지는 않는다.

## 2. 이미지 분할

분할은 픽셀 또는 객체별 영역을 예측한다. 시맨틱 분할은 픽셀의 클래스, 인스턴스 분할은 같은 클래스의 객체 구분까지, 파놉틱 분할은 객체 인스턴스와 배경 영역을 함께 다룬다. 출력 로짓·확률·정수 마스크의 차이를 구분한다.

**무엇을 푸는 문제인가**

- [이미지 분할의 기초](segmentation/segmentation_overview.md) — 과제의 갈래, 로스, IoU·Dice 와 집계

**인코더–디코더와 U-Net**

- [U-Net](segmentation/unet.md) — 줄였다 되살리고 스킵으로 잇는다
- [U-Net 구현](segmentation/31_u_net.md) — `padding=1` 과 BN 을 써서 **원 논문의 valid 컨볼루션과 다르다**
- [U-Net 이미지 분할 기초](segmentation/basic_unet_segmentation.md) — 합성 도형으로 학습·검증·체크포인트까지

**다른 구조들**

- [FCN](segmentation/32_fcn.md) — 분류기의 연산을 공간에 펴고 업샘플링한다
- [DeepLabV3](segmentation/33_deeplabv3.md) — dilated 컨볼루션과 ASPP · [PSPNet](segmentation/34_pspnet.md) — 여러 격자의 평균 풀링
- [Mask R-CNN](segmentation/mask_rcnn.md) — 탐지에 객체별 이진 마스크를 더한 **인스턴스** 분할

**실제로 돌려 보기**

- [사전 학습 모델을 이용한 이미지 분할](segmentation/pretrained_segmentation.md) — ResNet-50 인코더로 VOC2012 파인튜닝
- [의료 이미지 분할](segmentation/medical_segmentation.md) — **합성 영상**이다. 실제 환자 영상도 임상 검증도 아니다
- [이미지 분할 심화](segmentation/advanced_segmentation.md) — CBAM·로스 결합·해상도 변경·혼합 정밀도·TTA

이미지별 지표 평균과 데이터 전체의 누적 통계는 다를 수 있다. 빈 클래스·무시 픽셀·배경 포함 규칙을 명시하며, 합성 실험의 높은 겹침 점수를 실제 도메인의 유효성으로 확대하지 않는다.

## 3. 객체 탐지

탐지는 객체의 클래스·경계 상자·점수를 예측한다. 앵커 또는 위치별 거리 표현, raw 회귀 값의 디코딩, 목표 매칭·로스와 추론 후처리를 연결한다. 영역 제안과 영역별 예측을 나누는 두 단계 방식과 직접 예측하는 단일 단계 방식을 비교한다.

- [IoU와 NMS](detection/iou_nms.md): 겹침 측정·회귀 로스·중복 억제
- [YOLO](detection/yolo.md): 버전별 격자·앵커·거리 분포 표현
- [SSD](detection/ssd.md): 다중 해상도 피처·기본 상자·hard negative mining
- [Focal loss](detection/focal_loss.md): 쉬운 예제와 클래스의 웨이트
- [FCOS](detection/fcos.md): 위치별 거리·centerness와 목표 할당
- [탐지 기초 예제](detection/basic_object_detection.md): 직접 지정한 후보의 좌표·후처리
- [YOLO 추론 예제](detection/yolo_detection.md): Results의 상자·점수·클래스 읽기
- [합성 데이터 학습 구성](detection/custom_object_detection.md): 이미지·정규화 레이블·데이터 분할
- [추론 설정과 export](detection/advanced_detection.md): 크기 변경·배치·임계값·ONNX의 의미

평가 매칭 IoU·NMS IoU·신뢰도 임계값은 서로 다른 기준이다. 탐지 개수나 짧은 시간 측정으로 AP·정확도·실시간 성능을 대신하지 않는다.

## 4. 비디오 이해

비디오에서는 공간뿐 아니라 프레임의 순서·관찰 간격·유효 길이를 정의해야 한다. 프레임별 피처와 3D 컨볼루션·순환 모델·attention, RGB와 flow 또는 두 시간 해상도의 결합을 비교한다.

**비디오를 입력으로**

- [비디오 입력: 읽기·샘플링·전처리](video/video_overview.md) — 축 순서와 샘플링. $C$ 는 클래스 수가 아니라 **입력 채널 수**다
- [비디오 기초: 프레임 읽기와 샘플링](video/01_video_basics.md) — loader·sampler·crop
- [비디오 데이터 어그멘테이션과 시간적 일관성](video/04_data_augmentation_video.md) — 프레임마다 따로 흔들면 **없던 액션이 생긴다**

**시공간 컨볼루션**

- [3차원 컨볼루션과 시공간 피처](video/02_3d_convolution.md) — 세 번째 축은 깊이가 아니라 **프레임**이다
- [합성 비디오를 분류하는 3D CNN](video/03_simple_video_classifier.md) — 합성 자료라 지름길을 배울 수 있다

**움직임을 따로 보기**

- [Two-Stream 네트워크](video/two_stream.md) · [RGB와 옵티컬 플로우 결합](video/05_two_stream_network.md)
- [옵티컬 플로우의 의미와 추정 조건](video/06_optical_flow.md) — 실제 3차원 속도와 **같지 않다**

**시간을 모델에 넣기**

- [비디오의 CNN 피처와 LSTM 시간 모델](video/07_cnn_lstm_video.md) — 프레임별 예측을 평균하는 것과 다르다
- [시간 어텐션과 유효 프레임 마스크](video/09_temporal_attention.md) — 집계 웨이트를 입력에 따라 바꾼다

**두 시간 해상도와 토큰**

- [SlowFast](video/slowfast.md) · [SlowFast의 두 시간 해상도 경로](video/10_slowfast_network.md) — 느린 경로와 빠른 경로, 그리고 lateral 연결
- [비디오 트랜스포머의 토큰과 시공간 어텐션](video/11_video_transformer.md) — patch 와 tubelet

**과제로 묶기**

- [액션 인식의 과제·입력·평가 설계](video/08_action_recognition.md) — 클립 분류·동시 예측·시작과 끝은 다른 과제다
- [비디오 이해 프로젝트의 설계와 검토](video/12_video_understanding_project.md) — 구조를 고르는 것만으로 시스템이 되지 않는다

같은 원본에서 잘라낸 겹친 clip이 분할 사이에 섞이지 않도록 한다. clip과 비디오의 평가 단위, 여러 crop의 집계·비용, 온라인 추론의 관찰 지연과 미래 프레임 사용 여부도 명시한다. 일부 페이지는 개념 설계와 시작용 파일을 제공하므로 완성된 학습 시스템으로 해석하지 않는다.

## 정리하며

비전 모델을 읽을 때는 과제의 정답 단위와 입력·출력·좌표·로스·평가가 일치하는지 확인한다. 구조 아이디어와 제공된 축약 코드, 원 논문의 성능을 구분하고 동일한 데이터·학습·장치·평가 조건에서 비교한다.
