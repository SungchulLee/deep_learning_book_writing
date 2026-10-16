# 22장: 컴퓨터 비전

컴퓨터 비전의 이미지 분류·분할·객체 탐지·비디오 이해를 다룬다. 각 과제가 예측하는 대상과 입력·출력의 차이를 먼저 이해하고, 구조의 아이디어를 코드의 실제 호출 경로와 연결한다. 전체 모델, 일부 구성 요소를 보여 주는 축약 예제, 안내 문구만 있는 시작용 코드는 구현 범위가 다르다. 코드 이름이나 출력 메시지만으로 학습·성능·배포가 검증되었다고 해석하지 않는다.

## 1. 이미지 분류

이미지 전체의 클래스를 예측하는 구조를 살펴본다. 컨볼루션·잔차 연결·피처 재사용·그룹 및 깊이별 컨볼루션·attention과 모델 확장의 관계를 비교한다. 원 논문 모델과 축약 코드의 깊이·채널·head가 같은지 확인한다.

- [VGGNet](image_classification/vgg.md): 작은 컨볼루션을 반복하는 설계
- [ResNet](image_classification/04_resnet.md): 잔차 경로와 채널·stride 정렬
- [DenseNet](image_classification/05_densenet.md): 채널 연결을 통한 피처 재사용
- [EfficientNet](image_classification/07_efficientnet.md): 깊이·너비·입력 해상도의 확장

여러 구조의 파라메터·MAC·활성 메모리와 실제 지연시간은 같은 척도가 아니다. 출력 클래스 수와 입력 해상도·장치·학습 조건을 맞춰 비교한다. 이 장의 개별 구조 페이지는 설계 원리를 설명하며 모두 동일한 성능 실험을 제공하지는 않는다.

## 2. 이미지 분할

분할은 픽셀 또는 객체별 영역을 예측한다. 시맨틱 분할은 픽셀의 클래스, 인스턴스 분할은 같은 클래스의 객체 구분까지, 파놉틱 분할은 객체 인스턴스와 배경 영역을 함께 다룬다. 출력 로짓·확률·정수 마스크의 차이를 구분한다.

- [분할 개요](segmentation/segmentation_overview.md): 과제·로스·IoU·Dice와 집계
- [U-Net](segmentation/unet.md): 인코더-디코더와 스킵 연결
- [Mask R-CNN](segmentation/mask_rcnn.md): 영역별 탐지와 이진 마스크
- [기본 U-Net 예제](segmentation/basic_unet_segmentation.md): 합성 이미지의 이진 분할
- [사전 학습 인코더 예제](segmentation/pretrained_segmentation.md): 전이 학습과 레이블·평가 처리
- [의료 분할 예제](segmentation/medical_segmentation.md): 합성 자료와 픽셀 지표의 해석
- [고급 분할 예제](segmentation/advanced_segmentation.md): attention·경계 로스·다중 크기와 TTA의 범위

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

- [입력 개요](video/video_overview.md): 읽기·축 순서·샘플링·전처리
- [기초 예제](video/01_video_basics.md): loader·sampler·crop과 합성 시연
- [3D 컨볼루션](video/02_3d_convolution.md): 시공간 커널·풀링·고정 head의 입력
- [간단한 비디오 분류기](video/03_simple_video_classifier.md): 합성 데이터 학습과 지름길의 해석
- [Two-Stream](video/two_stream.md): RGB·flow의 정렬과 결합
- [SlowFast](video/slowfast.md): 두 시간 해상도와 lateral 연결
- [비디오 트랜스포머](video/11_video_transformer.md): 토큰·위치 정보·attention 설계
- [프로젝트 설계](video/12_video_understanding_project.md): 데이터·모델·평가·추론의 계약

같은 원본에서 잘라낸 겹친 clip이 분할 사이에 섞이지 않도록 한다. clip과 비디오의 평가 단위, 여러 crop의 집계·비용, 온라인 추론의 관찰 지연과 미래 프레임 사용 여부도 명시한다. 일부 페이지는 개념 설계와 시작용 파일을 제공하므로 완성된 학습 시스템으로 해석하지 않는다.

## 정리하며

비전 모델을 읽을 때는 과제의 정답 단위와 입력·출력·좌표·로스·평가가 일치하는지 확인한다. 구조 아이디어와 제공된 축약 코드, 원 논문의 성능을 구분하고 동일한 데이터·학습·장치·평가 조건에서 비교한다.
