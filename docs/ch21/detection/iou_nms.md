# IoU와 비최대 억제(NMS)

객체 탐지에서는 경계 상자의 겹침을 측정하고, 같은 객체에 대한 중복 후보를 정리해야 한다. IoU는 두 영역의 겹침 비율이며 NMS는 점수와 겹침에 따른 후보 선택 규칙이다. 높은 IoU가 곧 같은 객체라는 뜻은 아니고, NMS가 정답 객체를 알고 선택하는 것도 아니다.

## 1. IoU의 정의와 좌표

면적이 양수인 두 상자 A, B에 대해 IoU는 다음과 같다.

$$\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}.$$

```
         Box A              Box B           Intersection        Union
        ┌──────┐          ┌──────┐          ┌────┐         ┌──────────┐
        │      │          │      │          │████│         │          │
        │  ┌───┼──────────┼───┐  │    →     │████│    /    │          │
        │  │   │          │   │  │          └────┘         │          │
        └──┼───┘          └───┼──┘                         └──────────┘
           └──────────────────┘
```

이 절은 연속 좌표 `(x1,y1,x2,y2)`를 사용한다. 너비는 x2−x1, 높이는 y2−y1이며 픽셀 인덱스의 양 끝을 포함하는 관례처럼 1을 더하지 않는다. 유한한 좌표와 x2>x1, y2>y1을 가정한다. 분모에 ε를 더하는 것은 수치 안정화이지 잘못된 좌표를 유효한 상자로 바꾸는 검사가 아니다.

교집합의 왼쪽 위는 두 시작 좌표의 최댓값, 오른쪽 아래는 끝 좌표의 최솟값이다. 교집합 너비·높이를 각각 0 이상으로 잘라 곱하고 합집합은 두 면적의 합에서 교집합을 뺀다. 양의 면적 상자에서 IoU는 0~1이고 대칭이다. 경계만 접촉해도 교집합 면적이 0이므로 IoU=0이다. ε 없는 정의에서 IoU=1이면 두 상자가 일치한다.

두 상자를 같은 비율로 확대하거나 같은 양만큼 이동해도 IoU는 같다. 공통 좌표 축별 양의 스케일 변환에도 면적 배수가 약분된다. 한 상자만 바꾸거나 고정 ε의 영향을 무시할 수 없는 경우까지 불변인 것은 아니다. 큰 물체와 작은 물체에서 같은 픽셀 이동이 서로 다른 IoU를 만들 수 있다.

```python
import torch
from typing import Union

def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    두 상자 모음 사이의 짝짓기 겹침 비를 셈한다.
    
    여기 짠 것은 효율을 위해 배치 셈을 받쳐 준다.
    
    인수:
        boxes1: xyxy 꼴의, 꼴이 (N, 4)인 텐서
        boxes2: xyxy 꼴의, 꼴이 (M, 4)인 텐서
        eps: 0으로 나누지 않도록 하는 작은 엡실론
        
    반환값:
        짝짓기 겹침 비 값을 담은, 꼴이 (N, M)인 텐서
        
    보기:
        >>> boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
        >>> boxes2 = torch.tensor([[5, 5, 15, 15], [20, 20, 30, 30]], dtype=torch.float32)
        >>> iou = box_iou(boxes1, boxes2)
        >>> print(iou)
        tensor([[0.1429, 0.0000],
                [1.0000, 0.0000]])
    """
    # 자리표 뽑아내기
    # boxes1: 퍼뜨리기를 위해 (N, 4) -> (N, 1, 4)
    # boxes2: 퍼뜨리기를 위해 (M, 4) -> (1, M, 4)
    x1_1, y1_1, x2_1, y2_1 = boxes1.unsqueeze(1).unbind(-1)
    x1_2, y1_2, x2_2, y2_2 = boxes2.unsqueeze(0).unbind(-1)
    
    # 교집합 자리표 셈하기
    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    
    # 교집합 넓이 셈하기(안 겹치면 0으로 묶기)
    inter_width = (inter_x2 - inter_x1).clamp(min=0)
    inter_height = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_width * inter_height
    
    # 낱낱의 넓이 셈하기
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 합집합 넓이 셈하기
    union_area = area1 + area2 - inter_area
    
    # 겹침 비 셈하기
    iou = inter_area / (union_area + eps)
    
    return iou

def box_iou_single(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    낱낱의 두 상자 사이 겹침 비를 셈한다.
    
    인수:
        box1: xyxy 꼴의, 꼴이 (4,)인 텐서
        box2: xyxy 꼴의, 꼴이 (4,)인 텐서
        
    반환값:
        실수로 된 겹침 비 값
    """
    return box_iou(box1.unsqueeze(0), box2.unsqueeze(0)).item()
```

`box_iou`는 N×4와 M×4를 받아 모든 쌍의 N×M 행렬을 만든다. 아래 `batch_iou`는 이름과 달리 이미지 배치 전체의 모든 쌍을 계산하지 않는다. 같은 행끼리 대응된 N개 상자의 IoU를 계산한다. 예측과 정답의 매칭 규칙을 먼저 정해야 한다.

```python
def batch_iou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor
) -> torch.Tensor:
    """
    짝지어진 상자 사이의 겹침 비를 셈한다(짝짓기가 아님).
    
    어림이 이미 목표에 짝지어진 익히기 동안 쓸모 있다.
    
    인수:
        pred_boxes: xyxy 꼴의 (N, 4) 어림 상자
        target_boxes: xyxy 꼴의 (N, 4) 목표 상자
        
    반환값:
        (N,) 어림-목표 짝마다의 겹침 비 값
    """
    # 교집합
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    # 넓이
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                  (target_boxes[:, 3] - target_boxes[:, 1])
    
    # 합집합
    union_area = pred_area + target_area - inter_area
    
    return inter_area / (union_area + 1e-7)
```

평가에서 예측과 정답을 매칭하는 IoU 임계값, NMS에서 후보끼리 비교하는 IoU 임계값, 후보를 거르는 신뢰도 임계값은 목적이 다르다. 0.5나 0.75를 쓰는 평가가 있어도 특정 IoU 구간만으로 객체의 동일성이나 탐지의 품질을 보편적으로 판정할 수는 없다.

## 2. 상자 회귀 손실의 변형

보통 IoU 손실은 1−IoU다. 서로 떨어져 겹치지 않는 구간에서는 IoU가 계속 0이므로 상자를 움직일 방향을 주지 못한다. GIoU는 두 상자를 감싸는 최소 축 정렬 상자 C를 이용한다.

$$\operatorname{GIoU}=\operatorname{IoU}-\frac{|C|-|A\cup B|}{|C|}.$$

```python
def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    짝지어진 상자 사이의 넓힌 겹침 비를 셈한다.
    
    GIoU = 겹침 비 - (감싸는 상자의 넓이 - 합집합) / 감싸는 상자의 넓이
    
    인수:
        boxes1: xyxy 꼴의 (N, 4) 상자
        boxes2: xyxy 꼴의 (N, 4) 상자
        
    반환값:
        (N,) GIoU 값
        
    참고:
        Rezatofighi 외, "Generalized Intersection over Union", CVPR 2019
    """
    # 보통의 겹침 비 셈하기
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + eps)
    
    # 감싸는 상자
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + eps)
    
    return giou
```

GIoU는 감싸는 영역에서 비어 있는 면적을 벌점으로 주며 범위는 −1~1이다. 다만 모든 상황에서 유용한 기울기를 보장하지는 않는다. 한 상자가 다른 상자 안에 있으면 C가 합집합과 같아 GIoU=IoU이고, 두 상자가 일치하지 않아도 이 등식이 성립한다.

DIoU는 중심 거리, CIoU는 여기에 가로세로비 차이를 반영한다. 중심 거리 ρ와 C의 대각선 길이 c에 대해 다음과 같다.

$$\operatorname{DIoU}=\operatorname{IoU}-\rho^2/c^2,$$

$$\operatorname{CIoU}=\operatorname{DIoU}-\alpha v,\qquad
v=\frac{4}{\pi^2}\left(\arctan\frac{w_A}{h_A}-\arctan\frac{w_B}{h_B}\right)^2,\qquad
\alpha=\frac{v}{1-\operatorname{IoU}+v}.$$

정확히 일치하면 α 식이 0/0이 될 수 있어 해당 항을 0으로 정의하거나 수치 안정화한다. 회귀에서는 보통 1−DIoU 또는 1−CIoU를 최소화한다. 중심·비율 정보를 추가하는 목적과 특정 데이터에서 수렴이나 AP가 개선되는지는 구분한다. [DIoU 논문](https://ojs.aaai.org/index.php/AAAI/article/view/6999)의 실험 결과를 모든 구조의 성능 보장으로 확대하지 않는다.

```python
def complete_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    짝지어진 상자 사이의 온전한 겹침 비(CIoU)를 셈한다.
    
    CIoU는 겹침, 가운데점 거리, 가로세로비를 헤아린다.
    
    인수:
        boxes1: xyxy 꼴의 (N, 4) 어림 상자
        boxes2: xyxy 꼴의 (N, 4) 목표 상자
        
    반환값:
        (N,) CIoU 값
        
    참고:
        Zheng 외, "Distance-IoU Loss", AAAI 2020
    """
    # 겹침 비 셈하기
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / (union_area + eps)
    
    # 가운데점 거리
    center1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
    center1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
    center2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
    center2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    center_dist_sq = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
    
    # 감싸는 상자의 대각선
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # 가로세로비 항
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]
    
    v = (4 / (torch.pi ** 2)) * \
        (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    
    # CIoU
    ciou = iou - center_dist_sq / (enclose_diag_sq + eps) - alpha * v
    
    return ciou
```

위 구현은 `alpha`를 `no_grad` 안에서 계산한다. 따라서 α를 상수로 취급한 기울기를 사용하며, 표시한 CIoU 수식 전체를 그대로 미분한 결과와는 다르다. 양의 너비·높이와 적절한 dtype도 필요하다.

## 3. NMS로 중복 후보 정리하기

탐지기는 같은 객체 근처에 여러 후보를 출력할 수 있다. NMS는 점수가 가장 높은 후보를 남기고, 그것과 많이 겹치는 낮은 점수 후보를 제거하는 탐욕적 규칙이다. 점수는 모델의 예측이므로 가장 높은 점수가 실제로 가장 정확한 상자라는 보장은 없다. 겹친 서로 다른 객체가 함께 제거될 수도 있다.

```
Before NMS:                     After NMS:
┌─────┐                         
│ 0.9 │ ← High confidence       ┌─────┐
└─────┘                         │ 0.9 │ ← Keep best
  ┌─────┐                       └─────┘
  │ 0.8 │ ← Also high           
  └─────┘                       Removed: 0.8, 0.7 (overlapping)
    ┌─────┐
    │ 0.7 │ 
    └─────┘
```

```
Algorithm: Non-Maximum Suppression
Input: Boxes B, Scores S, IoU threshold τ
Output: Kept indices K

1. Sort boxes by score in descending order
2. Initialize K = []
3. While B is not empty:
   a. Select box with highest score, add to K
   b. Compute IoU of this box with all remaining boxes
   c. Remove boxes with IoU > τ from B
4. Return K
```

일반적으로 이미지별·클래스별로 적용한다. 클래스가 다른 후보를 함께 억제하는 class-agnostic NMS도 별도의 설계 선택이다. 같은 이미지인지, 같은 클래스인지와 좌표가 어느 이미지 크기를 기준으로 하는지 명시한다.

```python
def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    알아낸 상자에 최대가 아닌 것 누르기를 한다.
    
    이것은 배움을 위한 순수 파이썬 짜기이다.
    실전에서는 C++로 다듬은 torchvision.ops.nms를 쓴다.
    
    인수:
        boxes: xyxy 꼴의 (N, 4) 두름 상자
        scores: (N,) 믿음도 점수
        iou_threshold: 누르기용 겹침 비 문턱값
        
    반환값:
        남길 상자의 번호
        
    복잡도:
        시간: 상자 N개에 O(N² × 4)
        공간: 번호에 O(N)
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    # 점수로 정렬 (내림차순)
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    
    while sorted_indices.numel() > 0:
        # 점수가 가장 높은 상자 남기기
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        if sorted_indices.numel() == 1:
            break
        
        # 남은 번호 얻기
        remaining_indices = sorted_indices[1:]
        
        # 지금 상자와 남은 상자 사이 겹침 비 셈하기
        current_box = boxes[current_idx].unsqueeze(0)
        remaining_boxes = boxes[remaining_indices]
        
        ious = box_iou(current_box, remaining_boxes).squeeze(0)
        
        # 겹침 비가 문턱값 아래인 상자 남기기
        mask = ious < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    갈래마다 따로 NMS를 한다.
    
    갈래가 다른 상자끼리는 서로 누르지 않는다.
    
    인수:
        boxes: (N, 4) 두름 상자
        scores: (N,) 믿음도 점수
        labels: (N,) 갈래 이름표
        iou_threshold: 겹침 비 문턱값
        
    반환값:
        남길 상자의 번호
    """
    # 갈래끼리 누르지 않도록 갈래별로 상자를 밀기
    max_coord = boxes.max()
    offsets = labels.float() * (max_coord + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    return nms(boxes_for_nms, scores, iou_threshold)
```

위 직접 구현은 `ious < iou_threshold`만 남기므로 임계값과 **같은** IoU도 제거한다. 반면 [torchvision NMS](https://docs.pytorch.org/vision/master/generated/torchvision.ops.nms.html)는 임계값보다 **큰** IoU를 제거한다. 동점 점수의 선택 순서와 부동소수점 오차도 경계 사례에 영향을 줄 수 있다.

직접 구현한 `batched_nms`의 클래스별 좌표 이동은 비음수 좌표를 가정한다. 음수 좌표가 있으면 `max_coordinate+1` 간격만으로 클래스별 영역 분리를 보장하지 못하고, 빈 입력에서는 `boxes.max()`가 실패한다. 클래스 인덱스·좌표 범위·정밀도를 확인해야 한다.

```python
import torchvision.ops as ops

# 보통의 NMS
keep_indices = ops.nms(boxes, scores, iou_threshold=0.5)

# 배치 NMS(갈래마다)
keep_indices = ops.batched_nms(boxes, scores, labels, iou_threshold=0.5)

# 결과 거르기
final_boxes = boxes[keep_indices]
final_scores = scores[keep_indices]
final_labels = labels[keep_indices]
```

라이브러리 연산을 사용해도 입력 검증·이미지별 처리·점수 임계값의 선택은 호출하는 쪽의 책임이다. 실제 설치된 torch·torchvision과 지원 장치가 맞아야 한다.

## 4. Soft-NMS와 거리 기반 NMS

Soft-NMS는 후보를 즉시 삭제하는 대신 겹침에 따라 점수를 낮추고 갱신된 점수의 최댓값을 다시 선택한다. 가우스 방식은 σ>0에서 다음과 같다.

$$s_i\leftarrow s_i\exp[-\operatorname{IoU}(M,b_i)^2/\sigma].$$

선형 방식은 지정 임계값 이상에서 s_i에 1−IoU를 곱한다. 점수 감소와 최종 점수 필터링은 서로 다른 단계다.

```python
def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001,
    sigma: float = 0.5,
    method: str = 'gaussian'
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    부드러운 최대가 아닌 것 누르기.
    
    겹치는 상자를 없애는 대신 그 점수를 줄인다.
    물체가 겹치는 붐비는 장면에 더 낫다.
    
    인수:
        boxes: xyxy 꼴의 (N, 4) 두름 상자
        scores: (N,) 믿음도 점수
        iou_threshold: 선형 방식의 겹침 비 문턱값
        score_threshold: 남길 최소 점수
        sigma: 가우스 줄임 매개변수
        method: 'linear' 또는 'gaussian'
        
    반환값:
        (남긴 상자, 새 점수) 튜플
        
    참고:
        Bodla 외, "Soft-NMS", ICCV 2017
    """
    boxes = boxes.clone()
    scores = scores.clone()
    
    indices = torch.arange(boxes.shape[0], device=boxes.device)
    keep = []
    
    while scores.numel() > 0:
        # 점수가 가장 높은 상자 얻기
        max_idx = scores.argmax()
        keep.append(indices[max_idx])
        
        if scores.numel() == 1:
            break
        
        # 지금 상자를 얻어 나머지와 겹침 비 셈하기
        current_box = boxes[max_idx:max_idx+1]
        
        # 지금 상자를 헤아림에서 빼기
        mask = torch.ones(scores.numel(), dtype=torch.bool, device=boxes.device)
        mask[max_idx] = False
        boxes = boxes[mask]
        scores = scores[mask]
        indices = indices[mask]
        
        # 겹침 비 셈하기
        ious = box_iou(current_box, boxes).squeeze(0)
        
        # 겹침 비에 따라 점수 줄이기
        if method == 'gaussian':
            decay = torch.exp(-(ious ** 2) / sigma)
        else:  # 선형
            decay = torch.where(ious >= iou_threshold, 1 - ious, torch.ones_like(ious))
        
        scores = scores * decay
        
        # 점수가 낮은 상자 없애기
        keep_mask = scores >= score_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        indices = indices[keep_mask]
    
    keep = torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)
    return keep, scores
```

위 코드는 선택된 인덱스를 `keep`에 기록하지만 반환하는 `scores`는 반복문 끝의 **남은 후보 점수**다. 선택된 각 인덱스에 대응하는 갱신 점수 배열이 아니므로 두 반환값을 그대로 짝지으면 안 된다. 또한 최고 점수를 먼저 선택하고 나머지 후보만 점수 임계값으로 걸러 처음부터 모든 점수가 낮아도 하나를 남길 수 있다. 실제 후처리에는 선택 시점 점수의 별도 기록과 일관된 필터 규칙이 필요하다.

DIoU-NMS는 선택한 예측 M과 다른 예측 b_i 사이에 IoU−ρ²/c²를 계산한다. 추론 중 비교 대상은 정답 상자가 아니다. 중심 거리가 떨어져 있으면 같은 IoU에서도 억제가 달라질 수 있으나 혼잡 장면의 정확도 향상이 자동으로 보장되지는 않는다.

```python
def diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    beta: float = 0.6
) -> torch.Tensor:
    """
    DIoU에 바탕한 최대가 아닌 것 누르기.
    
    겹치는 물체를 더 잘 다루려 겹침 비 대신 DIoU를 쓴다.
    
    인수:
        boxes: xyxy 꼴의 (N, 4) 두름 상자
        scores: (N,) 믿음도 점수
        iou_threshold: 누르기용 DIoU 문턱값
        beta: DIoU의 지수(가운데점 거리 무게를 다스린다)
        
    반환값:
        남길 상자의 번호
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while sorted_indices.numel() > 0:
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        if sorted_indices.numel() == 1:
            break
        
        remaining_indices = sorted_indices[1:]
        
        # DIoU 셈하기
        current_box = boxes[current_idx].unsqueeze(0).expand(len(remaining_indices), -1)
        remaining_boxes = boxes[remaining_indices]
        
        dious = compute_diou(current_box, remaining_boxes)
        
        # DIoU에 따라 누르기
        mask = dious < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def compute_diou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """짝지어진 상자 사이의 DIoU를 셈한다."""
    # 겹침 비
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    iou = inter_area / (area1 + area2 - inter_area + 1e-7)
    
    # 가운데점 거리
    c1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
    c1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
    c2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
    c2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
    
    center_dist_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
    
    # 감싸는 상자의 대각선
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    
    diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
    
    return iou - center_dist_sq / (diag_sq + 1e-7)
```

위 함수의 `beta` 인수는 사용되지 않는다. 따라서 β를 바꿔도 결과가 달라지는 구현으로 설명할 수 없다. 여기서 구현한 거리 벌점과 논문·다른 라이브러리의 변형을 구분한다.

## 5. 탐지 후처리와 계산비용

다음 함수는 이미 xyxy로 디코딩한 상자·점수·클래스를 받아 신뢰도 필터, 클래스별 NMS, 최대 탐지 수 제한을 적용한다. raw 회귀 출력의 디코딩, 이미지 경계 clipping, resize·padding의 역변환은 포함하지 않는다. 앞에서 사용한 연산의 import와 좌표·dtype·device 조건도 필요하다.

```python
def detection_postprocess(
    predictions: dict,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.5,
    max_detections: int = 100
) -> dict:
    """
    물체 알아내기의 온전한 뒷손질 물길.
    
    인수:
        predictions: 'boxes', 'scores', 'labels' 텐서를 담은 사전
        conf_threshold: 거르기용 믿음도 문턱값
        nms_threshold: NMS의 겹침 비 문턱값
        max_detections: 돌려줄 알아냄의 최대 개수
        
    반환값:
        거른 'boxes', 'scores', 'labels'를 담은 사전
    """
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']
    
    # 1단계: 믿음도 문턱값 두기
    conf_mask = scores >= conf_threshold
    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    labels = labels[conf_mask]
    
    if boxes.numel() == 0:
        return {
            'boxes': torch.empty(0, 4, device=boxes.device),
            'scores': torch.empty(0, device=scores.device),
            'labels': torch.empty(0, dtype=torch.long, device=labels.device)
        }
    
    # 2단계: 갈래별 NMS
    keep = ops.batched_nms(boxes, scores, labels, nms_threshold)
    
    # 3단계: 알아냄 수 제한
    if len(keep) > max_detections:
        # 점수 기준 상위 k개 남기기
        _, top_k_indices = scores[keep].topk(max_detections)
        keep = keep[top_k_indices]
    
    return {
        'boxes': boxes[keep],
        'scores': scores[keep],
        'labels': labels[keep]
    }
```

CUDA 사용은 장치 지원과 데이터 위치에 달렸다. 적은 후보를 매번 CPU에서 GPU로 복사하면 복사 비용 때문에 오히려 느려질 수 있으므로 고정 FPS를 기대하지 않는다.

```python
# 텐서가 GPU 위에 있는지 확인
boxes = boxes.cuda()
scores = scores.cuda()

# torchvision의 NMS는 CUDA 알맹이를 저절로 쓴다
keep = ops.nms(boxes, scores, iou_threshold=0.5)
```

아래는 이미지별 Python 반복이다. 모든 이미지를 동시에 처리하는 단일 병렬 커널은 아니지만 서로 다른 이미지의 상자가 억제되는 것을 막는다.

```python
def batched_detection_postprocess(
    batch_predictions: list[dict],
    **kwargs
) -> list[dict]:
    """어림 배치를 다룬다."""
    return [detection_postprocess(pred, **kwargs) for pred in batch_predictions]
```

모든 쌍의 IoU는 O(NM) 시간과 결과 행렬을 저장하는 O(NM) 공간이 든다. 대응된 행만 비교하면 O(N)이다. 큰 행렬은 청크로 나누어 저장량을 줄일 수 있다. 탐욕적 NMS는 정렬 O(N log N)에 선택마다 남은 후보와의 비교가 추가되어 최악 O(N²)이다. 남긴 K개와의 비교를 O(KN)으로 표현해도 K가 N까지 커질 수 있다. Soft-NMS도 일반적인 직접 구현에서 최악 O(N²)다. 최적화된 CPU·GPU 커널은 상수와 병렬성이 다르며 내부 버퍼의 공간 복잡도까지 모두 O(N)이라고 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
IoU가 0인 경우와 1인 경우를 양의 면적 상자에 대해 설명하라. IoU만으로 같은 객체인지 알 수 있는가?

</div>

??? success "연습문제 1 풀이"

    IoU=0은 양의 면적 교집합이 없다는 뜻으로 경계 접촉도 포함한다. ε 없는 정의에서 IoU=1이면 두 상자가 일치한다. 서로 다른 객체의 상자가 겹칠 수 있으므로 객체의 동일성은 클래스·영상 내용·매칭 규칙과 함께 판단한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
상자가 다른 상자 안에 있을 때 GIoU와 IoU를 비교하고, 중심이 다를 때 DIoU가 추가하는 정보를 설명하라.

</div>

??? success "연습문제 2 풀이"

    포함 관계이면 최소 외접 상자 C가 합집합과 같아 GIoU의 면적 벌점이 0이다. DIoU는 중심 거리의 제곱을 C의 대각선 제곱으로 나눈 항을 빼므로 포함 관계에서도 중심이 다르면 다른 점수를 줄 수 있다. 어느 손실이 항상 더 빠른 학습을 보장하는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 NumPy NMS의 경계 비교와 의존 함수를 점검하고 본문의 PyTorch 직접 구현과 비교하라.

</div>

??? success "연습문제 3 풀이"

    아래 코드는 compute_iou 정의가 별도로 필요하며 독립적으로 완성된 예제가 아니다. `<= iou_threshold`를 남기므로 같은 값은 유지하지만 본문의 `<` 구현은 제거한다. 두 구현 모두 최고 점수 후보와의 겹침으로 선택하는 탐욕적 규칙이며 실제 객체 정답을 비교하지 않는다. 이 코드가 앞 함수와 같은 이름 nms를 다시 정의하는 점도 호출 환경에서 구분한다.

    ```python
    import numpy as np

    def nms(boxes, scores, iou_threshold=0.5):
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            remaining = order[1:]
            ious = compute_iou(boxes[i], boxes[remaining])
            mask = ious <= iou_threshold
            order = remaining[mask]
        return keep
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Soft-NMS의 keep과 scores를 바로 대응시킬 수 없는 이유와 클래스별 좌표 이동의 전제 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"

    keep은 이미 선택한 원래 인덱스이고 scores는 반복 후 남은 후보 배열이므로 길이·순서가 대응하지 않는다. 선택 때 갱신 점수를 별도로 기록해야 한다. 클래스별 좌표 이동은 좌표 범위보다 충분히 큰 간격과 적절한 정밀도가 필요하며 본문 방식은 비음수 좌표·비어 있지 않은 입력을 가정한다.



## 정리하며

IoU는 겹침의 척도이며 GIoU·DIoU·CIoU는 회귀에 추가 기하 정보를 제공한다. NMS는 예측 후보를 선택하는 후처리다. 매칭·신뢰도·억제 임계값을 구분하고, 경계 비교와 클래스·이미지별 처리, 반환 점수의 의미와 좌표 조건을 확인한다.

**참고 문헌**

1. Rezatofighi, H., et al. (2019). Generalized Intersection over Union: A Metric and a Loss for Bounding Box Regression. *CVPR*.
2. Zheng, Z., et al. (2020). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. *AAAI*.
3. Bodla, N., et al. (2017). Soft-NMS: Improving Object Detection with One Line of Code. *ICCV*.
4. Neubeck, A., & Van Gool, L. (2006). Efficient Non-Maximum Suppression. *ICPR*.
