# 겹침 비(IoU)와 최대가 아닌 것 누르기(NMS)
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 겹침 비(IoU)를 첫 원리에서 이끌어 내고 짠다
- 겹침 비의 변종(GIoU, DIoU, CIoU)과 그 좋은 점을 이해한다
- 최대가 아닌 것 누르기(NMS)를 짜고 알아내기에서의 몫을 이해한다
- 더 나은 결과를 얻으려 NMS 변종(Soft-NMS, DIoU-NMS)을 쓴다
- 이 연산의 셈 복잡도와 가장 좋게 하기 전략을 살핀다

## 겹침 비(IoU)

### 정의와 직관

**자카드 지수**라고도 하는 겹침 비(IoU)는 두 자리 사이의 겹침을 잰다. 두름 상자 $A$과 $B$에 대해:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

기하로 보면 이 비는 두 상자가 얼마나 잘 맞물리는지를 담아낸다:

```
         Box A              Box B           Intersection        Union
        ┌──────┐          ┌──────┐          ┌────┐         ┌──────────┐
        │      │          │      │          │████│         │          │
        │  ┌───┼──────────┼───┐  │    →     │████│    /    │          │
        │  │   │          │   │  │          └────┘         │          │
        └──┼───┘          └───┼──┘                         └──────────┘
           └──────────────────┘
```

### 겹침 비의 성질

1. **Bounded**: $0 \leq \text{IoU} \leq 1$
2. **Symmetric**: $\text{IoU}(A, B) = \text{IoU}(B, A)$
3. **잣수에 안 바뀜**: 결과는 절대 크기가 아니라 상대적인 겹침에만 달렸다
4. **IoU = 0**: 상자끼리 겹치지 않는다
5. **IoU = 1**: 완전히 겹친다(같은 상자)

### 풀이 지침

| 겹침 비 범위 | 읽는 법 | 흔한 쓰임 |
|-----------|----------------|-------------|
| 0.00 - 0.20 | 겹침이 나쁨 | 다른 물체일 가능성 |
| 0.20 - 0.50 | 일부 겹침 | 아리송한 경우 |
| 0.50 - 0.75 | 겹침이 좋음 | 보통의 알아내기 문턱값 |
| 0.75 - 0.90 | 겹침이 강함 | 빡빡한 값매김(AP@75) |
| 0.90 - 1.00 | 겹침이 아주 좋음 | 높은 정밀도가 필요한 쓰임 |

### 수학적 유도

$(x_{min}, y_{min}, x_{max}, y_{max})$ 꼴의 상자 둘에 대해

**Box A**: $(x_1^A, y_1^A, x_2^A, y_2^A)$

**Box B**: $(x_1^B, y_1^B, x_2^B, y_2^B)$

**교집합 자리표**:

$$x_1^I = \max(x_1^A, x_1^B), \quad y_1^I = \max(y_1^A, y_1^B)$$

$$x_2^I = \min(x_2^A, x_2^B), \quad y_2^I = \min(y_2^A, y_2^B)$$

**교집합 넓이**(겹치지 않으면 0):

$$\text{Area}_I = \max(0, x_2^I - x_1^I) \times \max(0, y_2^I - y_1^I)$$

**낱낱의 넓이**:

$$\text{Area}_A = (x_2^A - x_1^A) \times (y_2^A - y_1^A)$$

$$\text{Area}_B = (x_2^B - x_1^B) \times (y_2^B - y_1^B)$$

**합집합 넓이**(넣고 빼기 원리):

$$\text{Area}_U = \text{Area}_A + \text{Area}_B - \text{Area}_I$$

**마지막 겹침 비**:

$$\text{IoU} = \frac{\text{Area}_I}{\text{Area}_U}$$

### PyTorch 구현

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
    
    여기 짠 것은 효율을 위해 묶음 셈을 받쳐 준다.
    
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

### 익히기를 위한 벡터로 짜기

익히는 동안에는 흔히 어림 텐서와 참값 텐서 사이의 겹침 비가 필요하다:

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

## 손실 함수로서의 겹침 비

보통의 겹침 비를 손실 함수로 바로 쓰면 한계가 있다:

1. **겹치지 않으면 기울기가 0**: $\text{IoU} = 0$이면 잃음이 배움 신호를 주지 못한다
2. **어떻게 안 겹치는지를 가리지 못함**: 멀리 떨어져 안 겹치는 두 상자와 가깝지만 안 겹치는 상자가 똑같이 IoU=0이다

### 넓힌 겹침 비(GIoU)

GIoU는 둘을 감싸는 가장 작은 상자를 헤아려 기울기가 0이 되는 문제를 다룬다:

$$\text{GIoU}(A, B) = \text{IoU}(A, B) - \frac{|C \setminus (A \cup B)|}{|C|}$$

여기서 $C$은 $A$과 $B$을 모두 감싸는 가장 작은 상자이다.

**성질**:

- 범위: $[-1, 1]$(상자가 안 겹치면 음수일 수 있다)
- 상자가 완전히 겹치면 겹침 비와 같다
- 안 겹치는 상자에도 기울기 신호를 준다

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

### 거리 겹침 비(DIoU)

DIoU는 상자 가운데점 사이의 거리에 바탕한 벌을 더한다:

$$\text{DIoU}(A, B) = \text{IoU}(A, B) - \frac{\rho^2(A, B)}{c^2}$$

여기서 각 기호는 다음과 같다.

- $\rho(A, B)$은 상자 가운데 사이의 유클리드 거리다
- $c$은 감싸는 가장 작은 상자의 대각선 길이이다

**좋은 점**:

- GIoU보다 빨리 모인다
- 가운데점 거리를 곧바로 가장 작게 한다
- 상자 되돌리기에 더 낫다

### 온전한 겹침 비(CIoU)

CIoU는 가로세로비가 어긋나지 않게 하는 항을 더한다:

$$\text{CIoU}(A, B) = \text{IoU}(A, B) - \frac{\rho^2(A, B)}{c^2} - \alpha v$$

여기서 각 기호는 다음과 같다.

$$v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$$

$$\alpha = \frac{v}{(1 - \text{IoU}) + v}$$

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

### 겹침 비 변종 견줌

| 변종 | 식 | 범위 | 핵심 이점 |
|---------|---------|-------|-------------|
| **IoU** | $\frac{I}{U}$ | [0, 1] | 단순하고 느낌이 잡힌다 |
| **GIoU** | $\text{IoU} - \frac{C - U}{C}$ | [-1, 1] | 겹치지 않을 때도 기울기를 준다 |
| **DIoU** | $\text{IoU} - \frac{\rho^2}{c^2}$ | [-1, 1] | 더 빨리 모여든다 |
| **CIoU** | $\text{DIoU} - \alpha v$ | [-1, 1] | 가로세로 견줌까지 본다 |

## 최대가 아닌 것 누르기(NMS)

### 거듭 알아냄 문제

물체 알아내개는 그림 전체에 걸쳐 촘촘한 어림을 내놓는다. 실제 물체마다 믿음 점수가 비슷한, 겹치는 상자가 여럿 어림된다:

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

NMS는 가장 좋은 알아냄을 고르고 남아도는 겹친 상자를 없앤다.

### 보통의 NMS 알고리즘

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

### PyTorch 구현

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

### torchvision의 NMS 쓰기

실전 코드에는 가장 좋게 다듬은 C++ 짜기를 쓰라:

```python
import torchvision.ops as ops

# 보통의 NMS
keep_indices = ops.nms(boxes, scores, iou_threshold=0.5)

# 묶음 NMS(갈래마다)
keep_indices = ops.batched_nms(boxes, scores, labels, iou_threshold=0.5)

# 결과 거르기
final_boxes = boxes[keep_indices]
final_scores = scores[keep_indices]
final_labels = labels[keep_indices]
```

## NMS 변종

### Soft-NMS

보통의 NMS는 딱 자른다. 곧 겹침 비 문턱값을 넘는 상자를 아예 없앤다. Soft-NMS는 그 대신 겹치는 상자의 점수를 겹친 만큼 줄인다:

**가우스 무게 주기**:

$$s_i = s_i \cdot e^{-\frac{\text{IoU}(M, b_i)^2}{\sigma}}$$

**선형 무게 주기**:

$$s_i = \begin{cases} s_i & \text{if IoU} < N_t \\ s_i(1 - \text{IoU}) & \text{if IoU} \geq N_t \end{cases}$$

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

### DIoU-NMS

누를 때 겹침 비 대신 DIoU를 써서 가운데점 거리를 헤아린다:

$$R_{DIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2}$$

이러면 겹침 비는 비슷하되 가운데점 자리가 다른 상자를 가려낼 수 있어 붐비는 장면에서 잘못 누르는 일이 줄어든다.

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

### NMS 변종 견줌

| 방법 | 좋은 점 | 나쁜 점 | 알맞은 곳 |
|--------|------|------|----------|
| **딱 자르는 NMS** | 단순하고 빠르다 | 겹친 물체를 놓친다 | 성긴 장면 |
| **Soft-NMS** | 재현율이 더 좋다 | 느리고 다듬어야 한다 | 붐비는 장면 |
| **DIoU-NMS** | 가운데점을 헤아린다 | 셈이 더 든다 | 물체 크기가 들쭉날쭉 |

## 온전한 알아내기 뒷손질 물길

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

## 성능 가장 좋게 하기

### CUDA로 다듬은 NMS

GPU로 빠르게 하려면 상자가 CUDA 위에 있는지 확인하라:

```python
# 텐서가 GPU 위에 있는지 확인
boxes = boxes.cuda()
scores = scores.cuda()

# torchvision의 NMS는 CUDA 알맹이를 저절로 쓴다
keep = ops.nms(boxes, scores, iou_threshold=0.5)
```

### 묶음 단위 다루기

그림 여럿을 효율적으로 다룬다:

```python
def batched_detection_postprocess(
    batch_predictions: list[dict],
    **kwargs
) -> list[dict]:
    """어림 묶음을 다룬다."""
    return [detection_postprocess(pred, **kwargs) for pred in batch_predictions]
```

### 계산 복잡도

| 연산 | 시간 복잡도 | 공간 복잡도 |
|-----------|-----------------|------------------|
| 겹침 비(짝짓기) | O(N × M) | O(N × M) |
| NMS(막무가내) | O(N²) | O(N) |
| NMS(다듬음) | O(N log N + K × N) | O(N) |
| Soft-NMS | O(N²) | O(N) |

여기서 N은 상자 수, M은 참값 수, K는 남긴 상자 수이다.

## 요약

겹침 비와 NMS는 물체 알아내기의 근본 벽돌이다:

**겹침 비(IoU)**:

- 두름 상자 사이 겹침의 좋음을 잰다
- 범위는 [0, 1]이고 0.5가 흔한 문턱값이다
- 변종(GIoU, DIoU, CIoU)은 모서리 경우를 다루어 익히기를 낫게 한다

**최대가 아닌 것 누르기(NMS)**:

- 같은 물체를 거듭 알아낸 것을 없앤다
- 욕심쟁이 알고리즘: 가장 좋은 것을 남기고 겹치는 것을 없앤다
- 변종(Soft-NMS, DIoU-NMS)은 붐비는 장면을 더 잘 다룬다

**핵심 짜기 요점**:

- 효율을 위해 겹침 비 셈을 벡터로 한다
- 실전 코드에는 `torchvision.ops.nms`를 쓴다
- 갈래끼리 누르지 않도록 갈래마다 NMS를 쓴다
- 쓰임새에 맞게 문턱값을 다듬는다

## 참고 문헌

1. Rezatofighi, H., et al. (2019). Generalized Intersection over Union: A Metric and a Loss for Bounding Box Regression. *CVPR*.
2. Zheng, Z., et al. (2020). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. *AAAI*.
3. Bodla, N., et al. (2017). Soft-NMS: Improving Object Detection with One Line of Code. *ICCV*.
4. Neubeck, A., & Van Gool, L. (2006). Efficient Non-Maximum Suppression. *ICPR*.

## 연습문제

**연습문제 1.**
한 단계 알아내개와 두 단계 알아내개의 차이를 설명하여라. 빠르기와 정확도 사이의 근본 맞바꿈은 무엇인가?

??? success "연습문제 1 풀이"
    **두 단계 알아내개**(보기로 더 빠른 R-CNN)는 먼저 자리 제안을 만들고 제안마다 갈래를 매기고 다듬는다. 정확하지만 제안마다 다루기 때문에 느리다. **한 단계 알아내개**(보기로 YOLO, SSD)는 특징 지도에서 두름 상자와 갈래 확률을 한 번에 곧바로 어림하여 정확도를 조금 내주고 훨씬 빠른 미룸을 얻는다. 맞바꿈은 이렇다. 두 단계 알아내개는 작고 겹치는 물체를 잘 알아내지만 초당 5~15틀로 돌고, 한 단계 알아내개는 mAP가 조금 낮은 대신 초당 30~155틀 넘게 낸다.

---

**연습문제 2.**
겹침 비(교집합 나누기 합집합) 식을 이끌어 내고 두름 상자를 값매김할 때 왜 단순한 L2 거리보다 낫게 여기는지 설명하여라.

??? success "연습문제 2 풀이"
    두 두름 상자 $A$과 $B$에 대해:

    $$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

    겹침 비를 낫게 여기는 까닭은 이렇다. (1) 잣수에 안 바뀐다(화소 10개의 어긋남은 큰 물체보다 작은 물체에 더 크게 다가온다). (2) 자연스레 $[0, 1]$에 놓여 좋음 점수로 읽을 수 있다. (3) 상자 자리표 사이의 L2 거리는 겹침을 담아내지 못해 두 상자의 L2 거리가 작아도 겹침이 0일 수 있다(보기로 하나가 다른 하나 안에 있는 경우와 나란히 놓인 경우).

---

**연습문제 3.**
최대가 아닌 것 누르기(NMS)를 짜고 알아내기 물길에서 그것이 하는 몫을 설명하여라.

??? success "연습문제 3 풀이"
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
    NMS는 같은 물체를 거듭 알아낸 것을 없앤다. 후보 상자에 점수를 매긴 뒤 점수가 가장 높은 상자를 고르고 겹침 비가 문턱값을 넘는 상자(겹친 것일 가능성이 높다)를 모두 없애기를 되풀이한다.

---

**연습문제 4.**
물체 알아내기의 갈래 치우침 문제와 초점 손실이 그것을 어떻게 다루는지 설명하여라.

??? success "연습문제 4 풀이"
    한 단계 알아내개에서는 닻 상자 대부분이 바탕(쉬운 아님 보기)이고 물체를 담은 것은 몇 안 된다. 여느 엇결 엔트로피 잃음은 쉬운 아님 보기가 워낙 많아 그쪽에 휘둘리므로 어려운 맞음 보기의 기울기 신호가 묻힌다. **초점 잃음**은 조절 값을 더한다. $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$이다. $\gamma > 0$이면 쉬운 보기($p_t$이 높은 것)의 짐이 지수로 줄어 어려운 보기에 익힘이 몰린다. $\gamma = 2$과 $\alpha = 0.25$을 쓰면 RetinaNet은 한 단계의 빠르기를 지키면서 두 단계 알아내개에 맞먹는 맞음을 이룬다.
