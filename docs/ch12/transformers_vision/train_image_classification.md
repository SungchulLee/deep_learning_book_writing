# 이미지 분류 학습

ViT를 생성하고 무작위 이미지 배치의 순전파 모양을 확인한다. 함수 이름은 train_vit이지만 데이터셋·손실·역전파·옵티마이저 갱신을 포함하지 않아 실제 학습을 수행하지 않는다.

## 1. 코드

```python
"""
이미지 분류로 비전 트랜스포머 학습하기
"""
import torch
from vision_transformer import VisionTransformer

# ========================================================================
# 메인
# ========================================================================

def train_vit():
    model = VisionTransformer(img_size=224, patch_size=16, num_classes=10)
    
    # 임시 데이터
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    
    print(f"ViT Model created!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nReady for image classification training!")

if __name__ == '__main__':
    train_vit()
```

**출력:**

```
ViT Model created!
Input shape: torch.Size([4, 3, 224, 224])
Output shape: torch.Size([4, 10])
Parameters: 85,804,810

Ready for image classification training!
```

## 2. 논의

외부 vision_transformer.VisionTransformer를 RGB 224×224, 패치 16과 10개 클래스로 생성한다. (4,3,224,224) 입력은 패치 196개와 CLS를 거쳐 (4,10) 로짓이 된다. 외부 모델의 계수는 85,804,810개이며 같은 폴더의 Pre-LN vit_model 구현과는 구조·계수 수가 다르다.

eval이나 no_grad를 사용하지 않아 학습 모드와 경사 기록이 켜져 있지만, 이것만으로 가중치가 갱신되는 것은 아니다. 본문 출력은 모델 생성·순전파 메시지이며 정확도·수렴·학습 성공을 보여 주지 않는다.

실제 분류 학습에는 데이터·레이블·독립 검증 분할과 전처리, 교차 엔트로피와 갱신 루프를 연결해야 한다. 무작위 이미지의 출력 모양 확인과 학습한 이미지 특징의 품질은 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 함수가 하는 작업과 하지 않는 학습 단계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    모델 생성과 무작위 배치 순전파·출력만 수행한다. 손실·backward·step이 없어 학습 모드이더라도 가중치는 갱신되지 않는다. 준비 메시지를 학습 성능 결과로 읽어서는 안 된다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력과 외부 모델의 인터페이스에서 확인할 항목은 무엇인가?

</div>

??? success "연습문제 2 풀이"
    RGB 채널, 해상도·패치 수와 위치 표 길이, 클래스 수와 정답 ID를 맞춘다. 외부 모듈 검색 경로와 같은 이름의 다른 VisionTransformer 생성자도 구분한다. 입력·모델 장치와 자료형의 호환성도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
추론 시연과 실제 학습 루프를 각각 어떻게 구성해야 하는가?

</div>

??? success "연습문제 3 풀이"
    추론에는 eval·no_grad를 적용하고 같은 전처리를 사용한다. 학습에는 train 모드에서 로짓 손실·경사 초기화·역전파·갱신을 수행한다. 별도 검증으로 설정을 선택하며 무작위 입력 검사로 정확도를 대신하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_train image classification():
        model = Train Image Classification(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 손실이나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

이 페이지는 ViT의 생성과 순전파 모양 확인이다. 학습 모드·경사 기록·옵티마이저 갱신을 구분하고 실제 데이터 학습과 평가가 추가되어야 함을 이해한다.
