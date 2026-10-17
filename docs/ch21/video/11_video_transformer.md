# 비디오 트랜스포머의 토큰과 시공간 어텐션

비디오 트랜스포머는 프레임의 patch 또는 여러 프레임에 걸친 tubelet을 토큰으로 바꾸어 관계를 학습한다. 시간 축이 추가되면서 토큰 수·관찰 구간·위치 표현과 attention 비용이 중요해진다. 이 페이지의 기존 코드는 안내 메시지만 출력하며 모델·토큰화·학습을 구현하지 않는다.

## 1. 기존 코드와 출력

```python
"""영상 변환기"""
import torch

# ========================================================================
# 메인
# ========================================================================
print("Video Transformers module")


if __name__ == "__main__":
    pass
```

**기존 출력:**

```
Video Transformers module
```

## 2. Patch와 tubelet의 입력 계약

프레임 T개를 p_h×p_w patch로 자르면 공간 토큰 수는 S=(H/p_h)(W/p_w)이고 총수는 TS다. p_t프레임을 묶은 겹치지 않는 tubelet이면 T/p_t개의 시간 위치와 S개의 공간 위치로 토큰을 구성할 수 있다. 이 계산은 크기가 정확히 나누어지고 padding·겹침이 없는 조건이다.

각 토큰을 D차원으로 사영한 뒤 위치·시간 정보를 추가한다. 입력 순서를 배치·시간·공간 중 어떤 순서로 펼쳤는지와 위치 embedding의 순서가 같아야 한다. 단순 attention만으로 프레임의 물리 시간 간격을 알 수 없으므로 fps·stride·timestamp와 위치 표현의 관계도 설계한다.

## 3. 공동·분리 attention

전체 TS토큰에 공동 attention을 적용하면 쌍별 연산은 O(T²S²D)다. 시간과 공간 attention을 나누면 대표적으로 O(ST²D+TS²D)의 쌍별 비용으로 표현할 수 있지만 같은 연산을 단지 더 빠르게 계산한 것과는 다른 연결 구조다. projection·MLP 비용과 실제 커널도 추가된다.

[TimeSformer](https://arxiv.org/abs/2102.05095)와 [ViViT](https://arxiv.org/abs/2103.15691)는 시공간 attention과 인수분해 설계의 구체적인 예다. 모든 비디오 트랜스포머가 같은 tokenization·attention·head를 쓰는 것은 아니다. 국소 창·계층 구조·토큰 축소도 각자 관찰 범위와 근사를 바꾼다.

## 4. 패딩·집계·사전 학습

가변 길이·해상도 입력은 patch 크기와 마스크·위치 embedding을 일치시켜야 한다. 영상 경계 padding이나 시간 반복을 실제 관측과 구분하고 유효 토큰만 최종 집계한다. CLS 토큰·평균 풀링·clip별 결합은 서로 다른 요약 방식이며 최종 로짓의 클래스 순서와 레이블 계약도 필요하다.

이미지 사전 학습 웨이트를 비디오에 사용하려면 tubelet projection과 시간 위치 정보를 어떻게 초기화·적응하는지 정해야 한다. 큰 사전 학습 모델을 사용했다는 사실만으로 어떤 작은 데이터에서도 CNN보다 정확하다는 보장은 없다. 학습 데이터·사전 학습·관찰 구간과 평가 clip 수를 맞춰 비교한다.

## 5. 해상도와 실제 비용

T=16,H=W=224,p_h=p_w=16이면 S=196, 총3136토큰이다. p_t=2의 겹치지 않는 tubelet이면1568개로 줄어든다. 각 축을 얼마나 줄였는지가 정보와 비용을 함께 바꾼다. 프레임 수를 두 배로 늘리면 dense 쌍별 비용은 약4배지만 공간 해상도를 가로·세로 각각 두 배로 늘리면 토큰이4배가 되어 쌍별 항은 약16배다.

점수 행렬을 직접 저장하는 구현과 메모리 효율적인 exact attention의 저장량은 다를 수 있다. 전체 비용에는 비디오 디코딩·토큰 projection·MLP·여러 clip 평가도 포함한다. 미래 토큰을 보는 offline 설정은 온라인 인과 추론과 분리해 평가한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
16프레임224×224에서16×16 patch와2×16×16 tubelet의 토큰 수를 계산하라.

</div>

??? success "풀이"

    공간 위치는14×14=196개다. 프레임별 patch는16×196=3136, 시간2프레임 tubelet은8×196=1568개다. padding·겹침·추가CLS 토큰은 제외한 계산이다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
위치·시간 정보 없이 토큰을 attention에 넣으면 어떤 순서 문제가 생기는가?

</div>

??? success "풀이"

    self-attention 자체는 입력 순열에 대해 equivariant이고 평균 집계는 순서 불변이므로 시간 재배열을 구분하지 못할 수 있다. 토큰 순서에 맞는 시간·공간 위치 표현과 관찰 간격 정의가 필요하다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
공동 attention과 시간·공간 분리 attention의 비용식을 비교하라.

</div>

??? success "풀이"

    공동은O(T²S²D), 대표적인 분리는O(ST²D+TS²D)의 쌍별 항이다. 분리는 관계를 단계별로 구성하는 다른 모델링 선택이고 projection·MLP 비용까지 포함한 전체 실행 시간은 이 항만으로 정해지지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 기존 시험 조각의 한계와 비디오 트랜스포머에 필요한 확인 항목을 설명하라.

</div>

??? success "풀이"

    현재 모델·tokenizer가 없고 아래 공백 식별자·미정의 객체는 실행 불가능하다. 실제 구현에서는 patch 나눗셈·padding, 토큰·위치 순서, 유효 마스크·집계, 길이 변경·인과 마스크와 출력 클래스 계약을 명세로 확인해야 한다. 기존 조각은 자리 표시자다.

    ```python
    def test_module 34():
        model = Module 34(...)
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


## 정리하며

비디오 트랜스포머는 시공간 토큰 수·위치 정보·마스크와 attention 연결을 함께 설계한다. 공동·분리 attention의 구조와 비용, 사전 학습·관찰 구간·평가 횟수를 구분해 비교한다. 기존 파일은 모델 구현이 아닌 시작용 코드다.
