# 비디오의 CNN 특징과 LSTM 시간 모델

CNN-LSTM은 각 프레임의 공간 특징을 CNN으로 얻고 그 특징 시퀀스를 LSTM으로 처리하는 구성이다. 프레임별 예측을 평균하는 방식과 달리 시간 순서에 의존하는 상태를 유지할 수 있다. 이 페이지의 기존 코드는 메시지만 출력하며 CNN·LSTM·학습 루프는 구현하지 않는다. 아래는 구현 설계에 필요한 설명이다.

## 1. 기존 코드와 출력

```python
"""영상을 위한 CNN-LSTM"""
import torch

# ========================================================================
# 메인
# ========================================================================
print("CNN-LSTM for Video module")


if __name__ == "__main__":
    pass
```

**기존 출력:**

```
CNN-LSTM for Video module
```

## 2. 프레임 특징에서 시퀀스로

입력을 B×T×C×H×W로 정했다면 CNN에 BT개 프레임을 넣어 BT×D 특징을 만들고 B×T×D로 되돌린다. 이때 배치와 시간의 대응 순서를 유지해야 한다. LSTM에 batch_first=True를 쓰면 이 모양을 사용하고, 기본 시간 우선 계약이면 T×B×D로 바꾼다. Conv3d 입력 B×C×T×H×W와 혼동하지 않는다.

CNN이 프레임별 연산만 하더라도 뒤 LSTM이 순서를 결합할 수 있다. 반면 긴 간격의 균일 프레임 샘플은 빠른 움직임을 놓칠 수 있고, feature 하나로 줄이면서 잃은 공간 세부를 LSTM이 자동 복원하지는 않는다. fps·stride와 관찰 시간 범위를 함께 설계한다.

## 3. 상태·패딩과 시퀀스 요약

LSTM의 은닉 상태 h_t와 셀 상태 c_t는 서로 다른 역할을 한다. 단방향은 과거에서 현재로 진행하며, 양방향은 뒤 프레임도 사용하므로 전체 클립 분류에는 쓸 수 있어도 미래 정보를 사용할 수 없는 온라인 문제에는 그대로 적용할 수 없다.

길이가 다른 클립을 패딩하면 마지막 배열 위치의 출력을 곧바로 ‘마지막 실제 프레임’으로 사용하면 안 된다. 유효 길이·packed sequence·마스크를 이용해 마지막 유효 상태나 마스크 평균을 선택한다. 서로 독립인 비디오 사이에 상태를 무심코 공유하면 다른 표본의 정보가 섞인다. 연속 스트림의 상태 유지와 truncated backpropagation의 detach 규칙은 별도로 정한다.

최종 특징에서 K개 로짓을 만들고 단일 레이블 분류에는 교차 엔트로피를 사용할 수 있다. 여러 행동이 동시에 있는 multi-label 문제의 목표·손실은 다르다. 프레임 특징 CNN을 고정할지 학습할지도 명시한다. requires_grad=False만으로 BN의 running 통계가 고정되는 것은 아니다.

## 4. LSTM과 GRU의 매개변수

한 층·단방향·projection 없음에서 입력 차원 D, 은닉 H인 PyTorch LSTM은 네 묶음의 입력·은닉 가중치와 두 bias 벡터를 갖는다. 총수는 4DH+4H²+8H다. GRU는 세 묶음이라 3DH+3H²+6H다. 후보 셀·새 상태 연산까지 포함한 묶음 수이며 모두를 sigmoid 게이트라고 부르지는 않는다.

하나의 통합 bias를 쓰는 다른 수식에서는 마지막 항이 각각4H·3H여서 개수가 달라진다. 층·방향·projection과 뒤 분류기의 차원까지 맞춰 비교한다. 매개변수 수가 적다는 사실만으로 GRU나 LSTM의 정확도·실제 속도 순위를 보장할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
B×T×3×H×W에서 프레임 CNN 출력 차원 D를 LSTM에 연결하는 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"

    BT×3×H×W로 프레임을 묶어 BT×D 특징을 얻고 같은 배치·시간 순서로 B×T×D로 복원한다. batch_first 설정에 따라 시간·배치 축을 맞춘다. 단순 평균과 달리 LSTM은 순서를 상태에 반영할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
패딩된 시퀀스의 output[:,-1]을 무조건 쓰면 생기는 문제를 설명하라.

</div>

??? success "연습문제 2 풀이"

    짧은 비디오에서는 마지막 위치가 패딩일 수 있어 유효한 마지막 관측을 의미하지 않는다. 길이를 이용한 상태 선택·packing·마스크 요약을 정하고 독립 비디오 사이 상태도 분리한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
PyTorch 기본 bias를 포함한 단일 층 LSTM·GRU의 매개변수를 비교하라.

</div>

??? success "연습문제 3 풀이"

    LSTM은4DH+4H²+8H, GRU는3DH+3H²+6H다. 같은 D·H·한 층·단방향·projection 없음이면 GRU가3/4이다. 두 bias 벡터를 가진 구현과 bias 하나로 합친 수식을 구분해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 기존 시험 조각의 한계와 실제 CNN-LSTM에서 확인할 계약을 설명하라.

</div>

??? success "연습문제 4 풀이"

    현재 코드에는 모델이 없고 공백 식별자·미정의 변수 때문에 아래 조각도 실행 가능한 시험이 아니다. 구현한다면 배치·시간 정렬, 유효 길이·마스크, 상태 초기화와 단방향의 미래 비참조, CNN 고정 시 BN 모드·출력 로짓 모양을 확인하는 명세가 필요하다.

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

CNN-LSTM은 프레임 특징과 시간 상태를 연결한다. 축 순서·패딩·마지막 유효 상태·비디오별 상태 분리를 명확히 하고, 양방향의 미래 정보와 실제 bias 관례를 구분한다. 기존 파일은 설계의 시작용 코드다.
