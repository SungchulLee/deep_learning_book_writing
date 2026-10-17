# SlowFast의 두 시간 해상도 경로

[SlowFast](https://arxiv.org/abs/1812.03982)는 같은 시간 구간의 RGB 비디오를 낮은 시간 샘플링률의 Slow 경로와 높은 샘플링률의 Fast 경로로 처리한다. 외형과 빠른 시간 변화를 서로 다른 용량의 경로에서 표현하고 피처를 결합한다. 이 페이지의 기존 코드는 안내 문구만 출력하며 네트워크를 구현하지 않는다. 실제 축약 구조는 [SlowFast 상세 절](slowfast.md)에서 살펴본다.

## 1. 기존 코드와 출력

```python
"""느림빠름 넷트웍"""
import torch

# ========================================================================
# 메인
# ========================================================================
print("SlowFast Network module")


if __name__ == "__main__":
    pass
```

**기존 출력:**

```
SlowFast Network module
```

## 2. α와 β의 역할

Slow가 T개 프레임을 볼 때 Fast가 αT개 프레임을 보도록 시간 샘플링률을 높일 수 있다. α>1은 두 경로의 시간 해상도 비율이다. Fast의 채널 수를 Slow의 β배로 두는 β<1은 용량·비용을 줄이는 선택이다. 같은 공간 해상도를 쓰더라도 시간·채널 수는 서로 다르다.

두 경로는 같은 실제 구간을 관찰해야 한다. 프레임 수만 맞추고 시작 시각·stride·fps를 다르게 해 서로 다른 액션 구간을 넣으면 정렬이 깨진다. 경계·홀수 프레임 수에서는 인덱스 선택과 시간 다운샘플 결과 크기도 명시해야 한다. α=8,β=1/8 같은 값은 특정 설정이지 모델 이름이 강제하는 유일한 값은 아니다.

## 3. 피처 교환과 최종 결합

Fast 피처를 시간 축으로 줄여 Slow의 길이에 맞춘 뒤 채널로 연결하는 lateral connection 등을 사용할 수 있다. 단순 slicing·pooling·학습 가능한 컨볼루션은 서로 다른 연산이다. 시간 정렬뿐 아니라 출력 채널과 다음 Slow 블록의 입력 채널도 맞춰야 한다.

각 경로의 전역 시공간 피처를 연결해 분류 head에 넣을 수 있다. 중간 피처 교환과 마지막 출력 결합은 다른 설계 단계다. 두 RGB 경로이므로 RGB+optical flow를 쓰는 원래 Two-Stream과도 구분한다. Fast 경로가 flow 자체를 출력하거나 물리적 속도를 직접 추정하는 것은 아니다.

## 4. 비용과 평가 해석

같은 공간·시간 커널에서 입출력 채널이 모두 β배이고 출력 시간이 α배라면 Fast 컨볼루션 비용의 한 부분은 대략 αβ²에 비례한다. 입력 RGB 채널이 고정된 첫 레이어·lateral 연산·각 경로의 커널 설계와 실제 메모리 비용까지 모두 이 비율을 따르는 것은 아니다.

Slow와 Fast를 함께 쓰면 항상 더 정확하거나 정확히 두 배 비용이라는 보장은 없다. 동일 데이터·관찰 구간·crop·학습 조건에서 Slow 단독과 두 경로를 비교하고 실제 연산·메모리·추론 횟수를 보고한다. 미래 프레임을 포함한 클립 분류 결과와 온라인 지연 요구도 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Slow8프레임, α=4일 때 Fast 프레임 수와 시간 구간 정렬 조건을 설명하라.

</div>

??? success "풀이"

    Fast는32프레임을 사용한다. 두 입력은 같은 시작·끝 구간에서 서로 다른 간격으로 샘플링해야 한다. 숫자만8과32라고 같은 구간인 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Fast의 채널이 β배일 때 모든 연산 비용이 αβ²배인가?

</div>

??? success "풀이"

    아니다. 입출력 채널을 모두 β배로 줄이고 나머지 커널·공간 조건이 같은 컨볼루션에 대한 근사다. 첫 RGB레이어·lateral·서로 다른 시간 커널·후처리는 별도로 계산해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
SlowFast와 RGB+flow Two-Stream의 입력 차이를 설명하라.

</div>

??? success "풀이"

    SlowFast는 같은 RGB 구간의 두 시간 해상도를 사용한다. RGB+flow Two-Stream은 외형 이미지와 별도로 추정한 흐름 스택을 사용한다. Fast 피처를 optical flow의 정답 벡터로 해석하면 안 된다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 기존 시험 조각 대신 SlowFast에서 명확히 할 계약을 설명하라.

</div>

??? success "풀이"

    현재 본문에는 모델이 없고 아래 공백 식별자·미정의 객체도 실행 불가능하다. 실제 구현에는 두 구간의 인덱스·시간 정렬, lateral 출력 길이와 채널, 짧은 입력·홀수 길이, 분류 출력과 장치·dtype의 계약이 필요하다. 기존 조각은 자리 표시자다.

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

SlowFast는 같은 RGB 구간을 서로 다른 시간 해상도와 채널 용량으로 처리한다. 시간 정렬·lateral 채널 연결·최종 집계를 맞추고 비용 근사의 조건을 확인한다. 이 페이지의 코드는 네트워크 실행이 아닌 시작용 파일이다.
