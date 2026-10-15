# 시간 어텐션과 유효 프레임 마스크

시간 attention은 프레임·짧은 구간 특징 사이의 관계 또는 중요도 가중치를 학습하는 방식이다. 모든 프레임을 똑같이 평균하는 것과 달리 입력에 따라 집계를 바꿀 수 있다. 이 페이지의 코드는 메시지만 출력하며 attention 모듈·학습·시각화를 구현하지 않는다.

## 1. 기존 코드와 출력

```python
"""때에 걸친 어텐션 구조"""
import torch

# ========================================================================
# 메인
# ========================================================================
print("Temporal Attention Mechanisms module")


if __name__ == "__main__":
    pass
```

**기존 출력:**

```
Temporal Attention Mechanisms module
```

## 2. 가중 풀링과 self-attention

프레임 특징 h_t에 점수 e_t=f(h_t)를 매기고 α_t=softmax(e)_t로 정규화해 z=Σ_t α_t h_t를 만들면 시간 가중 풀링이다. 이는 모든 프레임 쌍의 관계를 계산하는 self-attention과는 다른 구조다. 학습된 가중치가 크다는 사실만으로 해당 프레임의 인과적 중요도나 설명의 충실성이 보장되지는 않는다.

self-attention에서 Q,K,V를 만들면 softmax(QKᵀ/√d_k+M)V로 시간 정보를 결합한다. 입력 순서를 표현할 위치·시간 정보가 없으면 self-attention 자체는 순열에 대해 equivariant하며 순서 없는 평균 같은 집계까지 거치면 시간 순서를 구분하지 못할 수 있다. 프레임 timestamp·상대 위치 등 필요한 시간 표현을 설계한다.

## 3. 패딩 마스크와 인과 마스크

패딩 프레임은 key 점수에서 제외하고 최종 풀링에서도 유효 위치만 집계해야 한다. key 마스크만으로 패딩 query의 출력이 자동으로 0이 되지는 않는다. 모든 key가 가려진 행에 −∞ softmax를 적용하면 정의되지 않는 값이 생길 수 있으므로 빈 clip의 처리 규칙이 필요하다.

온라인 예측에는 미래 위치를 보지 못하도록 causal mask를 적용한다. 패딩 마스크와 인과 마스크는 목적이 다르다. clip 전체를 쓰는 offline 분류와 과거만 쓰는 추론을 같은 입력 조건의 성능으로 비교하지 않는다. API마다 bool mask의 True 의미가 다를 수 있으므로 사용 연산의 계약을 확인한다.

## 4. Attention dropout의 해석

softmax 뒤 가중치에 p=0.1 dropout을 적용하면 일부 항을 0으로 만들고 남은 항은 1/0.9배 한다. 기대값은 원래 가중치와 같지만 개별 행의 합이 더 이상 항상1은 아니다. 따라서 적용 후 값들을 정규화된 확률 분포라고 해석하지 않는다. eval 모드에서는 일반 dropout이 비활성화된다.

특정 연결에 대한 의존을 줄이는 정규화 방법이지만 가중치가 자동으로 균등해지거나 항상 더 정확해지는 것은 아니다. dropout은 패딩·인과 마스크를 대체하지 않는다. 필요하면 검증 자료에서 강도를 정한다.

## 5. 길이와 계산비용

길이 T, 특징 차원 D의 표준 dense self-attention에서 쌍별 점수·값 결합은 O(T²D), 입력 projection은 O(TD²)다. 점수 행렬을 직접 저장하면 head별 O(T²) 메모리가 필요하다. 메모리 효율적인 exact 구현은 전체 행렬 저장을 피할 수 있어 모든 구현의 메모리가 반드시 T²라고 단정하지 않는다. 가중 풀링은 점수 함수에 따라 보통 시간 길이에 선형인 더 단순한 대조군이다.

긴 비디오에는 시간 샘플링·국소 창·희소 연결·계층적 요약 등을 사용할 수 있다. 각각 관찰 범위와 연결 구조·근사가 달라지므로 계산 절감만으로 동등한 표현이나 정확도를 보장하지 않는다. 프레임 수와 실제 시간 길이, 추가 특징 추출 비용도 함께 보고한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
시간 가중 풀링과 self-attention의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    가중 풀링은 각 프레임 점수로 하나의 요약을 만들고 self-attention은 query 위치별로 다른 key·value와 상호작용한 출력을 만든다. 둘 모두 위치 정보가 없고 순서 불변 집계를 쓰면 순서를 구별하지 못할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
softmax 뒤 p0.1 dropout을 적용하면 가중치 합은 항상1인가?

</div>

??? success "연습문제 2 풀이"

    아니다. 남은 항을1/0.9배 해 기대값은 보존하지만 실제 합은 선택된 항에 따라 변한다. dropout 뒤 값을 그대로 확률 분포라 부르지 않으며 평가 모드와 마스크 처리도 따로 정의한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
시간 길이를 두 배로 늘리면 dense 쌍별 attention 비용은 어떻게 변하는가?

</div>

??? success "연습문제 3 풀이"

    D와 head 구조가 같으면 T²D 항은 약4배, TD² projection 항은 약2배다. 전체 실행 시간과 메모리는 특징 추출·커널·저장 방식에도 달려 이 비율을 그대로 전체 시스템에 적용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 기존 시험 조각과 달리 시간 attention에 필요한 명세를 설명하라.

</div>

??? success "연습문제 4 풀이"

    본문에는 attention 구현이 없고 아래 미정의 객체·공백 식별자는 실행 가능한 코드가 아니다. 실제 구현은 패딩 값에 대한 유효 출력 불변성, 미래 입력의 비참조, 빈 clip·전부 가려진 행, 위치 정보·출력 모양·dropout 모드를 확인해야 한다. 기존 조각은 자리 표시자다.

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

시간 attention은 가중 풀링과 쌍별 self-attention을 구분하고 위치 정보·패딩·인과 마스크를 함께 설계해야 한다. dropout 후 가중치는 항상 확률 분포가 아니며 큰 가중치가 곧 인과 설명도 아니다. 기존 파일은 개념을 연결하기 위한 시작용 코드다.
