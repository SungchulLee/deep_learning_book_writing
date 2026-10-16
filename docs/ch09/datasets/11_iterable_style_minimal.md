# 반복자 방식 최소 예제

PyTorch의 map-style Dataset과 IterableDataset을 비교한다. 보존된 코드 블록은 주석과 빈 main만 담은 안내이며 실제 스트리밍 데이터셋 클래스는 구현하지 않는다.

## 1. 코드

```python
#!/usr/bin/env python3
# ========================================================
# 12_dataset_09_iterable_style_minimal.py
# ========================================================
"""
반복자 방식 데이터셋:
- __iter__으로 표본을 흘려보낸다. 설계상 임의 인덱싱이 없다.
- 길이를 모를 수 있다. __len__은 선택 사항이다.
- DataLoader는 IterableDataset에 표집기를 쓰지 않는다. 그저 반복자를 소비할 뿐이다.
- 섞기, 에포크 제어, 일꾼 분배가 필요하면 __iter__ 안에서 직접 구현한다.

다시 훑기:
- 교육용 예제에서는 __iter__ 안에서 씨앗을 다시 심어 매번 같은 열이
  나오게 하기도 한다.
- 실무에서는 에포크와 일꾼마다 씨앗을 달리한다 (torch.utils.data.get_worker_info 참고).

참고 — PyTorch의 두 가지 데이터셋 방식:
- 맵 방식 (torch.utils.data.Dataset): __len__과 __getitem__(i). Sampler(shuffle=True),
  random_split, Subset과 함께 쓸 수 있다.
- 반복자 방식 (torch.utils.data.IterableDataset): __iter__(흘려보내기).
  DataLoader가 표집하지 않으므로 섞기와 분배를 직접 처리한다.
"""


if __name__ == "__main__":
    pass
```

## 2. 논의

map-style 데이터셋은 키로 표본을 조회하며 샘플러로 순서를 정할 수 있다. IterableDataset은 반복자를 통해 표본을 제공하므로 스트림, 순차 파일 처리 등에서 유용하다. 유한한 메모리 데이터에도 사용할 수 있다.

여러 worker가 같은 IterableDataset을 복제하면 각 worker가 같은 표본을 내보낼 수 있다. get_worker_info 등을 이용해 데이터 구간을 나누고, 분산 학습에서는 rank별 분할까지 함께 고려한다.

shuffle이나 무작위 인덱스 샘플러를 map-style과 같은 방식으로 사용할 수 없으므로, 필요하면 데이터 소스나 버퍼에서 순서를 섞는다. 유한 스트림인지, 다음 에포크에서 다시 시작할 수 있는지도 정의해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
현재 코드가 생성하는 데이터와 출력은 무엇인가?

</div>

??? success "연습문제 1 풀이"
    실제 데이터셋이나 DataLoader를 만들지 않고 main에서 pass하므로 표본이나 출력이 없다. 문서의 인터페이스 설명과 실행 구현을 구분해야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
네 worker가 0부터 99까지의 유한 스트림을 중복 없이 처리할 분할을 제안하라.

</div>

??? success "연습문제 2 풀이"
    worker i가 25i부터 25(i+1)-1까지를 처리하거나, 인덱스를 4로 나눈 나머지가 i인 표본을 처리할 수 있다. 데이터 소스가 해당 분할을 지원하는지 확인하고 분산 rank까지 포함해 중복을 점검한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
스트림의 에포크 길이와 재시작 규칙을 왜 명시해야 하는가?

</div>

??? success "연습문제 3 풀이"
    무한 스트림은 끝이 없어 학습 스텝 수를 따로 정해야 하고, 한 번 소비하는 소스는 재시작이 불가능할 수 있다. 길이 추정과 drop_last도 worker별 처리에 영향을 줄 수 있으므로 표본 수를 가정하지 않는다.

## 정리하며

IterableDataset의 핵심은 표본을 순차 제공하는 인터페이스이다. 구현에는 worker·rank 분할, 순서 섞기, 길이와 재시작 폴리시가 필요하다. 이 페이지의 코드 자체는 그 구현을 제공하지 않는다.
