# 하향식
Merge Sort는 두 개의 정렬된 데이타를 합병하는 작업을 리커시브하게 진행한다.

```python
def merge_sort(arr):
    if len(arr)<=1:
        return
    
    mid = len(arr) // 2
    left = arr[:mid] 
    right = arr[mid:] 

    merge_sort(left)
    merge_sort(right)

    merge_sort_two_sorted_arrays(left, right, arr)
```

```python
def merge_sort_two_sorted_arrays(left, right, arr):
    len_left = len(left) 
    len_right = len(right) 
    i = j = k = 0
    while (i<len_left) and (j<len_right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1 
        else:
            arr[k] = right[j]
            j += 1 
        k += 1
        
    while (i<len_left): 
        arr[k] = left[i]
        i += 1
        k += 1
        
    while (j<len_right): 
        arr[k] = right[j]
        j += 1
        k += 1
```

```python
if __name__ == '__main__':
    test_cases = (
        [],
        [3],
        [3,5],
        [5,3],
        [3,5,2],
        [5,3,4],
        [10,3,15,7,8,23,98,29],
        [9,8,7,2],
        [1,2,3,4,5]
    )
    for arr in test_cases:
        merge_sort(arr)
        print(arr)
```

**출력:**
```
[]
[3]
[3, 5]
[3, 5]
[2, 3, 5]
[3, 4, 5]
[3, 7, 8, 10, 15, 23, 29, 98]
[2, 7, 8, 9]
[1, 2, 3, 4, 5]
```

<div align="center"><img src="https://gmlwjd9405.github.io/images/algorithm-merge-sort/sort-time-complexity.png" width="50%"></div>

[[알고리즘] 합병 정렬(merge sort)이란](https://gmlwjd9405.github.io/2018/05/08/algorithm-merge-sort.html)

# 참고 문헌

[Merge Sort vs Quick Sort](https://www.youtube.com/watch?v=es2T6KY45cA)

[Merge Sort - Data Structures & Algorithms Tutorial Python #17](https://www.youtube.com/watch?v=nCNfu_zNhyI&list=PLeo1K3hjS3uu_n_a__MI_KktGTLYopZ12&index=18)

[[알고리즘] 합병 정렬(merge sort)이란](https://gmlwjd9405.github.io/2018/05/08/algorithm-merge-sort.html)

[정렬 알고리즘](https://namu.wiki/w/정렬%20알고리즘#s-2.3.1)

## 연습문제

**연습문제 1.**
이 방법의 핵심 생각과 그것이 파국적 잊음을 어떻게 다루는지 설명하라.

??? success "연습문제 1 풀이"
    이 방법은 새 과제를 배울 때 모델의 매개변수나 표현이 바뀌는 방식을 옥죄어 파국적 잊음을 누그러뜨린다. (벌주기, 되살리기, 증류, 구조 갈라두기로) 배운 함수의 중요한 대목을 지켜 냄으로써, 앞선 과제의 성능을 지키면서도 새 과제에 맞추어 갈 수 있게 한다.

---

**연습문제 2.**
이 접근법의 셈과 기억 요구는 무엇인가?

??? success "연습문제 2 풀이"
    요구는 변형마다 다르지만 대체로 (a) 매개변수의 중요도 무게, (b) 학습 보기의 일부, (c) 스승 모델의 출력, (d) 과제마다의 망 모듈 가운데 하나를 담아 두어야 한다. 기억 비용과 잊음 막기의 효과 사이에서 맞바꿈이 일어난다.

---

**연습문제 3.**
이 방법을 효과와 셈 비용 면에서 EWC와 견주어라.

??? success "연습문제 3 풀이"
    EWC은 대각 피셔 정보로 중요한 가중치를 짚어낸다. 이 방법은 다른 맞바꿈을 준다. 옛 과제의 성능을 더 잘 지킬 수 있고, 기억 요구가 다르며, 과제 짜임에 대한 가정도 다르다. 실험으로 견주어 보면 잣대에 따라 서로 보완되는 강점을 보일 때가 많다.

---

**연습문제 4.**
이 방법을 간추린 판으로 파이토치에 구현하라.

??? success "연습문제 4 풀이"
    구현은 대개 새 과제를 익히는 동안 보통의 교차 엔트로피 손실에 벌주기 항을 더한다. 핵심 부품은 (1) 앞선 과제 학습에서 제약을 셈하기, (2) 필요한 정보(가중치, 본보기, 스승 출력)를 담아 두기, (3) 새 과제 학습 중에 그 제약을 씌우기이다.
