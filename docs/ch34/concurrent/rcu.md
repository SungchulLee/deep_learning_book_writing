# 읽고 베끼고 고치기

한꺼번에 쓰는 자료 얼개 가운데에는 적는 일보다 읽는 일이 훨씬 잦은 것이 많다. 이렇게 읽기가 판치는 일감에서 읽을 때마다 발맞추기 덧듦을 물면 성능을 버리게 된다. **읽고 베끼고 고치기**(RCU)는 읽는 쪽의 발맞추기를 아예 없앤다. 읽는 이는 잠금도, 기억 울짱도, 원자적인 명령도 없이 함께 쓰는 자료에 닿는다. 적는 이는 자료를 고친 벌을 만들어 손가락질을 원자적으로 바꿔 끼운 뒤, 앞서 있던 읽는 이가 모두 마치기를 기다렸다가 옛 판을 거둬들인다.

## 한가운데 이치

RCU는 장치 셋에 기댄다.

1. **펴내고 받아 보기**: 적는 이는 함께 쓰는 손가락질을 고쳐 자료 얼개의 새 판을 원자적으로 펴낸다. 읽는 이는 이 손가락질을 실어 받아 본다.
2. **말미**: 새 판을 펴낸 뒤 적는 이는 옛 판을 아직 가리키고 있을 만한 읽는 이가 모두 마칠 때까지 기다린다. 이 기다림이 *말미*다.
3. **거둬들이기**: 말미가 끝나면 어떤 읽는 이도 옛 판을 가리키지 않으므로 안전하게 놓아줄 수 있다.

## 어떻게 도는가

### 읽는 쪽

읽는 이는 **RCU 읽기 쪽 고비 구역**에 들어가 함께 쓰는 손가락질을 읽고 자료를 쓴 뒤 고비 구역을 나온다. 잠금을 하나도 얻지 않는다. 고비 구역은 그저 읽는 이를 "돌고 있음"이라 표시하여 적는 이가 아직 자료를 거둬들이면 안 됨을 알게 한다.

### 적는 쪽

적는 이는 걸음 넷을 밟는다.

1. 자료 얼개를 가리키는 지금 손가락질을 **읽는다**.
2. 자료를 **베끼고** 그 벌에 고침을 매긴다.
3. 함께 쓰는 손가락질을 원자적으로 갈음하여 그 벌을 **펴낸다**.
4. **발맞춘다**: 말미 장치를 불러 앞서 있던 읽는 이가 모두 제 고비 구역을 나올 때까지 기다린다.
5. 옛 벌을 **거둬들인다**.

!!! warning "읽는 이는 옛것이나 새것을 볼 뿐 반쪽을 보지 않는다"
    손가락질 바꿔 끼우기가 원자적이므로 읽는 이는 늘 온전한 옛 판이나 온전한 새 판 가운데 하나를 보며, 반쯤 고쳐진 상태는 결코 보지 않는다. 이로써 잠금 없이 한결같음이 보장된다.

## 구현

```python
"""
읽고 베끼고 고치기(RCU) 흉내 내기.

함께 쓰는, 바뀌지 않는 자료 찰나본으로 RCU를 흉내 낸다.
적는 이는 새 찰나본을 만들고, 읽는 이는 잠금 없이 지금
찰나본에 닿는다. 말미가 안전한 거둬들이기를 지킨다.
"""

import threading
import time
import copy

# ===================================================================
# RCU 흉내 내기
# ===================================================================

class RCUProtected:
    """RCU로 지키는, 함께 쓰는 자료.

    읽는 이는 잠금 없이 자료에 닿는다. 적는 이는 벌을 만들어
    고친 뒤 가리킴을 원자적으로 바꿔 끼운다.

    인수:
        initial_data: 함께 쓰는 자료의 처음 값
    """

    def __init__(self, initial_data):
        self._data = initial_data
        self._write_lock = threading.Lock()
        self._reader_count = 0
        self._reader_lock = threading.Lock()
        self._versions_reclaimed = 0

    def read(self):
        """RCU 읽기 쪽 고비 구역을 시작한다.

        돌려주는 값:
            지금 자료 찰나본에 대한 가리킴(바꿀 수 없는 봄)
        """
        with self._reader_lock:
            self._reader_count += 1
        return self._data

    def read_done(self):
        """RCU 읽기 쪽 고비 구역을 나온다."""
        with self._reader_lock:
            self._reader_count -= 1

    def update(self, modify_fn):
        """RCU 고침을 벌인다.

        인수:
            modify_fn: 옛 자료를 받아 새 자료를 돌려주는 함수

        돌려주는 값:
            새 자료 판
        """
        with self._write_lock:
            old_data = self._data
            new_data = modify_fn(copy.deepcopy(old_data))
            # 펴내기(원자적인 손가락질 바꿔 끼우기)
            self._data = new_data
            # 말미: 옛 판을 읽는 이들을 기다린다
            self._synchronize()
            # 옛 판 거둬들이기(파이썬에서는 쓰레기 거두개가 맡는다)
            self._versions_reclaimed += 1
            return new_data

    def _synchronize(self):
        """앞서 있던 읽는 이가 모두 마칠 때까지 기다린다."""
        while True:
            with self._reader_lock:
                if self._reader_count == 0:
                    return
            time.sleep(0.001)

# ===================================================================
# 메인
# ===================================================================

if __name__ == "__main__":
    data = {"users": ["Alice", "Bob"], "count": 2}
    rcu = RCUProtected(data)

    print("RCU simulation:")

    # 읽는 이: 잠금 없이 닿는다
    snapshot = rcu.read()
    print(f"  Reader sees: {snapshot}")
    rcu.read_done()

    # 적는 이: 새 판을 만든다
    def add_user(data):
        data["users"].append("Charlie")
        data["count"] += 1
        return data

    new_data = rcu.update(add_user)
    print(f"  After update: {new_data}")

    # 고치는 동안 읽는 이 여럿
    results = []
    barrier = threading.Barrier(3)

    def reader(reader_id):
        barrier.wait()
        snap = rcu.read()
        results.append((reader_id, len(snap["users"])))
        time.sleep(0.01)  # 일을 흉내 낸다
        rcu.read_done()

    def writer():
        barrier.wait()
        rcu.update(lambda d: {**d, "users": d["users"] + ["Dave"],
                               "count": d["count"] + 1})

    threads = [
        threading.Thread(target=reader, args=(1,)),
        threading.Thread(target=reader, args=(2,)),
        threading.Thread(target=writer),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = rcu.read()
    rcu.read_done()

    print(f"\n  Concurrent readers saw user counts: "
          f"{[r[1] for r in sorted(results)]}")
    print(f"  Final state: {final}")
    print(f"  Versions reclaimed: {rcu._versions_reclaimed}")
```

**출력:**
```
RCU simulation:
  Reader sees: {'users': ['Alice', 'Bob'], 'count': 2}
  After update: {'users': ['Alice', 'Bob', 'Charlie'], 'count': 3}

  Concurrent readers saw user counts: [3, 3]
  Final state: {'users': ['Alice', 'Bob', 'Charlie', 'Dave'], 'count': 4}
  Versions reclaimed: 2
```

## 복잡도

| 연산 | 비용 |
|---|---|
| 읽기 쪽 들고 남 | $O(1)$, 발맞추기 없음 |
| 적기(베끼고 펴내기) | 크기 $n$인 자료에 $O(n)$ |
| 말미(발맞추기) | 잠잠한 상태 좇기를 쓰면 $O(1)$ 나눠 갚음 |

## 맞바꿈

| 성질 | RCU | 읽는 이-적는 이 잠금 |
|---|---|---|
| 읽기 덧듦 | 없음(잠금도 울짱도 없음) | 함께 쓰는 잠금을 얻고 놓음 |
| 적기 덧듦 | 자료 베끼기 + 말미 | 홀로 쓰는 잠금을 얻음 |
| 알맞은 곳 | 읽기가 판치는 일감 | 읽기와 적기가 고른 일감 |
| 기억 | 고치는 동안 벌이 하나 더 든다 | 더 드는 벌이 없다 |
| 낡음 | 읽는 이가 잠깐 옛 판을 볼 수 있다 | 모두 같은 판을 본다 |

## 쓰임새

- **리눅스 낟알**: 길잡이 표, 파일 시스템 캐시, 꾸러미 목록에 RCU가 두루 쓰인다. 낟알의 RCU는 초당 수백만 번의 읽기를 다룬다.
- **한꺼번에 쓰는 자료 얼개**: RCU로 지키는 이음 목록, 해시 표, 나무가 잠금 없는 읽기를 이루게 한다.
- **얼개 고치기**: 응용 설정을 RCU로 고칠 수 있다. 읽는 이는 늘 한결같은 찰나본을 본다.

## 참고 문헌

- McKenney, P. E. (2004). "Exploiting Deferred Destruction: An Analysis of Read-Copy-Update Techniques in Operating System Kernels." *PhD Thesis, OGI*.
- McKenney, P. E. and Slingwine, J. D. (1998). "Read-Copy Update: Using Execution History to Solve Concurrency Problems." *PDCS*.

## 연습문제

**연습문제 1.**
RCU 고침의 세 마디인 베끼기, 고치기, 거둬들이기를 풀어라. 넘어가는 동안 옳음을 지켜 주는 것은 무엇인가?

??? success "연습문제 1 풀이"
    (1) **베끼기**: 적는 이가 자료 얼개(또는 걸맞은 마디)의 벌을 만들어 그 벌에 고침을 매긴다. 처음 것은 그대로 남아 읽을 수 있다. (2) **고치기**: 적는 이가 손가락질을 옛 판에서 새 판으로 원자적으로 바꿔 끼운다(보기로 `rcu_assign_pointer`을 쓴다). 바꾼 뒤 새 읽는 이는 고친 판을 보고, 앞서 있던 읽는 이는 아직 옛 판을 가리킬 수 있다. (3) **거둬들이기**: 적는 이가 `synchronize_rcu()`을 부르거나 (`call_rcu`으로) 되부름을 걸어 두어, 앞서 있던 읽는 이가 모두 읽기 쪽 고비 구역을 마칠 때까지("말미") 기다린다. 그런 뒤에야 옛 판을 놓아준다. 어떤 읽는 이도 반쯤 고쳐진 얼개를 보지 않고 온전한 옛 판이나 온전한 새 판을 보므로 옳음이 보장된다. $\square$

---

**연습문제 2.**
RCU에서 "말미"가 무엇인지 밝히고, 리눅스 낟알이 말미가 지났음을 어떻게 알아내는지 풀어라.

??? success "연습문제 2 풀이"
    말미는 손가락질을 바꿔 끼운 뒤 어떤 읽는 이가 아직 옛 자료를 가리키고 있을 수 있는 사이다. 바꿔 끼우던 때에 읽기 쪽 고비 구역에 있던 모든 실이 그 구역을 나오면 말미가 끝난다. 리눅스 낟알(밀려나지 않는 RCU)에서 읽기 쪽 고비 구역은 `rcu_read_lock()`과 `rcu_read_unlock()` 사이의 코드이며, 이 둘은 그저 밀려남을 끄고 켠다. 고침 뒤 모든 CPU가 적어도 한 번 흐름 바꿈을 벌였거나(또는 놀거나 쓰는 이 자리에 있었으면) 말미가 지났음을 알아채는데, 이는 어떤 CPU도 옛 고비 구역을 아직 돌리고 있지 않음을 보장한다. 낟알은 CPU마다 두는 세개로 이를 좇는다. 밀려날 수 있는 RCU(PREEMPT_RCU)에서는 그 대신 읽는 이를 드러내 좇는다. $\square$

---

**연습문제 3.**
밀려나지 않는 낟알에서 RCU의 읽기 쪽 덧듦이 없음을 증명하여라. 밀려날 수 있는 낟알에서는 무엇이 달라지는가?

??? success "연습문제 3 풀이"
    밀려나지 않는 낟알에서 `rcu_read_lock()`과 `rcu_read_unlock()`은 아무 일도 하지 않는다(또는 아무것으로도 엮이지 않는다). 어떤 CPU의 읽는 이가 읽기 쪽 고비 구역 동안 밀려날 수 없으므로 흐름 바꿈이 있었다면 그 읽는 이가 나온 것임이 보장된다. 이 연산이 말 그대로 비어 있으므로 읽기 쪽 덧듦이 없다. 기억 울짱도, 원자적인 명령도, 캐시 줄 튀어 다님도 없다. 밀려날 수 있는 낟알에서는 읽는 이가 고비 구역 도중에 밀려날 수 있으므로 `rcu_read_lock()`이 CPU별(또는 일별) 세개를 올리고 `rcu_read_unlock()`이 그것을 내려야 한다. 이는 (CPU를 넘나드는 발맞추기가 없어) 빠르지만 없는 것은 아니다. 제 자리 기억에 닿고 엮개가 그 테두리를 넘어 차례를 바꾸지 못하게 막는다. 밀려나지 않는 경우의 없음에 견주어 짝마다 몇 나노초가 든다. $\square$

---

**연습문제 4.**
RCU는 읽기가 거의 전부인 일감에 가장 좋다. 흔한 잠금 얻기 비용을 여길 때 RCU가 읽는 이-적는 이 잠금을 앞서는 읽기/적기 비의 문턱을 어림하여라.

??? success "연습문제 4 풀이"
    읽는 이-적는 이 잠금(`rwlock`)은 요즘 x86 기계에서 읽기 잠금을 얻고 놓는 짝마다 대략 20~50 ns가 든다(함께 쓰는 잠금 낱말에 원자적인 연산을 벌여 캐시 줄이 튀어 다니기 때문이다). RCU의 읽기 쪽은 (밀려나지 않으면) 0 ns이거나 (밀려날 수 있으면) 약 5 ns가 든다. RCU의 적기 쪽은 훨씬 비싸다. 손가락질 바꿔 끼우기(약 10 ns)에 말미를 기다리는 `synchronize_rcu()`(약 10~100 ms, 다만 나눠 갚을 수 있다)이 더해진다. 초당 읽기를 $R$, 초당 적기를 $W$이라 하자. rwlock의 온 때는 $(R + W) \times 30$ ns이다. RCU의 온 때는 $R \times 0$ ns $+ W \times$ (10 ns + 말미 비용)이다. RCU가 이기는 조건은 $R \times 30 > W \times$ 말미_비용, 곧 $R/W > \text{말미\_비용} / 30$ ns이다. 말미가 10 ms이면 $R/W > 3 \times 10^5$이다. 묶어 처리하는 `call_rcu`을 쓰면 적기마다의 실제 비용이 떨어져 문턱이 $R/W \gtrsim 100$~$1000$이 된다. $\square$

---

**연습문제 5.**
한꺼번에 읽기, 넣기, 지우기를 받쳐 주는, RCU로 지키는 이음 목록을 설계하여라. 마디를 지울 때 적는 이가 따르는 규약을 밝혀라.

??? success "연습문제 5 풀이"
    이 줄은 머리 손가락질을 둔 한 겹 이음 목록이다. **읽는 이**: `rcu_read_lock()`을 부르고 (알맞은 기억 차례를 얻으려 `rcu_dereference()`을 쓰며) `next` 손가락질을 따라 줄을 훑은 뒤 `rcu_read_unlock()`을 부른다. 잠금이 필요 없다. **넣기**: 새 마디를 마련해 그 `next`을 지금 뒤따르는 마디로 둔 다음, `rcu_assign_pointer()`으로 앞 마디의 `next` 손가락질을 원자적으로 고친다. 읽는 이는 (새 마디가 없는) 옛 줄이나 (그것이 있는) 새 줄 가운데 하나를 보며 둘 다 한결같다. **지우기 규약**: (1) 앞 마디의 `next`을 원자적으로 고쳐 겨눈 마디를 건너뛴다(`rcu_assign_pointer(prev->next, target->next)`). (2) `synchronize_rcu()`이나 `call_rcu()`을 불러, 앞서 있던 읽는 이가 모두 마칠 때까지 겨눈 마디 놓아주기를 미룬다. 1번과 2번 사이에서 겨눈 마디는 줄에서 떨어져 나왔지만, 떨어지기 앞서 그것을 가리킨 읽는 이가 아직 닿고 있을 수 있다. 말미가 기억을 놓아주기 전에 그 읽는 이들이 마치도록 지켜 준다. $\square$
