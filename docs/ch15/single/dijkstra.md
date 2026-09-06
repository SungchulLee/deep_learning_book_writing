# 데이크스트라 알고리즘
$$
\begin{array}{lccllll}
&\text{Graph}&\text{Complexity}\\
\text{BFS}&w=1\\
\text{Bellana Ford}&\text{Negative weights are allowed}&O(nm)\\
\text{Dijkstra}&w\ge 0&O(n^2)\ \text{or}\ O(m\log n)\\
\end{array}
$$

# 벨먼-포드의 최악의 경우

<img src='img/Screen Shot 2022-07-02 at 1.56.37 AM.png' width=70%>

[[알고리즘] 제16-2강 최단경로(shortest path problem) (계속)](https://www.youtube.com/watch?v=icqzGct4V1s&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=39)

# 데이크스트라

<img src='img/Screen Shot 2022-07-02 at 2.01.11 AM.png' width=70%>
<img src='img/Screen Shot 2022-07-02 at 2.01.55 AM.png' width=70%>
<img src='img/Screen Shot 2022-07-02 at 2.04.02 AM.png' width=70%>

[[알고리즘] 제16-2강 최단경로(shortest path problem) (계속)](https://www.youtube.com/watch?v=icqzGct4V1s&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=39)

<img src='img/Screen Shot 2022-07-02 at 2.06.38 AM.png' width=70%>
<img src='img/Screen Shot 2022-07-02 at 2.11.08 AM.png' width=70%>
<img src='img/Screen Shot 2022-07-02 at 2.12.10 AM.png' width=70%>

[[알고리즘] 제16-2강 최단경로(shortest path problem) (계속)](https://www.youtube.com/watch?v=icqzGct4V1s&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=39)

<div align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/2/23/Dijkstras_progress_animation.gif" width="20%"></div>

[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)

데이크스트라 알고리즘처럼 벨먼-포드도 늦추기로 나아간다. 곧 올바른 거리의 어림값을 더 나은 값으로 바꿔 가다가 마침내 풀이에 이른다. 두 알고리즘 모두 꼭짓점마다의 어림 거리가 늘 참된 거리를 넘겨 어림한 것이며, 그 옛 값과 새로 찾은 길의 길이 가운데 작은 것으로 바뀐다. 다만 데이크스트라 알고리즘은 우선순위 줄을 써서 아직 다루지 않은 가장 가까운 꼭짓점을 욕심껏 고르고 그 나가는 변 모두에 늦추기를 한다. 반면 벨먼-포드 알고리즘은 그냥 모든 변을 늦추며, 이를 $|V|-1$번 한다. 여기서 $|V|$은 그래프의 꼭짓점 개수이다.

$$\begin{array}{lll}
\text{Bellman Ford}&&\text{Relax using Pre-Determined Fixed Order}\\
\\
\text{Dijkstra}&&\text{Relax using Greedy Outgoing Edges}\\
\end{array}$$

이 되풀이마다 거리가 맞게 셈해진 꼭짓점의 개수가 늘어나며, 그로부터 마침내 모든 꼭짓점이 올바른 거리를 갖게 됨이 따라 나온다. 이 길 덕분에 벨먼-포드 알고리즘은 데이크스트라보다 더 넓은 갈래의 입력에 쓸 수 있다.

[Bellman–Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)

# 참고 문헌

[[알고리즘] 제16-2강 최단경로(shortest path problem) (계속)](https://www.youtube.com/watch?v=icqzGct4V1s&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=39)

[3.6 Dijkstra Algorithm - Single Source Shortest Path - Greedy Method](https://www.youtube.com/watch?v=XB4MIexjvY0&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=45)

[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)

[Dijkstra algorithm | Single source shortest path algorithm](https://www.youtube.com/watch?v=Sj5Z-jaE2x0)

[Dijkstra algorithm | Code implementation](https://www.youtube.com/watch?v=t2d-XYuPfg0)

## 연습문제

**연습문제 1.**
꼭짓점 4개와 변 $A \xrightarrow{2} B$, $A \xrightarrow{5} C$, $B \xrightarrow{1} C$, $B \xrightarrow{4} D$, $C \xrightarrow{1} D$을 갖는 그래프에서 꼭짓점 $A$부터 이 알고리즘이 굴러가는 것을 따라가라.

??? success "연습문제 1 풀이"
    처음 거리: $d(A) = 0$, $d(B) = \infty$, $d(C) = \infty$, $d(D) = \infty$.

    $A$에서 다루기: $d(B) = 2$, $d(C) = 5$.
    $B$에서 다루기(잠정 거리가 가장 작다): $d(C) = \min(5, 2+1) = 3$, $d(D) = 2+4 = 6$.
    $C$에서 다루기: $d(D) = \min(6, 3+1) = 4$.
    $D$에서 다루기: 새로 고침 없음.

    $A$에서의 마지막 최단 거리: $d(A)=0, d(B)=2, d(C)=3, d(D)=4$.

---

**연습문제 2.**
그래프에 음의 변 무게가 있으면 데이크스트라 알고리즘이 왜 무너지는지 설명하여라. 구체적인 어긋냄 보기를 들어라.

??? success "연습문제 2 풀이"
    데이크스트라 알고리즘은 앞으로 어떤 늦추기도 그것을 낫게 할 수 없다고 놓고 잠정 거리가 가장 작은 꼭짓점을 욕심껏 확정한다. 음의 변이 있으면 아직 다루지 않은 꼭짓점을 지나는 더 긴 길이 나중에 더 짧아질 수 있다. 이를테면 변 $A \xrightarrow{1} B$, $A \xrightarrow{3} C$, $C \xrightarrow{-4} B$에서 데이크스트라는 $B$을 거리 1으로 확정하지만, 길 $A \to C \to B$의 거리는 $3 + (-4) = -1 < 1$이다.

---

**연습문제 3.**
이진 힙을 쓴 데이크스트라 알고리즘의 시간 복잡도는 얼마인가? 피보나치 힙을 쓰면?

??? success "연습문제 3 풀이"
    **이진 힙**을 쓰면 최소 꺼내기 $V$번마다 $O(\log V)$, 열쇠 낮추기 $E$번마다 $O(\log V)$이 들어 $O((V + E) \log V)$이다. **피보나치 힙**을 쓰면 최소 꺼내기가 고르게 나눠 $O(\log V)$, 열쇠 낮추기가 고르게 나눠 $O(1)$이라 $O(V \log V + E)$이다. $E = \Theta(V^2)$인 빽빽한 그래프에서는 피보나치 힙이 이론상 낫다.

---

**연습문제 4.**
모든 변을 $|V| - 1$번 늦춘 뒤 벨먼-포드 알고리즘이 최단 경로를 맞게 셈함을 증명하여라.

??? success "연습문제 4 풀이"
    최단 경로 안 변의 개수 $k$에 대한 귀납으로 증명한다. $i$번 되풀이한 뒤 알고리즘은 변을 많아야 $i$개 쓰는 최단 경로를 모두 맞게 셈해 두었다. **바탕:** 0번 되풀이한 뒤 $d(s) = 0$이 맞다(변 0개짜리 길). **걸음:** $i$번 되풀이한 뒤 변이 $\leq i$개인 최단 경로가 모두 맞다고 놓자. $i+1$번째 되풀이에서 변이 $i+1$개인 최단 경로 $s \to \cdots \to u \to v$을 생각하자. $u$까지의 부분 길은 변이 $i$개이고 가정에 따라 맞다. 변 $(u,v)$을 늦추면 $d(v) = d(u) + w(u,v)$이 되고 이것이 가장 좋다. 최단 경로의 변이 많아야 $|V|-1$개이므로(음의 고리가 없다) $|V|-1$번 되풀이하면 넉넉하다. $\square$
