# 꼭짓점과 변
$$
\text{Graph}\left\{\begin{array}{l}
\text{Directed}\\
\text{Undirected}\\
\end{array}\right.
$$

$$
\text{Graph}\left\{\begin{array}{l}
\text{Weighted}\\
\text{Unweighted}\\
\end{array}\right.
$$

# 정규 그래프

그래프 이론에서 정규 그래프는 꼭짓점마다 이웃의 개수가 같은 그래프이다. 곧 꼭짓점마다 차수가 같다.

[wiki](https://en.wikipedia.org/wiki/Regular_graph)

# 완전 그래프

수학의 그래프 이론에서 완전 그래프는 서로 다른 꼭짓점 짝마다 하나뿐인 변으로 이어진 단순 무방향 그래프이다.

[wiki](https://en.wikipedia.org/wiki/Complete_graph)

# 이어짐

$$
\text{Graph}\left\{\begin{array}{l}
\text{Directed}
\left\{\begin{array}{l}
\text{Strongly Connected}\\
\text{Weakly Connected}\\
\text{Disconnected}
\end{array}\right.
\\
\\
\text{Undirected}
\left\{\begin{array}{l}
\text{Connected}\\
\text{Disconnected}
\end{array}\right.
\end{array}\right.
$$

# 이분 그래프

수학의 그래프 이론에서 이분 그래프는 꼭짓점을 서로 겹치지 않는 두 묶음 {\displaystyle U}U과 {\displaystyle V}V으로 나누어 변마다 {\displaystyle U}U의 꼭짓점과 {\displaystyle V}V의 꼭짓점을 잇게 할 수 있는 그래프이다. 꼭짓점 묶음 {\displaystyle U}U과 {\displaystyle V}V을 보통 그래프의 조각이라 한다. 마찬가지로 이분 그래프는 길이가 홀수인 고리를 하나도 갖지 않는 그래프이다.

[wiki](https://en.wikipedia.org/wiki/Bipartite_graph)

# 평면 그래프

그래프 이론에서 평면 그래프는 평면에 박을 수 있는 그래프이다. 곧 변이 끝점에서만 만나도록 평면 위에 그릴 수 있다.

[wiki](https://en.wikipedia.org/wiki/Planar_graph)

$$
\text{Graph}\left\{\begin{array}{l}
\text{Cyclic}\\
\text{Acyclic}\\
\end{array}\right.
$$

# 나무

그래프 이론에서 나무는 아무 두 꼭짓점이 꼭 하나의 길로 이어진 무방향 그래프이며, 마찬가지로 이어져 있고 고리가 없는 무방향 그래프이다. 숲은 아무 두 꼭짓점이 많아야 하나의 길로 이어진 무방향 그래프이며, 마찬가지로 고리가 없는 무방향 그래프, 곧 나무들의 겹치지 않는 합집합이다.

[wiki](https://en.wikipedia.org/wiki/Tree_(graph_theory))

# 참고 문헌

[Introduction to Graphs | Types of Graphs | Data Structures](https://www.youtube.com/watch?v=M1Z7B7IkCEE&list=PL1w8k37X_6L9IfRTVvL-tKnrZ_F-8HJQt&index=1)

## 연습문제

**연습문제 1.**
꼭짓점 $n$개의 단순 무방향 그래프는 변을 많아야 몇 개 갖는가? 답을 증명하여라.

??? success "연습문제 1 풀이"
    많아야 $\binom{n}{2} = \frac{n(n-1)}{2}$개이다. 변마다 서로 다른 꼭짓점 짝을 잇고 그런 짝이 $\binom{n}{2}$개 있다. 이 한계에 이르는 그래프가 완전 그래프 $K_n$이다. $\square$

---

**연습문제 2.**
차수 열이 $(3, 3, 3, 3, 3)$인 꼭짓점 5개의 단순 그래프가 있는가? 설명하여라.

??? success "연습문제 2 풀이"
    없다. 악수 보조정리에 따라 차수의 합은 $2|E|$이며 짝수여야 한다. 여기서 합은 $5 \times 3 = 15$으로 홀수이다. 그러므로 그런 그래프는 없다.

---

**연습문제 3.**
단순 그래프 $G = (V, E)$의 **여집합 그래프** $\bar{G}$을 정의하여라. $G$에 꼭짓점 7개와 변 10개가 있으면 $\bar{G}$의 변은 몇 개인가?

??? success "연습문제 3 풀이"
    여집합 그래프 $\bar{G} = (V, \bar{E})$은 $(u,v) \notin E$일 때 그리고 그때만 변 $(u,v)$을 갖는다. 꼭짓점 7개에서 있을 수 있는 변의 총 개수는 $\binom{7}{2} = 21$이다. 그러므로 $|\bar{E}| = 21 - 10 = 11$이다.

---

**연습문제 4.**
여섯 사람의 아무 무리에나 서로 다 아는 셋이 있거나 서로 다 모르는 셋이 있음을 증명하여라.

??? success "연습문제 4 풀이"
    변을 빨강(서로 앎)이나 파랑(서로 모름)으로 칠한 완전 그래프 $K_6$으로 본뜬다. 아무 꼭짓점 $v$을 잡자. 비둘기집 원리에 따라 $v$에는 같은 빛깔의 변이 적어도 3개 있으며, 빨강이라 하고 그 끝을 꼭짓점 $a, b, c$이라 하자. $\{a, b, c\}$ 사이의 변 가운데 하나라도 빨강이면 그 변과 $v$이 빨강 세모를 이룬다. $\{a, b, c\}$ 사이의 변 셋이 모두 파랑이면 그것들이 파랑 세모를 이룬다. 이것이 램지 수 $R(3,3) = 6$이다. $\square$
