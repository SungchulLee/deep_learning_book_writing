# Dijkstra's Algorithm

$$

\begin{array}{lccllll}
&\text{Graph}&\text{Complexity}\\
\text{BFS}&w=1\\
\text{Bellana Ford}&\text{Negative weights are allowed}&O(nm)\\
\text{Dijkstra}&w\ge 0&O(n^2)\ \text{or}\ O(m\log n)\\
\end{array}

$$

# Bellana Ford Worst Case

<img src='img/Screen Shot 2022-07-02 at 1.56.37 AM.png' width=70%>

[[알고리즘] 제16-2강 최단경로(shortest path problem) (계속)](https://www.youtube.com/watch?v=icqzGct4V1s&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=39)

# Dijkstra

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

Like Dijkstra's algorithm, Bellman–Ford proceeds by relaxation, in which approximations to the correct distance are replaced by better ones until they eventually reach the solution. In both algorithms, the approximate distance to each vertex is always an overestimate of the true distance, and is replaced by the minimum of its old value and the length of a newly found path. However, Dijkstra's algorithm uses a priority queue to greedily select the closest vertex that has not yet been processed, and performs this relaxation process on all of its outgoing edges; by contrast, the Bellman–Ford algorithm simply relaxes all the edges, and does this $|V|-1$ times, where $|V|$ is the number of vertices in the graph. 

$$\begin{array}{lll}
\text{Bellman Ford}&&\text{Relax using Pre-Determined Fixed Order}\\
\\
\text{Dijkstra}&&\text{Relax using Greedy Outgoing Edges}\\
\end{array}$$

In each of these repetitions, the number of vertices with correctly calculated distances grows, from which it follows that eventually all vertices will have their correct distances. This method allows the Bellman–Ford algorithm to be applied to a wider class of inputs than Dijkstra.

[Bellman–Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)

# Reference

[[알고리즘] 제16-2강 최단경로(shortest path problem) (계속)](https://www.youtube.com/watch?v=icqzGct4V1s&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=39)

[3.6 Dijkstra Algorithm - Single Source Shortest Path - Greedy Method](https://www.youtube.com/watch?v=XB4MIexjvY0&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=45)

[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)

[Dijkstra algorithm | Single source shortest path algorithm](https://www.youtube.com/watch?v=Sj5Z-jaE2x0)

[Dijkstra algorithm | Code implementation](https://www.youtube.com/watch?v=t2d-XYuPfg0)
