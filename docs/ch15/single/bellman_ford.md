# Bellman-Ford

$$

\begin{array}{lccllll}
&\text{Graph}&\text{Complexity}\\
\text{BFS}&w=1\\
\text{Bellana Ford}&\text{Negative weights are allowed}&O(nm)\\
\text{Dijkstra}&w\ge 0&O(n^2)\ \text{or}\ O(m\log n)\\
\end{array}

$$

<img src="img/Screen Shot 2022-05-02 at 12.22.53 PM.png" width="70%">
<img src="img/Screen Shot 2022-05-02 at 12.23.40 PM.png" width="70%">

Like Dijkstra's algorithm, Bellman–Ford proceeds by relaxation, in which approximations to the correct distance are replaced by better ones until they eventually reach the solution. In both algorithms, the approximate distance to each vertex is always an overestimate of the true distance, and is replaced by the minimum of its old value and the length of a newly found path. However, Dijkstra's algorithm uses a priority queue to greedily select the closest vertex that has not yet been processed, and performs this relaxation process on all of its outgoing edges; by contrast, the Bellman–Ford algorithm simply relaxes all the edges, and does this $|V|-1$ times, where $|V|$ is the number of vertices in the graph. 

$$\begin{array}{lll}
\text{Bellman Ford}&&\text{Relax using Pre-Determined Fixed Order}\\
\\
\text{Dijkstra}&&\text{Relax using Greedy Outgoing Edges}\\
\end{array}$$

In each of these repetitions, the number of vertices with correctly calculated distances grows, from which it follows that eventually all vertices will have their correct distances. This method allows the Bellman–Ford algorithm to be applied to a wider class of inputs than Dijkstra.

[Bellman–Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)

# Reference

[[알고리즘] 제16-1강 최단경로(shortest path problem)](https://www.youtube.com/watch?v=QH-Btq8SgLQ&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=38)

[3.6 Dijkstra Algorithm - Single Source Shortest Path - Greedy Method](https://www.youtube.com/watch?v=XB4MIexjvY0&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=45)

[Bellman–Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)
