# 0/1 Knapsack


<div align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Knapsack.svg/500px-Knapsack.svg.png" width="50%"></div>

[배낭 문제](https://ko.wikipedia.org/wiki/%EB%B0%B0%EB%82%AD_%EB%AC%B8%EC%A0%9C)


$$\begin{array}{ccccccccccccc}
m&[&i&,&w&]\\
\uparrow&&\uparrow&&\uparrow&\\
\text{max total value}&&\text{all the first $i$ items considered}&&\text{weight limit}&\\
\end{array}$$


$$\begin{array}{llll}
m[0,w]&=&0\\
\\
m[i,w]&=&m[i−1,w]&\text{if $w_i>w$}\\
\\
m[i,w]&=&\max\left(m[i−1,w],m[i−1,w−w_i]+p_i\right)&\text{if $w_i\le w$}
\end{array}$$


# Reference

[3.1 Knapsack Problem - Greedy Method](https://www.youtube.com/watch?v=oTTzNMHM05I&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=40)

[배낭 문제](https://ko.wikipedia.org/wiki/%EB%B0%B0%EB%82%AD_%EB%AC%B8%EC%A0%9C)
