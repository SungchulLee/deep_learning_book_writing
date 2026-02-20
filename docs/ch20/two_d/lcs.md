# Longest Common Subsequence


$$
{\displaystyle {\mathit {LCS}}(X_{i},Y_{j})={\begin{cases}\emptyset &{\mbox{if }}i=0{\mbox{ or }}j=0\\{\mathit {LCS}}(X_{i-1},Y_{j-1}){\hat {}}x_{i}&{\mbox{if }}i,j>0{\mbox{ and }}x_{i}=y_{j}\\\operatorname {\max } \{{\mathit {LCS}}(X_{i},Y_{j-1}),{\mathit {LCS}}(X_{i-1},Y_{j})\}&{\mbox{if }}i,j>0{\mbox{ and }}x_{i}\neq y_{j}.\end{cases}}}
$$

To find the LCS of $X_{i}$ and $Y_j$, compare $x_{i}$ and $y_{j}$. 

If they are equal, then the sequence ${\displaystyle {\mathit {LCS}}(X_{i-1},Y_{j-1})}$ is extended by that element, $x_{i}$. 

If they are not equal, then the longer of the two sequences, ${\displaystyle {\mathit {LCS}}(X_{i},Y_{j-1})}$, and ${\displaystyle {\mathit {LCS}}(X_{i-1},Y_{j})}$, is retained. (If they are the same length, but not identical, then both are retained.)

[Longest common subsequence problem](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)


# Reference

[Longest common subsequence problem](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)

[4.9 Longest Common Subsequence (LCS) - Recursion and Dynamic Programming](https://www.youtube.com/watch?v=sSno9rV8Rhg&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=60)
