# C++ for Algorithms

C++ offers high performance and STL containers, making it popular for competitive programming and systems algorithms.

## Key STL Containers

$$
\begin{array}{lll}
\texttt{vector} & \text{Dynamic array} & O(1) \text{ amortized append} \\
\texttt{set/map} & \text{Balanced BST} & O(\log n) \text{ operations} \\
\texttt{unordered\_set/map} & \text{Hash table} & O(1) \text{ average} \\
\texttt{priority\_queue} & \text{Max-heap} & O(\log n) \text{ push/pop} \\
\texttt{deque} & \text{Double-ended queue} & O(1) \text{ both ends}
\end{array}
$$

## Example: Two Sum in C++

```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }
    return {};
}

int main() {
    vector<int> nums = {2, 7, 11, 15};
    auto result = twoSum(nums, 9);
    cout << "[" << result[0] << ", " << result[1] << "]" << endl;
    return 0;
}
```

**Output:**
```
[0, 1]
```


# Reference

[C++ Reference — Containers](https://en.cppreference.com/w/cpp/container)

[Competitive Programmer's Handbook](https://cses.fi/book/book.pdf)
