# Conditional Statements

When an algorithm reaches a branch point — an `if`, `if-else`, or multiway
conditional — the number of operations it performs depends on which branch executes.
Since different inputs trigger different branches, analyzing conditionals is where the
distinction between best-case, worst-case, and average-case running time becomes
concrete.

## The Analysis Rule

Consider an `if-else` statement where the condition check takes $T_{\text{cond}}$
time, the `if` branch takes $T_{\text{if}}$ time, and the `else` branch takes
$T_{\text{else}}$ time.

**Worst-case rule.** Take the more expensive branch:

$$
T_{\text{worst}} = T_{\text{cond}} + \max(T_{\text{if}},\; T_{\text{else}})
$$

**Best-case rule.** Take the cheaper branch:

$$
T_{\text{best}} = T_{\text{cond}} + \min(T_{\text{if}},\; T_{\text{else}})
$$

**Average-case rule.** Weight each branch by its probability. Let $p$ be the
probability that the condition is true:

$$
T_{\text{avg}} = T_{\text{cond}} + p \cdot T_{\text{if}} + (1 - p) \cdot T_{\text{else}}
$$

!!! tip "When Branches Have Equal Cost"

    If both branches take the same asymptotic time, the conditional does not
    introduce case-dependent behavior. For instance, if
    $T_{\text{if}} = \Theta(n)$ and $T_{\text{else}} = \Theta(n)$, then the overall
    cost is $\Theta(n)$ regardless of which branch executes.

## Simple If Statement

An `if` statement without an `else` branch is a special case where
$T_{\text{else}} = 0$:

```
if condition:
    body
```

- **Worst case:** $T_{\text{cond}} + T_{\text{body}}$ (condition is true)
- **Best case:** $T_{\text{cond}}$ (condition is false, body skipped)

??? example "Finding the Maximum Element"

    ```
    FindMax(A, n):
        max_val = A[0]              # O(1)
        for i = 1 to n - 1:        # O(n) iterations
            if A[i] > max_val:      #   O(1) comparison
                max_val = A[i]      #   O(1) assignment
        return max_val              # O(1)
    ```

    The `if` statement on line 4 executes $n - 1$ times. In the **worst case** (array
    in ascending order), the assignment executes every iteration, costing $O(1)$ each
    time. In the **best case** (first element is the maximum), the assignment never
    executes. Either way, the loop runs $n - 1$ times, so the total is $\Theta(n)$.

## Multiway Conditionals

For an `if-elseif-else` chain with $k$ branches, the worst-case rule generalizes to:

$$
T_{\text{worst}} = T_{\text{cond}_1} + T_{\text{cond}_2} + \cdots + T_{\text{cond}_k} + \max_{1 \leq j \leq k} T_{\text{branch}_j}
$$

Note that the condition checks are evaluated sequentially until one succeeds.
In the worst case, we may evaluate all conditions before reaching the last branch.

??? example "Classifying a Number"

    ```
    Classify(x):
        if x > 0:                   # O(1)
            result = "positive"     # O(1)
        else if x < 0:              # O(1)
            result = "negative"     # O(1)
        else:
            result = "zero"         # O(1)
        return result               # O(1)
    ```

    All branches take $\Theta(1)$ time, and at most two conditions are checked. The
    entire construct runs in $\Theta(1)$.

## Conditionals Inside Loops

When a conditional appears inside a loop, the analysis depends on how often each
branch executes across all iterations. This interaction often determines whether a
loop runs in $O(n)$ or $O(n^2)$.

??? example "Conditional with Variable-Cost Branch"

    ```
    Process(A, n):
        for i = 0 to n - 1:
            if A[i] == 0:
                Scan(A, n)          # O(n)
            else:
                A[i] = A[i] + 1     # O(1)
    ```

    - **Best case:** No zeros in `A`. Every iteration takes $O(1)$, giving
      $T_{\text{best}}(n) = \Theta(n)$.
    - **Worst case:** Every element is zero. Each of the $n$ iterations calls
      `Scan` at cost $O(n)$, giving $T_{\text{worst}}(n) = \Theta(n^2)$.
    - **Average case:** If each element is zero with probability $p$, the
      expected number of `Scan` calls is $pn$, giving
      $T_{\text{avg}}(n) = n \cdot [p \cdot O(n) + (1 - p) \cdot O(1)] = O(pn^2 + n)$.

## Common Pitfall: Ignoring the Condition Cost

The cost of evaluating the condition itself is sometimes nontrivial. If the condition
involves a function call or a comparison over a data structure, it must be included in
the analysis.

!!! warning "Do Not Assume Conditions Are Free"

    ```
    if IsSorted(A, n):      # O(n) to check
        return A             # O(1)
    else:
        Sort(A, n)           # O(n log n)
    ```

    The worst-case cost is $O(n) + O(n \log n) = O(n \log n)$, not just $O(n \log n)$.
    The condition check contributes $O(n)$ even before the branch executes.

## Reference

- [Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
