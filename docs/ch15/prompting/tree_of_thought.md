# Tree-of-Thought Prompting

## Learning Objectives

- Understand the tree-of-thought (ToT) framework
- Compare ToT with chain-of-thought and self-consistency
- Implement a basic ToT reasoning system

## Core Idea

**Tree-of-Thought (ToT)** (Yao et al., 2023) generalizes CoT by exploring **multiple reasoning paths** simultaneously using search algorithms:

```
                    Problem
                   /   |   \
              Thought1  T2  T3        (Generate k candidates)
              /    \     |
           T1.1  T1.2  T2.1          (Expand promising nodes)
            |
         Solution                     (Best path)
```

## Algorithm

1. **Thought Generation**: Generate $k$ candidate intermediate steps
2. **State Evaluation**: Score each thought using the LLM as value function
3. **Search**: BFS or DFS with pruning of low-value branches

```python
def tree_of_thought(problem, llm, n_candidates=3, max_depth=5, beam_width=3):
    active_paths = [{"thoughts": [], "state": problem}]

    for depth in range(max_depth):
        all_candidates = []
        for path in active_paths:
            candidates = generate_thoughts(llm, path["state"], n=n_candidates)
            for thought in candidates:
                score = evaluate_thought(llm, path["state"], thought)
                new_path = {
                    "thoughts": path["thoughts"] + [thought],
                    "state": update_state(path["state"], thought),
                    "score": score,
                }
                all_candidates.append(new_path)

        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        active_paths = all_candidates[:beam_width]

        for path in active_paths:
            if is_solution(path["state"]):
                return path

    return active_paths[0]
```

## Comparison

| Method | Structure | Search | Cost |
|--------|----------|--------|------|
| Standard | Direct | None | 1x |
| CoT | Single chain | None | 1x |
| Self-Consistency | Multiple chains | Majority vote | kx |
| Tree-of-Thought | Tree with eval | BFS/DFS | 10-50x |

## References

1. Yao, S., et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with LLMs." *NeurIPS*.
