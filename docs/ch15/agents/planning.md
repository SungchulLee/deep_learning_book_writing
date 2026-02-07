# Planning

## Learning Objectives

- Understand task decomposition strategies
- Implement plan-and-execute architectures
- Apply reflexion for self-improvement

## Task Decomposition

Complex tasks require breaking down into manageable sub-tasks:

```python
def plan_and_execute(llm, task, tools):
    # Step 1: Generate plan
    plan_prompt = f"""Break down this task into sequential steps:
Task: {task}

Return a numbered list of steps, each achievable with available tools."""

    plan = llm(plan_prompt)
    steps = parse_plan(plan)

    # Step 2: Execute each step
    results = []
    for i, step in enumerate(steps):
        result = execute_step(llm, step, tools, context=results)
        results.append({"step": step, "result": result})

        # Step 3: Check if plan needs adjustment
        if needs_replanning(results):
            remaining = replan(llm, task, results, steps[i+1:])
            steps = steps[:i+1] + remaining

    # Step 4: Synthesize final answer
    return synthesize(llm, task, results)
```

## Reflexion

Shinn et al. (2023) introduced **Reflexion**: the agent reflects on failures and tries again with improved strategies:

```
Attempt 1 → Failure → Reflection: "I should have checked the date range"
Attempt 2 → Failure → Reflection: "The API returns JSON, not CSV"
Attempt 3 → Success
```

```python
def reflexion_loop(llm, task, tools, max_attempts=3):
    reflections = []

    for attempt in range(max_attempts):
        result = react_loop(llm, task, tools)

        if is_successful(result):
            return result

        # Reflect on failure
        reflection = llm(
            f"Task: {task}\n"
            f"Attempt: {result['trace']}\n"
            f"Previous reflections: {reflections}\n"
            f"What went wrong and how should I approach this differently?"
        )
        reflections.append(reflection)

    return {"status": "failed", "reflections": reflections}
```

## Financial Planning Example

```
Task: "Analyze the competitive landscape of cloud computing and
       recommend portfolio allocation among AMZN, MSFT, GOOGL"

Plan:
1. Retrieve latest revenue/growth data for each company's cloud segment
2. Compare market share and growth trajectories
3. Analyze pricing power and margin trends
4. Assess competitive moats and risks
5. Generate allocation recommendation with rationale
```

## References

1. Wang, L., et al. (2023). "Plan-and-Solve Prompting." *ACL*.
2. Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS*.
