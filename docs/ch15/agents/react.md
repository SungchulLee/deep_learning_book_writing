# ReAct: Reasoning + Acting

## Learning Objectives

- Understand the ReAct paradigm
- Implement the Thought-Action-Observation loop
- Apply ReAct to financial research tasks

## Core Idea

**ReAct** (Yao et al., 2023) interleaves reasoning traces with actions:

```
Thought: I need to find AAPL's latest earnings data
Action: search("AAPL Q3 2024 earnings results")
Observation: AAPL reported Q3 revenue of $85.8B...

Thought: Now I need to compare with analyst estimates
Action: search("AAPL Q3 2024 consensus estimate")
Observation: Consensus estimate was $84.5B...

Thought: Revenue beat by $1.3B (1.5%). Let me check guidance.
Action: search("AAPL Q4 2024 guidance")
Observation: Apple guided Q4 revenue of $89-93B...

Thought: I now have enough information to provide analysis.
Answer: AAPL beat Q3 estimates by 1.5%...
```

## Implementation

```python
def react_loop(llm, question, tools, max_steps=10):
    prompt = f"""Answer the following question using the available tools.

Format your response as:
Thought: [your reasoning]
Action: [tool_name(arguments)]
(wait for Observation)
... repeat as needed ...
Thought: I have enough information.
Answer: [final answer]

Question: {question}
"""
    history = prompt

    for step in range(max_steps):
        response = llm(history)
        history += response

        # Check if we have a final answer
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
            return {"answer": answer, "steps": step + 1, "trace": history}

        # Extract and execute action
        if "Action:" in response:
            action_str = response.split("Action:")[-1].strip().split("\n")[0]
            tool_name, args = parse_action(action_str)
            observation = tools[tool_name](**args)
            history += f"\nObservation: {observation}\n"

    return {"answer": "Could not determine answer", "steps": max_steps}
```

## ReAct vs. Other Approaches

| Method | Reasoning | Acting | Grounding |
|--------|----------|--------|-----------|
| CoT only | Yes | No | No |
| Action only | No | Yes | Yes |
| ReAct | Yes | Yes | Yes |

ReAct reduces hallucination by grounding reasoning in retrieved facts.

## References

1. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*.
