# LLM Agents Overview

## Learning Objectives

- Understand the LLM agent architecture
- Identify key components: planning, memory, tool use
- Classify agent architectures by complexity

## What is an LLM Agent?

An **LLM agent** is a system that uses a language model as a central reasoning engine to autonomously plan, execute actions, and interact with external tools to accomplish goals.

```
User Goal → LLM (Planning) → Action Selection → Tool Execution → Observation
                  ↑                                                    |
                  └──────────────── Feedback Loop ─────────────────────┘
```

## Key Components

### 1. Planning

The LLM decomposes complex goals into sub-tasks:

$$\text{Goal} \rightarrow \text{Plan} = (s_1, s_2, \ldots, s_n)$$

### 2. Memory

- **Short-term**: Conversation history and current task state
- **Long-term**: Retrieved knowledge from vector stores or databases
- **Working memory**: Intermediate results and scratchpad

### 3. Tool Use

External capabilities the agent can invoke:

```python
tools = {
    "search": "Search the web for information",
    "calculator": "Perform mathematical calculations",
    "code_executor": "Run Python code",
    "database_query": "Query SQL databases",
    "api_call": "Call external APIs",
}
```

### 4. Action Selection

The LLM decides which tool to use and with what arguments:

$$a_t = \text{LLM}(\text{goal}, \text{history}, \text{tools}, \text{observation}_{t-1})$$

## Financial Agent Applications

- **Research Agent**: Gathers data from multiple sources, synthesizes investment memos
- **Trading Signal Agent**: Monitors news feeds, extracts signals, generates alerts
- **Compliance Agent**: Reviews documents against regulatory requirements
- **Data Pipeline Agent**: Automates data collection, cleaning, and analysis

## References

1. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in LLMs." *ICLR*.
2. Wang, L., et al. (2024). "A Survey on LLM-based Autonomous Agents." *arXiv*.
