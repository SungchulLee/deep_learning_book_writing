# Multi-Agent Systems

## Learning Objectives

- Understand multi-agent architectures
- Compare supervisor, debate, and assembly line patterns
- Implement a financial analysis team

## Architectures

### Supervisor Pattern

A supervisor agent delegates tasks to specialist agents:

```
Supervisor → Research Agent → Data
           → Analysis Agent → Insights
           → Writing Agent → Report
```

### Debate Pattern

Multiple agents argue different positions, then a judge synthesizes:

```
Bull Agent → "Strong buy thesis based on..."
Bear Agent → "Key risks include..."
Judge → Balanced analysis incorporating both views
```

### Assembly Line Pattern

Agents process sequentially, each adding to the output:

```
Data Agent → Feature Agent → Model Agent → Report Agent
```

## Implementation

```python
class FinancialAnalysisTeam:
    def __init__(self, llm):
        self.llm = llm
        self.agents = {
            "researcher": "You are a financial researcher. Find and verify data.",
            "analyst": "You are a quantitative analyst. Analyze data and metrics.",
            "risk": "You are a risk analyst. Identify and quantify risks.",
            "writer": "You are a financial writer. Produce clear, concise reports.",
        }

    def analyze(self, query):
        # Step 1: Research
        research = self.run_agent("researcher",
            f"Research the following: {query}")

        # Step 2: Analysis
        analysis = self.run_agent("analyst",
            f"Analyze this data:\n{research}\nQuery: {query}")

        # Step 3: Risk assessment
        risks = self.run_agent("risk",
            f"Identify risks:\n{research}\n{analysis}")

        # Step 4: Final report
        report = self.run_agent("writer",
            f"Write a report combining:\n"
            f"Research: {research}\n"
            f"Analysis: {analysis}\n"
            f"Risks: {risks}")

        return report

    def run_agent(self, agent_name, prompt):
        system = self.agents[agent_name]
        return self.llm(system_prompt=system, user_prompt=prompt)
```

## Frameworks

| Framework | Architecture | Key Feature |
|-----------|-------------|-------------|
| LangGraph | Graph-based | Flexible state machines |
| CrewAI | Role-based | Simple multi-agent setup |
| AutoGen | Conversational | Agent-to-agent chat |
| MetaGPT | SOP-based | Software development focus |

## References

1. Wu, Q., et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications." *arXiv*.
2. Hong, S., et al. (2023). "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework." *arXiv*.
