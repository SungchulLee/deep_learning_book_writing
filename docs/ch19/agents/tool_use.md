# Tool Use

## Learning Objectives

- Understand how LLMs interact with external tools
- Define tool schemas for LLM consumption
- Implement a tool execution framework

## Tool Definition

Tools are defined with structured schemas that the LLM can understand:

```python
tool_schema = {
    "name": "get_stock_price",
    "description": "Get the current stock price for a given ticker symbol",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
            },
            "date": {
                "type": "string",
                "description": "Date in YYYY-MM-DD format. Default: today"
            }
        },
        "required": ["ticker"]
    }
}
```

## Tool Selection

The LLM selects tools based on the task requirements:

$$\text{tool}_t = \arg\max_{\text{tool} \in \mathcal{T}} P(\text{tool} \mid q, \text{history})$$

## Execution Framework

```python
class ToolExecutor:
    def __init__(self):
        self.tools = {}

    def register(self, name, func, schema):
        self.tools[name] = {"func": func, "schema": schema}

    def execute(self, tool_name, arguments):
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            result = self.tools[tool_name]["func"](**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def get_schemas(self):
        return [
            {"name": name, **tool["schema"]}
            for name, tool in self.tools.items()
        ]


# Register financial tools
executor = ToolExecutor()
executor.register("get_stock_price", get_stock_price, tool_schema)
executor.register("calculate_ratio", calculate_ratio, ratio_schema)
executor.register("query_sec_filing", query_sec_filing, sec_schema)
```

## References

1. Schick, T., et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." *NeurIPS*.
2. Qin, Y., et al. (2023). "Tool Learning with Foundation Models." *arXiv*.
