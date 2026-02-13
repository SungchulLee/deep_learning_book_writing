# Function Calling

## Learning Objectives

- Understand API-level function calling mechanisms
- Implement function calling with structured outputs
- Handle multi-turn function calling conversations

## How Function Calling Works

Modern LLM APIs support structured function calling:

1. **Define functions**: Provide JSON schemas describing available functions
2. **LLM decides**: The model decides when and which function to call
3. **Execute**: The application executes the function with provided arguments
4. **Return result**: Feed the result back to the LLM for further reasoning

```python
import json


def function_calling_loop(llm, messages, functions, max_iterations=10):
    for _ in range(max_iterations):
        response = llm.chat(messages=messages, functions=functions)

        # Check if the model wants to call a function
        if response.function_call:
            func_name = response.function_call.name
            func_args = json.loads(response.function_call.arguments)

            # Execute the function
            result = execute_function(func_name, func_args)

            # Add function result to conversation
            messages.append({"role": "assistant", "content": None,
                           "function_call": response.function_call})
            messages.append({"role": "function", "name": func_name,
                           "content": json.dumps(result)})
        else:
            # Model produced a final text response
            return response.content

    return "Max iterations reached"
```

## Multi-Turn Example

```
User: "Compare AAPL and MSFT P/E ratios"

LLM → function_call: get_financial_metric(ticker="AAPL", metric="pe_ratio")
Result: {"ticker": "AAPL", "pe_ratio": 28.5}

LLM → function_call: get_financial_metric(ticker="MSFT", metric="pe_ratio")
Result: {"ticker": "MSFT", "pe_ratio": 34.2}

LLM: "AAPL has a P/E ratio of 28.5, while MSFT's is 34.2.
      MSFT trades at a ~20% premium, likely reflecting higher
      growth expectations in cloud and AI segments."
```

## Parallel Function Calling

Some APIs support calling multiple functions simultaneously:

```python
# Model returns multiple function calls at once
function_calls = [
    {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}},
    {"name": "get_stock_price", "arguments": {"ticker": "MSFT"}},
    {"name": "get_stock_price", "arguments": {"ticker": "GOOGL"}},
]
# Execute all in parallel
results = await asyncio.gather(*[
    execute_function(fc["name"], fc["arguments"])
    for fc in function_calls
])
```

## References

1. OpenAI. (2023). "Function Calling and Other API Updates."
2. Anthropic. (2024). "Tool Use Documentation."
