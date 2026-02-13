# Retriever-Reader Architecture

## Learning Objectives

- Understand the separation of retriever and reader components
- Implement a complete RAG pipeline
- Compare RAG variants: naive, iterative, adaptive

## Architecture

The retriever-reader pattern separates concerns:

1. **Retriever**: Finds relevant documents (fast, approximate)
2. **Reader/Generator**: Produces answers from retrieved context (slow, accurate)

```python
class RAGPipeline:
    def __init__(self, retriever, generator, top_k=5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def __call__(self, query: str) -> str:
        # Step 1: Retrieve relevant documents
        documents = self.retriever.search(query, k=self.top_k)

        # Step 2: Build augmented prompt
        context = "\n\n".join([
            f"[Source {i+1}]: {doc['text']}"
            for i, doc in enumerate(documents)
        ])

        prompt = f"""Answer the question based on the provided context.
If the answer is not in the context, say "Not found in provided documents."

Context:
{context}

Question: {query}

Answer:"""

        # Step 3: Generate answer
        response = self.generator(prompt)
        return response
```

## RAG Variants

### Naive RAG

Single retrieval step followed by generation. Simple but may miss relevant information if the initial query doesn't capture all aspects.

### Iterative RAG

Multiple retrieval rounds, refining queries based on intermediate results:

```python
def iterative_rag(query, retriever, generator, max_iterations=3):
    context = []
    current_query = query

    for i in range(max_iterations):
        new_docs = retriever.search(current_query, k=3)
        context.extend(new_docs)

        # Generate intermediate answer
        answer = generator(build_prompt(query, context))

        # Check if answer is sufficient
        if is_confident(answer):
            return answer

        # Refine query based on what's missing
        current_query = generator(
            f"Original question: {query}\n"
            f"Current answer: {answer}\n"
            f"What additional information is needed? Generate a search query."
        )

    return answer
```

### Adaptive RAG

Dynamically decides whether retrieval is needed based on the query:

- **No retrieval**: Factual questions the LLM can answer directly
- **Single retrieval**: Straightforward lookup queries
- **Multi-step retrieval**: Complex questions requiring multiple sources

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.
2. Jiang, Z., et al. (2023). "Active Retrieval Augmented Generation." *EMNLP*.
