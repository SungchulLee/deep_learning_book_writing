<<<<<<< HEAD
# Chapter Overview

This chapter covers **Greedy Algorithms**.

# Reference

[Introduction to Algorithms (CLRS), Chapter 16](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 19: Large Language Models

This chapter provides a comprehensive treatment of large language models, covering their foundations, architectures, scaling laws, prompting techniques, retrieval-augmented generation, agentic systems, parameter-efficient fine-tuning, inference optimization, and alignment methods. Each section combines mathematical rigor with practical implementation guidance.

---

## LLM Foundations

Core concepts underlying large language models, from pretraining objectives to evaluation and conversational AI.

- [LLM Overview](llm_foundations/llm_overview.md) -- Definition, capabilities, and limitations of large language models
- [Origins and Development](llm_foundations/origins_development.md) -- Historical evolution from n-grams to modern LLM families
- [Decoder Architecture](llm_foundations/decoder_architecture.md) -- Decoder-only Transformer with causal self-attention
- [Next-Token Prediction](llm_foundations/next_token_prediction.md) -- Autoregressive language modeling objective and decoding strategies
- [Pretraining Objectives](llm_foundations/pretraining_objectives.md) -- CLM vs MLM, denoising objectives, and modern strategies like UL2
- [Tokenization and Scale](llm_foundations/tokenization_scale.md) -- BPE, WordPiece, SentencePiece, and vocabulary size trade-offs
- [Training Data at Scale](llm_foundations/training_data.md) -- Data curation, filtering, deduplication, and contamination risks
- [Evaluation Metrics](llm_foundations/evaluation_metrics.md) -- Intrinsic, task-specific, and human evaluation of LLMs
- [Conversational AI](llm_foundations/conversational_ai.md) -- Multi-turn dialogue systems from rule-based chatbots to LLM assistants
- [Challenges and Limitations](llm_foundations/challenges.md) -- Context management, factual reliability, bias, and ethical deployment
- [Future Directions](llm_foundations/future_directions.md) -- Architectural efficiency, reasoning, multimodal integration, and agentic behavior
- [Exercises](llm_foundations/exercises.md) -- Hands-on exercises for building and evaluating conversational AI systems

---

## Architectures

Key architectural innovations and model families in large language models.

- [Architectural Innovations](architectures/architectures.md) -- RMSNorm, SwiGLU, Rotary Embeddings, and Grouped-Query Attention
- [GPT Series](architectures/gpt_series.md) -- Evolution from GPT-1 through GPT-4 with emergent capabilities
- [LLaMA Family](architectures/llama.md) -- Open foundation models prioritizing inference efficiency

---

## Scaling

Empirical scaling laws governing LLM performance and compute-optimal training.

- [Scaling Laws](scaling/scaling_overview.md) -- Empirical laws governing model performance vs compute
- [Compute-Optimal Training](scaling/compute_optimal.md) -- Optimal allocation between parameters and training tokens
- [Chinchilla Scaling](scaling/chinchilla.md) -- Chinchilla loss parametrization and predictions
- [Model Size vs Data Trade-offs](scaling/model_vs_data.md) -- Three scaling regimes and data repetition effects
- [Emergent Abilities](scaling/emergent_abilities.md) -- Capabilities that appear abruptly at scale

---

## Prompting

Techniques for eliciting desired behavior from LLMs through prompt design.

- [Prompting Overview](prompting/prompting_overview.md) -- The paradigm shift from fine-tuning to prompting
- [Prompt Engineering](prompting/prompt_engineering.md) -- Core principles and structural patterns for effective prompts
- [Zero-Shot Prompting](prompting/zero_shot.md) -- Task completion without demonstration examples
- [Few-Shot Prompting](prompting/few_shot.md) -- In-context learning with example selection and ordering
- [Chain-of-Thought](prompting/chain_of_thought.md) -- Step-by-step reasoning for improved performance
- [Self-Consistency](prompting/self_consistency.md) -- Majority voting over multiple reasoning paths
- [Tree-of-Thought](prompting/tree_of_thought.md) -- Branching exploration of reasoning strategies
- [Prompt Optimization](prompting/prompt_optimization.md) -- Automated discrete and continuous prompt search

---

## RAG

Retrieval-Augmented Generation for grounding LLM outputs in external knowledge.

- [RAG Overview](rag/rag_overview.md) -- Architecture, motivation, and when RAG beats fine-tuning
- [Dense Retrieval](rag/dense_retrieval.md) -- Bi-encoder architecture and contrastive training
- [Vector Databases](rag/vector_databases.md) -- FAISS and vector DB technologies for similarity search
- [Document Chunking](rag/chunking.md) -- Chunking strategies and trade-offs for financial documents
- [Retriever-Reader Architecture](rag/retriever_reader.md) -- Complete RAG pipeline with naive, iterative, and adaptive variants
- [Reranking](rag/reranking.md) -- Two-stage retrieval with cross-encoder reranking
- [RAG Evaluation](rag/evaluation.md) -- Retrieval and generation quality metrics for RAG systems

---

## Agents

LLM-powered autonomous systems with planning, tool use, and multi-agent collaboration.

- [Agents Overview](agents/agent_overview.md) -- LLM agent architecture with planning, memory, and tool use
- [Tool Use](agents/tool_use.md) -- Defining tool schemas and building execution frameworks
- [Function Calling](agents/function_calling.md) -- API-level structured function calling mechanisms
- [ReAct](agents/react.md) -- Reasoning + Acting with Thought-Action-Observation loops
- [Planning](agents/planning.md) -- Task decomposition and plan-and-execute architectures
- [Multi-Agent Systems](agents/multi_agent.md) -- Supervisor, debate, and assembly line patterns
- [Case Studies](agents/case_studies.md) -- Real-world conversational AI deployments across industries
- [LLM Applications](agents/llm_applications.md) -- Practical applications of ChatGPT across domains

---

## Efficient LLM

Parameter-efficient fine-tuning methods for adapting LLMs with minimal trainable parameters.

- [Efficiency Overview](efficient_llm/efficiency_overview.md) -- Why PEFT is necessary and classification of methods
- [LoRA](efficient_llm/lora.md) -- Low-Rank Adaptation with mathematical foundations and implementation
- [QLoRA](efficient_llm/qlora.md) -- Quantized LoRA enabling fine-tuning on consumer hardware
- [Adapter Layers](efficient_llm/adapters.md) -- Serial, parallel, and AdapterFusion bottleneck modules
- [Prefix Tuning](efficient_llm/prefix_tuning.md) -- Soft prefixes for adapting models without weight modification
- [Prompt Tuning](efficient_llm/prompt_tuning.md) -- Continuous soft prompt optimization for task adaptation
- [BitFit](efficient_llm/bitfit.md) -- Bias-only fine-tuning for minimal parameter updates
- [PEFT Comparison](efficient_llm/peft_comparison.md) -- Comparative analysis of LoRA, QLoRA, adapters, and prefix tuning

---

## Inference

Optimizing LLM serving for throughput, latency, and memory efficiency.

- [Inference Overview](inference/inference_overview.md) -- Memory-bandwidth bottleneck, prefill vs decode phases
- [KV-Cache](inference/kv_cache.md) -- Caching key-value tensors to eliminate redundant computation
- [Flash Attention](inference/flash_attention.md) -- IO-aware exact attention with O(N) memory
- [Quantization](inference/quantization.md) -- Weight and activation quantization for deployment
- [Speculative Decoding](inference/speculative_decoding.md) -- Draft-then-verify acceleration with smaller models
- [Paged Attention](inference/paged_attention.md) -- Virtual memory concepts for KV cache management (vLLM)
- [Continuous Batching](inference/continuous_batching.md) -- Iteration-level scheduling for improved throughput
- [Model Sharding](inference/model_sharding.md) -- Distributing model weights across multiple devices
- [Tensor Parallelism](inference/tensor_parallelism.md) -- Column and row parallelism for linear and attention layers
- [Pipeline Parallelism](inference/pipeline_parallelism.md) -- Layer-wise distribution with pipeline bubble analysis
- [Model Compression](inference/model_compression_overview.md) -- Pruning, quantization, and distillation for deployment

---

## Alignment

Aligning LLM behavior with human preferences through RLHF and alternative methods.

- [Alignment Overview](alignment/alignment_overview.md) -- Why alignment is needed and the three-stage pipeline
- [RLHF](alignment/rlhf.md) -- Three-stage Reinforcement Learning from Human Feedback pipeline
- [Reward Modeling](alignment/reward_modeling.md) -- Bradley-Terry preference model and reward architecture
- [PPO for LLMs](alignment/ppo_llm.md) -- Proximal Policy Optimization adapted for language model training
- [DPO](alignment/dpo.md) -- Direct Preference Optimization without explicit reward models
- [Constitutional AI](alignment/constitutional.md) -- Self-supervised alignment with constitutional principles
- [Training Pipeline](alignment/training_pipeline.md) -- Pre-training and alignment phases with optimization strategies
>>>>>>> 96f31bd (...)
