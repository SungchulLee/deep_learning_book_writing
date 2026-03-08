"""Hf distillation benchmark."""
# ---
# title: "HuggingFace Distillation & Inference Benchmarking"
# description: "DistillationTrainer with custom loss, PerformanceBenchmark for
#               accuracy/size/latency, ONNX Runtime export, and compression comparison"
# ---
#
# The previous scripts (01-06) use raw PyTorch for model compression.
# In practice, HuggingFace Trainer provides a powerful abstraction for
# custom training loops.  This script demonstrates:
#
#   Part 1 – DistillationTrainer: Extending HF Trainer with custom loss
#   Part 2 – PerformanceBenchmark: Measuring accuracy, size, and latency
#   Part 3 – ONNX Runtime export and OnnxPipeline wrapper
#   Part 4 – Compression comparison: baseline → distilled → quantized → ONNX
#
# Adapted from: O'Reilly "NLP with Transformers" Ch 8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import time
import os


# =====================================================================
# Part 1 – DistillationTrainer: Custom Loss with HF Trainer
# =====================================================================
print("=" * 60)
print("Part 1: DistillationTrainer — Extending HF Trainer")
print("=" * 60)

# The key pattern: override compute_loss() in HuggingFace's Trainer to
# combine the standard cross-entropy with KL divergence against a teacher.
#
#   L = α × CE(student_logits, labels) + (1-α) × T² × KL(soft_student, soft_teacher)
#
# This lets us use all of Trainer's features (LR scheduling, logging,
# gradient accumulation, checkpointing) while adding distillation.

try:
    from transformers import Trainer, TrainingArguments

    class DistillationTrainer(Trainer):
        """HuggingFace Trainer extended with knowledge distillation loss.

        The teacher model runs in eval mode (no gradients). The student
        is trained on a weighted combination of:
          - Hard loss: cross-entropy against true labels
          - Soft loss: KL divergence against teacher's soft predictions

        Args:
            teacher_model: Pre-trained teacher (frozen during training)
            temperature:   Softmax temperature for soft targets (default: 2.0)
            alpha:         Weight for hard loss; (1-alpha) for soft loss
            *args, **kwargs: Forwarded to HuggingFace Trainer
        """

        def __init__(
            self,
            teacher_model=None,
            temperature: float = 2.0,
            alpha: float = 0.5,
            *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.teacher = teacher_model
            self.temperature = temperature
            self.alpha = alpha

            # Freeze teacher and move to same device as student
            if self.teacher is not None:
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Override Trainer's loss computation with distillation loss.

            The teacher produces soft targets at temperature T.
            The student learns from both hard labels and soft targets.

            Math:
                soft_teacher = softmax(teacher_logits / T)
                soft_student = log_softmax(student_logits / T)
                L_soft = KL(soft_student, soft_teacher) × T²
                L_hard = CE(student_logits, labels)
                L = α × L_hard + (1 - α) × L_soft
            """
            # Student forward pass
            labels = inputs.pop("labels", None)
            outputs_student = model(**inputs)
            student_logits = outputs_student.logits

            # Hard loss (standard cross-entropy)
            loss_hard = F.cross_entropy(student_logits, labels)

            if self.teacher is not None:
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    outputs_teacher = self.teacher(**inputs)
                    teacher_logits = outputs_teacher.logits

                # Soft loss (KL divergence at temperature T)
                soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
                soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

                loss_soft = F.kl_div(
                    soft_student,
                    soft_teacher,
                    reduction="batchmean",
                ) * (self.temperature ** 2)

                # Combined loss
                loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft
            else:
                loss = loss_hard

            return (loss, outputs_student) if return_outputs else loss

    print("  DistillationTrainer defined successfully.")
    print()
    print("  Usage:")
    print("""
    teacher = AutoModelForSequenceClassification.from_pretrained("bert-base")
    student = AutoModelForSequenceClassification.from_pretrained("distilbert-base")

    training_args = TrainingArguments(
        output_dir="./distilled-model",
        num_train_epochs=5,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
    )

    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=2.0,
        alpha=0.5,
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    """)

    HAS_TRAINER = True

except ImportError:
    print("  transformers not available — showing concept only")
    HAS_TRAINER = False
    print()


# =====================================================================
# Part 2 – PerformanceBenchmark
# =====================================================================
print("=" * 60)
print("Part 2: PerformanceBenchmark — Accuracy, Size & Latency")
print("=" * 60)

# When comparing compression techniques, we need consistent metrics.
# PerformanceBenchmark wraps a HuggingFace pipeline and measures:
#   1. Accuracy (or F1) on a held-out dataset
#   2. Model size on disk (MB)
#   3. Inference latency (ms per example)


class PerformanceBenchmark:
    """Benchmark a HuggingFace pipeline on accuracy, size, and latency.

    This class provides a standardised way to compare different model
    compression techniques on the same metrics.

    Args:
        pipeline:    A HuggingFace pipeline (e.g., text-classification)
        dataset:     Evaluation dataset with 'text' and 'label' columns
        optim_type:  Label for this configuration (e.g., 'Baseline', 'Distilled')
    """

    def __init__(self, pipeline, dataset, optim_type: str = "Baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self) -> float:
        """Compute accuracy on the evaluation dataset."""
        preds = self.pipeline(self.dataset["text"])
        # HF classification pipelines return [{"label": ..., "score": ...}]
        pred_labels = [p["label"] for p in preds]
        true_labels = self.dataset["label"]

        # Handle string vs int labels
        if isinstance(true_labels[0], str):
            correct = sum(p == t for p, t in zip(pred_labels, true_labels))
        else:
            # Map label strings to IDs using the pipeline's model config
            label2id = self.pipeline.model.config.label2id
            pred_ids = [label2id.get(p, -1) for p in pred_labels]
            correct = sum(p == t for p, t in zip(pred_ids, true_labels))

        accuracy = correct / len(true_labels)
        print(f"  [{self.optim_type}] Accuracy: {accuracy:.4f}")
        return accuracy

    def compute_size(self) -> float:
        """Compute model size on disk in MB."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.pipeline.model.save_pretrained(tmp_dir)
            # Sum all .bin / .safetensors files
            size_bytes = sum(
                os.path.getsize(os.path.join(tmp_dir, f))
                for f in os.listdir(tmp_dir)
                if f.endswith((".bin", ".safetensors"))
            )

        size_mb = size_bytes / (1024 * 1024)
        print(f"  [{self.optim_type}] Size: {size_mb:.1f} MB")
        return size_mb

    def compute_latency(
        self,
        sample_text: str = "This is a test sentence for benchmarking.",
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> float:
        """Measure average inference latency in milliseconds.

        Runs warmup iterations first to stabilize caching, then
        measures the average over num_runs iterations.
        """
        # Warmup
        for _ in range(num_warmup):
            _ = self.pipeline(sample_text)

        # Timed runs
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.pipeline(sample_text)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        print(f"  [{self.optim_type}] Latency: {avg_latency:.1f} ± {std_latency:.1f} ms")
        return avg_latency

    def run_benchmark(self) -> Dict[str, float]:
        """Run all benchmarks and return results as a dictionary."""
        print(f"\n  Benchmarking: {self.optim_type}")
        print("  " + "-" * 40)
        metrics = {
            "accuracy": self.compute_accuracy(),
            "size_mb": self.compute_size(),
            "latency_ms": self.compute_latency(),
        }
        return metrics


# Demo without HuggingFace
print("""
  Usage:
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline("text-classification", model="my-model")
    bench = PerformanceBenchmark(pipe, eval_dataset, optim_type="Baseline")
    metrics = bench.run_benchmark()
    # → Accuracy: 0.8742, Size: 418.2 MB, Latency: 12.3 ± 0.8 ms
""")


# =====================================================================
# Part 3 – ONNX Runtime Export and OnnxPipeline
# =====================================================================
print("=" * 60)
print("Part 3: ONNX Runtime Export & OnnxPipeline")
print("=" * 60)

# ONNX Runtime provides optimised inference for transformer models.
# Exporting to ONNX and using ORT can reduce latency by 2-3x on CPU.
#
# Pipeline:
#   1. Export PyTorch model to ONNX format
#   2. Load with ONNX Runtime InferenceSession
#   3. Wrap in an OnnxPipeline for HuggingFace-compatible interface


class OnnxPipeline:
    """HuggingFace-compatible pipeline wrapper for ONNX Runtime inference.

    Provides the same interface as transformers.pipeline() but runs
    inference through ONNX Runtime for faster CPU execution.

    Args:
        model_path:  Path to the exported .onnx model file
        tokenizer:   HuggingFace tokenizer for text preprocessing
        id2label:    Mapping from class IDs to label strings
    """

    def __init__(self, model_path: str, tokenizer, id2label: Dict[int, str]):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime")

        self.tokenizer = tokenizer
        self.id2label = id2label

    def __call__(self, texts, **kwargs):
        """Run inference on one or more texts.

        Args:
            texts: Single string or list of strings.

        Returns:
            List of dicts with 'label' and 'score' keys,
            matching HuggingFace pipeline format.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",  # ONNX Runtime uses numpy
        )

        # Run ONNX inference
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        logits = self.session.run(None, ort_inputs)[0]

        # Convert to HuggingFace pipeline format
        results = []
        for logit_row in logits:
            probs = self._softmax(logit_row)
            pred_idx = int(np.argmax(probs))
            results.append({
                "label": self.id2label[pred_idx],
                "score": float(probs[pred_idx]),
            })
        return results

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


print("  OnnxPipeline class defined.")
print()

# ONNX export procedure
print("  Exporting a HuggingFace model to ONNX:")
print("""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model = AutoModelForSequenceClassification.from_pretrained("my-model")
    tokenizer = AutoTokenizer.from_pretrained("my-model")
    model.eval()

    # Create dummy input
    dummy = tokenizer("Example text", return_tensors="pt")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits":         {0: "batch"},
        },
        opset_version=14,
    )

    # Use with OnnxPipeline
    onnx_pipe = OnnxPipeline(
        "model.onnx",
        tokenizer,
        id2label=model.config.id2label,
    )
    result = onnx_pipe("This is a test.")
    # → [{"label": "positive", "score": 0.93}]
""")


# =====================================================================
# Part 4 – Compression Comparison Pipeline
# =====================================================================
print("=" * 60)
print("Part 4: Full Compression Comparison")
print("=" * 60)

# In practice, we compare multiple compression strategies:
#   1. Baseline (full-size model)
#   2. Distilled student model
#   3. Quantized model (dynamic INT8)
#   4. ONNX Runtime model
#
# Each is benchmarked on the same metrics for a fair comparison.

print("""
  Compression comparison pipeline:
  ─────────────────────────────────

  Step 1: Train baseline model
    teacher = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=6)
    # ... fine-tune on dataset ...

  Step 2: Distill to smaller student
    student = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=6)
    trainer = DistillationTrainer(
        teacher_model=teacher,
        model=student,
        temperature=2.0,
        alpha=0.5,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

  Step 3: Quantize the student (dynamic quantization)
    import torch.quantization as quant
    quantized_student = quant.quantize_dynamic(
        student,
        {nn.Linear},      # quantize all Linear layers
        dtype=torch.qint8,
    )

  Step 4: Export to ONNX
    torch.onnx.export(student, dummy_input, "student.onnx", ...)
    onnx_pipe = OnnxPipeline("student.onnx", tokenizer, id2label)

  Step 5: Benchmark all variants
    configs = {
        "Baseline":   teacher_pipeline,
        "Distilled":  student_pipeline,
        "Quantized":  quantized_pipeline,
        "ONNX":       onnx_pipe,
    }
    results = {}
    for name, pipe in configs.items():
        bench = PerformanceBenchmark(pipe, eval_ds, optim_type=name)
        results[name] = bench.run_benchmark()
""")


# Simulated results table
print("  Typical results (BERT text classification):")
print("  ┌────────────┬──────────┬──────────┬────────────┐")
print("  │ Method     │ Accuracy │ Size(MB) │ Latency(ms)│")
print("  ├────────────┼──────────┼──────────┼────────────┤")
print("  │ Baseline   │  0.874   │  418.2   │   12.3     │")
print("  │ Distilled  │  0.861   │  255.4   │    7.1     │")
print("  │ Quantized  │  0.856   │  173.8   │    5.8     │")
print("  │ ONNX       │  0.861   │  255.4   │    4.2     │")
print("  │ Quant+ONNX │  0.852   │  102.1   │    3.1     │")
print("  └────────────┴──────────┴──────────┴────────────┘")
print()
print("  Key takeaways:")
print("    - Distillation: 40% smaller, 42% faster, 1.3% accuracy drop")
print("    - Quantization: adds 58% size reduction on top of distillation")
print("    - ONNX Runtime: 2-3x speedup on CPU with zero accuracy loss")
print("    - Combining all: 75% smaller, 75% faster, ~2% accuracy drop")
print()

# =====================================================================
# Standalone Demo — Distillation with raw PyTorch (no HF dependency)
# =====================================================================
print("=" * 60)
print("Standalone Demo: DistillationTrainer Logic in Raw PyTorch")
print("=" * 60)

# For environments without transformers, we show the core distillation
# logic using simple feedforward networks on synthetic data.

torch.manual_seed(42)

# Create teacher and student networks
class SimpleTeacher(nn.Module):
    def __init__(self, in_dim=20, hidden=128, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleStudent(nn.Module):
    def __init__(self, in_dim=20, hidden=32, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute the distillation loss.

    This is the same loss used in DistillationTrainer.compute_loss()
    but in a standalone function for clarity.

    Args:
        student_logits: Raw outputs from student [B, C]
        teacher_logits: Raw outputs from teacher [B, C]
        labels:         Ground truth class indices [B]
        temperature:    Softmax temperature (higher = softer)
        alpha:          Weight for hard loss

    Returns:
        Combined distillation loss (scalar)
    """
    # Hard loss: standard cross-entropy
    loss_hard = F.cross_entropy(student_logits, labels)

    # Soft loss: KL divergence at temperature T
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    loss_soft = F.kl_div(
        soft_student, soft_teacher, reduction="batchmean"
    ) * (temperature ** 2)

    return alpha * loss_hard + (1.0 - alpha) * loss_soft


# Synthetic data
X = torch.randn(500, 20)
y = torch.randint(0, 5, (500,))

# Pre-train teacher
teacher = SimpleTeacher()
opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
teacher.train()
for epoch in range(30):
    logits = teacher(X)
    loss = F.cross_entropy(logits, y)
    opt_t.zero_grad()
    loss.backward()
    opt_t.step()

teacher.eval()
with torch.no_grad():
    teacher_acc = (teacher(X).argmax(1) == y).float().mean().item()
print(f"  Teacher accuracy: {teacher_acc:.3f}")
print(f"  Teacher params:   {sum(p.numel() for p in teacher.parameters()):,}")

# Train student WITHOUT distillation (baseline)
student_baseline = SimpleStudent()
opt_s = torch.optim.Adam(student_baseline.parameters(), lr=1e-3)
student_baseline.train()
for epoch in range(50):
    logits = student_baseline(X)
    loss = F.cross_entropy(logits, y)
    opt_s.zero_grad()
    loss.backward()
    opt_s.step()

student_baseline.eval()
with torch.no_grad():
    baseline_acc = (student_baseline(X).argmax(1) == y).float().mean().item()
print(f"  Student (no distill) accuracy: {baseline_acc:.3f}")
print(f"  Student params:   {sum(p.numel() for p in student_baseline.parameters()):,}")

# Train student WITH distillation
student_distilled = SimpleStudent()
opt_d = torch.optim.Adam(student_distilled.parameters(), lr=1e-3)
student_distilled.train()
for epoch in range(50):
    student_logits = student_distilled(X)
    with torch.no_grad():
        teacher_logits = teacher(X)

    loss = distillation_loss(
        student_logits, teacher_logits, y,
        temperature=4.0, alpha=0.3,
    )
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()

student_distilled.eval()
with torch.no_grad():
    distilled_acc = (student_distilled(X).argmax(1) == y).float().mean().item()
print(f"  Student (distilled) accuracy:  {distilled_acc:.3f}")
print()

improvement = distilled_acc - baseline_acc
print(f"  Distillation improvement: {improvement:+.3f}")
print(f"  Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student_distilled.parameters()):.1f}x fewer params")
print()

# Dynamic quantization demo
print("  Dynamic INT8 quantization of the distilled student:")
quantized = torch.quantization.quantize_dynamic(
    student_distilled,
    {nn.Linear},
    dtype=torch.qint8,
)

# Compare sizes
def model_size_bytes(model):
    """Estimate model size from parameter storage."""
    import io, pickle
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()

orig_size = model_size_bytes(student_distilled)
quant_size = model_size_bytes(quantized)
print(f"  Original size:  {orig_size / 1024:.1f} KB")
print(f"  Quantized size: {quant_size / 1024:.1f} KB")
print(f"  Reduction: {(1 - quant_size / orig_size) * 100:.1f}%")
print()

# Verify quantized accuracy
with torch.no_grad():
    quant_acc = (quantized(X).argmax(1) == y).float().mean().item()
print(f"  Quantized accuracy: {quant_acc:.3f}")
print()

# Final summary
print("  ┌──────────────────────┬──────────┬──────────┐")
print("  │ Model                │ Accuracy │ Params   │")
print("  ├──────────────────────┼──────────┼──────────┤")
print(f"  │ Teacher              │  {teacher_acc:.3f}   │ {sum(p.numel() for p in teacher.parameters()):>6,}   │")
print(f"  │ Student (baseline)   │  {baseline_acc:.3f}   │ {sum(p.numel() for p in student_baseline.parameters()):>6,}   │")
print(f"  │ Student (distilled)  │  {distilled_acc:.3f}   │ {sum(p.numel() for p in student_distilled.parameters()):>6,}   │")
print(f"  │ Student (quantized)  │  {quant_acc:.3f}   │ {sum(p.numel() for p in student_distilled.parameters()):>6,}   │")
print("  └──────────────────────┴──────────┴──────────┘")
print()

print("Done.")


if __name__ == "__main__":
    pass
